# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

Data flow in the current WIP version:
- self.device = CPU. Buffers and scatter ops stay on CPU.
- _SpyreModelWrapper converts input_ids/positions to Spyre int64 at the
  model call boundary.
- _SpyreModelWrapper converts final hidden_states to CPU for downstream
  operations (logits indexing, lm_head, sampling).
- Embedding: Spyre int64 input → Spyre compute → float16 output on Spyre.
- Hidden states flow on Spyre between decoder layers.
- There are few exceptions where a CPU fallback is currently needed:
  - Attention block: Spyre input → CPU (and partial Spyre) compute → Spyre output.
  - Layers that are not yet wrapped for torch-spyre,
    for example RotaryEmbedding

As the TorchSpyreModelRunner is evolving, more layers will natively support inputs
arriving as a Spyre tensor and perform their operations on Spyre.
Thus, in the final state of the runner minimal D2H and H2D transfers will be necessary,
the SpyreCpuFallbackMixin will be obsolete and most operations will be performed on Spyre.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

import numpy as np

from vllm.config import VllmConfig, CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# Observed Spyre DMA failure threshold for encoder-only dummy batches with
# multiple sequences.  Pooling warmup stays below this limit.
SPYRE_ENCODER_DMA_TOKEN_LIMIT = 30
# Token count for pooling warmup (single sequence), kept under the DMA limit.
SPYRE_ENCODER_WARMUP_MAX_TOKENS = 16


# Cache of dynamically created CPU-embedding subclasses, keyed by the original
# embedding class, so ``__class__`` reassignment stays cheap and idempotent.
_CPU_EMBEDDING_SUBCLASSES: dict[type, type] = {}


class _CpuEmbeddingMixin:
    """``nn.Embedding`` forward that gathers on CPU and returns on the table's device.

    torch-spyre has no embedding kernel, so the gather runs on CPU and the
    result is moved back to the (Spyre) weight device. Installed via
    ``__class__`` reassignment so dispatch goes through ``nn.Module.__call__``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast(torch.Tensor, self.weight.data)
        weight_cpu = weight.cpu()
        out = F.embedding(
            input.cpu(),
            weight_cpu,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return convert(out, device=weight.device)


def _cpu_embedding_subclass(base: type) -> type:
    """Return (and cache) a ``base`` subclass mixing in the CPU-gather forward."""
    cls = _CPU_EMBEDDING_SUBCLASSES.get(base)
    if cls is None:
        cls = type(f"SpyreCpu{base.__name__}", (_CpuEmbeddingMixin, base), {})
        _CPU_EMBEDDING_SUBCLASSES[base] = cls
    return cls


def _patch_encoder_embeddings_cpu(model: nn.Module) -> None:
    """Retype each ``nn.Embedding`` in a pooling model to gather on CPU.

    The table stays on Spyre (moved with the rest of the model); the retyped
    forward copies it to CPU, gathers, and returns the result on the table's
    device. ``VocabParallelEmbedding`` handles this in its own ``forward``.
    """
    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Embedding) or isinstance(module, _CpuEmbeddingMixin):
            continue
        module.__class__ = _cpu_embedding_subclass(type(module))
        patched += 1

    logger.info("Patched %d nn.Embedding layer(s) to gather on CPU", patched)


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """Spyre-specific CpuGpuBuffer with Spyre-safe copies and split dtypes.
    This buffer is closely related to the CpuGpuBuffer in vllm/v1/utils.py.

    For float dtypes: .cpu on CPU, .gpu on Spyre (float16).
    For int/bool dtypes: .gpu aliased to .cpu (CPUModelRunner pattern).
    All copies are currently synchronous as torch-spyre does not yet support `non_blocking`.

    Inherits from `CpuGpuBuffer` (without invoking its `__init__`) so that
    `_make_buffer` overrides remain Liskov-compatible with `GPUModelRunner`.
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        cpu_dtype: torch.dtype,
        gpu_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=cpu_dtype, device="cpu", pin_memory=pin_memory)
        if device.type == "spyre":
            self.gpu = torch.zeros(*size, dtype=gpu_dtype, device=device)
        else:
            # int/bool: alias gpu = cpu (CPUModelRunner pattern)
            self.gpu = self.cpu
        self.np: np.ndarray
        if with_numpy:
            if cpu_dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call SpyreCpuGpuBuffer with "
                    "with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.gpu if n is None else self.gpu[:n]
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        dst.copy_(src)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        # Currently only the copy_to_gpu function is invoked.
        # If the copy_to_cpu also becomes required, override it here with
        # spyre-specific aspects.
        raise NotImplementedError("SpyreCpuGpuBuffer.copy_to_cpu is not implemented")


class _SpyreModelWrapper:
    """Converts model inputs/outputs at the device boundary, for every call site.

    Integer inputs move to Spyre int64, except under ``keep_int_inputs_on_cpu``
    (pooling/encoder) where they stay on CPU. Outputs move to CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        spyre_device: torch.device,
        *,
        keep_int_inputs_on_cpu: bool = False,
    ):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)
        object.__setattr__(self, "_keep_int_inputs_on_cpu", keep_int_inputs_on_cpu)

    def __call__(self, *args, **kwargs):
        def _convert_int(t):
            if (
                t is not None
                and isinstance(t, torch.Tensor)
                and t.dtype in (torch.int32, torch.int64)
            ):
                if self._keep_int_inputs_on_cpu:
                    # Pooling path: normalize to int64 on CPU (no Spyre H2D).
                    if t.dtype != torch.int64:
                        return t.to(dtype=torch.int64)
                    return t
                return convert(t, dtype=torch.int64, device=self._spyre_device)
            return t

        args_converted = []
        for arg in args:
            args_converted.append(_convert_int(arg))

        kwargs_converted = {}
        for key in kwargs:
            val = kwargs.get(key)
            kwargs_converted[key] = _convert_int(val)

        t0 = time.time()
        result = self._model(*args_converted, **kwargs_converted)

        def _to_cpu(x):
            return convert(x, device="cpu")

        result = tree_map(_to_cpu, result)

        input_ids = kwargs_converted.get("input_ids")
        num_tokens = input_ids.shape[0] if input_ids is not None else -1
        logger.debug("t_token: %.2fms [num tokens %d]", (time.time() - t0) * 1000, num_tokens)

        return result

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for Spyre.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre for floats, aliased to CPU for int/bool

    Inherits from GPUModelRunner to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Store the real Spyre device before super().__init__ so that
        # _make_buffer can place .gpu tensors on Spyre directly.
        self._spyre_device = device

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Keep self.device as CPU so buffer management (scatter, copy) stays
        # on CPU.  For generative models, _SpyreModelWrapper converts
        # input_ids/positions to Spyre int64 at the model boundary.  For
        # pooling/encoder models, integers stay on CPU and the embeddings
        # gather word vectors on CPU before activations go to Spyre.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernel with C++ CPU implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # Same replacement as CPUModelRunner._postprocess_triton().
        import vllm.utils.cpu_triton_utils as cpu_tl
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = cpu_tl.compute_slot_mapping_kernel

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load weights on CPU, move Spyre layers to device, compile, and wrap.

        For pooling models, patches nn.Embedding to gather on CPU (tables stay
        on Spyre) and enables CPU integer inputs at the boundary.
        """
        logger.info("Loading model %s...", self.model_config.model)
        t0 = time.time()

        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        model_loader = get_model_loader(self.load_config)

        # Load model on CPU
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        # Cases appearing in GPUModelRunner.
        # When needed, they can be implemented for Spyre.
        if self.lora_config:
            raise NotImplementedError("LoRA adapters are not yet implemented and tested for Spyre.")

        if hasattr(self, "drafter"):
            raise NotImplementedError(
                "Models with a drafter model are not yet implemented and tested for Spyre."
            )

        # Pin Attention buffers (_k_scale, _v_scale, ...) to CPU before
        # model.to("spyre") via an _apply no-op so the CPU attention backend
        # reads them without a device mismatch (these are nn.Module, not
        # PluggableLayer, so OOT registration isn't possible).
        is_pooling = self.model_config.runner_type == "pooling"
        for module in self.model.modules():
            if isinstance(module, Attention):
                module._apply = lambda fn, recurse=True, _m=module: _m

        # Move all weights to Spyre (embeddings included; the gather still runs
        # on CPU — torch-spyre has no embedding kernel yet).
        self.model.to(device=self._spyre_device)
        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)
        logger.info("Model loaded for Spyre in %.3fs.", time.time() - t0)

        if is_pooling:
            _patch_encoder_embeddings_cpu(self.model)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Device boundary: generative models move ints to Spyre; pooling models
        # keep ints on CPU.  All model outputs are returned on CPU.
        self.model = _SpyreModelWrapper(
            self.model,
            self._spyre_device,
            keep_int_inputs_on_cpu=is_pooling,
        )
        if is_pooling:
            logger.info("Encoder pooling model: keeping input_ids/positions on CPU")

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile for Spyre with static shapes.

        Spyre compilation is handled here (not by vLLM's @support_torch_compile)
        because Spyre requires static shapes — dynamic shapes (SymInt) are not
        supported by the Spyre Inductor backend.

        Supported modes:
        - enforce_eager=True: no compilation (eager execution)
        - CompilationMode.NONE: Spyre-managed compilation with torch.compile
        Other vLLM compilation modes (VLLM_COMPILE, STOCK_TORCH_COMPILE) are
        not supported — the platform forces CompilationMode.NONE in
        apply_config_platform_defaults().
        """
        mode = self.compilation_config.mode
        if mode != CompilationMode.NONE:
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. "
                f"Only CompilationMode.NONE is supported. Spyre handles "
                f"compilation internally via _compile_for_spyre(). "
                f"Use enforce_eager=True to disable compilation entirely."
            )

        if self.vllm_config.model_config.enforce_eager:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        # Custom ops (spyre_rmsnorm, spyre_cpu_fallback, etc.) are opaque
        # to dynamo but don't cause graph breaks — fullgraph=True is safe.
        # dynamic=False ensures static shapes (Spyre can't handle SymInt).
        t0 = time.time()
        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info("Model compiled for Spyre (backend=inductor) in %.3fs.", time.time() - t0)

    def warming_up_model(self) -> None:
        """Run a dummy forward pass to warm up kernels and optional compile.

        Generative models: ``_SpyreModelWrapper`` moves integer inputs to Spyre
        int64; activations run on Spyre and outputs return on CPU.

        Pooling models: integers stay on CPU, encoder SDPA metadata must match
        a single-sequence embed pass, and token count is capped (see
        ``SPYRE_ENCODER_WARMUP_MAX_TOKENS``) to stay under the Spyre DMA limit
        for encoder dummy batches.
        """
        logger.info("Warming up model...")
        t0 = time.time()
        num_tokens = min(
            max(16, self.max_num_reqs),
            self.scheduler_config.max_num_batched_tokens,
        )
        with _set_spyre_compilation_settings(self.vllm_config):
            if self.model_config.runner_type == "pooling":
                # Match single-sequence embed metadata; cap tokens for DMA.
                num_tokens = min(num_tokens, SPYRE_ENCODER_WARMUP_MAX_TOKENS)
                saved_max_num_seqs = self.scheduler_config.max_num_seqs
                try:
                    self.scheduler_config.max_num_seqs = 1
                    logger.info(
                        "Pooling warmup: %d tokens, max_num_seqs=1 (was %d)",
                        num_tokens,
                        saved_max_num_seqs,
                    )
                    self._dummy_run(num_tokens)
                finally:
                    self.scheduler_config.max_num_seqs = saved_max_num_seqs
            else:
                self._dummy_run(num_tokens)
        logger.info("Warmup done in %.3fs.", time.time() - t0)

    # --- KV cache allocation ---

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """Allocate KV cache as lists of individual page tensors on Spyre.

        Each layer gets its own SpyrePagedKVCache(k_pages, v_pages) where each
        is a list of tensors of shape [num_kv_heads, block_size, head_size] on
        the Spyre device. This matches upstream vLLM's paged model but uses
        list indices instead of tensor indices — enabling direct per-page bmm
        without advanced indexing.
        """
        from vllm.v1.worker.utils import bind_kv_cache
        from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

        # Iterate kv_cache_tensors (one entry per physical buffer)
        spec_by_layer = {
            ln: g.kv_cache_spec for g in kv_cache_config.kv_cache_groups for ln in g.layer_names
        }

        # vLLM's `bind_kv_cache` types this dict as `dict[str, torch.Tensor]`,
        # but the matching `SpyreAttentionImpl.forward` consumes the
        # SpyrePagedKVCache — see the suppression on `bind_kv_cache(...)` below.
        kv_caches: dict[str, SpyrePagedKVCache] = {}

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # All layers in `shared_by` use the same spec by construction.
            spec = spec_by_layer[kv_cache_tensor.shared_by[0]]
            num_blocks = kv_cache_tensor.size // spec.page_size_bytes

            # Default stickification splits head_size into 64-element sticks.
            # Alternative: stickify block_size or num_kv_heads for different
            # access patterns (would require explicit SpyreTensorLayout).
            k_pages: list[torch.Tensor] = [
                torch.zeros(
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_size,
                    dtype=torch.float16,
                    device=self._spyre_device,
                )
                for _ in range(num_blocks)
            ]
            v_pages: list[torch.Tensor] = [
                torch.zeros(
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_size,
                    dtype=torch.float16,
                    device=self._spyre_device,
                )
                for _ in range(num_blocks)
            ]

            page_cache = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = page_cache

        for layer_name, target in self.shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target]

        bind_kv_cache(
            kv_caches,  # ty: ignore[invalid-argument-type]
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )
        return kv_caches

    # --- Stubs copied from CPUModelRunner ---
    # These are trivial overrides that GPUModelRunner expects.

    def _init_device_properties(self) -> None:
        # No CUDA/GPU device properties to query for Spyre
        pass

    def _sync_device(self) -> None:
        # TODO: Replace with torch.spyre.synchronize() when available.
        # For now, all copies are synchronous (no non_blocking), so
        # explicit sync is not needed.
        pass

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        return 0, None

    def get_model(self) -> nn.Module:
        # Return the unwrapped model for isinstance checks
        # (e.g. is_text_generation_model in get_supported_tasks).
        model = self.model
        if isinstance(model, _SpyreModelWrapper):
            model = model._model
        # Unwrap torch.compile's OptimizedModule (has _orig_mod attribute)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        assert isinstance(model, nn.Module)
        return model

    # --- Buffer management ---

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> SpyreCpuGpuBuffer:
        """Create a SpyreCpuGpuBuffer with float tensors on Spyre.

        - Float dtypes: .cpu on CPU, .gpu on Spyre as float16
        - Int/bool dtypes: .gpu aliased to .cpu (stays on CPU)
        """
        if dtype.is_floating_point:
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=torch.float16,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Int/bool → CPU-only (aliased)
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=numpy,
        )


@contextmanager
def _set_spyre_compilation_settings(config: VllmConfig):
    """Context manager for Spyre-specific compilation settings during warmup.

    Similar to _set_global_compilation_settings in cpu_model_runner.py but
    adapted for Spyre's compilation requirements.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
