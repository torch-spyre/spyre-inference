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
the CPU fallbacks will be obsolete and most operations will be performed on Spyre.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

import numpy as np

from vllm.config import VllmConfig, CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.custom_ops.unfuse import analyze_and_unfuse
from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# Pure-PyTorch replacement for torch.ops._C.compute_slot_mapping_kernel_impl
# (unavailable with VLLM_TARGET_DEVICE=empty).

_PAD_SLOT_ID = -1


def _compute_slot_mapping_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = _PAD_SLOT_ID,
    BLOCK_SIZE: int = 1024,
) -> None:
    """Map each token position to its flat index in the paged KV cache.

    The upstream vLLM implementation is a Triton kernel (requires a GPU) and
    the CPU backend delegates to a C++ op in _C.so. Neither is available with
    VLLM_TARGET_DEVICE=empty, so we reimplement the logic in pure PyTorch.

    Correctness is validated indirectly by the upstream attention backend test
    (test_causal_backend_correctness) and end-to-end model generation tests.
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on Spyre."
    block_indices = (positions[:num_tokens] // block_size).to(torch.int64)
    block_offsets = (positions[:num_tokens] % block_size).to(torch.int64)

    num_reqs = query_start_loc.shape[0] - 1
    req_indices = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        req_indices[start:end] = i

    flat_indices = req_indices * block_table_stride + block_indices
    block_numbers = block_table.flatten()[flat_indices].to(torch.int64)
    slot_mapping[:num_tokens] = block_numbers * block_size + block_offsets
    if max_num_tokens > num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID


class _FuncWrapper:
    """Mimics Triton's grid-launch syntax: kernel[(grid,)](...) → kernel(...)."""

    def __init__(self, func):
        self.func = func

    def __getitem__(self, grid):
        return self.func


_compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_impl)


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
    """Transparent wrapper that converts model inputs/outputs at the boundary.

    Input conversion (CPU → Spyre):
        For example, input_ids and positions arrive as CPU tensors (int32/int64) because
        self.device=CPU in the runner and buffer scatter ops run on CPU.
        Convert them to int64 and provide them to the model.

    Output conversion (Spyre → CPU):
        The model's final hidden_states come out on Spyre. Downstream
        operations (indexing via logits_indices, sampling) run on CPU.
        The lm_head matmul runs on Spyre via SpyreParallelLMHead,
        which handles H2D/D2H for the sample_hidden_states subset.

    Wrapping at the model level ensures ALL call sites get the right
    device — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).
    """

    def __init__(self, model: nn.Module, spyre_device: torch.device):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)

    def __call__(self, *args, **kwargs):
        # Convert integer tensor inputs to Spyre int64
        def _convert_int(t):
            if (
                t is not None
                and isinstance(t, torch.Tensor)
                and t.dtype in (torch.int32, torch.int64)
            ):
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

    def compute_logits(self, hidden_states, *args, **kwargs):
        """Move hidden_states onto Spyre for the lm_head custom op.

        gpu_model_runner.execute_model slices `hidden_states[logits_indices]`
        on CPU (Spyre cannot slice), so the tensor handed to compute_logits
        is on CPU. Thus, convert here, perform the lm_head operation and
        then convert the resulting logits back to CPU
        for downstream sampling.
        """
        hidden_states = convert(hidden_states, device=self._spyre_device)
        # logits are returned on cpu
        logits = self._model.compute_logits(hidden_states, *args, **kwargs)
        return logits

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
        # on CPU. _SpyreModelWrapper converts input_ids/positions to Spyre
        # int64 at the model call boundary, so the embedding takes the Spyre
        # fast-path and hidden_states flow on Spyre between decoder layers.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernel with a pure-PyTorch implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # The upstream CPU backend uses a C++ kernel (torch.ops._C) as its
        # fallback, but we don't have _C.abi3.so with VLLM_TARGET_DEVICE=empty.
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = _compute_slot_mapping_kernel

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model and compile for Spyre."""
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

        # Un-fuse QKV / gate-up projections.
        analyze_and_unfuse(self.model)

        # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
        # Attention is nn.Module (not PluggableLayer) so OOT registration is
        # not possible. Patch _apply to no-op before model.to("spyre") so
        # the CPU attention backend can access scale buffers without device
        # mismatch.
        for module in self.model.modules():
            if isinstance(module, Attention):
                module._apply = lambda fn, recurse=True, _m=module: _m

        # Move layer weights to Spyre device. The Attention._apply no-op
        # patched above keeps attention scale buffers on CPU; every other
        # module (linear, embedding, RMSNorm, SiluAndMul, ParallelLMHead)
        # has its weights moved to Spyre.
        self.model.to(device=self._spyre_device)
        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)
        logger.info("Model loaded for Spyre in %.3fs.", time.time() - t0)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Wrap model so ALL forward() calls to the entire model,
        # for example in execute_model, _dummy_run, etc.,
        # automatically convert Spyre outputs to CPU. This ensures downstream
        # indexing (logits_indices), lm_head (CPU weights), and sampling all
        # receive CPU tensors without needing per-call-site overrides.
        self.model = _SpyreModelWrapper(self.model, self._spyre_device)

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
        """Run a dummy forward pass to warm up the model.

        _dummy_run creates CPU int inputs, but _SpyreModelWrapper converts
        input_ids/positions to Spyre int64 at the model boundary. The
        embedding thus runs on Spyre and hidden_states flow on Spyre.
        _SpyreModelWrapper also converts final outputs back to CPU.

        When enforce_eager=False, this also triggers torch.compile.
        """
        logger.info("Warming up model...")
        t0 = time.time()
        num_tokens = min(
            max(16, self.max_num_reqs),
            self.scheduler_config.max_num_batched_tokens,
        )
        with _set_spyre_compilation_settings(self.vllm_config):
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
