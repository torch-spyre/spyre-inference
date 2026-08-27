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
- Generative: D2H hidden_states for logits/sampling. Pooling: keep on Spyre;
  pooler D2Hs only the final pooled vectors in ``_pool``.
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

import os
import time
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

import numpy as np

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig, CompilationMode, CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.tasks import PoolingTask
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.custom_ops.head_pad import (
    fix_padded_attention_scale,
    fix_padded_rope,
    install_head_pad_weight_loader,
    install_padded_head_dim,
    reject_padded_qk_norm,
    verify_padded_head_dim,
)
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.pool import (
    TOKEN_POOLING_TASKS,
    configure_pooling_for_spyre,
    copy_pooler_output_to_cpu,
    select_rows,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer


logger = init_logger(__name__)

# Eager pooling warmup: one short sequence so the dummy stays compile-cheap.
SPYRE_ENCODER_WARMUP_MAX_TOKENS = 16

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
    KV_CACHE_BLOCK_SIZE: int | None = None,
    BLOCKS_PER_KV_BLOCK: int = 1,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = _PAD_SLOT_ID,
    # Triton tile width; unused here, kept for call compatibility.
    BLOCK_SIZE: int = 1024,
) -> None:
    """Map each token position to its flat index in the paged KV cache.

    The upstream vLLM implementation is a Triton kernel (requires a GPU) and
    the CPU backend delegates to a C++ op in _C.so. Neither is available with
    VLLM_TARGET_DEVICE=empty, so we reimplement the logic in pure PyTorch.

    Correctness is validated indirectly by the upstream attention backend test
    (test_causal_backend_correctness) and end-to-end model generation tests.

    ``block_size`` is the kernel's block size, ``KV_CACHE_BLOCK_SIZE`` the KV
    manager's, and ``BLOCKS_PER_KV_BLOCK`` the ratio between them (1 on Spyre).
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on Spyre."
    kv_block_size = block_size if KV_CACHE_BLOCK_SIZE is None else KV_CACHE_BLOCK_SIZE

    token_positions = positions[:num_tokens]
    virtual_block_indices = (token_positions // kv_block_size).to(torch.int64)
    local_block_offsets = (token_positions % kv_block_size).to(torch.int64)
    block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size

    num_reqs = query_start_loc.shape[0] - 1
    req_indices = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        req_indices[start:end] = i

    flat_indices = req_indices * block_table_stride + block_indices
    block_numbers = block_table.flatten()[flat_indices].to(torch.int64)
    slot_mapping[:num_tokens] = block_numbers * block_size + local_block_offsets % block_size
    if num_tokens < max_num_tokens:
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
    Float H2D uses ``non_blocking=True``; callers must sync via
    ``TorchSpyreModelRunner._sync_device`` (``torch.spyre.synchronize``)
    before consuming the Spyre tensors.

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
        # Async H2D via torch-spyre's aten::_copy_from / copyAsync path.
        # GPUModelRunner calls _sync_device before the tensors are consumed.
        dst.copy_(src, non_blocking=True)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        # Currently only the copy_to_gpu function is invoked.
        # If the copy_to_cpu also becomes required, override it here with
        # spyre-specific aspects.
        raise NotImplementedError("SpyreCpuGpuBuffer.copy_to_cpu is not implemented")


SPYRE_COMPILE_GRANULARITIES = ("block", "model")


def _compile_granularity() -> str:
    granularity = os.environ.get("SPYRE_COMPILE_GRANULARITY") or "block"
    if granularity not in SPYRE_COMPILE_GRANULARITIES:
        raise ValueError(
            f"Unsupported SPYRE_COMPILE_GRANULARITY={granularity!r}. "
            f"Expected one of {SPYRE_COMPILE_GRANULARITIES}."
        )
    return granularity


def _block_sharing_defeated_by() -> str | None:
    """Sharing needs vLLM to hoist layer names out of the graph; report when it cannot."""
    try:
        from vllm.utils.torch_utils import _USE_LAYERNAME
    except ImportError:
        return None
    if not _USE_LAYERNAME:
        return (
            "vLLM is not hoisting per-layer attention names (needs torch >= 2.11 and "
            "VLLM_USE_LAYERNAME=1), so each block compiles separately"
        )
    return None


def _repeated_block_lists(model: nn.Module) -> list[nn.ModuleList]:
    block_lists = []
    for module in model.modules():
        if not isinstance(module, nn.ModuleList):
            continue
        blocks = [b for b in module if not isinstance(b, PPMissingLayer)]
        if not blocks:
            continue
        # nn.Module.modules() yields the module itself, so a list of bare Attention
        # layers (Zamba2's dpa_list) would match and "compile" one opaque call per entry.
        if any(isinstance(b, Attention) for b in blocks):
            continue
        # Hybrid Mamba+attention stacks (Granite 4.0, Jamba) mix classes in one list;
        # each class shares a forward code object, so compiles scale per class, not depth.
        if any(isinstance(m, Attention) for b in blocks for m in b.modules()):
            block_lists.append(module)
    return block_lists


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

    def __init__(
        self,
        model: nn.Module,
        spyre_device: torch.device,
        keep_outputs_on_device: bool = False,
    ):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)
        object.__setattr__(self, "_keep_outputs_on_device", keep_outputs_on_device)

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

        # Pooling: keep on Spyre. Generative: D2H for sampling.
        if not self._keep_outputs_on_device:

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
        is on CPU; move it onto Spyre for the lm_head matmul. The logits are
        returned on CPU: SpyreParallelLMHead.forward_oot keeps them on Spyre
        for the TP all_gather, and SpyreLogitsProcessor._gather_logits
        converts back to CPU right after the gather (before the vocab slice
        and scale), so downstream sampling gets CPU logits.
        """
        hidden_states = convert(hidden_states, device=self._spyre_device)
        return self._model.compute_logits(hidden_states, *args, **kwargs)

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

        # Set by load_model: whether the pooler/classifier stay on Spyre.
        self._pooling_on_spyre = False

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Keep self.device as CPU so buffer management (scatter, copy) stays
        # on CPU. _SpyreModelWrapper converts input_ids/positions to Spyre
        # int64 at the model boundary.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Shape bucketer for runtime dispatch (initialized after model load)
        self.spyre_shape_bucketer: SpyreShapeBucketer | None = None

        # Replace Triton kernel with a pure-PyTorch implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # The upstream CPU backend uses a C++ kernel (torch.ops._C) as its
        # fallback, but we don't have _C.abi3.so with VLLM_TARGET_DEVICE=empty.
        from vllm.v1.worker import block_table

        # Deliberately swap the Triton JITFunction for the grid-launch-compatible
        # _FuncWrapper; the type mismatch is the point of the patch.
        block_table._compute_slot_mapping_kernel = _compute_slot_mapping_kernel

    @staticmethod
    def _install_pooling_model_patches(model_config) -> None:
        """Install model-specific pooling adapters (BERT/RoBERTa token_type, …)."""
        if model_config.runner_type != "pooling":
            return
        from spyre_inference.models import install_pooling_model_patches

        install_pooling_model_patches()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load weights on CPU, move Spyre layers to device, compile, and wrap."""
        logger.info("Loading model %s...", self.model_config.model)
        t0 = time.time()

        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        model_loader = get_model_loader(self.load_config)

        self._install_pooling_model_patches(self.model_config)

        # Pad attention weights (q/k/v/o) to the stick-aligned head_dim as they
        # stream in, when the platform overrode head_dim (e.g. head_size=64).
        # Must run before load_model builds+loads the (now 128-wide) params.
        install_padded_head_dim(self.model_config)
        install_head_pad_weight_loader(model_loader, self.model_config.hf_config)

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

        # Restore original RoPE frequencies and attention scale corrupted by the
        # head_dim width override (no-op unless the platform padded head_dim).
        verify_padded_head_dim(self.model, self.model_config.hf_config)
        reject_padded_qk_norm(self.model, self.model_config.hf_config)
        fix_padded_rope(self.model, self.model_config.hf_config)
        fix_padded_attention_scale(self.model, self.model_config.hf_config)

        # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
        # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
        # an nn.Module, but just the attention implementation.
        Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

        # Move layer weights to Spyre device.
        self.model.to(device=self._spyre_device)

        # CLS/LAST on Spyre via v1.pool; MEAN stays CPU.
        self._pooling_on_spyre = False
        if self.model_config.runner_type == "pooling":
            self._pooling_on_spyre = configure_pooling_for_spyre(self.model, self._spyre_device)

        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)
        logger.info("Model loaded for Spyre in %.3fs.", time.time() - t0)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Generative: D2H model outputs. Pooling: keep hidden_states on Spyre.
        self.model = _SpyreModelWrapper(
            self.model,
            self._spyre_device,
            keep_outputs_on_device=self._pooling_on_spyre,
        )

        # Initialize bucket dispatcher for shape bucketing at runtime.
        self.spyre_shape_bucketer = self._create_shape_bucketer()

    def _create_shape_bucketer(self) -> SpyreShapeBucketer | None:
        """Create SpyreShapeBucketer if compilation with bucketing is active.

        Returns None when enforce_eager=True, mode is NONE, or no
        compile_sizes are configured (e.g. pooling models skip bucketing
        because their token counts depend on variable input sequence lengths).
        """
        if self.vllm_config.model_config.enforce_eager:
            logger.info("Grarph Recorder disabled (enforce_eager=True)")
            return None

        if not self.compilation_config.compile_sizes:
            logger.info("Grarph Recorder disabled (no compile_sizes configured)")
            return None

        return SpyreShapeBucketer(self.vllm_config)

    def _compile_for_spyre(self) -> None:
        """Install torch.compile wrappers; tracing happens on the first forward.

        `dynamic=False` is mandatory: the Spyre backend rejects SymInt shapes.
        """
        mode = self.compilation_config.mode
        if mode not in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. Only "
                f"CompilationMode.NONE and CompilationMode.STOCK_TORCH_COMPILE "
                f"are supported."
            )

        # Validated before the eager short-circuit so a typo is not silently ignored.
        # Per-block is the default: identical blocks share a ``forward`` code object,
        # so the backend compiles one block rather than a program that grows with depth.
        granularity = _compile_granularity()

        if self.vllm_config.model_config.enforce_eager or mode is CompilationMode.NONE:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        model_name = type(self.model).__name__

        if granularity == "block":
            defeated_by = _block_sharing_defeated_by()
            if defeated_by:
                logger.warning(
                    "SPYRE_COMPILE_GRANULARITY=block will not share one artifact across "
                    "blocks: %s. Expect compile time to grow with layer count; "
                    "SPYRE_COMPILE_GRANULARITY=model may warm up faster.",
                    defeated_by,
                )
            num_blocks = self._compile_blocks()
            if num_blocks:
                logger.info(
                    "Wrapped %d transformer blocks of %s for per-block compile on Spyre.",
                    num_blocks,
                    model_name,
                )
                return
            logger.warning(
                "Found no attention-bearing block ModuleList in %s; falling back to a "
                "whole-model graph. Models whose attention is not a vLLM Attention "
                "(MLA, encoder-only vision towers) take this path.",
                model_name,
            )

        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info("Wrapped %s as a single graph for Spyre.", model_name)

    def _compile_blocks(self) -> int:
        num_blocks = 0
        for blocks in _repeated_block_lists(cast(nn.Module, self.model)):
            for block in blocks:
                if isinstance(block, PPMissingLayer):
                    continue
                # In place: rebinding blocks[i] to the returned OptimizedModule reparents
                # the block under `_orig_mod`, renaming every parameter and breaking
                # reload_weights and save_sharded_state.
                block.compile(backend="inductor", fullgraph=True, dynamic=False)
                num_blocks += 1
        return num_blocks

    def warming_up_model(self) -> None:
        """Run a dummy forward pass to warm up kernels and optional compile.

        In eager mode, pooling models cap token count
        (``SPYRE_ENCODER_WARMUP_MAX_TOKENS``) and force ``max_num_seqs=1``.
        Compile mode uses the normal warmup size so shapes match torch.compile.
        """
        if self.spyre_shape_bucketer is None:
            logger.info("Running single warmup pass (graph manager Disabled)...")
            t0 = time.time()
            num_tokens = min(
                max(16, self.max_num_reqs),
                self.scheduler_config.max_num_batched_tokens,
            )
            with _set_spyre_compilation_settings(self.vllm_config):
                use_eager_pooling_warmup = (
                    self.model_config.runner_type == "pooling"
                    and self.vllm_config.model_config.enforce_eager
                )
                if use_eager_pooling_warmup:
                    # Match single-sequence embed metadata; cap tokens for DMA.
                    num_tokens = min(num_tokens, SPYRE_ENCODER_WARMUP_MAX_TOKENS)
                    saved_max_num_seqs = self.scheduler_config.max_num_seqs
                    try:
                        self.scheduler_config.max_num_seqs = 1
                        logger.info(
                            "Pooling warmup (eager): %d tokens, max_num_seqs=1 (was %d)",
                            num_tokens,
                            saved_max_num_seqs,
                        )
                        self._dummy_run(num_tokens)
                    finally:
                        self.scheduler_config.max_num_seqs = saved_max_num_seqs
                else:
                    self._dummy_run(num_tokens)
            logger.info("Warmup done in %.3fs.", time.time() - t0)
            return

        bucket_sizes = self.spyre_shape_bucketer.bucket_sizes
        logger.info(
            "Warming up model with %d bucket sizes [%d..%d]...",
            len(bucket_sizes),
            bucket_sizes[0] if bucket_sizes else 0,
            bucket_sizes[-1] if bucket_sizes else 0,
        )
        t0 = time.time()
        with _set_spyre_compilation_settings(self.vllm_config):
            # Compile largest bucket first: Inductor's internal caches benefit
            # from seeing the most complex shape first, so subsequent smaller
            # shapes compile faster via partial cache hits.
            for size in sorted(bucket_sizes, reverse=True):
                self._dummy_run(size)
        self.spyre_shape_bucketer.mark_warmed_up()
        logger.info(
            "Warmup complete in %.3fs for %d buckets.",
            time.time() - t0,
            len(bucket_sizes),
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        """Inject bucket padding via upstream's padding mechanism.

        Upstream uses this method to determine CUDA graph padding. We reuse
        the same interface to return our shape-bucketed padding so that the
        rest of execute_model (slot_mapping, attention metadata, _preprocess)
        handles padded vs unpadded counts correctly without mutating
        scheduler_output.total_num_scheduled_tokens.
        """
        if self.spyre_shape_bucketer is not None and self.spyre_shape_bucketer.is_warmed_up:
            desc = self.spyre_shape_bucketer.dispatch(num_tokens)
            if desc is not None:
                return (
                    CUDAGraphMode.NONE,
                    BatchDescriptor(num_tokens=desc.padded_num_tokens),
                    False,
                    None,
                    None,
                )

        return super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    @torch.inference_mode()
    def _dummy_run(self, *args, **kwargs):
        """Force D2H for warmup: upstream ``hidden_states[logit_indices]`` needs CPU."""
        # Read out of the passthrough rather than named in the signature, which would
        # pin this override to upstream's parameter order across vLLM bumps.
        num_tokens = kwargs.get("num_tokens", args[0] if args else None)
        if num_tokens is not None:
            attn_layer.publish_null_slots(num_tokens)
        wrapper = self.model
        keep = isinstance(wrapper, _SpyreModelWrapper) and wrapper._keep_outputs_on_device
        if keep:
            object.__setattr__(wrapper, "_keep_outputs_on_device", False)
        try:
            hidden_states, last_hidden_states = super()._dummy_run(*args, **kwargs)
        finally:
            if keep:
                object.__setattr__(wrapper, "_keep_outputs_on_device", True)

        if (
            keep
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.numel() > 0
            and hidden_states.device.type != "spyre"
        ):
            hidden_states = convert(hidden_states, self._spyre_device)
        # Sampler warmup only needs last_hidden_states on CPU.
        return hidden_states, last_hidden_states

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """Drop token-level tasks on Spyre pooler (slice views are unsafe)."""
        tasks = super().get_supported_pooling_tasks()
        if not self._pooling_on_spyre:
            return tasks

        supported = [t for t in tasks if t not in TOKEN_POOLING_TASKS]
        if tasks and not supported:
            raise RuntimeError(
                f"Model {self.model_config.model} supports only token-level "
                "pooling, which is unsupported while the pooler runs on Spyre."
            )
        return supported

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Pool on the activation device; D2H only the pooled vectors.

        MEAN / FP32 heads keep the pooler on CPU — delegate to
        ``GPUModelRunner._pool``. On-Spyre CLS/LAST still overrides the private
        hook: dim-0 crop must use ``index_select`` (not ``[:n]``), and pooled
        D2H must use ``convert`` (not CUDA ``.to`` / AsyncGPU). Drop this once
        those ops are safe (fallback probes / #3507–#3508).
        """
        assert not self.use_async_scheduling, (
            "async scheduling is unsupported while pooling on Spyre"
        )

        if not self._pooling_on_spyre:
            return super()._pool(
                convert(hidden_states, "cpu"),
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                kv_connector_output,
            )

        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        for params in self.input_batch.pooling_params.values():
            if params.task in TOKEN_POOLING_TASKS:
                raise NotImplementedError(
                    f"Pooling task {params.task!r} returns per-sequence views "
                    "of hidden_states, which is unsupported while the pooler "
                    "runs on Spyre."
                )

        # Crop via index_select — Spyre dim-0 slice views are unsafe.
        hidden_states = convert(hidden_states, self._spyre_device)
        if hidden_states.shape[0] != num_scheduled_tokens:
            hidden_states = select_rows(
                hidden_states, torch.arange(num_scheduled_tokens, dtype=torch.int64)
            )

        # Mirror GPUModelRunner._pool after crop. Build the cursor on CPU:
        # upstream does ``cumsum[1:] - 1`` for last_token_indices; that offset-1
        # view is not stick-aligned on Spyre (copy_from_d2d fails). SpyreCLS/Last
        # only read host ``num_scheduled_tokens_cpu`` via cursor_row_indices_cpu.
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=torch.device("cpu"),
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        model_runner_output.pooler_output = copy_pooler_output_to_cpu(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )
        self._sync_device()
        return model_runner_output

    # --- KV cache allocation ---

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """Allocate KV cache as one dense paged tensor per layer on Spyre.

        Each layer gets its own SpyrePagedKVCache(k_pages, v_pages) where each
        is a single tensor of shape [num_blocks, block_size, num_kv_heads,
        head_size], matching the shape SpyreAttentionBackend.get_kv_cache_shape
        advertises. The attention kernel selects a page by indexing with a
        one-element device tensor, so the page read is a real indirect access.
        """
        from vllm.v1.worker.utils import bind_kv_cache
        from spyre_inference.v1.attention.backends.spyre_attn import (
            SpyrePagedKVCache,
            slot_major_kv_layout,
        )

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

            # Host-allocated then transferred: only .to() takes a device_layout.
            layout = slot_major_kv_layout(
                num_blocks * spec.block_size, spec.num_kv_heads, spec.head_size, torch.float16
            )

            k_pages = torch.zeros(
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                dtype=torch.float16,
            ).to(self._spyre_device, device_layout=layout)  # ty: ignore[no-matching-overload]
            v_pages = torch.zeros(
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                dtype=torch.float16,
            ).to(self._spyre_device, device_layout=layout)  # ty: ignore[no-matching-overload]

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
        # Wait for outstanding async H2D from SpyreCpuGpuBuffer.copy_to_gpu
        # (and any other non_blocking copies) before the runner consumes
        # Spyre tensors. torch.spyre is registered by torch-spyre autoload.
        torch.spyre.synchronize(self._spyre_device)

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
            # Freezing folds per-block weights into each block's graph, defeating
            # artifact sharing. Warn rather than override an explicit max_autotune.
            if _compile_granularity() == "block" and not config.model_config.enforce_eager:
                logger.warning(
                    "max_autotune enables Inductor freezing, which folds per-block "
                    "weights into each block's graph and defeats per-block artifact "
                    "sharing. Compile time will grow with layer count."
                )
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
