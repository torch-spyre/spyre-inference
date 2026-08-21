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

import time
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

import numpy as np

from vllm.config import VllmConfig, CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
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
from spyre_inference.v1.encoder_buckets import (
    EncoderBucketPad,
    encoder_bucket_valid_row_indices,
    expand_packed_to_encoder_bucket,
    pooling_warmup_pad_query_lens,
    pooling_warmup_shapes,
    runtime_encoder_bucket,
)
from spyre_inference.v1.pool import (
    TOKEN_POOLING_TASKS,
    configure_pooling_for_spyre,
    copy_pooler_output_to_cpu,
    select_rows,
)

logger = init_logger(__name__)

# Eager pooling dummy stays tiny (compile is a no-op; avoids a large DMA).
SPYRE_ENCODER_WARMUP_MAX_TOKENS = 16


def compilation_disabled_reason(enforce_eager: bool, mode: CompilationMode) -> str | None:
    """Why ``torch.compile`` is skipped, or ``None`` if we should compile.

    Spyre stays eager when ``compilation_config.mode`` is unset/NONE even if
    ``enforce_eager=False`` (vLLM's default). Do not report that as
    ``enforce_eager=True``.
    """
    if enforce_eager:
        return "enforce_eager=True"
    if mode is CompilationMode.NONE:
        return (
            "compilation mode is NONE; pass compilation_config "
            "mode=STOCK_TORCH_COMPILE to enable compile"
        )
    return None


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

    # KV manager block, then the kernel block within it.
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
        # Set during pooling execute_model when the batch is expanded to (B, L).
        self._encoder_bucket: EncoderBucketPad | None = None

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

        # Replace Triton kernel with a pure-PyTorch implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # The upstream CPU backend uses a C++ kernel (torch.ops._C) as its
        # fallback, but we don't have _C.abi3.so with VLLM_TARGET_DEVICE=empty.
        from vllm.v1.worker import block_table

        # Deliberately swap the Triton JITFunction for the grid-launch-compatible
        # _FuncWrapper; the type mismatch is the point of the patch.
        block_table._compute_slot_mapping_kernel = _compute_slot_mapping_kernel  # ty: ignore[invalid-assignment]

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

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile for Spyre with static shapes.

        Spyre requires static shapes — dynamic shapes (SymInt) are not yet supported.
        We therefore pass `dynamic=False` to torch.compile(...).

        Supported modes:

        - CompilationMode.NONE: eager execution
        - CompilationMode.STOCK_TORCH_COMPILE: whole-model torch.compile
        """
        mode = self.compilation_config.mode
        if mode not in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. Only "
                f"CompilationMode.NONE and CompilationMode.STOCK_TORCH_COMPILE "
                f"are supported."
            )

        reason = compilation_disabled_reason(self.vllm_config.model_config.enforce_eager, mode)
        if reason:
            logger.info("Compilation disabled (%s)", reason)
            return

        # Trigger whole-model compile:
        # a single fullgraph over the entire model using dynamic=False.
        t0 = time.time()
        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info(
            "Compiled model %s as a single graph for Spyre in %.3fs.",
            type(self.get_model()).__name__,
            time.time() - t0,
        )

    def warming_up_model(self) -> None:
        """Warm kernels / compile.

        Eager pooling: one short dummy (DMA-safe; compile is off).
        Compiled pooling: dummy each encoder bucket at full ``B × L``, then
        again at ``L-2`` / ``L-1`` so pad leftovers compile. Upstream dummy
        skips encoder attention unless ``force_attention=True``.
        """
        logger.info("Warming up model...")
        t0 = time.time()
        with _set_spyre_compilation_settings(self.vllm_config):
            is_pooling = self.model_config.runner_type == "pooling"
            compiled = (
                not self.vllm_config.model_config.enforce_eager
                and self.compilation_config.mode is CompilationMode.STOCK_TORCH_COMPILE
            )
            if is_pooling and compiled:
                self._warmup_pooling_bucket_shapes()
            elif is_pooling:
                # Eager: no graph to specialize, so dummy size is not a
                # batch-size policy. Compiled warmup uses
                # SPYRE_ENCODER_BUCKET_{LENS,BATCH_SIZES} instead.
                num_tokens = min(
                    SPYRE_ENCODER_WARMUP_MAX_TOKENS,
                    self.scheduler_config.max_num_batched_tokens,
                )
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
                num_tokens = min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
                self._dummy_run(num_tokens)
        logger.info("Warmup done in %.3fs.", time.time() - t0)

    def _warmup_pooling_bucket_shapes(self) -> None:
        """Dummy each ``(B, L)`` at full size, then ``L-2`` / ``L-1`` pad leftovers."""
        shapes = pooling_warmup_shapes(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
        )
        if not shapes:
            logger.warning("No pooling warmup shapes; falling back to a single dummy run")
            self._dummy_run(
                min(16, self.scheduler_config.max_num_batched_tokens),
                force_attention=True,
            )
            return

        saved_max_num_seqs = self.scheduler_config.max_num_seqs
        try:
            for batch_size, prompt_len in shapes:
                self.scheduler_config.max_num_seqs = batch_size
                num_tokens = batch_size * prompt_len
                logger.info(
                    "Pooling warmup: batch_size=%d prompt_len=%d (%d tokens)",
                    batch_size,
                    prompt_len,
                    num_tokens,
                )
                hidden_states, _ = self._dummy_run(num_tokens, force_attention=True)
                self._dummy_pooler_run(hidden_states)
                for orig_len in pooling_warmup_pad_query_lens(prompt_len):
                    orig_tokens = batch_size * orig_len
                    logger.info(
                        "Pooling warmup: batch_size=%d prompt_len=%d "
                        "(dummy %d seqs × %d tokens, pad to %d)",
                        batch_size,
                        prompt_len,
                        batch_size,
                        orig_len,
                        num_tokens,
                    )
                    self._seed_pooling_bucket_dummy(batch_size, prompt_len, orig_len)
                    try:
                        hidden_states, _ = self._dummy_run(orig_tokens, force_attention=True)
                        hidden_states = self._unpad_encoder_hidden(hidden_states, orig_tokens)
                        self._dummy_pooler_run(hidden_states)
                    finally:
                        self._encoder_bucket = None
        finally:
            self.scheduler_config.max_num_seqs = saved_max_num_seqs

    def _encoder_pad_token_id(self) -> int:
        pad_token_id = getattr(self.model_config.hf_config, "pad_token_id", None)
        return 0 if pad_token_id is None else int(pad_token_id)

    def _write_encoder_bucket_inputs(self, padded_ids: list[int], padded_pos: list[int]) -> None:
        num_tokens = len(padded_ids)
        self.input_ids.cpu[:num_tokens].copy_(
            torch.tensor(padded_ids, dtype=self.input_ids.cpu.dtype)
        )
        if self.input_ids.gpu is not self.input_ids.cpu:
            self.input_ids.copy_to_gpu(num_tokens)
        self.positions[:num_tokens].copy_(
            torch.tensor(
                padded_pos,
                dtype=self.positions.dtype,
                device=self.positions.device,
            )
        )

    def _seed_pooling_bucket_dummy(self, batch_size: int, prompt_len: int, orig_len: int) -> None:
        orig_lens = [orig_len] * batch_size
        orig_tokens = orig_len * batch_size
        positions = [offset for _ in range(batch_size) for offset in range(orig_len)]
        padded_ids, padded_pos = expand_packed_to_encoder_bucket(
            [0] * orig_tokens,
            positions,
            orig_lens,
            batch_size,
            prompt_len,
            pad_token_id=self._encoder_pad_token_id(),
        )
        self._write_encoder_bucket_inputs(padded_ids, padded_pos)
        self._encoder_bucket = EncoderBucketPad(
            batch_bucket=batch_size,
            len_bucket=prompt_len,
            orig_query_lens=orig_lens,
            orig_num_tokens=orig_tokens,
            orig_num_reqs=batch_size,
        )
        self._apply_encoder_bucket_attn_layout(self._encoder_bucket)

    def _apply_encoder_bucket_attn_layout(self, bucket: EncoderBucketPad) -> None:
        """Packed ``query_start_loc`` is ``B`` rows of ``L``; ``seq_lens`` stay real."""
        num_tokens = bucket.num_tokens
        batch_bucket = bucket.batch_bucket
        len_bucket = bucket.len_bucket
        num_reqs = bucket.orig_num_reqs
        qsl = np.arange(0, num_tokens + 1, len_bucket, dtype=self.query_start_loc.np.dtype)
        self.query_start_loc.np[: batch_bucket + 1] = qsl
        self.query_start_loc.np[batch_bucket + 1 :].fill(num_tokens)
        self.query_start_loc.copy_to_gpu()

        orig = torch.tensor(bucket.orig_query_lens, dtype=self.optimistic_seq_lens_cpu.dtype)
        self.optimistic_seq_lens_cpu[:num_reqs] = orig
        self.seq_lens[:num_reqs] = orig.to(device=self.seq_lens.device)
        dummy = torch.full(
            (batch_bucket - num_reqs,),
            len_bucket,
            dtype=self.optimistic_seq_lens_cpu.dtype,
        )
        self.optimistic_seq_lens_cpu[num_reqs:batch_bucket] = dummy
        self.seq_lens[num_reqs:batch_bucket] = dummy.to(device=self.seq_lens.device)
        self.optimistic_seq_lens_cpu[batch_bucket:].fill_(0)
        self.seq_lens[batch_bucket:].fill_(0)

    @torch.inference_mode()
    def _dummy_run(self, *args, **kwargs):
        """Force D2H during dummy forward (upstream logits index is CPU).

        Pooling must pass ``force_attention=True``. Upstream skips attention
        metadata unless that flag or a FULL cudagraph is set; encoder impl
        then does ``if attn_metadata is None: return output`` and never
        compiles pack/SDPA. Real ``execute_model`` always has metadata.
        """
        if self.model_config.runner_type == "pooling":
            kwargs.setdefault("force_attention", True)
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
        return hidden_states, last_hidden_states

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self._encoder_bucket = None
        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            self._encoder_bucket = None

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        result = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        self._maybe_expand_pooling_inputs_to_encoder_bucket(num_scheduled_tokens)
        return result

    def _maybe_expand_pooling_inputs_to_encoder_bucket(
        self, num_scheduled_tokens: np.ndarray
    ) -> None:
        """Rewrite packed inputs to ``B`` sequences of length ``L`` (``T = B×L``).

        Linear/LN compile on the flat token count; attention compiles on
        ``[B, H, L, D]``. One warmup per ``(B, L)`` covers both when the
        runtime batch is padded the same way. Attention still masks with the
        original lengths (kept in ``seq_lens``).
        """
        self._encoder_bucket = None
        if self.model_config.runner_type != "pooling":
            return

        num_reqs = self.input_batch.num_reqs
        orig_lens = [int(n) for n in num_scheduled_tokens[:num_reqs]]
        orig_tokens = int(sum(orig_lens))
        if orig_tokens <= 0:
            return

        bucket = runtime_encoder_bucket(
            num_seqs=num_reqs,
            max_query_len=max(orig_lens),
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
        )
        if bucket is None:
            return
        batch_bucket, len_bucket = bucket
        if (
            batch_bucket == num_reqs
            and orig_tokens == batch_bucket * len_bucket
            and all(length == len_bucket for length in orig_lens)
        ):
            return

        padded_ids, padded_pos = expand_packed_to_encoder_bucket(
            self.input_ids.cpu[:orig_tokens].tolist(),
            self.positions[:orig_tokens].detach().cpu().tolist(),
            orig_lens,
            batch_bucket,
            len_bucket,
            pad_token_id=self._encoder_pad_token_id(),
        )
        self._write_encoder_bucket_inputs(padded_ids, padded_pos)
        self._encoder_bucket = EncoderBucketPad(
            batch_bucket=batch_bucket,
            len_bucket=len_bucket,
            orig_query_lens=orig_lens,
            orig_num_tokens=orig_tokens,
            orig_num_reqs=num_reqs,
        )
        self._apply_encoder_bucket_attn_layout(self._encoder_bucket)

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices=None,
    ):
        bucket = self._encoder_bucket
        if bucket is not None:
            num_tokens_padded = bucket.num_tokens
            num_reqs_padded = bucket.batch_bucket
            num_tokens_unpadded = bucket.num_tokens
        return super()._get_slot_mappings(
            num_tokens_padded,
            num_reqs_padded,
            num_tokens_unpadded,
            ubatch_slices,
        )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        if self._encoder_bucket is not None:
            return self._encoder_bucket.num_tokens
        return super()._pad_for_sequence_parallelism(num_scheduled_tokens)

    def _build_attention_metadata(self, *args, **kwargs):
        bucket = self._encoder_bucket
        if bucket is not None:
            # dummy_run overwrites query_start_loc / seq_lens; restore (B, L).
            self._apply_encoder_bucket_attn_layout(bucket)
            kwargs["num_tokens"] = bucket.num_tokens
            kwargs["num_reqs"] = bucket.batch_bucket
            kwargs["max_query_len"] = bucket.len_bucket
            kwargs["num_tokens_padded"] = bucket.num_tokens
            kwargs["num_reqs_padded"] = bucket.batch_bucket
            args = ()
        return super()._build_attention_metadata(*args, **kwargs)

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        bucket = self._encoder_bucket
        saved_pos = None
        if bucket is not None:
            saved_pos = self.positions[: bucket.num_tokens].clone()
        result = super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
        if saved_pos is None:
            return result
        self.positions[: saved_pos.shape[0]].copy_(saved_pos)
        input_ids, inputs_embeds, _positions, *rest = result
        return (input_ids, inputs_embeds, self.positions[:num_input_tokens], *rest)

    def _unpad_encoder_hidden(
        self, hidden_states: torch.Tensor, num_scheduled_tokens: int
    ) -> torch.Tensor:
        """Gather real tokens out of a ``B×L`` packed hidden state."""
        bucket = self._encoder_bucket
        if bucket is None:
            if hidden_states.shape[0] != num_scheduled_tokens:
                hidden_states = select_rows(
                    hidden_states, torch.arange(num_scheduled_tokens, dtype=torch.int64)
                )
            return hidden_states
        indices = encoder_bucket_valid_row_indices(bucket.orig_query_lens, bucket.len_bucket)
        return select_rows(hidden_states, torch.tensor(indices, dtype=torch.int64))

    def _restore_orig_query_start_loc(self) -> None:
        bucket = self._encoder_bucket
        if bucket is None:
            return
        cu = np.cumsum([0, *bucket.orig_query_lens], dtype=self.query_start_loc.np.dtype)
        num_reqs = bucket.orig_num_reqs
        self.query_start_loc.np[: num_reqs + 1] = cu
        self.query_start_loc.np[num_reqs + 1 :].fill(cu[-1])
        self.query_start_loc.copy_to_gpu()

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        """Same as GPU dummy pooler, but the cursor stays on CPU like ``_pool``."""
        if not self._pooling_on_spyre:
            return super()._dummy_pooler_run_task(hidden_states, task)

        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs
        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            prompt_token_ids_cpu=dummy_token_ids.cpu(),
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )
        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=torch.device("cpu"),
        )
        return model.pooler(hidden_states=hidden_states, pooling_metadata=dummy_metadata)

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
            hidden_states = self._unpad_encoder_hidden(
                convert(hidden_states, "cpu"), num_scheduled_tokens
            )
            self._restore_orig_query_start_loc()
            return super()._pool(
                hidden_states,
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
        hidden_states = self._unpad_encoder_hidden(hidden_states, num_scheduled_tokens)
        self._restore_orig_query_start_loc()

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
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
