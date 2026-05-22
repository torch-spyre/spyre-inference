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

"""Shared lm_head matmul helpers for SpyreParallelLMHead.

Supports two execution shapes:

1. Single-chunk: one padded F.linear on Spyre. Used for small vocab × hidden
   matmuls (e.g. Granite 4.1).
2. N-chunk: weight split along vocab dim, one F.linear per chunk, concat on CPU.
   Required for large vocab × hidden (e.g. Qwen3-8B 151936×4096) because the
   single matmul exceeds torch-spyre's per-core 256 MB EAR span limit
   (see torch_spyre/_inductor/work_division.py).

Chunk count selection:
- SPYRE_LM_HEAD_NUM_CHUNKS env var overrides everything (e.g. "1", "8").
- Otherwise: 1 chunk if weight bytes <= SPYRE_LM_HEAD_SINGLE_THRESHOLD_MIB
  (default 200 MiB), else SPYRE_LM_HEAD_DEFAULT_CHUNKS (default 8) — mirroring
  hf_adapters/hf_common.py chunk_lm_head.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from .utils import convert

logger = init_logger(__name__)

# Defaults match hf_adapters chunk_lm_head behavior.
_DEFAULT_CHUNKS_FALLBACK = 8
_DEFAULT_SINGLE_THRESHOLD_MIB = 200


@dataclass
class LmHeadMatmulState:
    """Prepared lm_head sub-weights for a Spyre F.linear logits matmul.

    chunks[i] is the i-th padded sub-weight (shape [padded_rows_i, hidden]).
    real_sizes[i] is the unpadded row count for chunk i (used to slice logits).
    forward_fn is the compiled per-chunk F.linear.
    """

    chunks: list[torch.Tensor] = field(default_factory=list)
    real_sizes: list[int] = field(default_factory=list)
    forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor] | None = (
        None
    )

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


def compute_chunk_padding(num_rows: int) -> int:
    """Return additional padding rows for torch-spyre matmul shape rules.

    Rule: row count must be a multiple of ``64 * 32 = 2048``.
    """
    if num_rows % 64 != 0:
        raise ValueError(
            f"lm_head chunk row count {num_rows} is not a multiple of 64; "
            "upstream vocab padding should have aligned it."
        )
    blocks = num_rows // 64
    rem = blocks % 32
    if rem == 0:
        return 0
    return (32 - rem) * 64


def _resolve_num_chunks(weight: torch.Tensor) -> int:
    """Pick chunk count from SPYRE_LM_HEAD_NUM_CHUNKS or size heuristic."""
    env = os.environ.get("SPYRE_LM_HEAD_NUM_CHUNKS")
    if env is not None:
        try:
            n = int(env)
        except ValueError as exc:
            raise ValueError(f"SPYRE_LM_HEAD_NUM_CHUNKS must be an integer, got {env!r}") from exc
        return max(1, n)

    try:
        threshold_mib = int(
            os.environ.get(
                "SPYRE_LM_HEAD_SINGLE_THRESHOLD_MIB",
                _DEFAULT_SINGLE_THRESHOLD_MIB,
            )
        )
    except ValueError:
        threshold_mib = _DEFAULT_SINGLE_THRESHOLD_MIB

    weight_bytes = weight.numel() * weight.element_size()
    if weight_bytes <= threshold_mib * 1024 * 1024:
        return 1

    try:
        default_chunks = int(
            os.environ.get(
                "SPYRE_LM_HEAD_DEFAULT_CHUNKS",
                _DEFAULT_CHUNKS_FALLBACK,
            )
        )
    except ValueError:
        default_chunks = _DEFAULT_CHUNKS_FALLBACK
    return max(1, default_chunks)


def build_lm_head_chunks(
    weight: torch.Tensor,
    num_chunks: int,
    layer_name: str,
) -> tuple[list[torch.Tensor], list[int]]:
    """Split weight along dim 0 into ``num_chunks`` padded sub-weights.

    Each chunk is sized in units of 64 rows ("sticks") so it remains 64-aligned
    (required by torch-spyre), then padded so its row count is a multiple of
    ``64 * 32`` to satisfy the F.linear shape rule. Stick-count remainder is
    distributed across the first chunks so chunk sizes differ by at most 64
    rows.

    Returns ``(chunks, real_sizes)`` where ``real_sizes[i]`` is the unpadded
    row count of chunk i (used to slice logits after the matmul).
    """
    vocab = weight.shape[0]
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if vocab % 64 != 0:
        raise ValueError(
            f"{layer_name}: lm_head vocab dim {vocab} must be a multiple of 64 "
            "(upstream pad_vocab_size_to_multiple_of should align it)."
        )

    total_sticks = vocab // 64
    if num_chunks > total_sticks:
        raise ValueError(
            f"num_chunks={num_chunks} exceeds total 64-row sticks ({total_sticks}) for {layer_name}"
        )

    sticks_base = total_sticks // num_chunks
    sticks_remainder = total_sticks % num_chunks

    chunks: list[torch.Tensor] = []
    real_sizes: list[int] = []
    cursor = 0
    for i in range(num_chunks):
        sticks = sticks_base + (1 if i < sticks_remainder else 0)
        real_rows = sticks * 64
        end = cursor + real_rows
        sub = weight[cursor:end].clone()
        cursor = end
        # 64*32-block alignment for torch-spyre's F.linear work_division.
        pad_rows = compute_chunk_padding(sub.shape[0])
        if pad_rows > 0:
            sub = F.pad(sub, (0, 0, 0, pad_rows))
        chunks.append(sub)
        real_sizes.append(real_rows)

    if cursor != vocab:
        raise AssertionError(
            f"lm_head chunking lost rows for {layer_name}: cursor={cursor}, vocab={vocab}"
        )

    if num_chunks > 1 or chunks[0].shape[0] != vocab:
        # warning_once hashes *args for dedup, so pass tuples (not lists).
        logger.warning_once(
            "%s: lm_head split into %d chunk(s); real_sizes=%s, padded_sizes=%s "
            "(torch-spyre 256 MB per-core span workaround) — expect numerical "
            "differences to upstream vLLM.",
            layer_name,
            num_chunks,
            tuple(real_sizes),
            tuple(c.shape[0] for c in chunks),
        )
    return chunks, real_sizes


def _make_forward_spyre() -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
]:
    def forward_spyre(
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, w, bias)

    return forward_spyre


def build_lm_head_matmul_state(
    weight: torch.Tensor,
    layer_name: str,
    compile_owner: nn.Module,
    num_chunks: int | None = None,
) -> LmHeadMatmulState:
    """Build chunks + compiled F.linear for an lm_head weight."""
    n = num_chunks if num_chunks is not None else _resolve_num_chunks(weight)
    chunks, real_sizes = build_lm_head_chunks(weight, n, layer_name)
    forward_fn = maybe_compile_spyre_linear(compile_owner, _make_forward_spyre())
    return LmHeadMatmulState(chunks=chunks, real_sizes=real_sizes, forward_fn=forward_fn)


def setup_lm_head_padding(layer: nn.Module) -> None:
    """Build LmHeadMatmulState on a ParallelLMHead layer.

    Stores the state at ``layer._spyre_matmul_state``. For the single-chunk
    no-padding case, the first chunk aliases ``layer.weight`` to avoid a
    duplicate vocab-sized allocation.
    """
    state = build_lm_head_matmul_state(layer.weight, layer.__class__.__name__, layer)
    if state.num_chunks == 1 and state.chunks[0].shape[0] == state.real_sizes[0]:
        state.chunks[0] = layer.weight
    layer._spyre_matmul_state = state


def forward_lm_head_matmul(
    state: LmHeadMatmulState,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run lm_head F.linear on Spyre; input x is typically on CPU.

    For chunked paths, runs one F.linear per chunk and concatenates on CPU.
    """
    if state.forward_fn is None or not state.chunks:
        raise RuntimeError(
            "LmHeadMatmulState is not initialized; call build_lm_head_matmul_state first."
        )

    x_device = x.device
    weight_device = state.chunks[0].device
    x_spyre = convert(x, device=weight_device)

    if state.num_chunks == 1:
        chunk = state.chunks[0]
        real_size = state.real_sizes[0]
        padding = chunk.shape[0] - real_size
        if bias is not None and padding > 0:
            bias_chunk = convert(F.pad(bias, (0, padding)), device=weight_device)
        elif bias is not None:
            bias_chunk = convert(bias, device=weight_device)
        else:
            bias_chunk = None
        out = state.forward_fn(x_spyre, chunk.data, bias_chunk)
        out_cpu = convert(out, device="cpu")
        out_cpu_unpadded = out_cpu[..., :real_size] if padding > 0 else out_cpu
        return convert(out_cpu_unpadded, device=x_device)

    # N-chunk path: F.linear adds bias to the full (padded) chunk output, so
    # the per-chunk bias must be padded to chunk.shape[0]. Bias indexes into
    # the real (unpadded) vocab; pad rows get zero bias.
    parts: list[torch.Tensor] = []
    bias_cursor = 0
    for chunk, real_size in zip(state.chunks, state.real_sizes):
        if bias is not None:
            real_slice = bias[bias_cursor : bias_cursor + real_size]
            bias_cursor += real_size
            pad_len = chunk.shape[0] - real_size
            if pad_len > 0:
                bias_chunk = F.pad(real_slice, (0, pad_len))
            else:
                bias_chunk = real_slice
            bias_chunk = convert(bias_chunk, device=weight_device)
        else:
            bias_chunk = None
        out = state.forward_fn(x_spyre, chunk.data, bias_chunk)
        out_cpu = convert(out, device="cpu")
        parts.append(out_cpu[..., :real_size])
    logits = torch.cat(parts, dim=-1)
    return convert(logits, device=x_device)


def forward_lm_head_oot(
    layer: nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """ParallelLMHead path — dispatches to ``layer._spyre_matmul_state``."""
    state: LmHeadMatmulState | None = getattr(layer, "_spyre_matmul_state", None)
    if state is None:
        raise RuntimeError(
            f"{layer.__class__.__name__}: _spyre_matmul_state not built; "
            "process_weights_after_loading must run before forward_oot."
        )
    return forward_lm_head_matmul(state, x, bias)


class SpyreUnquantizedLMHeadMethod(UnquantizedEmbeddingMethod):
    """Routes ParallelLMHead logits through layer.forward_oot()."""

    def apply(self, layer, x, bias=None):
        return layer.forward_oot(x, bias)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        setup_lm_head_padding(layer)


def maybe_compile_spyre_linear(layer: nn.Module, fn):
    """Compile helper for SpyreParallelLMHead's F.linear.

    Spyre runs with ``enforce_eager=True`` today (CompilationMode.NONE), so any
    maybe_compile wrapping is effectively a no-op. We still route through
    ``layer.maybe_compile`` for CustomOp owners so the hook is kept consistent
    with the rest of spyre-inference. The fallback handles non-CustomOp
    owners defensively.
    """
    if isinstance(layer, CustomOp):
        try:
            return layer.maybe_compile(fn)
        except Exception:
            return fn
    return fn
