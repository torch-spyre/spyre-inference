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

"""Encoder-only (bidirectional) self-attention for Spyre, without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY
layers. Two paths over one body buffer of ``R`` rows (``encoder_budget_rows``): a
rectangle, when the batch fits one, where Q/K/V already *are* the ``[B, L]`` grid the
runner laid out; otherwise the packed buffer, with requests grouped by their own extent
and each group a fused gather/attend/scatter. The runner picks once per step and records
the choice as the type of ``attn_metadata.encoder_plan``, so no compiled region branches
on it. Both paths cross the opaque ``unified_attention_with_output``, so the enclosing
block graph is identical for either.

Three torch-spyre constraints shape the design:

* A compile input's ``storage_offset`` is a Dynamo guard (torch-spyre#4449, which
  closed #3770), and for int32 it is still dropped outright, so rows are gathered with
  ``index_select``: slicing would either recompile per offset or read the wrong rows.
* There is no on-device ``arange`` or ``full``, so index and mask tensors are host-built
  and reach the device in one ``convert`` per plan.
* SDPA decomposes to ``amax`` then ``exp(scores - max)``, which NaNs a fully masked row.
  Hence the ``finfo.min / 2`` mask fill and the one attendable key a pad lane gets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
)
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_group_shapes,
    encoder_group_width_caps,
    encoder_rectangle_for_batch,
    encoder_rectangles,
    encoder_shape_tables,
)

# One Spyre stick of fp16, in tokens. A padded length that is a multiple of this
# keeps the row-index table stick-aligned and the matmul's contraction dimension
# aligned, and it is the width of a shared mask tile. Encoder attention has no KV
# cache and so no block walk -- this is alignment, not a block size.
ENCODER_LEN_ALIGNMENT = 64


def _alignment_units_for(length: int) -> int:
    """Stick-aligned units covering ``length``, rounded up to a power of two.

    Encoder self-attention has ``q_len == kv_len``, so this fixes the sequence's
    padded extent (``units * ENCODER_LEN_ALIGNMENT``) -- the attention kernel's cache
    has one length axis, not two. Rounding to a power of two keeps that axis to a
    handful of buckets, at the cost of padding a request up to the next one: a
    260-token request attends over 512, not 320.
    """
    return next_power_of_2(cdiv(length, ENCODER_LEN_ALIGNMENT))


def encoder_index_dtype(device: torch.device) -> torch.dtype:
    """int32 on Spyre, which has no int64 and whose compiled ``index_copy_``
    takes it; int64 everywhere else, where eager ``index_copy_`` rejects int32.
    """
    return torch.int32 if device.type == "spyre" else torch.int64


def encoder_row_table(start: int, query_len: int, extent: int, dtype: torch.dtype) -> torch.Tensor:
    """One sequence's absolute rows, pad lanes clamped to its last real row.

    ``extent`` is a multiple of ``ENCODER_LEN_ALIGNMENT`` and therefore of
    ``INT32_ELEMS_PER_STICK``, so the table is stick-aligned with no extra pad.

    The clamp makes the scatter write one destination row repeatedly. That is benign
    -- the mask depends only on the KV column, so every duplicate lane carries the
    same value -- but ``index_copy_`` with duplicate indices is formally undefined in
    PyTorch, so ``build_encoder_plan`` asserts the intent rather than leaving it
    implicit.
    """
    return torch.arange(extent, dtype=dtype).clamp(max=query_len - 1) + start


def encoder_key_pad_mask(extent: int, kv_lens: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Additive key-pad ``[N, 1, 1, extent]``, one row per sequence, on the host.

    Head and query axes stay 1 and broadcast: an encoder mask depends only on the KV
    column, since every query row -- real or padding -- attends to exactly the real
    keys.

    ``finfo.min / 2``, not ``finfo.min`` or ``-inf``: the fill is *added* to a score,
    and in fp16 ``finfo.min + score`` saturates to ``-inf``, which then NaNs through
    SDPA's ``exp(scores - amax)``. Halving leaves the headroom.

    Built whole on the host so a group's rows concatenate there and reach the device
    in one ``convert``. Assembling it from cached device tiles instead left an eager
    ``cat`` on the device, which torch-spyre compiled per tile pattern -- and warmup
    could not cover those, having only ever built masks with ``kv_len == extent``.
    """
    lens = torch.tensor(list(kv_lens), dtype=torch.int32).unsqueeze(1)
    pos = torch.arange(extent, dtype=torch.int32).unsqueeze(0)
    row = torch.where(
        pos < lens,
        torch.zeros((), dtype=dtype),
        torch.tensor(torch.finfo(dtype).min / 2, dtype=dtype),
    )
    return row.view(len(lens), 1, 1, extent).contiguous()


def _host_pad_head_dim(x: torch.Tensor, padded: int) -> torch.Tensor:
    """Widen the head dim to a whole stick, via the host.

    Below one stick several heads share a stick, and no device op -- compiled or
    eager -- can touch the per-head view that implies ("Unexpected stick expression
    d2 + 32*(Mod(d1, 2))"). ``convert`` is opaque, so the round trip is what escapes
    it; ``F.pad`` and a device ``cat``/``contiguous`` cannot. Zeros leave ``QK^T``
    unchanged and zero the extra output columns.
    """
    if x.shape[-1] == padded:
        return x
    device = x.device
    on_host = convert(x, "cpu") if device.type == "spyre" else x
    on_host = F.pad(on_host.contiguous(), (0, padded - x.shape[-1]))
    return convert(on_host, device) if device.type == "spyre" else on_host


def _widen_head_dim(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    padded_head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V widened to ``padded_head_size``, plus a fresh output buffer that shape.

    Spyre addresses the head dim in 64-element sticks, so a sub-stick head size has to
    run at the padded width throughout, on our own output rather than the caller's.
    ``_narrow_head_dim_into`` puts the result back.
    """
    return (
        _host_pad_head_dim(query, padded_head_size),
        _host_pad_head_dim(key, padded_head_size),
        _host_pad_head_dim(value, padded_head_size),
        convert(
            torch.zeros((output.shape[0], num_heads, padded_head_size), dtype=output.dtype),
            output.device,
        ),
    )


def _narrow_head_dim_into(padded: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Copy ``padded``'s real head columns into ``output``, and return ``output``.

    Narrowed on the host, then written through the flattened views, whose rows are a
    whole number of sticks -- a device-side narrow of the head dim is the per-head view
    ``_host_pad_head_dim`` cannot express.
    """
    rows = output.shape[0]
    on_host = convert(padded, "cpu")[..., : output.shape[-1]].contiguous()
    output.reshape(rows, -1).copy_(convert(on_host.reshape(rows, -1), output.device))
    return output


def _encoder_rect_kernel(
    query,
    key,
    value,
    mask,
    scale,
    width,
    extent,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Rectangular path: view ``[B*L, H, D]`` as the grid, attend, un-view. One graph.

    The reshapes stay inside the graph so the layout change is the matmul's problem
    rather than standalone d2d copies; these kernels are dispatch-bound, so the
    launch count is what matters.

    ``enable_gqa`` lets SDPA broadcast the KV heads itself, keeping operands 4-D: the
    rank-5 tensor an explicit expand would build is rejected by
    ``insert_restickify_padding``.
    """
    q = query.view(width, extent, num_heads, head_size).transpose(1, 2)
    k = key.view(width, extent, num_kv_heads, head_size).transpose(1, 2)
    v = value.view(width, extent, num_kv_heads, head_size).transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        scale=scale,
        is_causal=False,
        enable_gqa=(num_heads != num_kv_heads),
    )
    return attn.transpose(1, 2).reshape(width * extent, num_heads, head_size)


def _encoder_rect_kernel_out(
    out,
    query,
    key,
    value,
    mask,
    scale,
    width,
    extent,
    num_heads,
    num_kv_heads,
    head_size,
):
    """As above, writing the layer's output buffer inside the same graph."""
    out.copy_(
        _encoder_rect_kernel(
            query, key, value, mask, scale, width, extent, num_heads, num_kv_heads, head_size
        )
    )
    return out


def _encoder_gather_kernel(query, key, value, row_index):
    """Pull one group's rows out of the step's body buffer.

    Compiled alone for the eager/no-store path only; the fused kernel below inlines
    it. Keyed on ``(query.shape[0], row_index.shape[0])`` -- the body buffer is one
    fixed size, so that is effectively the group's row count alone.
    """
    q_rows = query.index_select(0, row_index)
    k_rows = key.index_select(0, row_index)
    v_rows = value.index_select(0, row_index)
    return q_rows, k_rows, v_rows


def _encoder_sdpa_kernel(
    q_rows,
    k_rows,
    v_rows,
    mask,
    scale,
    group,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Masked bidirectional attention over ``group`` sequences of equal padded length.

    ``group == 1`` is the ordinary single-sequence case, so this serves both.
    """
    extent = q_rows.shape[0] // group
    q = q_rows.reshape(group, extent, num_heads, head_size).transpose(1, 2)
    k = k_rows.reshape(group, extent, num_kv_heads, head_size).transpose(1, 2)
    v = v_rows.reshape(group, extent, num_kv_heads, head_size).transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        scale=scale,
        is_causal=False,
        enable_gqa=(num_heads != num_kv_heads),
    )
    return attn.transpose(1, 2).reshape(group * extent, num_heads, head_size)


def _encoder_fused_kernel(
    out,
    row_index,
    query,
    key,
    value,
    mask,
    scale,
    group,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Ragged path: gather, attend and store one group of equal-extent requests.

    Attention accounts for most of a step's jobplan launches, and each launch carries
    its own parameter upload, so collapsing three graphs into one is a device-path
    saving rather than host bookkeeping. The copies themselves do not get cheaper:
    ``index_select`` is charged for its source and ``index_copy_`` for its
    destination whatever graph they sit in.

    Keyed on ``(out.shape[0], group, extent)``. ``out`` is the body buffer, which is
    now one fixed size, so the key is the declared ``(width, extent)`` pair alone --
    that is what took this family from one graph per (buffer, width, extent) triple
    to one per pair.
    """
    q_rows, k_rows, v_rows = _encoder_gather_kernel(query, key, value, row_index)
    attn = _encoder_sdpa_kernel(
        q_rows, k_rows, v_rows, mask, scale, group, num_heads, num_kv_heads, head_size
    )
    out.index_copy_(0, row_index, attn)
    return out


_encoder_rect_compiled = torch.compile(_encoder_rect_kernel, dynamic=False)
_encoder_rect_out_compiled = torch.compile(_encoder_rect_kernel_out, dynamic=False)
_encoder_gather_compiled = torch.compile(_encoder_gather_kernel, dynamic=False)
_encoder_sdpa_compiled = torch.compile(_encoder_sdpa_kernel, dynamic=False)
_encoder_fused_compiled = torch.compile(_encoder_fused_kernel, dynamic=False)

for _kernel in (
    _encoder_rect_kernel,
    _encoder_rect_kernel_out,
    _encoder_gather_kernel,
    _encoder_sdpa_kernel,
    _encoder_fused_kernel,
):
    compile_guard.watch(_kernel, f"encoder attention kernel {_kernel.__name__}")


@dataclass
class EncoderRectPlan:
    """Rectangular path: the whole batch is one ``[width, extent]`` rectangle."""

    extent: int
    width: int
    mask: torch.Tensor
    """``[width, 1, 1, extent]`` additive key-pad, already on the device."""
    query_lens: list[int]
    """Real length per request. The runner needs it to lay out the grid and to
    compact the hidden states back afterwards, and this is the one place the ragged
    lengths were already read off the metadata."""


@dataclass
class EncoderGroupPlan:
    """Ragged path: one group of equal-extent requests inside the packed buffer.

    A group of one is the ordinary single-request case, so this covers both.
    """

    starts: list[int]
    query_lens: list[int]
    extent: int
    row_table: torch.Tensor
    mask: torch.Tensor

    @property
    def group(self) -> int:
        return len(self.starts)


def build_encoder_plan(
    attn_metadata: SpyreAttentionMetadata,
    *,
    rectangles: Sequence[tuple[int, int]],
    width_cap_for: dict[int, int],
    device: torch.device,
    dtype: torch.dtype,
    batched: bool,
) -> EncoderRectPlan | list[EncoderGroupPlan]:
    """Choose the path for this step and build what its kernels need.

    Module-level and query-free so the runner can call it before the model runs: inside
    a traced region its D2H reads and H2D converts would become graph nodes.

    ``query_start_loc``, not ``seq_lens``: query length *is* kv length here, and
    upstream's ``_dummy_run`` puts the whole padded token count in ``seq_lens``.
    """
    query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
    seq_lens = attn_metadata.seq_lens.cpu().tolist()
    # The body pads past num_actual_tokens; those rows are not a request.
    num_tokens = attn_metadata.num_actual_tokens

    members: list[tuple[int, int, int]] = []  # (start, query_len, kv_len)
    for seq_idx in range(attn_metadata.num_seqs):
        start = int(query_start_loc[seq_idx])
        query_len = int(query_start_loc[seq_idx + 1]) - start
        if start >= num_tokens or query_len <= 0:
            continue
        query_len = min(query_len, num_tokens - start)
        members.append((start, query_len, min(int(seq_lens[seq_idx]), query_len)))

    max_len = max((m[1] for m in members), default=0)
    # A dropped request shifts every later lane, and the pooler addresses rows by its
    # own cumsum over all requests, so the grid would be misaligned rather than padded.
    rect = (
        encoder_rectangle_for_batch(len(members), max_len, rectangles)
        if len(members) == attn_metadata.num_seqs
        else None
    )
    if rect is not None:
        extent, width = rect
        # Batch-pad lanes get one attendable key, not zero: an all-masked query row
        # NaNs inside SDPA's softmax. Their output is never read.
        kv_lens = [m[2] for m in members] + [1] * (width - len(members))
        return EncoderRectPlan(
            extent=extent,
            width=width,
            mask=convert(encoder_key_pad_mask(extent, kv_lens, dtype), device),
            query_lens=[m[1] for m in members],
        )

    by_extent: dict[int, list[tuple[int, int, int]]] = {}
    for member in members:
        extent = _alignment_units_for(member[1]) * ENCODER_LEN_ALIGNMENT
        by_extent.setdefault(extent, []).append(member)

    index_dtype = encoder_index_dtype(device)

    def plan(chunk: list[tuple[int, int, int]], extent: int) -> EncoderGroupPlan:
        # The row table's pad lanes repeat the last real row, so the scatter writes it
        # several times. Sound only because the mask ignores the query axis.
        assert all(m[1] >= 1 for m in chunk), "a zero-length request has no row to clamp onto"
        return EncoderGroupPlan(
            starts=[m[0] for m in chunk],
            query_lens=[m[1] for m in chunk],
            extent=extent,
            row_table=convert(
                torch.cat([encoder_row_table(m[0], m[1], extent, index_dtype) for m in chunk]),
                device,
            ),
            # Concatenated, not stacked: member order along dim 0 is what the
            # kernel's batch dim indexes.
            mask=convert(encoder_key_pad_mask(extent, [m[2] for m in chunk], dtype), device),
        )

    plans: list[EncoderGroupPlan] = []
    for extent, group in sorted(by_extent.items()):
        if not batched:
            plans.extend(plan([m], extent) for m in group)
            continue
        # Descending power-of-two chunks, not padding up to `cap`: padding would attend
        # phantom sequences. Default 1 keeps an undeclared extent correct, at one compile.
        cap = width_cap_for.get(extent, 1)
        offset = 0
        while offset < len(group):
            chunk = min(1 << ((len(group) - offset).bit_length() - 1), cap)
            plans.append(plan(group[offset : offset + chunk], extent))
            offset += chunk
    return plans


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``). ``forward`` reads the step's plan
    off the metadata and runs the path that plan's type names; it never decides the
    path itself, so every layer in the stack agrees.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._compile_attn:
            self._rect_fn = _encoder_rect_compiled
            self._rect_out_fn = _encoder_rect_out_compiled
            self._gather_fn = _encoder_gather_compiled
            self._attn_fn = _encoder_sdpa_compiled
            self._fused_fn = _encoder_fused_compiled
        else:
            self._rect_fn = _encoder_rect_kernel
            self._rect_out_fn = _encoder_rect_kernel_out
            self._gather_fn = _encoder_gather_kernel
            self._attn_fn = _encoder_sdpa_kernel
            self._fused_fn = _encoder_fused_kernel
        # get_current_vllm_config() only works at construction time; forward() runs
        # through a custom-op boundary that loses the context.
        config = get_current_vllm_config()
        self._rectangles = encoder_rectangles(config)
        self._group_shapes = encoder_group_shapes(config)
        self._width_caps = encoder_group_width_caps(config)
        self._buffer_rows = encoder_shape_tables(config).budget
        self._warmed = False

    def record_graphs(self, *args, **kwargs) -> int:
        """Nothing to page: ``warm_kernels`` traces every declared shape instead."""
        return 0

    def _run_rect(self, out, query, key, value, plan_mask, width, extent, heads, kv_heads, dim):
        if out is None:
            return _call_kernel(
                "encoder_rect",
                self._rect_fn,
                query,
                key,
                value,
                plan_mask,
                self.scale,
                width,
                extent,
                heads,
                kv_heads,
                dim,
            )
        return _call_kernel(
            "encoder_rect_out",
            self._rect_out_fn,
            out,
            query,
            key,
            value,
            plan_mask,
            self.scale,
            width,
            extent,
            heads,
            kv_heads,
            dim,
        )

    def _run_gather(self, query, key, value, row_index):
        return _call_kernel("encoder_gather", self._gather_fn, query, key, value, row_index)

    def _run_attn(self, q_rows, k_rows, v_rows, mask, group, heads, kv_heads, dim):
        return _call_kernel(
            "encoder_sdpa",
            self._attn_fn,
            q_rows,
            k_rows,
            v_rows,
            mask,
            self.scale,
            group,
            heads,
            kv_heads,
            dim,
        )

    def _run_fused(self, out, row_index, query, key, value, mask, group, heads, kv_heads, dim):
        return _call_kernel(
            "encoder_fused",
            self._fused_fn,
            out,
            row_index,
            query,
            key,
            value,
            mask,
            self.scale,
            group,
            heads,
            kv_heads,
            dim,
        )

    def warm_kernels(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> int:
        """Trace every declared shape, on the caller's own tensors.

        A Spyre tensor's device layout is part of its cache key, and neither a fused-QKV
        ``query`` (a strided view) nor ``output`` (a view of a 2-D allocation) is
        reproduced by a same-shaped fresh tensor. Writing into the caller's ``output`` is
        safe: the real work right after overwrites every real request's rows.

        A buffer that is not the declared body shape warms nothing -- every declared
        shape is sized against that one buffer, so a smaller one indexes out of bounds.
        """
        if self._warmed or query.shape[0] != self._buffer_rows:
            return 0
        self._warmed = True
        dtype, device = query.dtype, query.device
        index_dtype = encoder_index_dtype(device)
        traced = 0

        for extent, width in self._rectangles:
            mask = convert(encoder_key_pad_mask(extent, [extent] * width, dtype), device)
            self._run_rect(
                output, query, key, value, mask, width, extent, num_heads, num_kv_heads, head_size
            )
            traced += 1

        for width, extent in self._group_shapes:
            rows = convert(
                torch.cat([encoder_row_table(0, extent, extent, index_dtype)] * width),
                device,
            )
            mask = convert(encoder_key_pad_mask(extent, [extent] * width, dtype), device)
            self._run_fused(
                output,
                rows,
                query,
                key,
                value,
                mask,
                width,
                num_heads,
                num_kv_heads,
                head_size,
            )
            traced += 1
        return traced

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]

        # Everything runs where the result lands. A real step already has all four on
        # the same device; unit tests hand in host activations.
        if query.device != output.device:
            query = convert(query, output.device)
            key = convert(key, output.device)
            value = convert(value, output.device)

        plan = attn_metadata.encoder_plan
        if plan is None:
            # Fallback for a caller that did not pre-build one. Ragged path only: the grid
            # layout is a contract with ``_preprocess``, so a rectangle the runner did
            # not lay out would read the wrong rows.
            plan = build_encoder_plan(
                attn_metadata,
                rectangles=(),
                width_cap_for=self._width_caps,
                device=query.device,
                dtype=query.dtype,
                batched=self._compile_attn,
            )
            attn_metadata.encoder_plan = plan

        # A sub-stick head size runs the whole attention widened to a stick, on our own
        # output buffer; `caller_output` is the narrow one to write back into at the end.
        caller_output = output
        head_pad = -head_size % ENCODER_LEN_ALIGNMENT
        if head_pad:
            query, key, value, output = _widen_head_dim(
                query, key, value, output, num_heads, head_size + head_pad
            )
            head_size += head_pad

        # Re-checked per call: vLLM hands out a fresh output buffer per layer.
        fused_store_ok = (
            self._compile_attn
            and output.dtype == query.dtype
            # A nonzero offset is a separate compiled variant (torch-spyre#4449), so
            # store through our own buffer instead of specialising per buffer.
            and output.storage_offset() == 0
            and output.is_contiguous()
        )

        if fused_store_ok:
            # Here, not in the runner's warmup: a Spyre tensor's device layout is part of
            # the compile cache key, so the declared shapes must be traced against these
            # buffers. Self-guarded, so only the first call of the run does the work.
            self.warm_kernels(query, key, value, output, num_heads, num_kv_heads, head_size)

        if isinstance(plan, EncoderRectPlan):
            self._forward_rect(plan, query, key, value, output, fused_store_ok, head_size)
        else:
            self._forward_groups(plan, query, key, value, output, fused_store_ok, head_size)

        if output is not caller_output:
            return _narrow_head_dim_into(output, caller_output)
        return output

    def _forward_rect(self, plan, query, key, value, output, fused_store_ok, head_size) -> None:
        num_heads, num_kv_heads = query.shape[1], key.shape[1]
        if fused_store_ok:
            self._run_rect(
                output,
                query,
                key,
                value,
                plan.mask,
                plan.width,
                plan.extent,
                num_heads,
                num_kv_heads,
                head_size,
            )
            return
        attn = self._run_rect(
            None,
            query,
            key,
            value,
            plan.mask,
            plan.width,
            plan.extent,
            num_heads,
            num_kv_heads,
            head_size,
        )
        output.copy_(attn)

    def _forward_groups(self, plans, query, key, value, output, fused_store_ok, head_size) -> None:
        """Each group writes a disjoint row set, so order is immaterial."""
        num_heads, num_kv_heads = query.shape[1], key.shape[1]
        for plan in plans:
            if fused_store_ok:
                self._run_fused(
                    output,
                    plan.row_table,
                    query,
                    key,
                    value,
                    plan.mask,
                    plan.group,
                    num_heads,
                    num_kv_heads,
                    head_size,
                )
                continue
            q_rows, k_rows, v_rows = self._run_gather(query, key, value, plan.row_table)
            attn = self._run_attn(
                q_rows,
                k_rows,
                v_rows,
                plan.mask,
                plan.group,
                num_heads,
                num_kv_heads,
                head_size,
            )
            for i, (start, query_len) in enumerate(zip(plan.starts, plan.query_lens)):
                base = i * plan.extent
                output[start : start + query_len] = attn[base : base + query_len]


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
