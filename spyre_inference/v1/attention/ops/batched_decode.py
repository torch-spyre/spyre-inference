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

"""Batched multi-sequence decode over a head-major KV cache.

Each index selects a whole page containing all KV heads. The page is already
head-major, so it needs no per-chunk permutation, and the gathered page is LX-resident
as it stands. The per-sequence kernel's ``(page, kv_head)``
fold exists to win that residency, so there is nothing for it to buy here: folding anyway
measured roughly 1.5-2x slower.

With multiple chunks, or a chunk count only known at run time, each block slot keeps
its own running softmax across chunks; the slots merge once afterward. For two slots,
slot 0 processes blocks 0, 2, 4, ... and slot 1 processes blocks 1, 3, 5, ... of each
sequence.
Only the temporary softmax state grows; the permanent KV cache is unchanged.

In shared-slot mode the chunk's 32 block slots are not tied to a sequence: the host
assigns each one to a sequence per step, and a ``[slots, rows]`` one-hot then merges each
sequence's block slots into its output row, so no shape depends on the batch size.
"""

import torch

from spyre_inference.v1.attention.ops.sinks import normalize


def _entry_local_update(
    carry,
    sc,
    v_page,
    entries,
    blocks_per_chunk,
    num_seqs,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
):
    """Update each slot's maximum, unnormalized sum and weighted-value sum."""

    def pv(probs):
        return torch.matmul(
            probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size), v_page
        ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    # A slot can be masked in every chunk. A finite sentinel avoids -inf - -inf;
    # its weight vanishes at the final merge against a slot with valid scores.
    # The metadata builder gives every real sequence at least one valid token.
    sc = torch.clamp(sc, min=torch.finfo(sc.dtype).min)
    pos_max = torch.amax(sc, dim=-1, keepdim=True)
    tile_max, tile_sum, tile_output = carry
    rescale = torch.exp(-torch.relu(pos_max - tile_max))
    new_max = torch.maximum(tile_max, pos_max)
    probs = torch.exp(sc - new_max)
    return (
        new_max,
        tile_sum * rescale + torch.sum(probs, dim=-1, keepdim=True),
        tile_output * rescale + pv(probs),
    )


def _merge_entry_local(
    *,
    tile_max,
    tile_sum,
    tile_output,
    num_seqs,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    sinks=None,
):
    """Rescale slots to a common maximum, sum them, then normalize once.

    The sink joins after the merge, so a sequence counts it once rather than once per slot.
    """
    merged_max = torch.amax(tile_max, dim=0, keepdim=True)
    weight = torch.exp(tile_max - merged_max)
    merged_sum = torch.sum(weight * tile_sum, dim=0)
    merged_out = torch.sum(weight * tile_output, dim=0)
    return normalize(merged_max[0], merged_sum, merged_out, sinks).reshape(
        num_seqs, num_kv_heads * num_queries_per_kv, head_size
    )


def _merge_shared_slots(*, tile_max, tile_sum, tile_output, slot_onehot, sinks=None):
    """Rescale each row's block slots to the row's maximum, sum them, then normalize once.

    ``slot_onehot`` [E, R, 1, 1, 1] marks the rows each block slot counts towards, so it is
    ``_merge_entry_local`` with the ``(J, B)`` split replaced by a one-hot: every input
    keeps the carry's rank and the sums run over dim 0. A row past the batch repeats row
    0's slots, so all of a slot's rows share one maximum. An idle slot is in no row.
    """
    a = slot_onehot
    neg = torch.finfo(tile_max.dtype).min
    # Masked, not multiplied out: a slot outside a row must not lift its maximum.
    row_max = torch.amax(a * tile_max + (1 - a) * neg, dim=0, keepdim=True)
    slot_ref = torch.amax(a * row_max + (1 - a) * neg, dim=1, keepdim=True)
    weight = torch.exp(-torch.relu(slot_ref - tile_max))
    row_sum = torch.sum(a * (weight * tile_sum), dim=0)
    row_out = torch.sum(a * (weight * tile_output), dim=0)
    return normalize(row_max[0], row_sum, row_out, sinks)


def batched_decode_kernel(
    *,
    query,
    rep_row_ids,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
    num_seqs,
    blocks_per_chunk,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
    v_scale=None,
    sinks=None,
    entry_local,
    slot_onehot=None,
):
    """Read head-major pages and compute one decode output per sequence/query head.

    K/V: [num_pages_total, num_kv_heads, block_size, head_size]. Let C be the
    number of chunks, J=blocks_per_chunk, B=num_seqs and E=J*B. Within a chunk,
    entry j*B+s selects block slot j for sequence s; rep_row_ids repeats the
    sequence's query in that same order.

    Page IDs [C, E, 1], mask [C, J, B, num_kv_heads, 1, block_size]: the builder's
    [C*J, B] metadata, split per chunk on upload. The temporary split-index layout
    places each page ID in a separate stick so entries can split across cores.

    A float8 page is cast to the query dtype after its gather and v_scale scales the
    output; the caller folds k_scale into ``scale``. sinks: [num_kv_heads,
    num_queries_per_kv, 1] per-head sink logits, or None. entry_local keeps the per-slot
    softmax state, which more than one chunk needs; one chunk has no cross-slot merge to
    defer. A dynamic chunk count must keep it, since branching on the count would be a
    guard.

    Shared-slot mode: ``slot_onehot`` [E, R, 1, 1, 1] (see ``_merge_shared_slots``), with
    num_seqs=1, blocks_per_chunk=E and entry_local, since only the merge assigns the slots
    to sequences. The output is [R, num_heads, head_size].
    """
    from torch_spyre._inductor.wsr import for_each_tile

    shared = slot_onehot is not None
    assert not shared or (num_seqs == 1 and entry_local)
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    q = query.index_select(0, rep_row_ids).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    def reduce_chunk(probs, v_page):
        """Sum the chunk's blocks; its slots share one max, so this needs no rescale."""
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=0, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
            dim=0,
            keepdim=True,
        )
        return chunk_sum, chunk_out

    def chunk_body(carry, tiles):
        page_ids, mask_rows, k_pages, v_pages, q = tiles
        # Consume the chunk axis -> [E, 1] pages the gather splits on.
        page_ids = page_ids[0]
        mask_rows = mask_rows.reshape(blocks_per_chunk, num_seqs, *mask_rows.shape[3:])
        # Subscripting, not index_select: behind a 1-D index the entry axis splits only in
        # whole 32-entry sticks, and flattening the int32 tile first needs an unsupported
        # staging layout. Costs the eager path, which the preconditions decline.
        k_page = k_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
        v_page = v_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
        if k_page.dtype != q.dtype:
            k_page = k_page.to(q.dtype)
            v_page = v_page.to(q.dtype)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # Leading-axis split only: torch-spyre rejects merging a permuted axis pair, and
        # the mask's advancing read window must stay unflattened.
        sc = scores.reshape(
            blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, block_size
        )
        sc = sc + mask_rows
        if entry_local:
            return (
                _entry_local_update(
                    carry,
                    sc,
                    v_page,
                    entries,
                    blocks_per_chunk,
                    num_seqs,
                    num_kv_heads,
                    num_queries_per_kv,
                    block_size,
                    head_size,
                ),
                None,
            )
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past a
        # sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        tile_max, tile_sum, tile_output = carry
        # Read tile_max before the maximum that supersedes it, or the tiled lowering
        # copies the whole carry every trip. Identical to exp(tile_max - new_max).
        rescale = torch.exp(-torch.relu(chunk_max - tile_max))
        new_max = torch.maximum(tile_max, chunk_max)
        chunk_sum, chunk_out = reduce_chunk(torch.exp(sc - new_max), v_page)
        return (
            new_max,
            tile_sum * rescale + chunk_sum,
            tile_output * rescale + chunk_out,
        ), None

    slots = (
        (blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv)
        if entry_local
        else (1, num_seqs, num_kv_heads, num_queries_per_kv)
    )
    state_shape = (*slots, 1)
    out_shape = (*slots, head_size)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    carry, _ = for_each_tile(
        chunk_body,
        (chunk_page_ids, mask_by_chunk, k_pages, v_pages, q),
        dims=(0, 0, None, None, None),
        tile_size=1,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(out_shape, **state_kwargs),
        ),
    )
    tile_max, tile_sum, tile_output = carry
    out_rows = num_seqs
    if slot_onehot is not None:
        out_rows = slot_onehot.shape[1]
        attn = _merge_shared_slots(
            tile_max=tile_max,
            tile_sum=tile_sum,
            tile_output=tile_output,
            slot_onehot=slot_onehot,
            sinks=sinks,
        ).reshape(out_rows, num_heads, head_size)
    elif entry_local:
        attn = _merge_entry_local(
            tile_max=tile_max,
            tile_sum=tile_sum,
            tile_output=tile_output,
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            sinks=sinks,
        )
    else:
        attn = normalize(tile_max, tile_sum, tile_output, sinks).reshape(
            num_seqs, num_heads, head_size
        )
    if v_scale is not None and v_scale != 1.0:
        attn = attn * v_scale
    if out is not None:
        # Offset 0, so torch-spyre#3770 does not apply; rows past the batch are
        # don't-care and kept finite by the builder.
        out[:out_rows].copy_(attn)
        return out
    return attn
