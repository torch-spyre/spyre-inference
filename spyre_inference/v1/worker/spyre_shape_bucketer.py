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

"""Spyre shape bucketer for compilation warmup and runtime dispatch.

Body (1D, decoder): sorted ``compile_sizes`` token counts; pad the packed batch to
the nearest bucket ``>=`` actual ``num_tokens``. Linear / LN compile on ``[T, …]``.

Pooling's token budget ``R`` (``encoder_budget_rows``) is the most rows a step may
run, not the row count of every step. A step pads to the next multiple of 64 of its
real token count, and attention gathers each sequence out to its own stick length.
Q/K/V and the FFN therefore see that packed total, not a rectangle of the longest
prompt. The ragged kernels are staged on one fixed buffer so they do not recompile
for every total.

Both families come from one stick-aligned length ladder (every 64 tokens up to
``max_model_len``):

* ``encoder_rectangles`` -- the rectangular path. Several ``(L, W)`` per length, ``W`` a
  power of two up to the widest batch that length can hold (plus that cap when it is
  not itself a power of two).
* ``encoder_group_shapes`` -- attention's gather shapes. One ``(width, extent)`` per
  group of equal-extent requests. The body itself is not one of these.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from spyre_inference import envs

logger = init_logger(__name__)

# Spyre stick (64 fp16 elements), and the encoder attention KV block width.
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def _floor_pow2(n: int) -> int:
    return 1 << (max(1, int(n)).bit_length() - 1)


def next_bucket(n: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= n``. If ``n`` exceeds every bucket, stick-align ``n``."""
    if n < 1:
        n = 1
    ordered = sorted({b for b in buckets if b > 0})
    for bucket in ordered:
        if bucket >= n:
            return bucket
    return _align_up(n)


@dataclass(frozen=True)
class EncoderShapeTables:
    """Every encoder shape a pooling run can reach, derived from the engine limits."""

    budget: int
    """``R``: the most rows a step may run, and the ragged path's buffer."""
    lengths: tuple[int, ...]
    """The length ladder: every stick from 64 up to ``max_model_len``."""
    rectangles: tuple[tuple[int, int], ...]
    """Rectangular path ``(L, W)``, several widths per length."""
    groups: tuple[tuple[int, int], ...]
    """The ragged path's shapes, as ``(width, extent)``.

    A *group* is the unit the ragged path splits a batch into: the requests that share one
    padded extent, dispatched together as a single ``width``-wide kernel call. So one
    ragged step usually has several groups, and these pairs are every group shape it can
    produce. Empty when no batch can miss the rectangular path.
    """


def encoder_shape_tables(vllm_config: VllmConfig) -> EncoderShapeTables:
    """Resolve the length ladder and both shape families for this config.

    The platform hook, the runner and every attention layer re-derive these from the same
    config, so a 12-layer model paid for the derivation 14 times at startup. Cached on the
    limits it actually reads (env override included) rather than on the unhashable config.
    """
    return _encoder_shape_tables(
        int(vllm_config.model_config.max_model_len),
        int(vllm_config.scheduler_config.max_num_batched_tokens),
        max(1, int(vllm_config.scheduler_config.max_num_seqs)),
        envs.SPYRE_ATTN_QUERY_BUCKETS,
    )


@lru_cache(maxsize=8)
def _encoder_shape_tables(
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    query_buckets_override: str | None,
) -> EncoderShapeTables:
    del query_buckets_override  # read from envs below; a cache key only
    lengths = _encoder_lengths(max_model_len)
    budget = encoder_budget_rows(max_model_len, max_num_batched_tokens, max_num_seqs)

    # Several widths per length. The widest is the most sequences of that length the
    # budget can hold; narrower powers of two let a short batch skip the empty lanes.
    # Those lanes are real GEMM rows (QKV, attention, FFN), which is most of an
    # embedding step, so running them for requests that are not there is the cost.
    rectangles = tuple(
        (length, width)
        for length in lengths
        for width in _encoder_widths(min(max_num_seqs, budget // length))
    )

    return EncoderShapeTables(
        budget=budget,
        lengths=lengths,
        rectangles=rectangles,
        groups=_encoder_groups(lengths, budget, max_num_seqs),
    )


def _encoder_widths(cap: int) -> tuple[int, ...]:
    """Powers of two up to ``cap``, plus ``cap`` itself when it is not one of them.

    A batch of ``cap`` sequences has to land on a rectangle. Padding it to the next
    power of two would attend a phantom lane, and dropping the odd cap would push a
    full batch onto the ragged path.
    """
    if cap < 1:
        return ()
    widths: list[int] = []
    width = 1
    while width <= cap:
        widths.append(width)
        width *= 2
    if widths[-1] != cap:
        widths.append(cap)
    return tuple(widths)


def _encoder_lengths(max_model_len: int) -> tuple[int, ...]:
    """Every stick from 64 up to ``max_model_len``.

    ``SPYRE_ATTN_QUERY_BUCKETS`` overrides, clamped the same way the decoder's
    ladders are: entries above ``max_model_len`` are dropped as unreachable and the
    limit is appended when missing, so every schedulable length has a bucket.

    It is the only knob over either shape family, because both are derived from this
    ladder: one width family per entry. Fewer, coarser entries mean a shorter warmup
    and more padding per request; there is deliberately no separate override for the
    groups, which would let the two families disagree.

    The limit is rounded *up* to a whole stick (``_align_up``). An extent is a matmul
    dimension, so it cannot be the raw ``max_model_len`` when that is not a whole
    number of sticks, and rounding down would leave the longest requests uncovered.
    The step is one stick, not a power of two: a 300-token prompt padded to 512
    runs the FFN on 70% more rows than the same prompt padded to 320.
    """
    limit = _align_up(max_model_len)
    override = envs.SPYRE_ATTN_QUERY_BUCKETS
    if override:
        raw = {_align_up(int(v)) for v in override.split(",") if v.strip() and int(v) > 0}
        kept = sorted(v for v in raw if v <= limit)
        dropped = sorted(v for v in raw if v > limit)
        if dropped:
            logger.warning(
                "SPYRE_ATTN_QUERY_BUCKETS entries %s exceed max_model_len %d and are "
                "unreachable for a pooling batch; dropping.",
                dropped,
                max_model_len,
            )
        if limit not in kept:
            kept.append(limit)
        return tuple(kept)

    ladder: list[int] = []
    size = ENCODER_SEQ_ALIGNMENT
    while size < limit:
        ladder.append(size)
        size += ENCODER_SEQ_ALIGNMENT
    ladder.append(limit)
    return tuple(ladder)


def _encoder_groups(
    lengths: tuple[int, ...], budget: int, max_num_seqs: int
) -> tuple[tuple[int, int], ...]:
    """Ragged-path ``(width, extent)`` pairs, widths powers of two up to ``B(e)``.

    Serving pads the body to the real token count, so attention always gathers
    equal-extent groups out of that packed buffer. The family is declared even when
    a rectangle could also hold the batch: the rectangle would pad every sequence
    to the longest one, and that padding is what the GEMMs would spend the step on.

    Several widths per extent. A ragged step has several groups, and padding each
    up to ``B(e)`` would cost a full extra sequence per group.
    """
    pairs: list[tuple[int, int]] = []
    for extent in lengths:
        cap = _floor_pow2(min(max_num_seqs, budget // extent))
        width = 1
        while width <= cap:
            pairs.append((width, extent))
            width *= 2
    return tuple(pairs)


def encoder_budget_rows(max_model_len: int, max_num_batched_tokens: int, max_num_seqs: int) -> int:
    """``R``: the most rows one pooling step may run, and the ragged path's buffer.

    Floored at the top of the length ladder: encoder prefill cannot be chunked, so a
    budget below the longest declared length head-of-line blocks the scheduler forever
    -- and no rectangle would hold even one max-length sequence.

    Capped at ``max_num_seqs`` sequences of that length, which is the most tokens a
    step can carry. A rectangular step then pads only up to its own ``W * L``, which
    is at most this budget; the ragged path is the one that still fills it.

    Floored to a whole multiple of that longest length, so a full-length rectangle's
    width is exact. That is the only place a user-supplied ``max_num_batched_tokens``
    is rounded, so an awkward one costs rows rather than correctness.

    A formula rather than a field on the tables because ``check_and_update_config``
    needs it before the config the tables memoise on is final.
    """
    longest = _align_up(max_model_len)
    rows = min(int(max_num_batched_tokens), max(1, int(max_num_seqs)) * longest)
    return max(longest, rows // longest * longest)


def encoder_len_ladder(vllm_config: VllmConfig) -> list[int]:
    """The declared prompt lengths."""
    return list(encoder_shape_tables(vllm_config).lengths)


def encoder_width_for(length: int, vllm_config: VllmConfig) -> int:
    """Widest batch of ``length``-token sequences the budget and the engine allow."""
    tables = encoder_shape_tables(vllm_config)
    return min(int(vllm_config.scheduler_config.max_num_seqs), tables.budget // max(1, length))


def encoder_rectangles(vllm_config: VllmConfig) -> list[tuple[int, int]]:
    """Rectangular-path ``(L, W)`` table."""
    return list(encoder_shape_tables(vllm_config).rectangles)


def encoder_body_sizes(vllm_config: VllmConfig) -> list[int]:
    """Row counts a pooling step can compile: every stick from 64 through the budget.

    A step pads to the next one of these that holds its real tokens. The budget is
    the last entry, so the scheduler cap stays ``R``.
    """
    budget = encoder_shape_tables(vllm_config).budget
    sizes = list(range(ENCODER_SEQ_ALIGNMENT, budget, ENCODER_SEQ_ALIGNMENT))
    sizes.append(budget)
    return sizes


def encoder_group_shapes(vllm_config: VllmConfig) -> list[tuple[int, int]]:
    """Ragged-path ``(width, extent)`` table."""
    return list(encoder_shape_tables(vllm_config).groups)


def encoder_group_width_caps(vllm_config: VllmConfig) -> dict[int, int]:
    """Widest declared group width per extent -- what a group is chunked down to."""
    caps: dict[int, int] = {}
    for width, extent in encoder_shape_tables(vllm_config).groups:
        caps[extent] = max(caps.get(extent, 1), width)
    return caps


def encoder_rectangle_for_batch(
    num_seqs: int,
    max_len: int,
    rectangles: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """The ``(extent, width)`` this batch runs in, or ``None`` to take the ragged path.

    Shortest declared length that covers ``max_len``, then the narrowest width that
    covers ``num_seqs``. A longer length would pad every sequence further, and a wider
    batch would run GEMMs for lanes whose output is never read.

    ``None`` is a routine outcome, not an error: the scheduler is upstream's, so the
    backend cannot refuse a batch and must have a path for every one it can form.
    """
    if num_seqs < 1 or max_len < 1 or not rectangles:
        return None
    fitting = [
        (length, width) for length, width in rectangles if length >= max_len and width >= num_seqs
    ]
    if not fitting:
        return None
    return min(fitting, key=lambda item: (item[0], item[1]))


def encoder_dense_row_indices(query_lens: Sequence[int], len_bucket: int) -> torch.Tensor:
    """Dense row of every packed token: ``seq_idx * L + offset within the sequence``.

    Both directions of the grid build these rows inline in Python; this is the
    independent statement of the layout the round-trip test gathers with.
    """
    lens = torch.as_tensor(list(query_lens), dtype=torch.int64)
    if lens.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    if int(lens.max()) > len_bucket:
        raise ValueError(f"a query length exceeds len_bucket={len_bucket}: {list(query_lens)}")
    # Each sequence's rows shift by a constant, so one `repeat_interleave` of the
    # per-sequence shift beats gathering a start per token.
    packed_starts = torch.cumsum(lens, 0) - lens
    shifts = torch.arange(lens.numel(), dtype=torch.int64) * len_bucket - packed_starts
    packed = torch.arange(int(lens.sum()), dtype=torch.int64)
    return packed + torch.repeat_interleave(shifts, lens)


def encoder_cls_rows(num_seqs: int, extent: int) -> list[int]:
    """CLS row of each sequence in a rectangular grid: sequence ``i`` starts at ``i * extent``."""
    return [seq_idx * extent for seq_idx in range(num_seqs)]


def expand_packed_to_encoder_grid(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_lens: Sequence[int],
    batch_bucket: int,
    len_bucket: int,
    pad_token_id: int = 0,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each sequence to ``L`` and the batch to ``B``; return two ``[B*L]`` tensors.

    Real pad tokens continue positions from the true length. Batch-pad sequences are
    ``pad_token_id`` with positions ``0 .. L-1``. ``position_offset`` is RoBERTa's
    ``padding_idx + 1``, added here while the values are still a Python list so the
    embedding gather never round-trips them through the host.
    """
    if len(query_lens) > batch_bucket:
        raise ValueError(f"num_seqs={len(query_lens)} exceeds batch_bucket={batch_bucket}")

    # Host Python, not tensor ops: every extra CPU aten op on the per-step path
    # lengthens the shared eager-op guard chain and costs more than the loop (#981).
    id_list = input_ids.tolist()
    pos_list = positions.tolist()
    total = batch_bucket * len_bucket
    padded_ids = [int(pad_token_id)] * total
    padded_pos = [0] * total
    src = 0
    for seq_idx, length in enumerate(query_lens):
        dst = seq_idx * len_bucket
        padded_ids[dst : dst + length] = id_list[src : src + length]
        padded_pos[dst : dst + length] = [
            int(pos) + position_offset for pos in pos_list[src : src + length]
        ]
        for offset in range(length, len_bucket):
            padded_pos[dst + offset] = offset + position_offset
        src += length
    for seq_idx in range(len(query_lens), batch_bucket):
        dst = seq_idx * len_bucket
        for offset in range(len_bucket):
            padded_pos[dst + offset] = offset + position_offset
    return (
        torch.tensor(padded_ids, dtype=input_ids.dtype),
        torch.tensor(padded_pos, dtype=positions.dtype),
    )


def expand_packed_token_types(
    token_type_ids: torch.Tensor,
    query_lens: Sequence[int],
    batch_bucket: int,
    len_bucket: int,
) -> torch.Tensor:
    """Scatter packed segment ids into the ``[B*L]`` grid; every pad slot is segment 0.

    One value per packed token, so it takes the same layout as ``input_ids``. Left packed
    it would pair each sequence's segment ids with another sequence's tokens, and
    silently: the buffer still matches ``input_ids`` in shape, so the all-zeros fallback
    in ``spyre_token_type_ids_for`` never fires.
    """
    values = token_type_ids.tolist()
    grid = [0] * (batch_bucket * len_bucket)
    src = 0
    for seq_idx, length in enumerate(query_lens):
        dst = seq_idx * len_bucket
        grid[dst : dst + length] = values[src : src + length]
        src += length
    return torch.tensor(grid, dtype=token_type_ids.dtype)


def encoder_grid_source_rows(
    query_lens: Sequence[int], batch_bucket: int, len_bucket: int
) -> list[int]:
    """Packed row each ``[B*L]`` grid row reads from; every pad row reads row 0.

    The inverse of ``encoder_dense_row_indices``, as a gather index rather than a
    scatter, so a multimodal pooling model (e.g. CLIP) whose preprocessing produces
    embeddings can build its grid with one ``index_select`` on the device. Pad rows
    carry a real token's embedding rather than zeros: they are masked out of attention
    and never pooled, so only finiteness matters.
    """
    if len(query_lens) > batch_bucket:
        raise ValueError(f"num_seqs={len(query_lens)} exceeds batch_bucket={batch_bucket}")
    rows = [0] * (batch_bucket * len_bucket)
    src = 0
    for seq_idx, length in enumerate(query_lens):
        if length > len_bucket:
            raise ValueError(f"a query length exceeds len_bucket={len_bucket}: {list(query_lens)}")
        dst = seq_idx * len_bucket
        rows[dst : dst + length] = range(src, src + length)
        src += length
    return rows


def logits_row_buckets(bucket_sizes: Sequence[int], max_num_reqs: int) -> list[int]:
    """Row widths the lm_head can see: each body bucket clipped to ``max_num_reqs``."""
    cap = max(1, max_num_reqs)
    return sorted({min(size, cap) for size in bucket_sizes if size > 0})


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


class SpyreShapeBucketer:
    """Dispatches runtime batches to pre-compiled 1D body token buckets."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        compilation_config = vllm_config.compilation_config
        sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
        self._bucket_sizes = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False
        logger.info(
            "SpyreShapeBucketer initialized with %d body token buckets: min=%d, max=%d",
            len(self._bucket_sizes),
            self._bucket_sizes[0] if self._bucket_sizes else 0,
            self._max_bucket_size,
        )

    @classmethod
    def for_config(cls, vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
        """The runner's bucketer for 1D body token sizes, or None when it needs none.

        Decoder and pooling share 1D ``compile_sizes``. Pooling's is every
        rectangle's ``W * L`` plus the token budget.

        Pooling keeps a bucketer in eager *and* compile so *runtime* always
        1D-pads the body. Warmup still differs: compile dummies each 1D size
        (attention compiles through ``force_attention``); eager does one
        dummy then ``mark_warmed_up()``. Decoder skips a bucketer when eager
        because 1D pad exists only to hit compiled graphs.
        """
        if vllm_config.model_config.runner_type == "pooling":
            return cls.for_pooling(vllm_config)

        if vllm_config.model_config.enforce_eager:
            logger.info("Graph Recorder disabled (enforce_eager=True)")
            return None

        if not vllm_config.compilation_config.compile_sizes:
            logger.info("Graph Recorder disabled (no compile_sizes configured)")
            return None

        return cls(vllm_config)

    @classmethod
    def for_pooling(cls, vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
        """Pooling bucketer: one bucket per rectangle row count, plus the budget.

        Encoder attention shapes live on the tables above, not here.
        """
        model_config = vllm_config.model_config
        if getattr(model_config, "runner_type", None) != "pooling":
            return None
        compile_sizes = [int(s) for s in (vllm_config.compilation_config.compile_sizes or [])]
        if not compile_sizes:
            return None
        return cls(vllm_config)

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def max_bucket_size(self) -> int:
        return self._max_bucket_size

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest 1D bucket size >= num_tokens.

        Returns None if num_tokens exceeds the largest compiled bucket.
        The caller (execute_model) handles the None case by running the
        forward pass without bucket padding, which may trigger Dynamo
        recompilation for the unseen shape.
        """
        idx = bisect.bisect_left(self._bucket_sizes, num_tokens)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

    def warmed_bucket(self, num_tokens: int) -> int | None:
        """The padded token count, or None before warmup or past the last bucket."""
        if not self._is_warmed_up:
            return None
        desc = self.dispatch(num_tokens)
        return None if desc is None else desc.padded_num_tokens

    def dispatch(self, num_tokens: int) -> SpyreBucketDescriptor | None:
        """Compute padded batch descriptor for the given token count.

        Returns None if no suitable bucket exists.
        """
        padded = self.find_bucket(num_tokens)
        if padded is None:
            return None
        return SpyreBucketDescriptor(
            actual_num_tokens=num_tokens,
            padded_num_tokens=padded,
        )
