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

"""Compiled, bucketed vision encoders on upstream's ``SupportsEncoderCudaGraph`` protocol.

A vision model subclasses its upstream class with ``SpyreVisionEncoderMixin`` ahead of it and
implements the hooks in the mixin's first section; everything else (the protocol, warmup,
serving) is shared. The encoder runs in three graphs, compiled once per upstream token budget:

    host:   vision_inputs     items -> buffers, one entry per patch
    graph:  vision_embed      patches -> the blocks' input embeddings
    graph:  vision_blocks     the repeated encoder layers
    graph:  vision_project    merge or pool, then project into the text model's space

Units follow upstream: an *image token* is one position of ``vision_project``'s output, the unit
of the token budgets, and every image token is ``patches_per_token`` patches.

The image tokens then merge into the prompt embeddings in ``embed_input_ids``. With an
``EncoderOutputTable`` the encoder output stays on the card: the cache holds row handles, and
``vision_gather_merge`` gathers them per token bucket. Without one (tests, and models that
still return embeddings) the rows are placed on the host and combined by ``vision_merge``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.models import utils as vllm_utils
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from spyre_inference.custom_ops.utils import convert, place_row_gathered
from spyre_inference.multimodal.utils import STICK, align_up
from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class VisionBuffer:
    """One encoder input, ``[P]`` or ``[P, width]`` over ``P`` patch slots.

    ``pad`` is the value of the slots a budget adds: a scalar, or one ``[width]`` row.
    ``dtype`` is fp16 for activations; integer buffers carry indices (e.g. positions).
    """

    width: int | None
    pad: float | torch.Tensor = 0.0
    dtype: torch.dtype = torch.float16


class GatherTable(nn.Module):
    """A ``[rows, width]`` lookup table that a vision graph gathers rows from by index.

    Stays on the host until the model moves to the card, then is placed with its rows
    outermost, the layout torch-spyre's gather needs. Use it for per-patch lookups such as
    RoPE, so the host uploads indices instead of the gathered rows.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table

    def _apply(self, fn, recurse=True):
        self.table = place_row_gathered(self.table, fn, type(self).__name__)
        return self

    def gather(self, index: torch.Tensor) -> torch.Tensor:
        """The rows at ``index``."""
        return self.table.index_select(0, index)


def unbind_items(value: Any) -> list[torch.Tensor]:
    """An ``mm_kwargs`` field as one tensor per item, whether batched in one tensor or a list."""
    return list(value.unbind(0)) if isinstance(value, torch.Tensor) else list(value)


def offset_major(t: torch.Tensor, merge: int) -> torch.Tensor:
    """``[hp, wp, ...]`` -> ``[merge**2, tokens, ...]``: one group per offset in a merge window.

    Tokens run row-major over the merged grid, ``unfold``'s order. Patches laid out this way make
    a merge or pool in ``vision_project`` a free view plus a batched matmul or mean.
    """
    hp, wp = t.shape[:2]
    t = t.reshape(hp // merge, merge, wp // merge, merge, *t.shape[2:])
    t = t.permute(1, 3, 0, 2, *range(4, t.dim()))
    return t.reshape(merge * merge, (hp // merge) * (wp // merge), *t.shape[4:])


# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2
# Most items per call with `segment_buffer`: the one-hot is a stick wide, its last column padding.
SEGMENT_ITEMS = STICK - 1


def segment_buffer() -> VisionBuffer:
    """Each patch's item as a one-hot row; padding slots take the last column."""
    return VisionBuffer(STICK, pad=F.one_hot(torch.tensor(STICK - 1), STICK).half())


def one_hot_segments(item_ids: torch.Tensor) -> torch.Tensor:
    return F.one_hot(item_ids, STICK).to(torch.float16)


def _index_dtype(device: torch.device) -> torch.dtype:
    """int32 on Spyre (no int64; compiled ``index_copy_`` takes it); int64 elsewhere.

    Eager ``index_copy_`` rejects int32 and falls back to the CPU, so a Spyre write goes
    through the compiled function below, including under ``enforce_eager``.
    """
    return torch.int32 if device.type == "spyre" else torch.int64


def _copy_encoder_rows(
    table: torch.Tensor, index: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Overwrite ``table`` rows.

    On CPU a repeated index keeps the last row. The compiled Spyre kernel cannot repeat one:
    dbo-opt rejects it ("external allocations with repeated dimensions"), so every compiled
    write uses unique indices.
    """
    table.index_copy_(0, index, rows)
    return table


def _gather_cached_rows(
    table: torch.Tensor, text: torch.Tensor, src_idx: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``where(mask, table.index_select(0, src_idx), text)``.

    ``table`` is an argument so the compiled graph reads it on every call, including rows
    written after the graph was compiled. ``mask`` is ``[T, 1]``, built on the host.
    """
    return torch.where(mask, table.index_select(0, src_idx), text)


class EncoderOutputTable:
    """Device-resident ``[C + pad, H]`` table of encoder outputs, plus one trash row.

    ``C`` is upstream's encoder cache bound, in image tokens. The trash row is last, so a
    gather of a prompt is a strict subset of the table (torch-spyre#4033). Rows between
    ``C`` and the trash row are not cache slots: a short write pads onto them so a partly
    full cache reuses a warmed ``index_copy_`` instead of compiling a new one. The host
    cache stores ``[n, 1]`` row handles, not the embeddings.
    """

    def __init__(
        self, cache_size: int, hidden: int, dtype: torch.dtype, device: torch.device, *, rows: int
    ) -> None:
        if cache_size <= 0 or hidden <= 0 or rows <= cache_size:
            raise ValueError(
                f"encoder output table needs cache_size > 0, hidden > 0 and rows > cache_size; "
                f"got cache_size={cache_size}, hidden={hidden}, rows={rows}"
            )
        self.cache_size = cache_size
        self.hidden = hidden
        self.trash = rows - 1
        self._in_use: set[int] = set()
        blank = torch.zeros(rows, hidden, dtype=dtype)
        try:
            if device.type == "spyre":
                # Rows outermost, the layout a gather needs. CPU tests never take this branch.
                self.table = place_row_gathered(
                    blank, lambda t: t.to(device=device, dtype=dtype), "encoder output table"
                )
            else:
                self.table = blank.to(device=device, dtype=dtype)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot allocate the encoder output table ({rows} x {hidden} {dtype} on "
                f"{device}). It holds {cache_size} image tokens, the scheduler's encoder "
                "cache (the larger of --encoder-cache-size and the biggest multimodal item), "
                "and does not change the KV cache."
            ) from exc
        self._free = list(range(cache_size))
        self._writes: dict[int, Any] = {}

    def store(
        self, rows: torch.Tensor, *, keep: int | None = None, exact: bool = False
    ) -> torch.Tensor:
        """Copy ``rows`` ``[n, H]`` into free slots. Returns host handles ``[keep, 1]``.

        ``keep`` defaults to every row. The copy is a length warmup already compiled:
        the budget tensor itself when that length was warmed, otherwise the next warmed
        length. Padding past the free list uses the spare rows, which are not cache slots,
        so a partly full cache does not compile a new graph. A short tensor is padded with
        a copied last row. Pointing the tail at one repeated index, including the trash
        row, does not compile. ``exact`` is warmup, which records its own length.
        """
        if rows.ndim != 2 or rows.shape[1] != self.hidden:
            raise RuntimeError(
                f"encoder output is {tuple(rows.shape)}, table rows are [*, {self.hidden}]"
            )
        total = rows.shape[0]
        keep = total if keep is None else keep
        if keep < 0 or keep > total:
            raise RuntimeError(f"encoder output has {total} rows, this call keeps {keep}")
        if keep == 0:
            return torch.empty(0, 1, dtype=torch.int64)
        if keep > len(self._free):
            raise RuntimeError(
                f"Encoder output table has {len(self._free)} free rows, this call needs {keep}. "
                f"The table holds {self.cache_size} image tokens (the scheduler's encoder cache)."
            )
        target = total if exact else self._write_count(keep, total)
        spare = self.trash - self.cache_size
        use_spare = not exact and target > len(self._free)
        if (exact and target > len(self._free)) or (use_spare and target - keep > spare):
            raise RuntimeError(
                f"Encoder output table has {len(self._free)} free rows and {spare} spare "
                f"rows, this write needs {target}."
            )
        body = self._rows_for_write(rows, target)
        # A host projection (Gemma 4's tail) arrives on the CPU. The write runs where the table is.
        if body.device != self.table.device:
            body = convert(body, device=self.table.device, dtype=self.table.dtype)
        n_cache = keep if use_spare else target
        chosen = [self._free.pop() for _ in range(n_cache)]
        self._in_use.update(chosen)
        if use_spare:
            # Distinct rows, so the index does not repeat. They are not cache slots.
            chosen.extend(range(self.cache_size, self.cache_size + (target - keep)))
        index = torch.tensor(chosen, dtype=_index_dtype(body.device), device=body.device)
        self._write(body, index)
        handles = torch.tensor(chosen, dtype=torch.int64).view(-1, 1)
        if not use_spare and keep < target:
            self.release(handles[keep:])
        return handles[:keep]

    def _write_count(self, keep: int, total: int) -> int:
        """Rows this write copies: at least ``keep``.

        A length warmup compiled is written whole. Anything else grows to the smallest
        already-compiled write that can hold ``keep``. The extra rows come from the free
        list when it has room, and otherwise from the spare rows, so the real token
        count does not compile a graph of its own.
        """
        free = len(self._free)
        spare = self.trash - self.cache_size

        def fits(n: int) -> bool:
            if n < keep:
                return False
            extra = n - keep
            return n <= free or extra <= spare

        if not self._writes:
            return total if total <= free else keep
        if total in self._writes and fits(total):
            return total
        warmed = [n for n in self._writes if fits(n)]
        if warmed:
            return min(warmed)
        return keep

    def _rows_for_write(self, rows: torch.Tensor, target: int) -> torch.Tensor:
        total = rows.shape[0]
        if target == total:
            return rows
        if target < total:
            # Prefix at offset 0. A non-zero device offset reads as row 0 (torch-spyre#3770).
            return rows[:target]
        # A copied tail, not expand(): a zero-stride view is the same repeated-dimension
        # failure as pointing every padding index at the trash row.
        if rows.device.type != "cpu":
            rows = convert(rows, device="cpu")
        return torch.cat([rows, rows[-1:].repeat(target - total, 1)], dim=0)

    def release(self, handles: torch.Tensor) -> None:
        """Return stored rows to the free list. The trash row is not a stored row."""
        for row in handles.detach().cpu().view(-1).tolist():
            row = int(row)
            if row not in self._in_use:
                logger.debug(
                    "Encoder output table skipped releasing row %s; it is not in use.", row
                )
                continue
            self._in_use.remove(row)
            self._free.append(row)

    def warmup_write(self, rows: torch.Tensor) -> None:
        """Compile the largest write this output can make, then free it.

        The encoder's padded budget can be thousands of rows while the cache holds fewer.
        The scheduler never stores more real tokens than the cache, and a compiled
        ``index_copy_`` cannot repeat an index, so this writes one unique row per free
        slot. Every budget that fits is its own shape. Every budget past the cache shares
        that one cache-sized write, so those do not add another graph.
        """
        n = min(rows.shape[0], len(self._free))
        if n == 0:
            return
        # Prefix at offset 0. A non-zero device offset reads as row 0 (torch-spyre#3770).
        # exact: a larger warmed write must not swallow this budget, or serving a
        # smaller budget would compile it on the first request.
        prefix = rows if n == rows.shape[0] else rows[:n]
        self.release(self.store(prefix, exact=True))

    def _write(self, rows: torch.Tensor, index: torch.Tensor) -> None:
        # Eager int32 index_copy_ falls back to the CPU, so the card always runs the compiled
        # function, including under enforce_eager. One function per warmed row count.
        # store() pads onto a warmed count, using spare rows when the free list is short,
        # so serving does not compile a new length. The size is recorded on every device
        # so a later call can see that this length was already written.
        n = rows.shape[0]
        if rows.device.type != "spyre":
            _copy_encoder_rows(self.table, index, rows)
            self._writes.setdefault(n, None)
            return
        fn = self._writes.get(n)
        if fn is None:
            fn = torch.compile(
                _copy_encoder_rows, backend="inductor", fullgraph=True, dynamic=False
            )
            self._writes[n] = fn
        fn(self.table, index, rows)


class EncoderOutputCache(dict):
    """``encoder_cache`` that frees table rows when upstream pops or clears an entry.

    Values are ``[n, 1]`` host handles. A floating-point value (prompt embeds, or a model
    that still returns embeddings) is left alone.
    """

    def __init__(self, table: EncoderOutputTable) -> None:
        super().__init__()
        self.table = table

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        previous = self.get(key)
        super().__setitem__(key, value)
        if previous is not None and previous is not value:
            self._release(previous)

    def __delitem__(self, key: str) -> None:
        value = self[key]
        super().__delitem__(key)
        self._release(value)

    def pop(self, key: str, *args: Any) -> Any:
        if key not in self and args:
            return args[0]
        value = super().pop(key)
        self._release(value)
        return value

    def clear(self) -> None:
        values = list(self.values())
        super().clear()
        for value in values:
            self._release(value)

    def popitem(self) -> tuple[Any, Any]:
        key, value = super().popitem()
        self._release(value)
        return key, value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        # A plain dict, so this does not recurse into the releasing ``__setitem__``.
        incoming: dict[Any, Any] = {}
        incoming.update(*args, **kwargs)
        for key, value in incoming.items():
            self[key] = value

    def _release(self, value: Any) -> None:
        if isinstance(value, torch.Tensor) and not value.is_floating_point():
            self.table.release(value)


def segment_mask(segments: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive `[1, 1, P, P]` mask keeping each patch to its own item."""
    # A matmul: Spyre cannot lay out a column-minus-row broadcast.
    segments = segments.to(dtype)
    return ((1.0 - torch.matmul(segments, segments.t())) * _MASK_MIN)[None, None]


def _mixed_embedding_kinds(embeddings: Any) -> bool:
    """True when one step holds both floating embeddings and integer handles.

    ``_flatten_embeddings`` concatenates every item. A prompt-embeds tensor is
    ``[m, H]`` float and an image handle is ``[n, 1]`` int, so the concat either
    fails on the shape or promotes the handles to float and merges them as embeddings.
    """
    floating: set[bool] = set()

    def walk(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            floating.add(value.is_floating_point())
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(embeddings)
    return len(floating) > 1


class SpyreVisionEncoderMixin:
    """Implements upstream's ``SupportsEncoderCudaGraph`` for a vision model on Spyre.

    Structural rather than a subclass of the protocol: a protocol base would put its
    placeholder ``__init__`` ahead of the model's in the MRO. ``select_encoder_cudagraph_items``
    and ``get_input_modality`` are not defined here, so an upstream class that has them keeps
    them; a model whose upstream class does not writes them itself.
    """

    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    # --- 1. What a model provides ---

    # Host inputs by name, each ``[P]`` or ``[P, width]``: fp16 activations or integer indices.
    vision_buffers: dict[str, VisionBuffer]
    # Patches per image token: the merge or pool factor (``M**2``), 1 if none.
    patches_per_token: int
    # Width of vision_project's output: the text model's hidden size.
    out_hidden_size: int

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        """Image tokens each item in ``mm_kwargs`` produces, in item order.

        Computed on the host from the item shapes. ``sum(tokens) * patches_per_token`` must equal
        the patch count ``vision_inputs`` returns for the same items.
        """
        raise NotImplementedError

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Host-side preparation: one unpadded tensor per ``vision_buffers`` key, of its dtype.

        All tensors share one patch count; the mixin pads them to a budget. When
        ``vision_project`` merges or pools, lay patches out offset-major (``offset_major``).
        Put everything that depends on values here (patching, position gathers, image ids), so
        the device graphs depend on shapes only.
        """
        raise NotImplementedError

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Graph 1: the padded device buffers -> ``(x, block_kwargs)``.

        ``x`` is the patch embedding the blocks consume (plus any norm before them).
        ``block_kwargs`` holds what every block shares, such as the attention mask and RoPE,
        and reaches each block through ``vision_call_block``.
        """
        raise NotImplementedError

    def vision_blocks(self) -> nn.ModuleList:
        """Graph 2: the model's own encoder layers, compiled in place.

        Return the existing list, not a copy, so weight names stay unchanged. Swap in Spyre
        layers (e.g. attention) at construction, before weights load.
        """
        raise NotImplementedError

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """Graph 3: the last layer's output -> ``[tokens, out_hidden_size]``.

        Merge or pool, then project into the text model's space. ``taps`` are the hidden states
        named by ``vision_taps``, in that order. Padding tokens may stay; they are dropped later.
        """
        raise NotImplementedError

    # --- 2. Optional overrides ---
    # Two methods in section 4 are also meant to be overridden, as upstream models do:
    # get_encoder_cudagraph_budget_range and postprocess_encoder_output.

    # Modalities run through the compiled graphs; any other goes to upstream's embed_multimodal.
    vision_modalities: ClassVar[tuple[str, ...]] = ("image",)
    # Hidden states passed to vision_project: 0 is vision_embed's output, i is layer i's,
    # negative indices count from the last layer.
    vision_taps: ClassVar[tuple[int, ...]] = ()
    # Most items one encoder call can carry, when the layout bounds it (e.g. a one-hot width).
    vision_max_items_per_call: ClassVar[int | None] = None

    @property
    def vision_pad_groups(self) -> int:
        """Equal groups every buffer's patches are split into, each padded at its own tail.

        Default ``patches_per_token``, one group per merge offset, which keeps the merge a free
        view. Use 1 when items pad as whole units at the end (e.g. tiles).
        """
        return self.patches_per_token

    @property
    def vision_patch_multiple(self) -> int:
        """Every padded patch count is a whole multiple of this; budgets round up to it.

        Default: the least common multiple of the 64-wide stick, ``vision_pad_groups`` and
        ``patches_per_token``. Override for a larger natural unit (e.g. a 576-patch tile).
        """
        return math.lcm(STICK, self.vision_pad_groups, self.patches_per_token)

    def vision_call_block(
        self, block: nn.Module, x: torch.Tensor, block_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Run one encoder layer and return its output tensor.

        Default ``block(x, **block_kwargs)``. Override when the layer takes other arguments or
        returns a tuple (e.g. SigLIP's ``(hidden, None)``).
        """
        return block(x, **block_kwargs)

    def vision_merge(
        self, text: torch.Tensor, image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Graph 4: combine the prompt's text and image embeddings, both ``[T, H]``.

        ``image`` holds each image token at its placeholder row and zeros elsewhere; ``mask``
        is ``[T, 1]``, true at those rows. Default: the image rows where ``mask`` is set.
        Override when the model's upstream merge does more (e.g. Granite Vision's
        deepstack features).
        """
        return torch.where(mask, image, text)

    def vision_gather_merge(
        self, text: torch.Tensor, src_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Graph 4 when encoder outputs live in ``output_table``.

        ``src_idx`` is ``[T]`` into the table (the trash row everywhere that is not an image
        token). ``mask`` is ``[T, 1]``, built on the host: unsqueezing it on the card faults
        a later ``index_select`` (torch-spyre#5230).
        """
        table = self.output_table
        assert table is not None, "vision_gather_merge needs an encoder output table"
        return _gather_cached_rows(table.table, text, src_idx, mask)

    # --- 3. Used by the runner ---

    # Image tokens of the largest item: set by the runner from upstream's processor budget before
    # the manager is built, and read by get_encoder_cudagraph_budget_range.
    max_tokens_per_item: int
    # On-card rows of encoder outputs. Set by the runner when the model loads; absent means the
    # host still receives the embeddings.
    output_table: EncoderOutputTable | None = None

    def compile_vision_graphs(self) -> None:
        """Compile ``vision_embed``, every layer, ``vision_project`` and ``vision_merge`` with
        static shapes and register them with the compile guard. Called at warmup; a repeat call
        is a no-op."""
        if hasattr(self, "_spyre_vision_embed"):
            return
        compile_kwargs: dict[str, Any] = dict(backend="inductor", fullgraph=True, dynamic=False)
        self._spyre_vision_embed = torch.compile(self.vision_embed, **compile_kwargs)
        self._spyre_vision_project = torch.compile(self.vision_project, **compile_kwargs)
        self._spyre_vision_merge = torch.compile(self.vision_merge, **compile_kwargs)
        self.compile_output_merge()
        # vision_gather_merge is watched inside compile_output_merge.
        for name in ("vision_embed", "vision_project", "vision_merge"):
            compiled = getattr(self, f"_spyre_{name}", None)
            if compiled is not None:
                compile_guard.watch(compiled, f"{type(self).__name__}.{name}")
        for block in self.vision_blocks():
            block.compile(**compile_kwargs)
            compile_guard.watch(block, f"{type(block).__name__} (vision block)")

    def compile_output_merge(self, *, watch: bool = True) -> None:
        """Compile the table gather. The tower may stay eager; this op cannot.

        Eager ``index_select`` rejects the int32 index and falls back to the CPU, and an
        eager gather of this layout has faulted the card. ``enforce_eager`` has no shape
        bucketer, so nothing warms a prompt length ahead of time: the wrapper is installed
        with the table, and the first merge of each prompt length traces it. That trace
        is not watched. A compiled tower warms every token bucket before the guard arms.
        """
        if getattr(self, "output_table", None) is None:
            return
        if hasattr(self, "_spyre_vision_gather_merge"):
            return
        self._spyre_vision_gather_merge = torch.compile(
            _gather_cached_rows, backend="inductor", fullgraph=True, dynamic=False
        )
        if watch:
            compile_guard.watch(
                self._spyre_vision_gather_merge, f"{type(self).__name__}.vision_gather_merge"
            )

    # --- 4. Upstream's SupportsEncoderCudaGraph, in the order the manager calls it ---

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        """Modalities, buffer keys and output width, and per-group padding for every buffer
        (upstream's default pads only at the tail)."""
        return EncoderCudaGraphConfig(
            modalities=list(self.vision_modalities),
            buffer_keys=list(self.vision_buffers),
            out_hidden_size=self.out_hidden_size,
            padding_logics={
                key: partial(_pad_groups, pad=buf.pad, groups=self.vision_pad_groups)
                for key, buf in self.vision_buffers.items()
            },
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        """Smallest and largest token budget: one ``vision_patch_multiple``, and the largest item.

        Overridable: a model whose processor does not count in image tokens out of the encoder
        (e.g. it adds newline tokens) returns its largest item's encoder tokens as the maximum.
        """
        return self.vision_patch_multiple // self.patches_per_token, self.max_tokens_per_item

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        """Padding-filled host buffers at one budget's size, to compile that budget. The manager
        keeps them as staging buffers; only the shape matters."""
        return EncoderCudaGraphCaptureInputs(
            values=self._vision_empty(token_budget * self.patches_per_token)
        )

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        """Each item's size, for packing into budgets. Replaces upstream's, which counts in its
        own buffer layout."""
        return [
            EncoderItemSpec(input_size=tokens * self.patches_per_token, output_tokens=tokens)
            for tokens in self.vision_item_tokens(mm_kwargs)
        ]

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        """The validated ``vision_inputs`` of one packed batch. Replaces upstream's, whose
        buffers would not match ``vision_buffers``."""
        return EncoderCudaGraphReplayBuffers(values=dict(self._checked_inputs(mm_kwargs)))

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        """Run the three graphs on padded device buffers.

        With an ``output_table`` the output stays on the card and the cache stores row handles.
        Otherwise it is copied to the host: a device view at a non-zero offset reads as offset 0
        (torch-spyre#3770), and the host merge needs the rows.
        """
        embed = getattr(self, "_spyre_vision_embed", self.vision_embed)
        project = getattr(self, "_spyre_vision_project", self.vision_project)
        blocks = self.vision_blocks()
        depth = len(blocks) + 1
        if any(not -depth <= t < depth for t in self.vision_taps):
            raise ValueError(
                f"vision_taps {self.vision_taps} is out of range for {depth} hidden states"
            )
        wanted = {t % depth for t in self.vision_taps}

        x, block_kwargs = embed(inputs)
        hidden = {0: x} if 0 in wanted else {}
        for i, block in enumerate(blocks, start=1):
            x = self.vision_call_block(block, x, block_kwargs)
            if i in wanted:
                hidden[i] = x
        taps = [hidden[t % depth] for t in self.vision_taps]
        out = project(x, taps)
        if getattr(self, "output_table", None) is None:
            return convert(out, device="cpu")
        return out

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        """The encoder without a budget, padded only to ``vision_patch_multiple``.

        Serves ``enforce_eager`` and the profiling run before warmup. Unreached once compiled:
        the manager's startup check makes the largest budget cover every item.
        """
        values = self._checked_inputs(mm_kwargs)
        num_patches = next(iter(values.values())).shape[0]
        padded = self._vision_empty(align_up(num_patches, self.vision_patch_multiple))
        padding = self.get_encoder_cudagraph_config().padding_logics
        for key, src in values.items():
            padding[key](padded[key], src)
        device = next(self.vision_blocks().parameters()).device
        return self.encoder_cudagraph_forward(
            {key: convert(value, device=device) for key, value in padded.items()}
        )

    def encoder_cache_rows(self, output: torch.Tensor) -> torch.Tensor:
        """The rows stored in the output table.

        Default: ``vision_project`` already returns ``[tokens, out_hidden_size]``. Override when
        the cached rows are a projection of that output (Gemma 4's tail widens the vision width
        to the text width). Warmup writes whatever this returns, so the compiled write matches
        the rows serving stores.
        """
        return output

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Upstream's default: slice the output into ``dest``, one item at a time.

        Overridable, as upstream models do, when placeholders differ from the encoder's image
        tokens (e.g. unpadding or newline tokens). With an ``output_table`` each item is a
        ``[n, 1]`` host tensor of table rows; otherwise it is the embedding.
        """
        output = outputs["default"]
        table = getattr(self, "output_table", None)
        if table is not None:
            num_tokens = sum(per_item_out_tokens[i] for i in indices)
            # The whole budget tensor, not a device slice: store writes that warmed shape
            # and frees the padding rows. Slicing on the card first is a narrowed view.
            output = table.store(self.encoder_cache_rows(output), keep=num_tokens)
        scatter_output_slices(output, indices, per_item_out_tokens, dest, clone)

    def embed_multimodal(self, **kwargs: object) -> Any:
        """Upstream's entry point when the manager is not used (eager mode, profiling): the same
        hooks, uncompiled. Modalities outside ``vision_modalities`` go to upstream's."""
        if self.get_input_modality(kwargs) not in self.vision_modalities:
            return super().embed_multimodal(**kwargs)
        tokens = self.vision_item_tokens(kwargs)
        if not tokens:
            return ()
        dest: list[torch.Tensor | None] = [None] * len(tokens)
        # The manager caps items per call when compiled; without it the cap applies here.
        cap = self.vision_max_items_per_call or len(tokens)
        for start in range(0, len(tokens), cap):
            indices = list(range(start, min(start + cap, len(tokens))))
            batch = (
                kwargs
                if len(indices) == len(tokens)
                else self.select_encoder_cudagraph_items(  # ty: ignore[unresolved-attribute]
                    kwargs, indices
                )
            )
            output = self.encoder_eager_forward(batch)
            self.postprocess_encoder_output(
                {"default": output}, indices, tokens, dest, batch_mm_kwargs=batch
            )
        return tuple(t for t in dest if t is not None)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upstream's text embedding, then the image merge.

        Upstream's chain runs with no image tokens, so model overrides and its handling of
        out-of-vocabulary placeholders still apply, and it returns before its own merge.
        Handles from the output table are gathered on the card, one graph per token bucket.
        Embedding tensors are still placed on the host.
        """
        text = super().embed_input_ids(  # ty: ignore[unresolved-attribute]
            input_ids, (), is_multimodal=is_multimodal
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return text
        assert is_multimodal is not None, "image tokens need is_multimodal"
        if getattr(self, "output_table", None) is not None and _mixed_embedding_kinds(
            multimodal_embeddings
        ):
            raise RuntimeError(
                "This step mixes floating-point prompt embeddings with integer image handles. "
                "The encoder output table stores handles, so the two cannot share one merge."
            )
        mask = convert(is_multimodal, device="cpu")
        rows = convert(vllm_utils._flatten_embeddings(multimodal_embeddings), device="cpu")
        if rows.shape[0] != int(mask.sum()):
            expr = vllm_utils._embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {rows.shape[0]} multimodal tokens to "
                f"{int(mask.sum())} placeholders"
            )
        mask_col = mask[:, None]
        if not rows.is_floating_point():
            return self._merge_handles(text, rows.view(-1), mask, mask_col)
        image = torch.zeros(text.shape, dtype=text.dtype)
        image[mask] = rows.to(text.dtype)
        merge = getattr(self, "_spyre_vision_merge", self.vision_merge)
        return merge(
            text, convert(image, device=text.device), convert(mask_col, device=text.device)
        )

    def _merge_handles(
        self,
        text: torch.Tensor,
        handles: torch.Tensor,
        mask: torch.Tensor,
        mask_col: torch.Tensor,
    ) -> torch.Tensor:
        """Gather table rows for ``handles`` and merge. ``mask_col`` was unsqueezed on the host."""
        table = self.output_table
        src_idx = torch.full((text.shape[0],), table.trash, dtype=_index_dtype(text.device))
        src_idx[mask] = handles.to(dtype=src_idx.dtype)
        merge = getattr(self, "_spyre_vision_gather_merge", None)
        if text.device.type == "spyre":
            # Installed with the table. Compiling here would be an unwatched graph on
            # the serving path.
            assert merge is not None, (
                "encoder output table gather was not compiled when the table was installed"
            )
        else:
            merge = _gather_cached_rows
        return merge(
            table.table,
            text,
            convert(src_idx, device=text.device),
            convert(mask_col, device=text.device),
        )

    # --- Private helpers ---

    def _vision_empty(self, num_patches: int) -> dict[str, torch.Tensor]:
        """Host buffers of ``num_patches`` slots, filled with each buffer's padding value."""
        out = {}
        for key, buf in self.vision_buffers.items():
            shape = (num_patches,) if buf.width is None else (num_patches, buf.width)
            out[key] = torch.empty(shape, dtype=buf.dtype)
            out[key][:] = buf.pad
        return out

    def _checked_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """``vision_inputs``, validated: every declared key, its declared dtype, one patch count."""
        values = self.vision_inputs(mm_kwargs)
        if set(values) != set(self.vision_buffers):
            raise ValueError(
                f"{type(self).__name__}.vision_inputs returned {sorted(values)}, but "
                f"vision_buffers declares {sorted(self.vision_buffers)}"
            )
        counts = {v.shape[0] for v in values.values()}
        wrong = [k for k, v in values.items() if v.dtype != self.vision_buffers[k].dtype]
        if len(counts) != 1 or wrong:
            raise ValueError(
                f"vision_inputs must return one patch count and the declared dtypes; got "
                f"counts {sorted(counts)}, wrong dtype for {wrong}"
            )
        return values


def _pad_groups(
    dst: torch.Tensor, src: torch.Tensor, pad: float | torch.Tensor, groups: int
) -> None:
    """Copy ``src`` into the larger ``dst`` group by group, padding each group's tail."""
    d = dst.view(groups, -1, *dst.shape[1:])
    s = src.view(groups, -1, *src.shape[1:])
    d[:] = pad
    d[:, : s.shape[1]].copy_(s)
