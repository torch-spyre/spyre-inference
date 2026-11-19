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

"""The model wrapper the Spyre model runner holds as ``self.model``.

The runner keeps its buffers on the host (``self.device`` is CPU); the wrapper moves
integer inputs to Spyre int64 at the call boundary. Generative hidden states stay on
Spyre as ``_SpyreHiddenStates``, so the sampled rows are gathered and projected there
and only the logits come back; device poolers keep their outputs on Spyre too.
"""

from __future__ import annotations

import bisect
import time
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert, convert_tensor_tree
from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin
from spyre_inference.models.mistral import reset_llama4_scale_cache
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer

logger = init_logger(__name__)


def model_dtype(model_config) -> torch.dtype:
    """The activation dtype the platform settled on (float16 for every engine run;
    see ``TorchSpyrePlatform.apply_config_platform_defaults``)."""
    dtype = model_config.dtype
    return dtype if isinstance(dtype, torch.dtype) else torch.float16


def unwrap_model(model: Any) -> nn.Module:
    """The bare model under the wrapper and any whole-model OptimizedModule."""
    if isinstance(model, _SpyreModelWrapper):
        model = model._model
    return getattr(model, "_orig_mod", model)


class _SpyreHiddenStates(torch.Tensor):
    """Select sampled rows on Spyre, reusing the padded body output for decode.

    A 1-D integer index goes through ``select_rows``, padded onto a warmed row bucket,
    which a plain ``x[idx]`` cannot do. Upstream reaches it from ``execute_model``'s
    ``hidden_states[logits_indices]`` and ``_dummy_run``'s
    ``hidden_states[logit_indices_device]``. Slices such as
    ``hidden_states[:num_scheduled_tokens]`` and every other op see a plain tensor.
    """

    _row_buckets: list[int]

    @classmethod
    def wrap(cls, hidden_states: torch.Tensor, row_buckets: list[int]) -> torch.Tensor:
        out = hidden_states.as_subclass(cls)
        out._row_buckets = row_buckets
        return out

    @classmethod
    def __torch_function__(cls, func, types, args: tuple[Any, ...] = (), kwargs=None):
        if func is torch.Tensor.__getitem__ and len(args) == 2:
            src, index = args
            if (
                isinstance(src, cls)
                and isinstance(index, torch.Tensor)
                and index.ndim == 1
                and index.dtype in (torch.int32, torch.int64)
            ):
                return src._gather_rows(index)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))

    def _gather_rows(self, row_indices: torch.Tensor) -> torch.Tensor:
        source = self.as_subclass(torch.Tensor)
        idx = row_indices.long()
        num_rows = idx.numel()
        buckets = self._row_buckets
        pos = bisect.bisect_left(buckets, num_rows)
        padded_rows = buckets[pos] if pos < len(buckets) else num_rows
        if num_rows == 0:
            return source[:0]
        # Decode already has one row per request, followed only by bucket padding.
        if source.shape[0] == padded_rows and torch.equal(idx, torch.arange(num_rows)):
            rows = source
        else:
            # index_select compiles per index length, so use the warmed row buckets.
            if padded_rows != num_rows:
                idx = torch.cat([idx, idx[-1:].expand(padded_rows - num_rows)])
            rows = select_rows(source, idx)
        # A plain view of the real rows. ``_spyre_padded_rows`` carries the padded gather, which
        # already sits on a warmed row bucket, so ``compute_logits`` projects it on Spyre as is.
        # The cast only lets the type checker accept that extra attribute on a Tensor.
        out = cast(Any, rows[:num_rows])
        out._spyre_padded_rows = rows
        return out


def warm_sample_row_gathers(
    model: _SpyreModelWrapper, hidden_states: torch.Tensor, row_widths: list[int]
) -> None:
    """Compile the sampled-row gather and the lm_head on its output, per row bucket.

    A multi-row gather returns a padded device layout that ``_dummy_sampler_run``'s
    rows lack, and the lm_head graph guards on layout.
    """
    if not isinstance(hidden_states, _SpyreHiddenStates):
        return
    for rows in row_widths:
        if rows <= hidden_states.shape[0]:
            indices = torch.arange(rows, dtype=torch.int64)
            model.compute_logits(hidden_states[indices])
            # On an exact body bucket, arange takes the decode path and reuses the body.
            # A mixed step can still gather there: tokens [1, 1, 2] give a 4-token body
            # and index [0, 1, 3], padded to [0, 1, 3, 3]. Warm that gather with a
            # reversed index; it compiles per shape, not per value. From two rows up:
            # without a 1 bucket, one sampled row pads to [1, 1].
            if rows == hidden_states.shape[0] and rows > 1:
                model.compute_logits(hidden_states[indices.flip(0)])


class _SpyreModelWrapper:
    """Transparent wrapper that converts model inputs/outputs at the boundary.

    Input conversion (CPU → Spyre):
        For example, input_ids and positions arrive as CPU tensors (int32/int64) because
        self.device=CPU in the runner and buffer scatter ops run on CPU.
        Convert them to int64 and provide them to the model.

    Output conversion:
        Generative hidden_states stay on Spyre as ``_SpyreHiddenStates``, so the
        sampled rows are gathered and projected on-device and only the logits D2H.

    Wrapping at the model level ensures ALL call sites get the right
    device — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).
    """

    def __init__(
        self,
        model: nn.Module,
        spyre_device: torch.device,
        keep_outputs_on_device: bool = False,
        logits_row_buckets: list[int] | None = None,
        shape_bucketer: SpyreShapeBucketer | None = None,
        *,
        model_dtype: torch.dtype,
        inputs_embeds_buffer: torch.Tensor | None = None,
        sample_rows_on_device: bool = False,
        keep_aux_hidden_states_on_device: bool = False,
    ):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)
        object.__setattr__(self, "_keep_outputs_on_device", keep_outputs_on_device)
        object.__setattr__(self, "_sample_rows_on_device", sample_rows_on_device)
        object.__setattr__(self, "_logits_row_buckets", logits_row_buckets or [])
        object.__setattr__(self, "_shape_bucketer", shape_bucketer)
        object.__setattr__(self, "_model_dtype", model_dtype)
        object.__setattr__(self, "_inputs_embeds_buffer", inputs_embeds_buffer)
        object.__setattr__(
            self, "_keep_aux_hidden_states_on_device", keep_aux_hidden_states_on_device
        )

    @contextmanager
    def outputs_on_host(self):
        """D2H the outputs for the duration, for upstream's host-side dummy-run gather."""
        keep = self._keep_outputs_on_device
        object.__setattr__(self, "_keep_outputs_on_device", False)
        try:
            yield keep
        finally:
            object.__setattr__(self, "_keep_outputs_on_device", keep)

    def __call__(self, *args, **kwargs):
        # Convert integer tensor inputs to Spyre int64. Do not use int32:
        # stock torch-spyre SDSC cannot schedule integer add (warmup crash
        # ``0_add``). RoBERTa ``position_ids + padding_idx`` is applied on the
        # host in ``_preprocess``, before this copy.
        is_integer = lambda t: t.dtype in (torch.int32, torch.int64)
        args_converted = convert_tensor_tree(
            args, device=self._spyre_device, dtype=torch.int64, predicate=is_integer
        )
        kwargs_converted = convert_tensor_tree(
            kwargs, device=self._spyre_device, dtype=torch.int64, predicate=is_integer
        )

        # The Llama-4 scale cache keys on `positions` identity, blind to an in-place rewrite.
        reset_llama4_scale_cache()

        t0 = time.time()
        result = self._model(*args_converted, **kwargs_converted)

        # Device poolers keep their outputs; only generation opts into sampled-row gathering.
        if not self._keep_outputs_on_device:
            if self._keep_aux_hidden_states_on_device:
                hidden_states, aux_hidden_states = result
                result = (convert_tensor_tree(hidden_states, device="cpu"), aux_hidden_states)
            elif (
                isinstance(result, torch.Tensor)
                and result.device.type == "spyre"
                and self._sample_rows_on_device
            ):
                result = _SpyreHiddenStates.wrap(result, self._logits_row_buckets)
            else:
                result = convert_tensor_tree(result, device="cpu")

        input_ids = kwargs_converted.get("input_ids")
        num_tokens = input_ids.shape[0] if input_ids is not None else -1
        logger.debug("t_token: %.2fms [num tokens %d]", (time.time() - t0) * 1000, num_tokens)

        return result

    def _to_spyre(self, t):
        return convert(t, device=self._spyre_device) if isinstance(t, torch.Tensor) else t

    def _to_cpu(self, t):
        return convert(t, device="cpu") if isinstance(t, torch.Tensor) else t

    def compute_logits(self, hidden_states, *args, **kwargs):
        """Run the lm_head on Spyre over the sampled rows, padded to a warmed row bucket.

        Logits come back on CPU: SpyreLogitsProcessor converts them after the head, in
        ``_apply_head`` at TP=1 and right after the all_gather in ``_gather_logits`` at TP>1.
        """
        num_rows = hidden_states.shape[0]
        buckets = self._logits_row_buckets
        idx = bisect.bisect_left(buckets, num_rows)
        padded_rows = buckets[idx] if idx < len(buckets) else num_rows
        gathered = getattr(hidden_states, "_spyre_padded_rows", None)
        if gathered is not None:
            hidden_states = gathered
        else:
            if hidden_states.device.type == "spyre":
                # A view of the body output (prompt logprobs) changes the head's work
                # division, and with it the fp16 accumulation order.
                hidden_states = convert(hidden_states, device="cpu")
            if padded_rows != num_rows:
                hidden_states = F.pad(hidden_states, (0, 0, 0, padded_rows - num_rows))
            hidden_states = convert(hidden_states, device=self._spyre_device)

        logits = self._model.compute_logits(hidden_states, *args, **kwargs)

        if padded_rows != num_rows and logits is not None:
            logits = logits[:num_rows]
        return logits

    def embed_multimodal(self, **kwargs):
        """Move float multimodal inputs (e.g. ``pixel_values``) onto Spyre.

        The runner reaches this through ``__getattr__``, bypassing ``__call__``'s
        input conversion, so pixel tensors would otherwise arrive on CPU while the
        vision weights are on Spyre.
        """

        def _to_spyre_float(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return convert(t, dtype=self._model_dtype, device=self._spyre_device)
            return t

        # These prepare their inputs on the host and upload once themselves.
        if not isinstance(self._model, SpyreVisionEncoderMixin):
            kwargs = tree_map(_to_spyre_float, kwargs)
        # The parts of a vision tower outside a compiled block run eager, so each Spyre
        # op with a decomposition there reaches it through torch-spyre's lazily-compiled
        # PrivateUse1 kernel, which compiles without fullgraph. Decompositions built on
        # for_each_tile (SDPA since torch-spyre#4550) emit a scan whose while_loop
        # lowering reads the loop index with .item(); without fullgraph that needs
        # capture_scalar_outputs, or the trace dies with DataDependentOutputException.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            return self._model.embed_multimodal(**kwargs)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        """Move input_ids/is_multimodal/multimodal_embeddings onto Spyre.

        gpu_model_runner._preprocess calls this directly on `self.model`,
        bypassing __call__, so it needs its own CPU <-> Spyre boundary
        conversion: input_ids and is_multimodal are CPU buffers, while the text
        embedding table (and the merge with multimodal_embeddings) lives on
        Spyre.

        Bucket the token count: this runs on the raw scheduled count, so at TP>1
        the vocab-parallel all_reduce is `num_tokens * hidden` for every distinct
        prompt length, and some of those collective schedules fail to build. Pad
        on CPU and trim after; padding inside a compiled collective corrupts
        output. `is_multimodal` is padded the same way so it still lines up with
        `input_ids` for the merge; the padded rows are never multimodal.

        Everything comes back on device, merged or not. Upstream copies what we
        return into a row-prefix view of its persistent inputs_embeds buffer
        (`gpu_model_runner._preprocess`: `inputs_embeds.gpu[:n].copy_(...)`), and
        torch-spyre#4730 rejects an H2D copy whose destination is narrowed -- it
        requires `dev_sizes == dma_sizes`, while the view is `[n, hidden]` against
        a base allocated at the full token budget. Returning a CPU tensor makes
        that copy H2D and kills the engine on every multimodal request; keeping it
        on device makes the same copy d2d, which the check allows. It also avoids
        a D2H that upstream's H2D immediately undoes.

        The result is staged in that buffer at the bucket's shape and returned as its
        prefix view: upstream's copy is then a self-copy torch-spyre skips, instead of
        a `copy_from_d2d` compiled for every distinct `n`.

        A merged result previously came back on CPU because the merge's
        `torch.where` output layout did not survive that d2d `copy_`. If a merged
        multimodal prompt starts producing garbage rather than failing, suspect
        that layout again before anything else here.
        """
        # Generic path: model.embed_input_ids only does text embedding, or
        # boolean-mask ops are handled inside the model's own patch (e.g.
        # patch_embed_input_ids for Granite4Vision).
        num_tokens = input_ids.shape[0]
        vision = isinstance(self._model, SpyreVisionEncoderMixin)
        # An image step stages at least two rows: a one-row copy out of the merge's layout
        # fails (torch-spyre#4883).
        merges = vision and multimodal_embeddings is not None and len(multimodal_embeddings) > 0
        target = max(num_tokens, 2) if merges else num_tokens
        bucketer = self._shape_bucketer
        if bucketer is not None:
            padded_tokens = bucketer.find_bucket(target)
        else:
            padded_tokens = target if target != num_tokens else None
        if padded_tokens is not None and padded_tokens != num_tokens:
            input_ids = F.pad(input_ids, (0, padded_tokens - num_tokens))
            if is_multimodal is not None:
                is_multimodal = F.pad(is_multimodal, (0, padded_tokens - num_tokens))

        input_ids = convert(input_ids, dtype=torch.int64, device=self._spyre_device)
        # These place their image tokens on the host (``SpyreVisionEncoderMixin.embed_input_ids``),
        # where their encoder output already is, so neither goes to the card here.
        if not vision:
            is_multimodal = self._to_spyre(is_multimodal)
            multimodal_embeddings = tree_map(self._to_spyre, multimodal_embeddings)

        result = self._model.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )
        buf = self._inputs_embeds_buffer
        if (
            padded_tokens is not None
            and buf is not None
            and isinstance(result, torch.Tensor)
            and result.shape[0] == padded_tokens <= buf.shape[0]
            and result.shape[1:] == buf.shape[1:]
            and result.dtype == buf.dtype
        ):
            buf[:padded_tokens].copy_(result)
            return buf[:num_tokens]
        if padded_tokens is not None and padded_tokens != num_tokens:
            # A plain prefix slice, not select_rows: the padding above always
            # appends at the end, so the real rows are always 0..num_tokens
            # contiguously -- no gather needed, and a merged result's torch.where
            # output layout select_rows's own compiled index_select does not accept.
            result = tree_map(
                lambda t: t[:num_tokens] if isinstance(t, torch.Tensor) else t, result
            )
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        # `__init__` fills our `__dict__` via `object.__setattr__`, so a name in it is
        # ours, not the model's.
        if name in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)
