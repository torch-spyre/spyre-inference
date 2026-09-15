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

"""Spyre pooler: CLS/LAST via index_select, MEAN via one packed D2H, L2 via rsqrt."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import (
    CLSPool,
    LastPool,
    MeanPool,
    SequencePoolingMethod,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.v1.outputs import PoolerOutput

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    next_bucket,
)

logger = init_logger(__name__)


class SpyreEmbeddingPoolerHead(EmbeddingPoolerHead):
    """D2H before ``.to(head_dtype)`` when dtype changes; rest is upstream.

    Pooling defaults ``head_dtype=float32``. Spyre fp16→fp32 cast after CLS
    corrupts embeddings; keep gather on Spyre and cast on CPU.
    """

    def forward(self, pooled_data, pooling_metadata):
        if self.head_dtype is not None:
            sample = (
                pooled_data[0] if isinstance(pooled_data, list) and pooled_data else pooled_data
            )
            if (
                isinstance(sample, torch.Tensor)
                and sample.device.type == "spyre"
                and sample.dtype != self.head_dtype
            ):
                if isinstance(pooled_data, list):
                    pooled_data = torch.stack(pooled_data)
                # Upstream ``.to(head_dtype)`` is then a no-op on CPU.
                pooled_data = convert(pooled_data, "cpu").to(self.head_dtype)
        return super().forward(pooled_data, pooling_metadata)


def _pooler_output_on_cpu(raw_pooler_output: PoolerOutput) -> PoolerOutput:
    """Materialize Spyre pooled tensors on CPU; leave host tensors unchanged."""
    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.device.type == "spyre":
            return convert(raw_pooler_output, "cpu")
        return raw_pooler_output
    assert isinstance(raw_pooler_output, list)
    return [
        convert(t, "cpu") if isinstance(t, torch.Tensor) and t.device.type == "spyre" else t
        for t in raw_pooler_output
    ]


def copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    """vLLM ``_copy_pooler_output_to_cpu`` after Spyre→CPU via ``convert``.

    Upstream uses ``.to("cpu", non_blocking=True)``, which is not a valid Spyre
    D2H path. Convert first so the shared finished-mask / partial-batch logic
    stays in vLLM.
    """
    from vllm.v1.worker.gpu_model_runner import (
        _copy_pooler_output_to_cpu as _vllm_copy_pooler_output_to_cpu,
    )

    return _vllm_copy_pooler_output_to_cpu(
        _pooler_output_on_cpu(raw_pooler_output),
        finished_mask,
    )


def cursor_row_indices_cpu(pooling_cursor, *, last: bool) -> torch.Tensor:
    """First/last row indices from CPU counts (device cumsum slices are unsafe)."""
    counts = pooling_cursor.num_scheduled_tokens_cpu.to(torch.int64)
    ends = torch.cumsum(counts, dim=0)
    return ends - 1 if last else ends - counts


def select_rows(hidden_states: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Row gather via ``index_select`` (no Spyre ``aten::index.Tensor``).

    ``row_indices`` may be 1-D (CLS/LAST, unpack) or ``[B, L]`` (pack), on CPU
    or already on ``hidden_states.device``. Spyre has no int64 index kernel.
    """
    flat_idx = row_indices.reshape(-1)
    device = hidden_states.device
    if device.type != "spyre":
        return torch.index_select(hidden_states, 0, flat_idx.to(device=device, dtype=torch.long))

    if flat_idx.device.type == "spyre":
        if flat_idx.dtype != torch.int32:
            flat_idx = convert(flat_idx, dtype=torch.int32)
        return torch.index_select(hidden_states, 0, flat_idx)

    # convert() H2D is blocking (copy_tensor non_blocking=False).
    return torch.index_select(hidden_states, 0, convert(flat_idx.to(torch.int32), device))


class SpyreCLSPool(CLSPool):
    """CLS via ``index_select`` (keeps upstream ``isinstance`` checks)."""

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise RuntimeError("partial prefill is not supported with CLS pooling")
        return select_rows(hidden_states, cursor_row_indices_cpu(cursor, last=False))


class SpyreLastPool(LastPool):
    """LAST via ``index_select``."""

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        return select_rows(hidden_states, cursor_row_indices_cpu(cursor, last=True))


class SpyreMeanPool(MeanPool):
    """MEAN after one packed D2H; the segment sum is not on Spyre.

    Device fp32 lives staggered (torch-spyre#2971). Raw ``convert`` of an
    fp32 sum is garbage, and destagger via ``to(fp16)`` then convert then
    upcast is also garbage (e5/roberta cosine ~-0.02). After cropping
    encoder pad past ``sum(lens)``, copy the valid prefix as fp16 and let
    ``MeanPool`` reduce — empty batch, fp32 accumulator, zero-length
    ``nan``. ``convert`` is a no-op when the tensor is already on the host.
    """

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        prompt_lens = cursor.prompt_lens_cpu.to(torch.int64)
        total = int(prompt_lens.sum().item()) if prompt_lens.numel() else 0
        # Crop on the host, after the D2H: SpyreDispatchPooler leaves
        # hidden_states at its bucketed length, so this crop now fires on every
        # padded batch. Cropping on device would need a real-length index_select,
        # adding a torch.compile specialization per distinct prompt length -- the
        # defect the dispatcher exists to remove. The D2H is bucket-sized rather
        # than the real length it used to be while upstream still sliced, so it
        # is slightly larger; a host slice after it costs nothing.
        hidden_states = convert(hidden_states, "cpu")
        if hidden_states.shape[0] > total:
            hidden_states = hidden_states[:total]
        return super().forward(hidden_states, pooling_metadata)


class SpyreDispatchPooler(DispatchPooler):
    """``DispatchPooler`` that leaves ``hidden_states`` at its bucketed length.

    Upstream slices ``hidden_states`` down to the group's *real* token count
    before handing it to the sub-pooler (``DispatchPooler.forward``:
    ``hidden_states[token_offset : token_offset + num_group_tokens]``). On Spyre
    that makes ``select_rows``' ``index_select`` source shape track the prompt
    length, so ``torch.compile(dynamic=False)`` adds a specialization per distinct
    length and Dynamo rescans a growing guard chain on every later call —
    throughput decays as more distinct lengths are seen, which is why
    ``TorchSpyreModelRunner._pool`` deliberately hands this a padded tensor in the
    first place. Undoing the slice here is what makes that intent hold.

    Safe because the Spyre seqwise poolers address rows through
    ``cursor_row_indices_cpu``, which only ever names rows inside the real range —
    they never read the padding the slice would have removed.

    Only the single-task-group case is handled. With several groups each one
    starts at a nonzero token offset and upstream rebases the cursor's row
    indices onto its slice; without the slice those indices would need the offset
    added back instead, so anything else defers to upstream unchanged.
    """

    def forward(self, hidden_states, pooling_metadata):
        tasks = list(pooling_metadata.tasks)
        if (
            hidden_states.device.type != "spyre"
            or pooling_metadata.pooling_cursor is None
            or len(set(tasks)) != 1
        ):
            return super().forward(hidden_states, pooling_metadata)

        task = tasks[0]
        if not (pooler := self.poolers_by_task.get(task)):
            raise ValueError(
                f"Unsupported task: {task!r} Supported tasks: {self.get_supported_tasks()}"
            )
        # Mirror upstream's accumulation: a sub-pooler may return a stacked
        # tensor, which upstream flattens into one entry per request.
        outputs: list[torch.Tensor | None] = []
        outputs.extend(pooler(hidden_states, pooling_metadata))
        return outputs


class SpyreNormalize(PoolerNormalize):
    """L2 via ``rsqrt``; ``clamp_min`` missing. ``finfo.tiny`` keeps fp16 zeros."""

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        if pooled_data.device.type != "spyre":
            return super().forward_chunk(pooled_data)

        eps = torch.finfo(pooled_data.dtype).tiny
        sumsq = pooled_data.pow(2).sum(-1, keepdim=True)
        return pooled_data * sumsq.add(eps).rsqrt()


class SpyreAllPool(AllPool):
    """Per-request rows via ``index_select``; ``torch.split`` gives unsafe views."""

    def __init__(
        self,
        enable_chunked_prefill: bool,
        defer_trim: bool = False,
        len_ladder: list[int] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.enable_chunked_prefill = enable_chunked_prefill
        # Only a SpyreTokenPooler that will trim afterwards may set this; on its
        # own this class keeps AllPool's contract of one real-length chunk per
        # request, so an unpaired use cannot silently ship padded rows.
        self.defer_trim = defer_trim
        # Passed in from configure_pooling_for_spyre, which runs inside
        # load_model's set_current_vllm_config context. Resolving it here from
        # get_current_vllm_config() would not work: forward runs outside that
        # context (WorkerWrapperBase wraps __init__/init_device/
        # initialize_from_config, not execute_model), so the lookup raises and
        # the ladder would silently stay empty for the life of the process.
        # Empty means plain stick alignment, which still bounds the
        # specialization count but at every 64-multiple instead of the
        # powers of two.
        self.len_ladder = list(len_ladder) if len_ladder else []

    def forward(self, hidden_states, pooling_metadata):
        if self.enable_chunked_prefill:
            raise NotImplementedError(
                "chunked prefill is unsupported with token-level pooling on Spyre"
            )
        # Gather a bucketed row count, not the request's real token count: an
        # index sized on the real length makes index_select's output shape track
        # it, which adds a torch.compile specialization per distinct prompt
        # length and leaves Dynamo rescanning a growing guard chain on every
        # later call. Rows past the real length clamp to the last real row, so
        # they stay in bounds and cost only duplicate work; SpyreTokenPooler
        # drops them after the head, whose ops are all row-wise.
        counts = pooling_metadata.get_pooling_cursor().num_scheduled_tokens_cpu.tolist()
        out = []
        start = 0
        for n in counts:
            # n == 0 has no last real row to clamp onto (the clamp would index
            # start - 1), so it takes the plain path and yields an empty chunk.
            # Not reachable while chunked prefill is rejected, but the plain path
            # handled it and this keeps that.
            if self.defer_trim and n > 0:
                aligned = next_bucket(n, self.len_ladder)
                idx = start + torch.arange(aligned, dtype=torch.int64).clamp(max=n - 1)
            else:
                idx = torch.arange(start, start + n, dtype=torch.int64)
            out.append(select_rows(hidden_states, idx))
            start += n
        return out


class SpyreTokenPooler(TokenPooler):
    """Trim ``SpyreAllPool``'s bucketed rows back to real lengths, after the head.

    ``SpyreAllPool`` gathers a bucketed row count so ``index_select``'s shape does
    not track the request's token count. Every op in the token head is row-wise
    (``to(head_dtype)``, the ST projector, the last-dim slice, and
    normalize over ``dim=-1``), so the duplicate rows past the real length change
    nothing for the real ones and the head keeps running on device at a bucketed
    shape. They are dropped here instead, at the D2H every token-pooling output
    has to make anyway — a Spyre dim-0 slice view would not be safe.
    """

    def forward(self, hidden_states, pooling_metadata):
        pooled = super().forward(hidden_states, pooling_metadata)
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor is None:
            return pooled
        counts = cursor.num_scheduled_tokens_cpu.tolist()
        trimmed: list[torch.Tensor | None] = []
        for item, n in zip(pooled, counts):
            if item is None:
                trimmed.append(item)
                continue
            # Unconditional, including when the shape already matches: skipping
            # the D2H for an item that happens to land on a bucket would leave
            # that one on device while its neighbours came back on CPU, and _pool
            # runs late_interaction_runner.postprocess_pooler_output before
            # copy_pooler_output_to_cpu. The runner caches the query with
            # output.clone(), keeping its device, and scoring rejects a
            # query/document device mismatch -- which a fixed-length query
            # against variable-length documents would hit routinely. convert
            # short-circuits a same-device call, so this is free on CPU.
            item = convert(item, "cpu")
            trimmed.append(item if item.shape[0] == n else item[:n])
        return trimmed


def prepare_token_head_for_spyre(
    model: nn.Module, pooler: nn.Module, spyre_device: torch.device
) -> None:
    """Keep the token-level tail in fp16 so it can run on Spyre.

    Heads cast per chunk to a float32 ``head_dtype`` and the model casts before
    its own classifier; both are wrong on device, and Spyre has no fp32 matmul.
    """
    # Scope to the token sub-poolers: a DispatchPooler can also hold a sequence
    # pooler whose fp32 head is handled by SpyreEmbeddingPoolerHead instead.
    targets = [m for m in pooler.modules() if isinstance(m, TokenPooler)]
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        targets.append(classifier)
    if getattr(model, "head_dtype", None) is not None:
        model.head_dtype = torch.float16
    for target in targets:
        for module in target.modules():
            if getattr(module, "head_dtype", None) is not None:
                module.head_dtype = torch.float16  # ty: ignore[invalid-assignment]
        # A dtype cast on device returns wrong data; convert() detours via host.
        for param in target.parameters(recurse=True):
            if param.dtype == torch.float32:
                param.data = convert(param.data, spyre_device, torch.float16)


class SpyreCpuClassifier(nn.Module):
    """D2H wrapper for a classifier the model applies in its own forward.

    Token-classification models call ``self.classifier`` themselves, so moving it
    to CPU with the pooler leaves it receiving Spyre activations.
    """

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier
        self.param_dtype = next(classifier.parameters()).dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(convert(hidden_states, "cpu").to(self.param_dtype))


def run_pooling_tail_on_cpu(model: nn.Module, pooler: nn.Module) -> None:
    """Move pooler and classifier to CPU, wrapping a model-applied classifier."""
    pooler.to("cpu")
    classifier = getattr(model, "classifier", None)
    if classifier is None or isinstance(classifier, SpyreCpuClassifier):
        return
    # A reranker head owns the classifier and applies it after the pooler moves;
    # anything else is applied by the model itself and needs the D2H wrapper.
    owned = any(getattr(m, "classifier", None) is classifier for m in pooler.modules())
    classifier.to("cpu")
    if owned:
        return
    # An on-device fp16->fp32 cast returns wrong data, so neutralize the model's
    # head_dtype cast and upcast on the host instead.
    if getattr(model, "head_dtype", None) is not None:
        model.head_dtype = torch.float16
    model.classifier = SpyreCpuClassifier(classifier)


def _module_has_float32_params(module: nn.Module) -> bool:
    return any(p.dtype == torch.float32 for p in module.parameters())


def patch_normalize_for_spyre(pooler: nn.Module) -> int:
    """Replace ``PoolerNormalize`` with ``SpyreNormalize``. Recurses ``DispatchPooler``."""
    if isinstance(pooler, DispatchPooler):
        return sum(patch_normalize_for_spyre(sub) for sub in pooler.poolers_by_task.values())

    num_patched = 0
    for module in list(pooler.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, PoolerNormalize) and not isinstance(child, SpyreNormalize):
                setattr(module, name, SpyreNormalize())
                num_patched += 1
    return num_patched


def patch_embedding_heads_for_spyre(pooler: nn.Module) -> int:
    """Swap ``EmbeddingPoolerHead`` so fp32 ``head_dtype`` cast runs on CPU."""
    if isinstance(pooler, DispatchPooler):
        return sum(patch_embedding_heads_for_spyre(sub) for sub in pooler.poolers_by_task.values())

    num_patched = 0
    for module in list(pooler.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, EmbeddingPoolerHead) and not isinstance(
                child, SpyreEmbeddingPoolerHead
            ):
                setattr(
                    module,
                    name,
                    SpyreEmbeddingPoolerHead(
                        projector=child.projector,
                        head_dtype=child.head_dtype,
                        activation=child.activation,
                    ),
                )
                num_patched += 1
    return num_patched


def patch_pooler_for_spyre(
    pooler: nn.Module, len_ladder: list[int] | None = None
) -> tuple[int, list[str]]:
    """Install Spyre CLS, LAST, MEAN, and token AllPool. Returns ``(n_patched, unsupported)``.

    A pooler class this does not recognise is reported as unsupported rather than
    ignored: returning ``(0, [])`` for it made it invisible to both of
    ``configure_pooling_for_spyre``'s gates, so a DispatchPooler with one patched
    sub-pooler and one unrecognised one passed, and the SpyreDispatchPooler swap
    then handed the unrecognised one padded ``hidden_states``.
    """
    num_patched = 0
    unsupported: list[str] = []

    if isinstance(pooler, SequencePooler):
        pooling = pooler.pooling
        if isinstance(pooling, SpyreCLSPool | SpyreLastPool | SpyreMeanPool):
            num_patched += 1  # already swapped (shared under DispatchPooler)
        elif isinstance(pooling, CLSPool):
            pooler.pooling = SpyreCLSPool()
            num_patched += 1
        elif isinstance(pooling, LastPool):
            pooler.pooling = SpyreLastPool()
            num_patched += 1
        elif isinstance(pooling, MeanPool):
            pooler.pooling = SpyreMeanPool()
            num_patched += 1
        elif isinstance(pooling, SequencePoolingMethod):
            unsupported.append(type(pooling).__name__)
    elif isinstance(pooler, TokenPooler):
        pooling = pooler.pooling
        if isinstance(pooling, SpyreAllPool):
            num_patched += 1
        elif type(pooling) is AllPool:
            pooler.pooling = SpyreAllPool(pooling.enable_chunked_prefill, len_ladder=len_ladder)
            num_patched += 1
        else:
            unsupported.append(type(pooling).__name__)
        # Bucketing the gather is only safe when something trims afterwards, so
        # the two are switched on together and never independently.
        if isinstance(pooler.pooling, SpyreAllPool) and type(pooler) is TokenPooler:
            pooler.__class__ = SpyreTokenPooler
            pooler.pooling.defer_trim = True
    elif isinstance(pooler, DispatchPooler):
        for sub in pooler.poolers_by_task.values():
            sub_patched, sub_unsupported = patch_pooler_for_spyre(sub, len_ladder)
            num_patched += sub_patched
            unsupported.extend(sub_unsupported)
    else:
        unsupported.append(type(pooler).__name__)

    return num_patched, unsupported


def configure_pooling_for_spyre(
    model: nn.Module, spyre_device: torch.device, max_model_len: int | None = None
) -> bool:
    """Patch CLS/LAST/MEAN/token AllPool. True if hidden states stay on Spyre.

    CLS/LAST gather on device. MEAN copies packed ``[T, H]`` as fp16 and
    reduces with ``MeanPool`` on the host: destagger of a device fp32 sum
    is garbage (torch-spyre#2971). False if the method is unknown or the
    head is an FP32 linear.

    ``max_model_len`` builds the token-count ladder handed to ``SpyreAllPool``.
    It is a parameter rather than a ``get_current_vllm_config()`` lookup inside
    the pooler because only the caller is guaranteed to run inside a
    ``set_current_vllm_config`` context; token pooling degrades to plain stick
    alignment without it.
    """
    pooler = getattr(model, "pooler", None)
    if pooler is None:
        logger.info("Pooling: model has no pooler; leaving outputs on CPU")
        return False

    len_ladder = default_encoder_len_buckets(max_model_len) if max_model_len else []
    num_patched, unsupported = patch_pooler_for_spyre(pooler, len_ladder)
    if unsupported or num_patched == 0:
        reason = ", ".join(sorted(set(unsupported))) if unsupported else type(pooler).__name__
        logger.info(
            "Pooling: %s has no Spyre path; running the pooler on CPU",
            reason,
        )
        run_pooling_tail_on_cpu(model, pooler)
        return False

    classifier = getattr(model, "classifier", None)
    token_level = any(isinstance(m, SpyreAllPool) for m in pooler.modules())
    if token_level:
        if not len_ladder:
            logger.warning(
                "Pooling: token pooling has no length ladder (max_model_len was "
                "not passed); gathers round to every 64-multiple instead of the "
                "power-of-two buckets, so more shapes compile than necessary"
            )
        prepare_token_head_for_spyre(model, pooler, spyre_device)

    # torch-spyre SPYRE_FP32_OPS has add/mul/sum/mean, but not batchmatmul
    # (torch-spyre#1794). Reranker / classifier heads stay float32, so those
    # stay on CPU.
    fp32_head = _module_has_float32_params(pooler) or (
        classifier is not None and _module_has_float32_params(classifier)
    )
    if fp32_head:
        run_pooling_tail_on_cpu(model, pooler)
        logger.info(
            "Pooling: FP32 classifier/head unsupported on Spyre "
            "(no FP32 batchmatmul); running pooler on CPU"
        )
        return False

    num_norm = patch_normalize_for_spyre(pooler)
    num_heads = patch_embedding_heads_for_spyre(pooler)
    # Keep hidden_states bucketed through the dispatcher; see SpyreDispatchPooler.
    if type(pooler) is DispatchPooler:
        pooler.__class__ = SpyreDispatchPooler
    staying = ["hidden states"]
    if classifier is not None:
        staying.append("classifier")
    logger.info(
        "Pooling: %s stay on %s (%d method(s), %d normalize, %d embed heads)",
        ", ".join(staying),
        spyre_device,
        num_patched,
        num_norm,
        num_heads,
    )
    return True
