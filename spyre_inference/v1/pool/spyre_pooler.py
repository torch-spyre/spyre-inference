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


def select_rows(hidden_states: torch.Tensor, row_indices_cpu: torch.Tensor) -> torch.Tensor:
    """Row gather via ``index_select`` (no Spyre ``aten::index.Tensor``).

    ``row_indices_cpu`` may be 1-D (CLS/LAST, unpack) or ``[B, L]`` (pack).
    """
    flat_idx = row_indices_cpu.reshape(-1)
    if hidden_states.device.type != "spyre":
        return torch.index_select(
            hidden_states, 0, flat_idx.to(device=hidden_states.device, dtype=torch.long)
        )

    # Spyre has no int64 index kernel. convert() H2D is blocking
    # (copy_tensor non_blocking=False); no extra synchronize.
    indices = convert(flat_idx.to(torch.int32), hidden_states.device)
    return torch.index_select(hidden_states, 0, indices)


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
        if hidden_states.shape[0] > total:
            hidden_states = select_rows(hidden_states, torch.arange(total, dtype=torch.int64))
        return super().forward(convert(hidden_states, "cpu"), pooling_metadata)


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

    def __init__(self, enable_chunked_prefill: bool) -> None:
        nn.Module.__init__(self)
        self.enable_chunked_prefill = enable_chunked_prefill

    def forward(self, hidden_states, pooling_metadata):
        if self.enable_chunked_prefill:
            raise NotImplementedError(
                "chunked prefill is unsupported with token-level pooling on Spyre"
            )
        counts = pooling_metadata.get_pooling_cursor().num_scheduled_tokens_cpu.tolist()
        out = []
        start = 0
        for n in counts:
            out.append(select_rows(hidden_states, torch.arange(start, start + n)))
            start += n
        return out


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


def patch_pooler_for_spyre(pooler: nn.Module) -> tuple[int, list[str]]:
    """Install Spyre CLS, LAST, MEAN, and token AllPool. Returns ``(n_patched, unsupported)``."""
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
            pooler.pooling = SpyreAllPool(pooling.enable_chunked_prefill)
            num_patched += 1
        else:
            unsupported.append(type(pooling).__name__)
    elif isinstance(pooler, DispatchPooler):
        for sub in pooler.poolers_by_task.values():
            sub_patched, sub_unsupported = patch_pooler_for_spyre(sub)
            num_patched += sub_patched
            unsupported.extend(sub_unsupported)

    return num_patched, unsupported


def configure_pooling_for_spyre(model: nn.Module, spyre_device: torch.device) -> bool:
    """Patch CLS/LAST/MEAN/token AllPool. True if hidden states stay on Spyre.

    CLS/LAST gather on device. MEAN copies packed ``[T, H]`` as fp16 and
    reduces with ``MeanPool`` on the host: destagger of a device fp32 sum
    is garbage (torch-spyre#2971). False if the method is unknown or the
    head is an FP32 linear.
    """
    pooler = getattr(model, "pooler", None)
    if pooler is None:
        logger.info("Pooling: model has no pooler; leaving outputs on CPU")
        return False

    num_patched, unsupported = patch_pooler_for_spyre(pooler)
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
