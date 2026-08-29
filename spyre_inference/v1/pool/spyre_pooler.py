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

"""Spyre pooler: CLS/LAST via index_select, L2 via rsqrt. MEAN stays CPU (#3507)."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import (
    CLSPool,
    LastPool,
    SequencePoolingMethod,
)
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.v1.outputs import PoolerOutput

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# AllPool uses slice views; unsafe on Spyre.
TOKEN_POOLING_TASKS = frozenset({"token_embed", "token_classify"})


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


class SpyreNormalize(PoolerNormalize):
    """L2 via ``rsqrt``; ``clamp_min`` missing. ``finfo.tiny`` keeps fp16 zeros."""

    def forward_chunk(self, pooled_data: torch.Tensor) -> torch.Tensor:
        if pooled_data.device.type != "spyre":
            return super().forward_chunk(pooled_data)

        eps = torch.finfo(pooled_data.dtype).tiny
        sumsq = pooled_data.pow(2).sum(-1, keepdim=True)
        return pooled_data * sumsq.add(eps).rsqrt()


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
    """Swap CLS/LAST to Spyre forms. Returns ``(n_patched, unsupported)`` e.g. MEAN (#3507)."""
    num_patched = 0
    unsupported: list[str] = []

    if isinstance(pooler, SequencePooler):
        pooling = pooler.pooling
        if isinstance(pooling, SpyreCLSPool | SpyreLastPool):
            num_patched += 1  # already swapped (shared under DispatchPooler)
        elif isinstance(pooling, CLSPool):
            pooler.pooling = SpyreCLSPool()
            num_patched += 1
        elif isinstance(pooling, LastPool):
            pooler.pooling = SpyreLastPool()
            num_patched += 1
        elif isinstance(pooling, SequencePoolingMethod):
            unsupported.append(type(pooling).__name__)
    elif isinstance(pooler, DispatchPooler):
        for sub in pooler.poolers_by_task.values():
            sub_patched, sub_unsupported = patch_pooler_for_spyre(sub)
            num_patched += sub_patched
            unsupported.extend(sub_unsupported)

    return num_patched, unsupported


def configure_pooling_for_spyre(model: nn.Module, spyre_device: torch.device) -> bool:
    """Patch pooler for Spyre. True if on-device; False if MEAN/unknown/FP32 head → CPU."""
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
        pooler.to("cpu")
        if hasattr(model, "classifier"):
            model.classifier.to("cpu")
        return False

    classifier = getattr(model, "classifier", None)
    # Spyre has no FP32 batch matmul (reranker RobertaClassificationHead is
    # float32 via head_dtype). Run pooler + classifier on CPU like MEAN.
    fp32_head = _module_has_float32_params(pooler) or (
        classifier is not None and _module_has_float32_params(classifier)
    )
    if fp32_head:
        pooler.to("cpu")
        if classifier is not None:
            classifier.to("cpu")
        logger.info(
            "Pooling: FP32 classifier/head unsupported on Spyre "
            "(no FP32 batchmatmul); running pooler on CPU"
        )
        return False

    num_norm = patch_normalize_for_spyre(pooler)
    num_heads = patch_embedding_heads_for_spyre(pooler)
    on_spyre = ["pooler"]
    if classifier is not None:
        on_spyre.append("classifier")
    logger.info(
        "Pooling: keeping %s on %s (%d CLS/LAST method(s) on "
        "index_select, %d normalize head(s) on rsqrt, %d embed head(s) "
        "with CPU dtype cast)",
        ", ".join(on_spyre),
        spyre_device,
        num_patched,
        num_norm,
        num_heads,
    )
    return True
