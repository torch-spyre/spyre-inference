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

"""Spyre-safe replacement for vLLM's ``_merge_multimodal_embeddings``.

``vllm.model_executor.models.utils._merge_multimodal_embeddings`` merges
multimodal embeddings into the text embedding tensor via a boolean-mask
index_put: ``inputs_embeds[is_multimodal] = mm_embeds_flat``. That lowers to
``aten::_index_put_impl_``, which has no Spyre kernel at all (hard
``NotImplementedError``, not a CPU ``FallbackWarning``). Its own exception
handler is also unreachable on Spyre: computing the diagnostic message calls
``is_multimodal.sum()``, whose bool->int64 promotion torch-spyre's Inductor
backend can't lower either.

Fix: do the placement on CPU (index_put is fine there) and combine with the
Spyre-resident text embeddings via ``torch.where``, the same broadcast-select
pattern already used for attention masking (see ``spyre_attn.py``).
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.models import utils as vllm_utils
from vllm.multimodal import NestedTensors

from .utils import convert

logger = init_logger(__name__)

_orig_merge_multimodal_embeddings = vllm_utils._merge_multimodal_embeddings


def _spyre_merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    if inputs_embeds.device.type != "spyre":
        return _orig_merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )

    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    is_multimodal_cpu = convert(is_multimodal, device="cpu")
    mm_embeds_flat_cpu = convert(mm_embeds_flat, dtype=input_dtype, device="cpu")
    scattered_cpu = torch.zeros(inputs_embeds.shape, dtype=input_dtype)
    try:
        scattered_cpu[is_multimodal_cpu] = mm_embeds_flat_cpu
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal_cpu.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = vllm_utils._embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during index put operation") from e

    scattered = convert(scattered_cpu, device=inputs_embeds.device)
    mask = convert(is_multimodal_cpu, device=inputs_embeds.device).unsqueeze(-1)
    return torch.where(mask, scattered, inputs_embeds)


def register() -> None:
    """Monkeypatch ``_merge_multimodal_embeddings`` to avoid the unimplemented
    ``aten::_index_put_impl_`` op on Spyre."""
    vllm_utils._merge_multimodal_embeddings = _spyre_merge_multimodal_embeddings
    logger.debug_once("Patched vllm._merge_multimodal_embeddings for Spyre")
