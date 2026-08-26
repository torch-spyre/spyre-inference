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

"""Spyre patches for Gemma4 MoE routing.

Two upstream seams do not lower on Spyre:

1. ``gemma4_routing_function_torch`` uses ``one_hot`` (scatter), ``gather`` and
   advanced indexing, none of which lower. The replacement computes the *dense*
   ``[T, E]`` combine (renormalized top-k probs folded with the per-expert
   scale) from lowerable ops and returns THAT as ``topk_weights`` -- rebuilding
   ``[T, E]`` from a ``[T, K]`` selection needs a scatter Spyre cannot express.

2. ``CustomRoutingRouter._compute_routing`` casts weights to fp32 for CUDA. On
   Spyre fp16 and fp32 use different stick widths and the eager cast does not
   re-tile, so ``.to(float32)`` returns garbage; keep weights fp16 throughout.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.models import gemma4 as _gemma4

logger = init_logger(__name__)


def _spyre_gemma4_routing_function_torch(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma4 routing from Spyre-lowerable ops, returning the dense [T, E] combine.

    ``topk_weights`` is the dense ``[T, E]`` combine, not the customary
    ``[T, K]`` (see module docstring). ``topk_ids`` is returned for the contract
    but unused by the dense expert compute.
    """
    probs = torch.nn.functional.softmax(gating_output, dim=-1)  # [T, E]

    # A single-row (T==1 decode) reduction cannot be materialized on device
    # (topk's Inductor layout pass and eager amax/sum normalization both choke).
    # Run the whole chain on >=2 rows and slice back at the very end.
    tokens = probs.shape[0]
    probs_w = probs.expand(2, -1).contiguous() if tokens == 1 else probs  # [R, E]

    topk_vals, topk_ids = torch.topk(probs_w, k=topk, dim=-1)  # [R, K]

    # k-th largest per row via amax: amin is unimplemented, min falls back to
    # CPU, and the offset-(k-1) slice is stick-unaligned.
    kth = -(-topk_vals).amax(dim=-1, keepdim=True)  # [R, 1]

    mask = (probs_w >= kth).to(probs_w.dtype)  # [R, E]
    kept = probs_w * mask

    denom = kept.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    dense = kept / denom * per_expert_scale.to(probs_w.dtype)  # [R, E]

    return dense[:tokens], topk_ids[:tokens]


def _spyre_compute_routing(
    self: CustomRoutingRouter,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    *,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``CustomRoutingRouter._compute_routing`` without the fp32 weight cast."""
    topk_weights, topk_ids = self.custom_routing_function(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
    )
    return topk_weights, topk_ids.to(torch.int32 if indices_type is None else indices_type)


def _patch() -> None:
    current = getattr(_gemma4, "gemma4_routing_function_torch", None)
    if current is not None and not getattr(current, "_spyre_patched", False):
        _spyre_gemma4_routing_function_torch._spyre_patched = True
        _gemma4.gemma4_routing_function_torch = _spyre_gemma4_routing_function_torch
        logger.info("Patched gemma4_routing_function_torch for Spyre (dense [T,E]).")

    if not getattr(CustomRoutingRouter._compute_routing, "_spyre_patched", False):
        _spyre_compute_routing._spyre_patched = True
        CustomRoutingRouter._compute_routing = _spyre_compute_routing
        logger.info("Patched CustomRoutingRouter._compute_routing for Spyre (fp16).")


_patch()
