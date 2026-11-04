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

"""gpt-oss on the Spyre MoE backend.

gpt-oss takes a softmax over its top-k router logits, the ``topk_softmax`` recipe. Its experts
are ``swigluoai`` with gate/up and down biases, and the checkpoint interleaves gate and up rows
in ``w13``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.gpt_oss import GptOssForCausalLM, MLPBlock

from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer, deinterleave_gate_up

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn
    from vllm.config import VllmConfig

logger = init_logger(__name__)

GPT_OSS_MOE_RECIPE = SpyreMoERecipe(
    activation="swigluoai",
    routing="topk_softmax",
    prepare_gate_up_weight=deinterleave_gate_up,
)



def configure_gpt_oss_moe_layers(layers: Iterable[nn.Module]) -> int:
    """Opt every gpt-oss MoE block into ``GPT_OSS_MOE_RECIPE``; returns the count."""
    configured = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, MLPBlock):
            continue
        configure_spyre_moe_layer(mlp.experts.routed_experts, GPT_OSS_MOE_RECIPE)
        configured += 1
    if configured:
        logger.info("Spyre: configured %d gpt-oss MoE layers.", configured)
    return configured


class SpyreGptOssForCausalLM(GptOssForCausalLM):
    """gpt-oss on Spyre with Spyre MoE expert dispatch."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not configure_gpt_oss_moe_layers(self.model.layers):
            raise RuntimeError(
                "gpt-oss, but no decoder layer has an MLPBlock: vLLM has moved the experts; "
                "update configure_gpt_oss_moe_layers."
            )
