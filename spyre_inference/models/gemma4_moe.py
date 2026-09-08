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

"""Gemma-4's routing recipe for the generic Spyre MoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn


logger = init_logger(__name__)


def configure_gemma4_moe_layers(layers: Iterable[nn.Module]) -> None:
    """Register Gemma-4's full-softmax, scaled GELU expert recipe.

    Per-expert output scaling is explicitly requested by this recipe, not the
    generic backend. The generic backend compiles only its MoE regions, so this
    works with both compiled and eager outer model execution.
    """
    configured = 0
    for decoder in layers:
        moe = getattr(decoder, "moe", None)
        if moe is None:
            continue
        configure_spyre_moe_layer(
            moe.experts.routed_experts,
            SpyreMoERecipe(
                activation="gelu_tanh",
                routing="full_softmax",
                expert_scale=moe.per_expert_scale,
                fold_expert_scale_into_down=True,
            ),
        )
        configured += 1
    if configured:
        logger.info("Spyre: configured %d Gemma-4 MoE layers.", configured)
