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

"""Per-tensor FP8 loading without upstream's ``_C`` requant, which the ``+empty`` vLLM build
lacks (#1458): shard scales go to the kernel as ``[1]`` or per-channel instead. Covers
``quant_method: fp8`` only; other per-tensor formats still reach the requant."""

from __future__ import annotations

from typing import cast

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.model_executor.utils import is_weights_pre_processed, replace_parameter

logger = init_logger(__name__)


def shard_scales_to_kernel_scale(
    weight_scale: torch.Tensor, logical_widths: list[int]
) -> torch.Tensor:
    """Per-shard scales -> ``[1]``, or ``[N, 1]`` if they differ."""
    if weight_scale.ndim == 0:
        return weight_scale.reshape(1)
    # weight_loader_v2 fills every slot from a fused-on-disk scale, so a slot left at the
    # float32-min sentinel means the checkpoint has no scale for that shard.
    unloaded = weight_scale <= torch.finfo(torch.float32).min
    if bool(unloaded.any()):
        missing = unloaded.nonzero().flatten().tolist()
        raise ValueError(f"FP8 checkpoint has no weight scale for shard(s) {missing}")
    if bool((weight_scale == weight_scale[0]).all()):
        return weight_scale[:1].clone()
    # Upstream annotates a tuple; it returns the [N, 1] tensor.
    return cast(torch.Tensor, convert_to_channelwise(weight_scale, logical_widths))


# vLLM picks weight_loader_v2 by class name; the v1 loader rejects per-shard scalar scales.
@register_weight_loader_v2_supported_method
class SpyreFp8LinearMethod(Fp8LinearMethod):
    """``Fp8LinearMethod`` whose per-tensor path skips the ``_C`` requant."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Pre-processed weights, Marlin and block scales never reach the requant.
        if is_weights_pre_processed() or self.use_marlin or self.block_quant:
            super().process_weights_after_loading(layer)
            return

        weight = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        logical_widths = cast(list[int], layer.logical_widths)
        scale = shard_scales_to_kernel_scale(weight_scale.data, logical_widths)
        if scale.numel() > 1:
            logger.info_once(
                "FP8 shard weight scales differ; using a per-channel scale instead of "
                "requantizing to the max."
            )
        # (N, K) -> (K, N), as upstream.
        replace_parameter(layer, "weight", weight.data.t())
        replace_parameter(layer, "weight_scale", scale)

        if self.act_q_static:
            input_scale = cast(torch.Tensor, layer.input_scale)
            replace_parameter(layer, "input_scale", input_scale.data.max())
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)


class SpyreFp8Config(Fp8Config):
    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase) and not is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            match_mode=self.ignored_layers_match_mode,
        ):
            return SpyreFp8LinearMethod(self)
        return super().get_quant_method(layer, prefix)


def register_spyre_fp8_config() -> None:
    register_quantization_config("fp8")(SpyreFp8Config)
    logger.info("Registered SpyreFp8Config for quantization='fp8' (no _C requant)")
