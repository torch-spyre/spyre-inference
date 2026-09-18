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

"""Granite 4 Vision workarounds for Spyre."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)


def patch_interpolate_downsampler() -> None:
    """Run InterpolateDownsampler on CPU.

    InterpolateDownsampler uses F.interpolate(mode="area") which lowers to
    aten::_adaptive_avg_pool2d — not supported on Spyre.
    The permute/view/mean involves non-contiguous strides that copy_from_d2d
    cannot restickify on-device, so run on CPU.
    """
    try:
        from vllm.model_executor.models.granite4_vision import InterpolateDownsampler
    except ImportError:
        return

    if getattr(InterpolateDownsampler.__call__, "_spyre_patched", False):
        return

    def _interpolate_downsampler_call(
        self: InterpolateDownsampler,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        dev = image_features.device
        image_features_cpu = convert(image_features, device="cpu")
        batch_size, _, dim = image_features_cpu.size()
        up_shape = [batch_size, self.orig_image_side, self.orig_image_side, dim]
        large = image_features_cpu.view(up_shape).permute(0, 3, 1, 2)
        small = torch.nn.functional.adaptive_avg_pool2d(
            large, (self.new_image_side, self.new_image_side)
        )
        out_cpu = small.permute(0, 2, 3, 1).flatten(1, 2)
        return convert(out_cpu, device=dev)

    _interpolate_downsampler_call._spyre_patched = True  # type: ignore[attr-defined]
    InterpolateDownsampler.__call__ = _interpolate_downsampler_call  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched InterpolateDownsampler to run on CPU"
        " (permute/mean not restickifiable on Spyre)."
    )


def patch_pack_and_unpad_image_features() -> None:
    """Run Granite4VisionForConditionalGeneration._pack_and_unpad_image_features on CPU.

    permute(4,0,2,1,3) on a 5-D tensor produces a stick expression (e.g. 12*d1+d2)
    that Spyre's work_division pass cannot lower. No parameters touched — run on CPU.
    """
    try:
        from vllm.model_executor.models.granite4_vision import (
            Granite4VisionForConditionalGeneration,
        )
    except ImportError:
        return

    if getattr(
        Granite4VisionForConditionalGeneration._pack_and_unpad_image_features,
        "_spyre_patched",
        False,
    ):
        return

    _orig_pack_and_unpad = Granite4VisionForConditionalGeneration._pack_and_unpad_image_features

    def _pack_and_unpad_cpu(self, image_features, image_sizes):
        dev = image_features[0].device if image_features else None
        image_features_cpu = [convert(f, device="cpu") for f in image_features]
        image_sizes_cpu = convert(image_sizes, device="cpu")
        result_cpu = _orig_pack_and_unpad(self, image_features_cpu, image_sizes_cpu)
        if dev is not None and dev.type != "cpu":
            result_cpu = [convert(f, device=dev) for f in result_cpu]
        return result_cpu

    _pack_and_unpad_cpu._spyre_patched = True  # type: ignore[attr-defined]
    Granite4VisionForConditionalGeneration._pack_and_unpad_image_features = _pack_and_unpad_cpu  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched Granite4VisionForConditionalGeneration._pack_and_unpad_image_features "
        "to run on CPU (5-D permute not lowerable on Spyre)."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply Granite 4 Vision workarounds."""
    patch_interpolate_downsampler()
    patch_pack_and_unpad_image_features()
