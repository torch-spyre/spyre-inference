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
        small = torch.nn.functional.interpolate(
            large,
            size=(self.new_image_side, self.new_image_side),
            mode=self.mode,
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


def patch_embed_input_ids() -> None:
    """Replace the two boolean-mask index_puts in embed_input_ids; keep the lookup on card.

    Upstream's Granite4VisionForConditionalGeneration.embed_input_ids has two
    ``aten::_index_put_impl_`` calls that Spyre cannot execute:

    1. ``text_embeds[is_multimodal] = 0.0``   (zero image-token positions)
    2. ``target[is_multimodal] = level_features[level_idx]``  (fill deepstack buffers)

    Fix for (1): replace with ``torch.where(mask, zeros, text_embeds)`` — a
    broadcast select that stays on Spyre.

    Fix for (2): scatter on CPU into a staging tensor the same size as the
    buffer slice, then copy back with ``target.copy_(staged)``.  The persistent
    ``_ds_buffers`` remain full-size (``[max_tokens, lm_hidden]``) so forward's
    ``self._ds_buffers[lvl][:n]`` slices are always valid.

    ``all_packed.split(lm_h, dim=-1)`` produces last-dim views of the device
    tensor; fetching each selected slice to CPU is cheap (image tokens only).
    """
    try:
        from vllm.model_executor.models.granite4_vision import (
            Granite4VisionForConditionalGeneration,
        )
    except ImportError:
        return

    if getattr(
        Granite4VisionForConditionalGeneration.embed_input_ids,
        "_spyre_patched",
        False,
    ):
        return

    def _embed_input_ids_spyre(
        self: Granite4VisionForConditionalGeneration,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        lm_inner = self.language_model.model

        has_vision = (
            multimodal_embeddings is not None
            and is_multimodal is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal.any()
        )

        # 1. Text embeddings (on Spyre)
        text_embeds = lm_inner.embed_input_ids(input_ids)
        dev = text_embeds.device

        # Ensure persistent buffers are on the right device/dtype (first call or
        # after model.to()).  This must run on both the vision and text-only paths:
        # forward() passes _ds_buffers[:n] directly to IntermediateTensors regardless
        # of whether there are images, so CPU buffers cause a device-mismatch crash
        # even on pure-text decode steps.
        buf0 = self._ds_buffers[0]
        if buf0.device != dev or buf0.dtype != text_embeds.dtype:
            self._ds_buffers = [b.to(device=dev, dtype=text_embeds.dtype) for b in self._ds_buffers]

        if not has_vision:
            self._ds_num_tokens = 0
            return text_embeds * lm_inner.config.embedding_multiplier

        # 2. Zero image positions via torch.where (no index_put on Spyre).
        #    mask shape: [N] → unsqueeze to [N, 1] for broadcast over hidden dim.
        mask = convert(is_multimodal, device=dev).unsqueeze(-1)
        text_embeds = torch.where(mask, torch.zeros_like(text_embeds), text_embeds)

        # 3. Apply embedding_multiplier
        inputs_embeds = text_embeds * lm_inner.config.embedding_multiplier

        # 4. Split packed tensors → per-level features; fill _ds_buffers on CPU.
        N, lm_h = inputs_embeds.shape
        assert multimodal_embeddings is not None
        # Move packed mm features to CPU for the scatter (image tokens only — cheap).
        all_packed_cpu = torch.cat(
            [convert(t, dtype=inputs_embeds.dtype, device="cpu") for t in multimodal_embeddings],
            dim=0,
        )
        level_features_cpu = all_packed_cpu.split(lm_h, dim=-1)  # num_levels tensors on CPU

        is_multimodal_cpu = convert(is_multimodal, device="cpu")
        buf_len = self._ds_buffers[0].shape[0]
        for level_idx in range(len(self._ds_layer_indices)):
            # Stage the full buffer size on CPU (same shape as the on-device
            # allocation).  Spyre's DMA validates against the physical allocation
            # size, not the Python slice — copying a sub-slice [:N] raises
            # "Invalid dma sizes".  Staging buf_len rows and copying the whole
            # buffer avoids the mismatch; rows beyond N stay zero and are harmless.
            staged = torch.zeros(buf_len, lm_h, dtype=inputs_embeds.dtype)
            staged[:N][is_multimodal_cpu] = level_features_cpu[level_idx]
            self._ds_buffers[level_idx].copy_(staged)

        self._ds_num_tokens = N
        return inputs_embeds

    _embed_input_ids_spyre._spyre_patched = True  # type: ignore[attr-defined]
    Granite4VisionForConditionalGeneration.embed_input_ids = _embed_input_ids_spyre  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched Granite4VisionForConditionalGeneration.embed_input_ids"
        " (boolean-mask index_put replaced with torch.where / CPU scatter)."
    )


def migrate_ds_buffers(model: torch.nn.Module, device: torch.device) -> None:
    """Move Granite4Vision _ds_buffers to the target device and model dtype.

    _ds_buffers are plain tensors (not nn.Parameter, not register_buffer), so
    model.to() does not touch them.  They must be on the same device as
    hidden_states before the first forward call — including the warmup pass,
    which calls forward() directly without going through embed_input_ids.
    """
    ds_buffers = getattr(model, "_ds_buffers", None)
    if ds_buffers is None:
        return
    # Infer dtype from the first parameter (embedding table is always present).
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float16
    model._ds_buffers = [b.to(device=device, dtype=dtype) for b in ds_buffers]
    logger.info("Spyre: moved Granite4Vision _ds_buffers to %s (%s).", device, dtype)


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply Granite 4 Vision workarounds."""
    patch_interpolate_downsampler()
    patch_pack_and_unpad_image_features()
    patch_embed_input_ids()
    migrate_ds_buffers(model, device)
