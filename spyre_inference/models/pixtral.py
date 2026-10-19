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

"""Native Mistral-format Pixtral (e.g. Ministral-3) with a compiled, bucketed vision encoder.

``SpyrePixtralForConditionalGeneration`` keeps upstream's processor, weight loading and
decoder, and adds the ``SpyreVisionEncoderMixin`` hooks:

- blocks: ``SpyrePixtralTransformerBlock``, whose attention does RoPE as a matmul and the
  stick-aligned SDPA;
- inputs: patches laid out offset-major on the host, with each patch's image and position;
- RoPE: cos and sin gathered on the card by position, from a ``GatherTable``;
- patch embedding: the conv folded into a linear at load time;
- patch merge: a batched matmul over the offset groups, in place of upstream's ``unfold``.

The encoder is permutation-equivariant while RoPE and the image mask follow the patches, so
the offset-major order changes how the merge lowers, not the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.models import pixtral
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import is_vit_use_data_parallel

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
)
from spyre_inference.multimodal.utils import (
    STICK,
    align_up,
    padded_sdpa,
    rope_perm_matrix,
    rope_rotate_matmul,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2


class SpyrePixtralAttention(pixtral.Attention):
    """Upstream's vision attention with RoPE as a matmul and the stick-aligned SDPA.

    ``freqs_cis`` is a ``(cos, sin)`` pair of real tables instead of upstream's complex one,
    since Spyre has no complex dtype.
    """

    rope_perm: torch.Tensor

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "rope_perm",
            rope_perm_matrix("pair", self.head_dim, torch.device("cpu")),
            persistent=False,
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = freqs_cis
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = (
            t.reshape(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        q = rope_rotate_matmul(q, cos, sin, self.rope_perm)
        k = rope_rotate_matmul(k, cos, sin, self.rope_perm)
        out = padded_sdpa(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


class SpyrePixtralTransformerBlock(pixtral.TransformerBlock):
    """Upstream's block with ``SpyrePixtralAttention``, called as ``(x, mask, freqs_cis)``.

    Its own forward, since upstream's passes packed sequence lengths where these blocks take
    the same-image mask built once per encoder call in ``vision_embed``.
    """

    def __init__(self, args, prefix: str = "", disable_tp: bool = False) -> None:
        super().__init__(args, prefix=prefix, disable_tp=disable_tp)
        self.attention = SpyrePixtralAttention(
            args, prefix=f"{prefix}.attention", disable_tp=disable_tp
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


def pixtral_rope_tables(
    head_dim: int, max_side: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[max_side**2, head_dim]`` cos and sin tables, indexed by ``row * max_side + col``.

    The sin table carries the pair rotation's sign, so ``rope_rotate_matmul`` reproduces
    upstream's complex multiply.
    """
    fc = pixtral.precompute_freqs_cis_2d(dim=head_dim, height=max_side, width=max_side, theta=theta)
    cos = fc.real.repeat_interleave(2, dim=-1)
    sin = torch.stack([-fc.imag, fc.imag], dim=-1).flatten(-2)
    return cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)


def offset_major_merge_weight(weight: torch.Tensor, merge: int) -> torch.Tensor:
    """The merging layer's ``[D_out, D * merge**2]`` weight (columns in ``unfold``'s
    ``(c, ki, kj)`` order) as ``[merge**2, D, D_out]``, one slice per offset group."""
    out_dim, in_dim = weight.shape
    return weight.reshape(out_dim, in_dim // merge**2, merge**2).permute(2, 1, 0).contiguous()


def _linear_weight(layer: nn.Module) -> torch.Tensor:
    """``[out, in]``, undoing the Spyre linear's post-load transpose and row padding."""
    weight = cast(torch.Tensor, layer.weight).detach()
    padding = getattr(layer, "spyre_row_padding", None)
    if padding is None:
        return weight
    weight = weight.t()
    return weight[: weight.shape[0] - padding]


def _images(mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """The ``[C, H, W]`` images in ``mm_kwargs``, whether batched in one tensor or a list."""
    images = mm_kwargs.get("images")
    if images is None:
        return []
    return list(images.unbind(0)) if isinstance(images, torch.Tensor) else list(images)


class SpyrePixtralForConditionalGeneration(
    SpyreVisionEncoderMixin, pixtral.PixtralForConditionalGeneration
):
    """Upstream Pixtral with its vision encoder compiled per token budget."""

    # The one-hot image matrix is 64 wide, and its last column belongs to the padding.
    vision_max_items_per_call = STICK - 1

    # --- construction ---

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision(prefix)

    def _init_spyre_vision(self, prefix: str) -> None:
        """Set the mixin's attributes and swap in the Spyre blocks. Separate from ``__init__``
        so tests can run it on upstream modules without a full ``VllmConfig``."""
        args = self.vision_args
        self._merge = args.spatial_merge_size if self.patch_merger is not None else 1
        self.patches_per_token = self._merge**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._max_side = args.image_size // args.patch_size
        head_dim = args.hidden_size // args.num_attention_heads
        self._patch_width = align_up(args.num_channels * args.patch_size**2)
        self.vision_buffers = {
            "patches": VisionBuffer(self._patch_width),
            "segments": VisionBuffer(STICK, pad=F.one_hot(torch.tensor(STICK - 1), STICK).half()),
            "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
        }
        # [cos | sin] per position, gathered on the card in vision_embed.
        cos, sin = pixtral_rope_tables(head_dim, self._max_side, args.rope_theta)
        self.spyre_rope_table = GatherTable(torch.cat([cos, sin], dim=-1).half())
        self._head_dim = head_dim

        # With an image limit of 0, upstream leaves a placeholder instead of the tower.
        if not isinstance(self.vision_encoder, pixtral.VisionTransformer):
            return
        layers = self.vision_encoder.transformer.layers
        for i in range(len(layers)):
            layers[i] = SpyrePixtralTransformerBlock(
                args,
                prefix=maybe_prefix(prefix, f"vision_encoder.transformer.layers.{i}"),
                disable_tp=is_vit_use_data_parallel(),
            )

    def process_weights_after_loading(self) -> None:
        """Reshape the patch-conv and merging weights for the vision graphs, in place, as the
        Spyre linears do, so one copy of each reaches the card. Runs on the host, before the
        runner moves the model to the card."""
        if not isinstance(self.vision_encoder, pixtral.VisionTransformer):
            return
        tower = self.vision_encoder
        weight = tower.patch_conv.weight.detach().flatten(1)
        weight_t = F.pad(weight, (0, self._patch_width - weight.shape[1])).t().contiguous()
        # A plain holder, not the conv layer: nothing runs the conv, so the runner must not
        # place a tiled conv copy of the weight on the card as well.
        tower.patch_conv = nn.Module()
        tower.patch_conv.weight = nn.Parameter(weight_t, requires_grad=False)
        if self.patch_merger is not None:
            layer = self.patch_merger.merging_layer
            merge_weight = offset_major_merge_weight(_linear_weight(layer), self._merge)
            layer.weight = nn.Parameter(merge_weight, requires_grad=False)

    # --- protocol methods upstream Pixtral does not provide ---

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        images = _images(mm_kwargs)
        return {**mm_kwargs, "images": [images[i] for i in indices]}

    # --- SpyreVisionEncoderMixin hooks ---

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        p = self.vision_args.patch_size
        return [
            (img.shape[-2] // p) * (img.shape[-1] // p) // self.patches_per_token
            for img in _images(mm_kwargs)
        ]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Offset-major patches of every image, each flattened as ``(c, py, px)`` like the conv
        weight, with its one-hot image and its RoPE position."""
        images = _images(mm_kwargs)
        p, m = self.vision_args.patch_size, self._merge
        flat_patches, seg, pos = [], [], []
        for i, img in enumerate(images):
            c, h, w = img.shape
            assert h % (p * m) == 0 and w % (p * m) == 0, f"{h}x{w} is not a multiple of {p * m}"
            hp, wp = h // p, w // p
            patches = convert(img, device="cpu").reshape(c, hp, p, wp, p).permute(1, 3, 0, 2, 4)
            flat_patches.append(offset_major(patches.reshape(hp, wp, -1), m))
            grid = torch.arange(hp)[:, None] * self._max_side + torch.arange(wp)[None, :]
            pos.append(offset_major(grid, m))
            seg.append(torch.full((m * m, (hp // m) * (wp // m)), i))
        flat = torch.cat(flat_patches, dim=1).flatten(0, 1).to(torch.float16)
        return {
            "patches": F.pad(flat, (0, self._patch_width - flat.shape[1])),
            "segments": F.one_hot(torch.cat(seg, dim=1).flatten(), STICK).to(torch.float16),
            "positions": torch.cat(pos, dim=1).flatten().to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_encoder
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, tower.patch_conv.weight)
        x = tower.ln_pre(torch.matmul(values["patches"], patch_weight_t).unsqueeze(0))
        # Same-image mask as a matmul: Spyre cannot lay out a column-minus-row broadcast.
        segments = values["segments"].to(x.dtype)
        mask = ((1.0 - torch.matmul(segments, segments.t())) * _MASK_MIN)[None, None]
        rope = self.spyre_rope_table.gather(values["positions"])
        freqs_cis = (rope[:, : self._head_dim], rope[:, self._head_dim :])
        return x, {"mask": mask, "freqs_cis": freqs_cis}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_encoder.transformer.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        x = x.squeeze(0)
        if self.pre_mm_projector_norm is not None:
            x = self.pre_mm_projector_norm(x)
        if self.patch_merger is not None:
            # One [tokens, D] group per merge offset, each with its slice of the weight.
            x = x.reshape(self.patches_per_token, -1, x.shape[-1])
            # [merge**2, D, D_out] since process_weights_after_loading.
            merge_weight = cast(torch.Tensor, self.patch_merger.merging_layer.weight)
            x = torch.matmul(x, merge_weight).sum(dim=0)
        return self.vision_language_adapter(x)
