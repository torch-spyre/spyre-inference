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

"""Gemma 4 with its vision encoder compiled per token budget (math from hf-adapters#495)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma4 import modeling_gemma4
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.utils import scatter_output_slices

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    SEGMENT_ITEMS,
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
    one_hot_segments,
    segment_buffer,
    segment_mask,
    unbind_items,
)
from spyre_inference.models.pixtral import _linear_weight
from spyre_inference.multimodal.utils import (
    STICK,
    align_up,
    padded_sdpa,
    rope_perm_matrix,
    rope_rotate_matmul,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _host(t: torch.Tensor) -> torch.Tensor:
    """Detach a weight onto the host before reshaping or padding it.

    The padding helpers rebuild weights out of composed strided slice-assignments, which
    lower silently wrong on a Spyre-resident tensor -- finite but inaccurate, and nothing
    raises. Each helper pulls its source here, wherever it lives, and the caller moves the
    finished weight back.
    """
    return convert(t.detach(), device="cpu")


def _as_plain_linear(layer: nn.Module) -> nn.Linear:
    """Bounce a vLLM linear layer into a plain `nn.Linear`.

    vLLM's `recursive_replace_linear` makes `Gemma4ClippableLinear.linear` a
    `SpyreReplicatedLinear` whose weight is stored transposed (`[in, out]`); the padding
    helpers assume nn.Linear's `[out, in]`, and `F.linear` lowers fine for these
    projections.

    This discards the layer's `quant_method`, hence the guard: a quantized tower must
    fail rather than be silently dequantized.
    """
    if isinstance(layer, nn.Linear):
        return layer
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    quant_method = getattr(layer, "quant_method", None)
    if quant_method is not None and not isinstance(quant_method, UnquantizedLinearMethod):
        raise NotImplementedError(
            "Gemma 4 vision head-dim padding cannot preserve the quantization method "
            f"on {type(layer).__name__} ({type(quant_method).__name__}); it rebuilds "
            "the projection as a plain nn.Linear. Run the vision tower unquantized."
        )
    # [in, out] once vLLM's Spyre OOT method has run. cast: `nn.Module.weight` is
    # `Tensor | Module` generically; this layer's is always a real Tensor.
    weight_t = _host(cast(torch.Tensor, layer.weight))
    in_features, out_features = weight_t.shape
    plain = nn.Linear(in_features, out_features, bias=layer.bias is not None)
    plain.weight = nn.Parameter(weight_t.t().contiguous(), requires_grad=False)
    if layer.bias is not None:
        bias = _host(cast(torch.Tensor, layer.bias))
        plain.bias = nn.Parameter(bias.clone(), requires_grad=False)
    return plain


def _assert_clipping_preserves_zero(proj, name: str) -> None:
    """A clipped projection's output range must contain zero when we pad its heads.

    Padding relies on the padded lanes being exactly zero: `_padded_rms_norm` rescales
    the variance on that basis and the zero columns of `o_proj`/`down_proj` ignore them.
    A clamp range excluding zero would map those lanes to a nonzero bound, invalidating
    both while still returning plausible numbers.
    """
    if not getattr(proj, "use_clipped_linears", False):
        return
    lo = _host(proj.output_min)
    hi = _host(proj.output_max)
    if not bool(torch.all((lo <= 0) & (hi >= 0)).item()):
        raise NotImplementedError(
            f"{name}: output clipping must include zero when the head dim is padded, "
            f"or padded channels stop being zero before RMSNorm. Got "
            f"output_min={lo.tolist()}, output_max={hi.tolist()}."
        )


def _pad_qk_linear(proj, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> nn.Linear:
    """Pad and reorder two-axis RoPE channels into one matrix-RoPE layout."""
    linear = proj.linear
    weight = _host(linear.weight).view(num_heads, orig_head_dim, -1)
    new_weight = torch.zeros(num_heads, padded_head_dim, weight.shape[-1], dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    new_weight[:, :quarter] = weight[:, :quarter]
    new_weight[:, quarter : 2 * quarter] = weight[:, 2 * quarter : 3 * quarter]
    new_weight[:, padded_half : padded_half + quarter] = weight[:, quarter : 2 * quarter]
    new_weight[:, padded_half + quarter : padded_half + 2 * quarter] = weight[:, 3 * quarter :]
    padded = nn.Linear(
        linear.in_features, num_heads * padded_head_dim, bias=linear.bias is not None
    )
    padded.weight = nn.Parameter(
        new_weight.reshape(num_heads * padded_head_dim, -1), requires_grad=False
    )
    if linear.bias is not None:
        bias = _host(linear.bias).view(num_heads, orig_head_dim)
        new_bias = torch.zeros(num_heads, padded_head_dim, dtype=bias.dtype)
        new_bias[:, :quarter] = bias[:, :quarter]
        new_bias[:, quarter : 2 * quarter] = bias[:, 2 * quarter : 3 * quarter]
        new_bias[:, padded_half : padded_half + quarter] = bias[:, quarter : 2 * quarter]
        new_bias[:, padded_half + quarter : padded_half + 2 * quarter] = bias[:, 3 * quarter :]
        padded.bias = nn.Parameter(new_bias.reshape(-1), requires_grad=False)
    return padded


def _pad_proj_output_simple(
    proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int
) -> nn.Linear:
    """End-pad each head of a [n_heads*head_dim, hidden] output projection (V)."""
    w = _host(proj.weight)
    hidden = w.shape[1]
    new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
    new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        bias = _host(proj.bias)
        new_b = torch.zeros(n_heads * padded_head_dim, dtype=bias.dtype)
        for h in range(n_heads):
            s, d = h * orig_head_dim, h * padded_head_dim
            new_b[d : d + orig_head_dim] = bias[s : s + orig_head_dim]
        new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def _pad_proj_input_simple(
    proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int
) -> nn.Linear:
    """End-pad each head along the input dim of an O-style projection."""
    w = _host(proj.weight)
    hidden = w.shape[0]
    new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
    new_proj = nn.Linear(n_heads * padded_head_dim, hidden, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(_host(proj.bias).clone(), requires_grad=False)
    return new_proj


def _pad_norm_weight(norm, orig_head_dim: int, padded_head_dim: int) -> nn.Parameter:
    weight = _host(norm.weight)
    padded = torch.ones(padded_head_dim, dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    padded[:quarter] = weight[:quarter]
    padded[quarter : 2 * quarter] = weight[2 * quarter : 3 * quarter]
    padded[padded_half : padded_half + quarter] = weight[quarter : 2 * quarter]
    padded[padded_half + quarter : padded_half + 2 * quarter] = weight[3 * quarter :]
    return nn.Parameter(padded, requires_grad=False)


def _padded_rms_norm(
    hidden_states: torch.Tensor, weight, eps: float, orig_head_dim: int | None
) -> torch.Tensor:
    """RMSNorm with the denominator scaled back to ``orig_head_dim``, so the zero padding
    lanes do not deflate the variance. ``None`` means the input is not padded.

    Kept in the storage dtype rather than promoted to fp32 like the hf-adapters
    reference: an fp32 round trip changes the stick tiling of the result, and the eager
    elementwise op that follows then fails to lower at all (mixed-EA broadcast). Measured
    on device with the input held identical, this norm is exact to 2-byte rounding --
    cosine 1.00002 against an fp32 host reference, per-op relative error 5e-3 -- so the
    2-byte variance is not what costs this tower accuracy.
    """
    dtype = hidden_states.dtype
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    if orig_head_dim is not None and orig_head_dim != hidden_states.shape[-1]:
        variance = variance * (hidden_states.shape[-1] / orig_head_dim)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        # fp32 weights (transformers builds them at the default dtype), so an unguarded
        # multiply promotes the activation.
        hidden_states = hidden_states * weight.to(dtype)
    return hidden_states


def _fp32_inv_freq(rotary_emb, config) -> tuple[torch.Tensor, float]:
    """Rope frequencies in fp32 plus the initializer's attention scaling, recomputed rather
    than read off the module buffer.

    `model.to(...)` downcasts `inv_freq` to the model dtype, and these frequencies span
    1.0 down to ~1e-4 -- the angle is `position * inv_freq`, so the relative error a
    16-bit buffer carries is amplified by the position. Upcasting it back cannot recover
    the lost bits.

    The initializer is resolved off the rotary module before the global registry:
    transformers renames this tower's rope type from `default` to `axial` after 5.16.1, and
    the axial initializer is a class-local staticmethod absent from `ROPE_INIT_FUNCTIONS`.
    """
    params = getattr(config, "rope_parameters", None) or {}
    rope_type = params.get("rope_type", "default")
    init_fn = getattr(rotary_emb, f"compute_{rope_type}_rope_parameters", None)
    if init_fn is None:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attention_scaling = init_fn(config)
    return inv_freq.float(), float(attention_scaling)


def _padded_head_dim(orig_head_dim: int) -> int:
    """Pad to two sticks, not one: rope rotates each half independently, so both
    halves have to be stick-aligned."""
    return align_up(orig_head_dim, 2 * STICK)


def _pad_mlp(layer, orig_intermediate: int, padded_intermediate: int) -> None:
    """Zero-extend the MLP's intermediate width onto the stick, once per layer.

    Gate/up gain zero output rows and down matching zero input columns, so the padding
    cannot change the result.
    """
    if padded_intermediate == orig_intermediate:
        return
    if getattr(layer.mlp, "_spyre_padded_intermediate", None) == padded_intermediate:
        return
    mlp = layer.mlp
    for _name in ("gate_proj", "up_proj", "down_proj"):
        _assert_clipping_preserves_zero(getattr(mlp, _name), f"vision mlp {_name}")

    device = mlp.gate_proj.linear.weight.device
    for name in ("gate_proj", "up_proj"):
        proj = getattr(mlp, name)
        proj.linear = _pad_proj_output_simple(
            _as_plain_linear(proj.linear), 1, orig_intermediate, padded_intermediate
        ).to(device)
    mlp.down_proj.linear = _pad_proj_input_simple(
        _as_plain_linear(mlp.down_proj.linear), 1, orig_intermediate, padded_intermediate
    ).to(device)
    mlp._spyre_padded_intermediate = padded_intermediate


def _prepare_attention(attn, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> None:
    """Pad one `Gemma4VisionAttention`'s projections/norms to `padded_head_dim`, once."""
    if getattr(attn, "_spyre_padded_head_dim", None) == padded_head_dim:
        return
    if attn.v_norm.with_scale:
        raise NotImplementedError(
            "Scaled Gemma 4 vision V normalization is not supported on Spyre."
        )
    for _name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _assert_clipping_preserves_zero(getattr(attn, _name), f"vision attention {_name}")

    # The helpers pad on the host (see `_host`), so every result is moved back.
    device = attn.q_norm.weight.device
    attn.q_proj.linear = _as_plain_linear(attn.q_proj.linear)
    attn.k_proj.linear = _as_plain_linear(attn.k_proj.linear)
    attn.v_proj.linear = _as_plain_linear(attn.v_proj.linear)
    attn.o_proj.linear = _as_plain_linear(attn.o_proj.linear)
    attn.q_proj.linear = _pad_qk_linear(attn.q_proj, num_heads, orig_head_dim, padded_head_dim).to(
        device
    )
    attn.k_proj.linear = _pad_qk_linear(attn.k_proj, num_heads, orig_head_dim, padded_head_dim).to(
        device
    )
    attn.v_proj.linear = _pad_proj_output_simple(
        attn.v_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.o_proj.linear = _pad_proj_input_simple(
        attn.o_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.q_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.q_norm, orig_head_dim, padded_head_dim).to(device),
        requires_grad=False,
    )
    attn.k_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.k_norm, orig_head_dim, padded_head_dim).to(device),
        requires_grad=False,
    )
    attn._spyre_padded_head_dim = padded_head_dim


_NOT_PADDED = (
    "Gemma 4 vision weights are not padded for this head dim; "
    "`process_weights_after_loading` pads them and has to precede the first forward."
)


class SpyreGemma4VisionAttention(modeling_gemma4.Gemma4VisionAttention):
    """Padded rope + `padded_sdpa` over the head dim `_prepare_attention` padded to."""

    spyre_rope_swap: torch.Tensor

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        weight = self.q_proj.linear.weight
        self.register_buffer(
            "spyre_rope_swap",
            rope_perm_matrix("half", _padded_head_dim(self.head_dim), weight.device, weight.dtype),
            persistent=False,
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del position_ids, kwargs
        cos, sin = position_embeddings
        bsz, seq_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        orig_head_dim = self.head_dim
        padded_head_dim = _padded_head_dim(orig_head_dim)
        if getattr(self, "_spyre_padded_head_dim", None) != padded_head_dim:
            raise RuntimeError(_NOT_PADDED)

        # Rope at [B, L, H, D], then transpose to [B, H, L, D] for SDPA.
        q = self.q_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        q = _padded_rms_norm(q, self.q_norm.weight, self.q_norm.eps, orig_head_dim)
        q = rope_rotate_matmul(q, cos, sin, self.spyre_rope_swap).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        k = _padded_rms_norm(k, self.k_norm.weight, self.k_norm.eps, orig_head_dim)
        k = rope_rotate_matmul(k, cos, sin, self.spyre_rope_swap).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
        v = _padded_rms_norm(v, None, self.v_norm.eps, orig_head_dim).transpose(1, 2)

        attn_out = padded_sdpa(q, k, v, attention_mask, scale=float(self.scaling))
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_out), None


class SpyreGemma4RMSNorm(modeling_gemma4.Gemma4RMSNorm):
    """``Gemma4RMSNorm`` without fp32 promotion, and with ``rsqrt`` instead of
    ``pow(x, -0.5)``, which has no lowering here. (Unlike ``SpyreGemmaRMSNorm``, which keeps
    upstream's fp32 path by compiling it; see ``_padded_rms_norm`` for why this tower cannot.)

    ``forward`` is the override point, not ``_norm``: stock casts to fp32 in both, and both
    casts have to go.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.with_scale else None
        return _padded_rms_norm(hidden_states, weight, self.eps, orig_head_dim=None)


_LAYER_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)


def gemma4_rope_tables(
    inv_freq: torch.Tensor, size: int, padded_head_dim: int, attention_scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[2, size, D]` cos and signed-sin per axis; patch (x, y) sums rows [0, x], [1, y].

    Gemma 4 vision rotates its two position axes independently, each with its own
    `rotate_half` (stock `apply_multidimensional_rope`). `_pad_qk_linear` packs the padded
    head as `[X, Y, zeros | X, Y, zeros]`, so one first-half/second-half swap serves both
    axes at once: each axis's angle is duplicated into both blocks, with `rotate_half`'s
    sign flip baked into `sin`.

    `attention_scaling` (stock applies it to the whole table) scales the real lanes only;
    the padded ones stay an exact identity rotation.
    """
    quarter = inv_freq.shape[0]
    half = padded_head_dim // 2
    angles = torch.arange(size, dtype=torch.float32)[:, None] * inv_freq.float()
    cos_axis = angles.cos() * attention_scaling
    sin_axis = angles.sin() * attention_scaling
    cos = torch.zeros(2, size, padded_head_dim)
    sin = torch.zeros(2, size, padded_head_dim)
    for axis in range(2):
        lanes = slice(axis * quarter, (axis + 1) * quarter)
        upper = slice(half + axis * quarter, half + (axis + 1) * quarter)
        cos[axis, :, lanes] = cos_axis
        cos[axis, :, upper] = cos_axis
        sin[axis, :, lanes] = -sin_axis
        sin[axis, :, upper] = sin_axis
    cos[0, :, 2 * quarter : half] = 1.0
    cos[0, :, half + 2 * quarter :] = 1.0
    return cos, sin


# Not a module, so the model's move to the card leaves it on the host.
@dataclass
class _VisionTail:
    scale: float
    std_bias: torch.Tensor | None
    std_scale: torch.Tensor | None
    eps: float
    weight: torch.Tensor  # embed_vision's projection, [out, in]
    dtype: torch.dtype


def _images(mm_kwargs: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pixel_values = mm_kwargs.get("pixel_values")
    if pixel_values is None:
        return []
    positions = mm_kwargs["pixel_position_ids"]
    return list(zip(unbind_items(pixel_values), unbind_items(positions), strict=True))


def _valid_patches(
    pixel_values: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = convert(pixel_values, device="cpu")
    positions = convert(positions, device="cpu")
    valid = (positions != -1).any(dim=-1)
    return pixel_values[valid], positions[valid]


class SpyreGemma4ForConditionalGeneration(SpyreVisionEncoderMixin, Gemma4ForConditionalGeneration):
    vision_max_items_per_call = SEGMENT_ITEMS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Transformers builds the tower at vision_config.dtype (bf16), and bf16 + fp16 in a graph
        # promotes to an fp32 convert Spyre cannot lay out; loading converts the checkpoint.
        model_config = vllm_config.model_config
        model_config.hf_config.vision_config.dtype = model_config.dtype
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision()

    def _init_spyre_vision(self) -> None:
        vision_config = self.config.vision_config
        self._pool = vision_config.pooling_kernel_size
        self.patches_per_token = self._pool**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._patch_width = align_up(3 * vision_config.patch_size**2)
        self.vision_buffers = {
            "patches": VisionBuffer(self._patch_width),
            "segments": segment_buffer(),
            "pos_x": VisionBuffer(None, pad=0, dtype=torch.int64),
            "pos_y": VisionBuffer(None, pad=0, dtype=torch.int64),
        }

        # With image and video limits of 0, upstream leaves a placeholder instead of the tower.
        if not isinstance(self.vision_tower, modeling_gemma4.Gemma4VisionModel):
            return
        if vision_config.num_key_value_heads != vision_config.num_attention_heads:
            raise NotImplementedError(
                "Gemma 4 vision GQA is not supported on Spyre; num_key_value_heads "
                "must equal num_attention_heads."
            )
        for layer in self.vision_tower.encoder.layers:
            stock = layer.self_attn
            attn = SpyreGemma4VisionAttention(vision_config, stock.layer_idx)
            # Upstream has already replaced these with vLLM linears; keep them.
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, getattr(stock, name))
            layer.self_attn = attn
            for name in _LAYER_NORMS:
                norm = SpyreGemma4RMSNorm(vision_config.hidden_size, eps=vision_config.rms_norm_eps)
                setattr(layer, name, norm)

    def process_weights_after_loading(self) -> None:
        """Rebuild the vision weights in place, in the graphs' layouts, on the host."""
        tower = self.vision_tower
        if not isinstance(tower, modeling_gemma4.Gemma4VisionModel):
            return
        config = tower.config
        padded = _padded_head_dim(config.head_dim)
        for layer in tower.encoder.layers[: config.num_hidden_layers]:
            _prepare_attention(layer.self_attn, config.num_attention_heads, config.head_dim, padded)
            _pad_mlp(layer, config.intermediate_size, align_up(config.intermediate_size))

        embedder = tower.patch_embedder
        weight = _linear_weight(embedder.input_proj).float()
        # Not the tower's dtype: see __init__.
        dtype = cast(torch.dtype, self.model_dtype)
        weight_t = F.pad(weight, (0, self._patch_width - weight.shape[1])).t().contiguous()
        # A plain holder: nothing runs the linear, so no second copy of it goes to the card.
        embedder.input_proj = nn.Module()
        embedder.input_proj.weight = nn.Parameter(weight_t.to(dtype), requires_grad=False)

        # One row per coordinate and axis: [position embedding | cos | sin].
        table = _host(cast(torch.Tensor, embedder.position_embedding_table)).float()
        inv_freq, attention_scaling = _fp32_inv_freq(tower.encoder.rotary_emb, config)
        cos, sin = gemma4_rope_tables(inv_freq, table.shape[1], padded, attention_scaling)
        rows = torch.cat([table, cos, sin], dim=-1).to(dtype)
        self.spyre_vision_x = GatherTable(rows[0].contiguous())
        self.spyre_vision_y = GatherTable(rows[1].contiguous())
        del embedder.position_embedding_table

        standardize = getattr(config, "standardize", False)
        projection = self.embed_vision.embedding_projection
        self._spyre_vision_tail = _VisionTail(
            # vision_project sums the k*k patches; the pooler averages and scales them.
            scale=tower.pooler.root_hidden_size / self.patches_per_token,
            std_bias=_host(tower.std_bias).float() if standardize else None,
            std_scale=_host(tower.std_scale).float() if standardize else None,
            eps=self.embed_vision.eps,
            weight=_host(_linear_weight(projection)).float(),
            dtype=dtype,
        )
        # Only upstream's eager image and video paths ran it; the tail above replaces it.
        self.embed_vision = None

    def embed_multimodal(self, **kwargs: object) -> Any:
        if "pixel_values_videos" in kwargs:
            raise NotImplementedError(
                "Gemma 4 video input is not supported on Spyre; send the frames as images."
            )
        return super().embed_multimodal(**kwargs)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """The tail in fp32 on the host, as upstream: its scale can overflow fp16."""
        num_tokens = sum(per_item_out_tokens[i] for i in indices)
        tail = self._spyre_vision_tail
        x = outputs["default"][:num_tokens].float() * tail.scale
        if tail.std_bias is not None and tail.std_scale is not None:
            x = (x - tail.std_bias) * tail.std_scale
        # embed_vision: a weightless RMSNorm, then the projection.
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tail.eps)
        tokens = (x @ tail.weight.t()).to(tail.dtype)
        scatter_output_slices(tokens, indices, per_item_out_tokens, dest, clone)

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return [
            int((pos != -1).any(dim=-1).sum()) // self.patches_per_token
            for _, pos in _images(mm_kwargs)
        ]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Valid patches laid out offset-major, so the pool is a sum over offset groups."""
        k = self._pool
        patches, segments, xs, ys = [], [], [], []
        for i, (pixel_values, positions) in enumerate(_images(mm_kwargs)):
            values, pos = _valid_patches(pixel_values, positions)
            wp, hp = (int(n) + 1 for n in pos.max(dim=0).values)
            if hp * wp != pos.shape[0] or hp % k or wp % k:
                raise ValueError(
                    f"Gemma 4 image {i}: {pos.shape[0]} valid patches do not fill a "
                    f"{hp}x{wp} grid divisible by the {k}x{k} pooling kernel"
                )
            grid = torch.empty(hp, wp, values.shape[-1], dtype=values.dtype)
            grid[pos[:, 1], pos[:, 0]] = values
            patches.append(offset_major(grid, k))
            xs.append(offset_major(torch.arange(wp).expand(hp, wp), k))
            ys.append(offset_major(torch.arange(hp)[:, None].expand(hp, wp), k))
            segments.append(torch.full((k * k, hp * wp // (k * k)), i))
        # Stock's patch embedder starts with 2 * (x - 0.5).
        flat = (torch.cat(patches, dim=1).flatten(0, 1).float() * 2 - 1).to(torch.float16)
        return {
            "patches": F.pad(flat, (0, self._patch_width - flat.shape[1])),
            "segments": one_hot_segments(torch.cat(segments, dim=1).flatten()),
            "pos_x": torch.cat(xs, dim=1).flatten().to(torch.int64),
            "pos_y": torch.cat(ys, dim=1).flatten().to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_tower
        hidden = tower.config.hidden_size
        head_dim = _padded_head_dim(tower.config.head_dim)
        rows = self.spyre_vision_x.gather(values["pos_x"]) + self.spyre_vision_y.gather(
            values["pos_y"]
        )
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, tower.patch_embedder.input_proj.weight)
        x = (torch.matmul(values["patches"], patch_weight_t) + rows[:, :hidden]).unsqueeze(0)
        cos = rows[:, hidden : hidden + head_dim][None, :, None]
        sin = rows[:, hidden + head_dim :][None, :, None]
        mask = segment_mask(values["segments"], x.dtype)
        return x, {"position_embeddings": (cos, sin), "attention_mask": mask}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_tower.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """The pool only: the tail runs in ``postprocess_encoder_output``."""
        x = x.squeeze(0)
        return x.reshape(self.patches_per_token, -1, x.shape[-1]).sum(dim=0)
