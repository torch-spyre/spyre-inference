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

"""Gemma 4 vision-tower workarounds for Spyre.

vLLM loads the tower with plain ``AutoModel.from_config``, so it is stock
transformers code outside vLLM's layer registries -- same category as
``multimodal/pixtral.py``. Every fix is a guarded, idempotent monkeypatch and
``apply()`` is the only entry point. The tensor math follows hf-adapters#495.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.utils import STICK, align_up, padded_sdpa

logger = init_logger(__name__)


def _host(t: torch.Tensor) -> torch.Tensor:
    """Detach a weight onto the host before reshaping or padding it.

    The padding helpers rebuild weights out of composed strided slice-assignments, which
    lower silently wrong on a Spyre-resident tensor -- finite but inaccurate, and nothing
    raises. vLLM moves the model before our patches run, so each helper pulls its source
    here and the caller moves the finished weight back.
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


def _gemma4_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    padded_head_dim: int,
    dtype: torch.dtype,
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[bsz, seq, 1, padded_head_dim]` cos and signed-sin tables for `_apply_rope`.

    Gemma 4 vision rotates its two position axes independently, each with its own
    `rotate_half` (stock `apply_multidimensional_rope`). `_pad_qk_linear` packs the padded
    head as `[X, Y, zeros | X, Y, zeros]`, so one first-half/second-half swap serves both
    axes at once: each axis's angle is duplicated into both blocks, with `rotate_half`'s
    sign flip baked into `sin`.

    `attention_scaling` (stock applies it to the whole table) scales the real lanes only;
    the padded ones stay an exact identity rotation.
    """
    positions = position_ids.to("cpu").clamp(min=0).float()
    angles = positions[..., None] * inv_freq.to("cpu").float()  # [bsz, seq, 2, quarter]
    cos_axis = angles.cos() * attention_scaling
    sin_axis = angles.sin() * attention_scaling
    bsz, seq_len, _, quarter = cos_axis.shape
    half = padded_head_dim // 2
    cos_half = torch.ones(bsz, seq_len, half)
    sin_half_neg = torch.zeros(bsz, seq_len, half)
    sin_half_pos = torch.zeros(bsz, seq_len, half)
    for axis in range(2):
        start, end = axis * quarter, axis * quarter + quarter
        cos_half[..., start:end] = cos_axis[:, :, axis, :]
        sin_half_neg[..., start:end] = -sin_axis[:, :, axis, :]
        sin_half_pos[..., start:end] = sin_axis[:, :, axis, :]
    cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(2)  # [bsz, seq, 1, D]
    sin_full = torch.cat([sin_half_neg, sin_half_pos], dim=-1).unsqueeze(2)
    return cos_full.to(dtype), sin_full.to(dtype)


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


# Permutation matrices for `_apply_rope`, keyed by (dim, dtype, device).
_ROPE_SWAP: dict[tuple, torch.Tensor] = {}


def _rope_swap_matrix(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dim, dtype, str(device))
    swap = _ROPE_SWAP.get(key)
    if swap is None:
        half = dim // 2
        rows = torch.cat([torch.arange(half, dim), torch.arange(0, half)])
        swap = torch.zeros(dim, dim, dtype=dtype)
        swap[rows, torch.arange(dim)] = 1.0
        if device.type != "cpu":
            swap = convert(swap, device=device, dtype=dtype)
        _ROPE_SWAP[key] = swap
    return swap


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x*cos + rotate_half(x)*sin`, with the half-swap done as a permutation matmul.

    `torch.cat([x[..., half:], x[..., :half]], -1)` is silently wrong on device whenever
    `x` came from a matmul, as q/k always have: both slices read back correctly, the `cat`
    returns uncorrelated data, and nothing warns. It cost this tower its sight -- final
    vision-embedding cosine 0.12 against a host fp32 reference, versus 0.997 through the
    permutation. Slice-free rewrites (`stack`/`flip`, `index_select`, integer-index
    gather) are either wrong the same way or fail to lower, so the swap goes through a
    fixed permutation matrix.

    `sin` already carries `rotate_half`'s sign flip, so the permutation is a plain swap.
    """
    swapped = x @ _rope_swap_matrix(x.shape[-1], x.dtype, x.device)
    return x * cos + swapped * sin


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


def _run_attention(
    attn,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor,
    num_heads: int,
    orig_head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape

    # Rope at [B, L, H, D], then transpose to [B, H, L, D] for SDPA -- Pixtral's own
    # order (`multimodal/pixtral.py::patch_vision_attention`).
    q = attn.q_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    q = _padded_rms_norm(q, attn.q_norm.weight, attn.q_norm.eps, orig_head_dim)
    q = _apply_rope(q, cos, sin).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    k = _padded_rms_norm(k, attn.k_norm.weight, attn.k_norm.eps, orig_head_dim)
    k = _apply_rope(k, cos, sin).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    v = _padded_rms_norm(v, None, attn.v_norm.eps, orig_head_dim).transpose(1, 2)

    attn_out = padded_sdpa(q, k, v, attn_mask, scale=float(attn.scaling))
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return attn.o_proj(attn_out)


def _run_layer(
    layer,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor,
    num_heads: int,
    orig_head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_out = _run_attention(
        layer.self_attn,
        hidden_states,
        cos,
        sin,
        attn_mask,
        num_heads,
        orig_head_dim,
        padded_head_dim,
    )
    hidden_states = residual + layer.post_attention_layernorm(attn_out)

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    return residual + hidden_states


def patch_vision_encoder() -> None:
    """Replace `Gemma4VisionEncoder.forward` with a Spyre-safe walk over its layers.

    Three things in the stock forward do not lower: the mask built by
    `create_bidirectional_mask`, rope over a head_dim that is not stick-aligned, and
    attention over a patch count coprime with the stick. So pad the head dim, rotate
    with `_apply_rope`, and attend through `padded_sdpa`.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionEncoder", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        config = self.config
        if config.num_key_value_heads != config.num_attention_heads:
            raise NotImplementedError(
                "Gemma 4 vision GQA is not supported on Spyre; num_key_value_heads "
                "must equal num_attention_heads."
            )
        if pixel_position_ids is None:
            # Optional only to match the stock signature; every caller supplies it.
            raise NotImplementedError("Gemma 4 vision requires pixel_position_ids on Spyre.")
        num_heads = config.num_attention_heads
        orig_head_dim = config.head_dim
        padded_head_dim = _padded_head_dim(orig_head_dim)

        if getattr(self.layers[0].self_attn, "_spyre_padded_head_dim", None) != padded_head_dim:
            raise RuntimeError(
                "Gemma 4 vision weights are not padded; `pad_vision_weights` runs from "
                "`apply()` at load time and has to precede the first forward."
            )

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        inv_freq, attention_scaling = _fp32_inv_freq(self.rotary_emb, config)
        cos, sin = _gemma4_rope_cos_sin(
            inv_freq, pixel_position_ids, padded_head_dim, dtype, attention_scaling
        )
        cos = convert(cos, device=device)
        sin = convert(sin, device=device)

        # padded_sdpa takes one key-validity mask for the whole batch, so a batch mixing
        # valid-patch counts per row is refused rather than attended through row 0's mask.
        seq_len = attention_mask.shape[-1]
        mask_host = convert(attention_mask, device="cpu")
        if not bool((mask_host == mask_host[0]).all()):
            raise NotImplementedError(
                "Gemma 4 vision needs one key-validity mask shared by the whole batch "
                "on Spyre; this batch mixes valid-patch counts per row."
            )
        attn_mask = mask_host[0].bool().unsqueeze(0).expand(seq_len, seq_len)

        hidden_states = inputs_embeds
        for layer in self.layers[: config.num_hidden_layers]:
            hidden_states = _run_layer(
                layer,
                hidden_states,
                cos,
                sin,
                attn_mask,
                num_heads,
                orig_head_dim,
                padded_head_dim,
            )

        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states  # ty: ignore[invalid-argument-type]
        )

    _forward._spyre_patched = True
    cls.forward = _forward
    logger.info_once("Spyre: patched Gemma4VisionEncoder to head_dim-padded rope + padded SDPA.")


def patch_rms_norm() -> None:
    """Give ``Gemma4RMSNorm`` the treatment ``SpyreGemmaRMSNorm`` gives vLLM's: no fp32
    promotion, and ``rsqrt`` instead of ``pow(x, -0.5)``, which has no lowering here.

    ``forward`` is the patch point, not ``_norm``: stock casts to fp32 in both, and both
    casts have to go.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4RMSNorm", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    def forward(self, hidden_states):
        weight = self.weight if self.with_scale else None
        return _padded_rms_norm(hidden_states, weight, self.eps, orig_head_dim=None)

    forward._spyre_patched = True
    cls.forward = forward
    logger.info_once(
        "Spyre: Gemma4RMSNorm runs without fp32 promotion and uses torch.rsqrt "
        "instead of torch.pow(x, -0.5); expect small numerical differences."
    )


def _pool_weights(
    pixel_position_ids: torch.Tensor,
    padding_positions: torch.Tensor,
    length: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-built `[bsz, patches, length]` averaging weights, and the validity mask.

    Everything in the stock pooler the device cannot run, and all of it integer geometry
    over `pixel_position_ids`, which already live on the host. Stock's `masked_fill` of
    padding patches folds in by zeroing those patches' weight *rows* instead --
    `out[j] = sum_i W[i,j] * h[i]`, so zeroing either factor drops the term.

    The mask comes from the unzeroed weights, matching stock: an output cell fed only by
    padding patches still counts as valid.

    TODO: rebuilt per image, so a multi-image batch pays for it once per image even when
    the patch geometry repeats. Cache it keyed on that geometry before raising
    ``limit_mm_per_prompt`` above 1.
    """
    clamped = pixel_position_ids.clamp(min=0)
    max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    raw = F.one_hot(kernel_idxs.long(), length).float() / (k * k)
    mask = torch.logical_not((raw == 0).all(dim=1))
    return raw.masked_fill(padding_positions.unsqueeze(-1), 0.0), mask


def patch_pooler() -> None:
    """Run the ``Gemma4VisionPooler`` average as one matmul on Spyre.

    Stock pools with an fp32 batchmatmul against a one-hot weight matrix, which the device
    has no kernel for (`SPYRE_FP32_OPS` carries no matmul, torch-spyre#1794 -- the same
    limit `v1/pool`'s MEAN pooler cites), so we run it in the input dtype instead. Stock
    rounds that matmul's fp32 result straight back to the input dtype one line later, so
    only the accumulation differs. The `sqrt(hidden_size)` scale still happens on the host
    in fp32, as stock does it.

    The weights and the validity mask stay host-side (`_pool_weights`), and the pooled
    rows come back to the host because the tail after this is host-side anyway
    (`place_vision_tail_on_cpu`).
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionPooler", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    orig_forward = cls.forward

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
        num_patches = hidden_states.shape[1]
        k = int((num_patches // output_length) ** 0.5) if output_length else 0
        if (
            hidden_states.device.type != "spyre"
            or not output_length
            # `!=`, mirroring stock's own guard: at equal lengths it skips pooling
            # entirely and hands back `padding_positions` as the mask, so taking the
            # pooling path here would return a differently-derived mask.
            or num_patches == output_length
            or output_length > num_patches
            or k * k * output_length != num_patches
        ):
            # Nothing to pool (stock then only masks and scales, and its `masked_fill`
            # has no Spyre kernel), or a ratio stock itself rejects -- let it raise.
            return orig_forward(
                self,
                convert(hidden_states, device="cpu"),
                convert(pixel_position_ids, device="cpu"),
                convert(padding_positions, device="cpu"),
                output_length,
            )

        weights, mask = _pool_weights(
            convert(pixel_position_ids, device="cpu"),
            convert(padding_positions, device="cpu"),
            output_length,
            k,
        )
        weights = convert(weights.to(hidden_states.dtype), device=hidden_states.device)
        pooled = torch.matmul(weights.transpose(1, 2), hidden_states)
        return convert(pooled, device="cpu").float() * self.root_hidden_size, mask

    forward._spyre_patched = True
    cls.forward = forward
    logger.info_once(
        "Spyre: Gemma4VisionPooler averages on device as one matmul; its one_hot/"
        "masked_fill geometry stays on CPU."
    )


def patch_patch_embedder() -> None:
    """Run ``Gemma4VisionPatchEmbedder``'s position-embedding gather on the host.

    ``F.embedding`` needs index and weight on one device, and ``pixel_position_ids`` stays
    on the host (``embed_multimodal`` only moves float inputs) while the table lives on
    Spyre.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionPatchEmbedder", None)
    if cls is None or getattr(cls._position_embeddings, "_spyre_patched", False):
        return

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        device = self.position_embedding_table.device
        clamped_positions = convert(pixel_position_ids, device="cpu").clamp(min=0)
        table = convert(self.position_embedding_table, device="cpu")
        x_emb = F.embedding(clamped_positions[..., 0], table[0])
        y_emb = F.embedding(clamped_positions[..., 1], table[1])
        position_embeddings = x_emb + y_emb
        padding_cpu = convert(padding_positions, device="cpu")
        position_embeddings = torch.where(padding_cpu.unsqueeze(-1), 0.0, position_embeddings)
        return convert(position_embeddings, device=device)

    _position_embeddings._spyre_patched = True
    cls._position_embeddings = _position_embeddings
    logger.info_once("Spyre: Gemma4VisionPatchEmbedder position-embedding gather runs on CPU.")


def pad_vision_weights(model: torch.nn.Module) -> None:
    """Pad the encoder's attention projections, norms and MLP width to the stick.

    Load-time weight surgery rather than a first-forward side effect: `apply()` already
    runs after the weights land and before compile, and a forward that padded in place
    would both probe every layer forever and mutate the module mid-trace.
    """
    encoder = getattr(getattr(model, "vision_tower", None), "encoder", None)
    if encoder is None:
        return
    config = encoder.config
    padded_head_dim = _padded_head_dim(config.head_dim)
    padded_intermediate = align_up(config.intermediate_size)
    for layer in encoder.layers[: config.num_hidden_layers]:
        _prepare_attention(
            layer.self_attn, config.num_attention_heads, config.head_dim, padded_head_dim
        )
        _pad_mlp(layer, config.intermediate_size, padded_intermediate)
    logger.info_once(
        "Spyre: padded the Gemma 4 vision tower's head dim to %d and its MLP width to %d.",
        padded_head_dim,
        padded_intermediate,
    )


def place_vision_tail_on_cpu(model: torch.nn.Module) -> None:
    """Keep the post-pooler tail on the host: standardize buffers and ``embed_vision``.

    ``_process_image_input`` boolean-selects the pooled rows (no Spyre kernel), applies the
    fp32 standardize affine, then projects to text space -- all on the pooler's output,
    which ``patch_pooler`` already returns on the host. These operands have to follow or
    each step trips a device mismatch.
    """
    tower = getattr(model, "vision_tower", None)
    if tower is not None and getattr(tower.config, "standardize", False):
        for name in ("std_bias", "std_scale"):
            buf = getattr(tower, name, None)
            if buf is not None and buf.device.type != "cpu":
                setattr(tower, name, buf.to("cpu"))
    embed_vision = getattr(model, "embed_vision", None)
    if embed_vision is not None:
        embed_vision.to("cpu")
    logger.info_once(
        "Spyre: Gemma 4 vision tail (standardize buffers + embed_vision) placed on CPU."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install every Gemma 4 vision-tower workaround."""
    del device
    patch_rms_norm()
    patch_patch_embedder()
    patch_pooler()
    patch_vision_encoder()
    pad_vision_weights(model)
    place_vision_tail_on_cpu(model)
