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

    The padding helpers rebuild weights out of strided slice-assignments, and composed
    on a Spyre-resident tensor those lower incorrectly and silently -- finite but wrong,
    costing most of the encoder's accuracy. Individual assignments are fine, so only
    the composite is affected and nothing raises. vLLM moves the model before our
    patches run, so each helper pulls its source here and the caller moves the
    finished weight back.
    """
    return convert(t.detach(), device="cpu")


def _as_plain_linear(layer: nn.Module) -> nn.Linear:
    """Bounce a vLLM linear layer into a plain `nn.Linear`.

    vLLM runs `recursive_replace_linear` over the tower, so `Gemma4ClippableLinear.linear`
    is a `SpyreReplicatedLinear` whose weight is stored transposed (`[in, out]`, for the
    Spyre-fast `x @ Wᵀ` GEMM). The padding helpers assume nn.Linear's `[out, in]`, and
    `F.linear` lowers fine for these projections, so bounce rather than special-case.

    This also discards the layer's `quant_method`, hence the guard: a quantized tower
    must fail rather than be silently dequantized.
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
    the variance on that basis and the zero columns of `o_proj`/`down_proj` ignore
    them. A clamp range excluding zero would map those lanes to a nonzero bound after
    the projection, invalidating both while still returning plausible numbers.
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
    hidden_states: torch.Tensor, weight, eps: float, orig_head_dim: int
) -> torch.Tensor:
    """RMSNorm with the denominator scaled back to the unpadded head_dim, so the zero
    padding lanes do not deflate the variance the real channels are normalized by.

    No fp32 promotion, unlike the hf-adapters reference: torch-spyre does not support
    it (``custom_ops/rms_norm.py``), and an on-device round trip through fp32 leaves a
    stick-tiling state a later eager elementwise op cannot broadcast against. Same
    trade as ``SpyreRMSNorm``: expect small numerical differences from upstream.
    """
    dtype = hidden_states.dtype
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    variance = variance * (hidden_states.shape[-1] / orig_head_dim)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        # These weights are fp32 (transformers builds them at the default dtype), so an
        # unguarded multiply would promote the activation -- the promotion above.
        hidden_states = hidden_states * weight.to(dtype)
    return hidden_states


def _gemma4_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    padded_head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[bsz, seq, 1, padded_head_dim]` cos and signed-sin tables for `_apply_rope`.

    Gemma 4 vision rotates its two position axes independently, each with its own
    `rotate_half` (stock `apply_multidimensional_rope`). `_pad_qk_linear` packs the
    padded head as `[X, Y, zeros | X, Y, zeros]`, so one first-half/second-half swap
    serves both axes at once: each axis's angle is duplicated into both blocks, with
    `rotate_half`'s sign flip baked into `sin`.

    Built on the host in fp32 and cast on the way out, so only the rounded result
    reaches the device.
    """
    positions = position_ids.to("cpu").clamp(min=0).float()
    angles = positions[..., None] * inv_freq.to("cpu").float()  # [bsz, seq, 2, quarter]
    cos_axis = angles.cos()
    sin_axis = angles.sin()
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


def _fp32_inv_freq(rotary_emb, config) -> torch.Tensor:
    """Rope frequencies in fp32, recomputed rather than read off the module buffer.

    `model.to(bfloat16)` downcasts `inv_freq`, and these frequencies span 1.0 down to
    ~1e-4 where bf16 costs real precision -- amplified because the angle is
    `position * inv_freq`. Upcasting the buffer cannot recover the lost bits, so
    recompute; stock HF keeps the whole rope computation in fp32 for the same reason.
    """
    params = getattr(config, "rope_parameters", None) or {}
    rope_type = params.get("rope_type", "default")
    if rope_type == "default":
        inv_freq, _ = type(rotary_emb).compute_default_rope_parameters(config)
    else:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](config, None)
    return inv_freq.float()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x*cos + rotate_half(x)*sin`, with the half-swap done by slicing.

    Slicing is legal here where Pixtral needs a matmul instead: the padded head_dim
    puts each half on a whole stick, whereas Pixtral's 64-wide heads give 32-wide
    halves. It is also required, not just simpler -- at this tower's shapes the
    matmul form's reduction fails to tile after the RMSNorm reduction preceding it.

    `sin` already carries `rotate_half`'s sign flip, so the swap is a plain
    `cat([x2, x1])`.
    """
    half = x.shape[-1] // 2
    swapped = torch.cat([x[..., half:], x[..., :half]], dim=-1)
    return x * cos + swapped * sin


def _padded_head_dim(orig_head_dim: int) -> int:
    """Pad to two sticks, not one: rope rotates each half independently, so both
    halves have to be stick-aligned."""
    return align_up(orig_head_dim, 2 * STICK)


def _pad_mlp(layer, orig_intermediate: int, padded_intermediate: int) -> None:
    """Zero-extend the MLP's intermediate width onto the stick, once per layer.

    A no-op for an already-aligned width. Gate/up gain zero output rows and down
    matching zero input columns, so the padding cannot change the result.
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
            # Kept optional only to match the stock signature; every caller supplies
            # it (rope needs real patch positions to do anything).
            raise NotImplementedError("Gemma 4 vision requires pixel_position_ids on Spyre.")
        num_heads = config.num_attention_heads
        orig_head_dim = config.head_dim
        padded_head_dim = _padded_head_dim(orig_head_dim)

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        cos, sin = _gemma4_rope_cos_sin(
            _fp32_inv_freq(self.rotary_emb, config), pixel_position_ids, padded_head_dim, dtype
        )
        cos = convert(cos, device=device)
        sin = convert(sin, device=device)

        # One shared key-validity mask for the whole batch (padded_sdpa's contract):
        # correct for the batch=1 / uniform-padding case this has been validated
        # against; a batch mixing different valid-patch counts per row would need
        # padded_sdpa extended to a per-row mask.
        seq_len = attention_mask.shape[-1]
        key_valid = convert(attention_mask[0], device="cpu").bool()
        attn_mask = key_valid.unsqueeze(0).expand(seq_len, seq_len)

        orig_intermediate = config.intermediate_size
        padded_intermediate = align_up(orig_intermediate)

        hidden_states = inputs_embeds
        for layer in self.layers[: config.num_hidden_layers]:
            _prepare_attention(layer.self_attn, num_heads, orig_head_dim, padded_head_dim)
            _pad_mlp(layer, orig_intermediate, padded_intermediate)
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
    logger.info_once(
        "Spyre: patched Gemma4VisionEncoder to head_dim-padded rope (Pixtral's "
        "matmul-rotate form) + padded SDPA (pad L/D to 64, mask, crop)."
    )


def patch_rms_norm() -> None:
    """Give ``Gemma4RMSNorm`` the treatment ``SpyreGemmaRMSNorm`` gives vLLM's: no fp32
    promotion, and ``rsqrt`` instead of ``pow(x, -0.5)``, which has no lowering here.

    ``forward`` is the patch point, not ``_norm``: stock casts to fp32 in both, and
    both casts have to go (see ``_padded_rms_norm``). Expect small numerical
    differences from upstream, as with the sibling Spyre norms.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4RMSNorm", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    def forward(self, hidden_states):
        # The trailing cast is stock's `.type_as(hidden_states)`, and it is
        # load-bearing rather than cosmetic: these weights are fp32 (transformers
        # builds them at the default dtype), so an unguarded `* self.weight` would
        # promote the activation to fp32 -- the very thing torch-spyre does not
        # support, and a dtype mismatch against the fp16 projections downstream.
        dtype = hidden_states.dtype
        mean_squared = (hidden_states * hidden_states).mean(-1, keepdim=True) + self.eps
        normed_output = hidden_states * torch.rsqrt(mean_squared)
        if self.with_scale:
            normed_output = normed_output * self.weight.to(dtype)
        return normed_output.to(dtype)

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

    This is everything in the stock pooler the device cannot run, and all of it is
    integer geometry over `pixel_position_ids`, which already live on the host: the
    coordinate arithmetic, the `one_hot`, and the `masked_fill` of padding patches.
    The `masked_fill` folds in by zeroing those patches' weight *rows* instead --
    `out[j] = sum_i W[i,j] * h[i]`, so zeroing either factor drops the term.

    The mask comes from the unzeroed weights, matching stock: an output cell fed only
    by padding patches still counts as valid.

    TODO: this is rebuilt per image, so a multi-image batch pays for it (and its
    upload) once per image even when the patch geometry repeats. Cache it keyed on
    that geometry before raising ``limit_mm_per_prompt`` above 1.
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

    Stock pools with an fp32 batchmatmul against a one-hot weight matrix, which the
    device has no kernel for (`SPYRE_FP32_OPS` carries no matmul, torch-spyre#1794 --
    the same limit `v1/pool`'s MEAN pooler cites). **We run that matmul in bf16
    instead of fp32, which is worth ~2x on the whole post-encoder tail** (8.6 ms vs
    17.4 ms measured at 26B-A4B's 2520 patches), and moves a ~0.8 GMAC GEMM off the
    host while shrinking the encoder-output copy from 5.8 MB to 0.6 MB.

    The precision given up is smaller than it looks: stock computes that matmul in
    fp32 but rounds the result straight back to the input dtype one line later, so
    the values it passes on are bf16 either way. Only the accumulation differs, and
    it measures below one bf16 ULP. The `sqrt(hidden_size)` scale still happens on
    the host in fp32, exactly as stock does it.

    The weights and the validity mask stay host-side (`_pool_weights`), and the
    pooled rows come back to the host because the tail after this -- the caller's
    `pooled[mask]`, the standardize affine, `embed_vision` -- is host-side anyway
    (`place_vision_tail_on_cpu`); measurement says moving that too gives the gain
    straight back.
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
    logger.info_once("Spyre: Gemma4VisionPooler runs on CPU (masked_fill/one_hot geometry).")


def patch_accelerator_memory_info() -> None:
    """Fall back to host RAM for ``torch.accelerator.get_memory_info()``.

    ``_process_image_input`` calls this to size its encoder-chunking budget, and Spyre
    registers no accelerator memory-info hook, so the native call always raises. Host
    RAM is the right substitute: the transients this budget guards run on the host.
    """
    orig = torch.accelerator.get_memory_info
    if getattr(orig, "_spyre_patched", False):
        return

    def _get_memory_info(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except NotImplementedError:
            import psutil

            vm = psutil.virtual_memory()
            return (vm.available, vm.total)

    _get_memory_info._spyre_patched = True
    torch.accelerator.get_memory_info = _get_memory_info  # ty: ignore[invalid-assignment]
    logger.info_once(
        "Spyre: torch.accelerator.get_memory_info() falls back to host RAM "
        "(psutil) when the native accelerator call is unimplemented."
    )


def patch_patch_embedder() -> None:
    """Run ``Gemma4VisionPatchEmbedder``'s position-embedding gather on the host.

    ``F.embedding`` needs index and weight on one device, and ``pixel_position_ids``
    stays on the host (``embed_multimodal`` only moves float inputs) while the table
    lives on Spyre. Same doctrine as the other integer gathers here.
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


def place_vision_tail_on_cpu(model: torch.nn.Module) -> None:
    """Keep the post-pooler tail on the host: standardize buffers and ``embed_vision``.

    ``_process_image_input`` boolean-selects the pooled rows (no Spyre kernel), applies
    the fp32 standardize affine, then projects to text space -- all on the pooler's
    output, which ``patch_pooler`` already returns on the host. These operands have to
    follow or each step trips a device mismatch. It also avoids a round trip: the
    projection is the cheap end of the tower and its output feeds a host-side merge.
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
    patch_accelerator_memory_info()
    patch_rms_norm()
    patch_patch_embedder()
    patch_pooler()
    patch_vision_encoder()
    place_vision_tail_on_cpu(model)
