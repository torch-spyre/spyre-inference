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

"""Native-path attention head padding to a stick-aligned head_dim.

A head_dim whose half is not a multiple of the 64-element fp16 stick (e.g.
head_size=64) cannot restickify after RoPE, so the KV write-back fails to lower
on Spyre. ``TorchSpyrePlatform._maybe_pad_head_dim`` overrides ``head_dim`` to a
128-multiple before the model is built (sizing QKV/o_proj/Attention/KV-cache/RoPE
at the padded width); the passes here fill the padded region on load and restore
the two things the width override would otherwise corrupt — the RoPE frequencies
and the attention scale.

Padding is interleaved (RoPE-compatible) for Q/K and end-of-head for V/O, and the
rotation cache keeps the original frequencies. The Transformers backend shares the
weight passes here; it rebuilds its own rotation cache in ``transformers_backend``,
since HF's rotary module is not the one ``fix_padded_rope`` reaches.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope

logger = init_logger(__name__)

_ORIG_ATTR = "_spyre_orig_head_dim"


def original_head_dim(hf_config) -> int | None:
    """The pre-pad head_dim if the platform padded this model, else None."""
    return getattr(hf_config, _ORIG_ATTR, None)


def head_padding_active(hf_config) -> bool:
    """True when the platform padded this model's head_dim for stick alignment."""
    return original_head_dim(hf_config) is not None


def reduced_rotary_dim_reason(cfg) -> str | None:
    """If any custom rope configs exist that would rotate fewer than `head_dim` dims,
    this returns a string with the offending configs.

    transformers 5.x carries all RoPE config in ``rope_parameters`` (an absolute
    ``rope_dim`` override, or a ``partial_rotary_factor`` < 1), so that is the only
    place to look.
    """
    rope_params = getattr(cfg, "rope_parameters", None)
    rope_params = rope_params if isinstance(rope_params, dict) else {}
    if rope_params.get("rope_dim") is not None:
        return f"rope_parameters.rope_dim={rope_params['rope_dim']}"
    factor = rope_params.get("partial_rotary_factor")
    if factor is not None and factor != 1.0:
        return f"partial_rotary_factor={factor}"
    return None


def _pad_qk_interleaved(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """Interleaved padding on the output dim (dim 0), RoPE half-split compatible.

    Per head: ``[first_half | zeros | second_half | zeros]`` so that the padded
    dims pair with zeros under the ``[2, D/2]`` RoPE reshape.
    """
    orig_half, padded_half = orig // 2, padded // 2
    w = w.view(n_heads, orig, *w.shape[1:])
    new = w.new_zeros(n_heads, padded, *w.shape[2:])
    new[:, :orig_half] = w[:, :orig_half]
    new[:, padded_half : padded_half + orig_half] = w[:, orig_half:orig]
    return new.reshape(n_heads * padded, *w.shape[2:])


def _pad_output_end(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """End-pad each head on the output dim (dim 0). Used for V (no RoPE)."""
    w = w.view(n_heads, orig, *w.shape[1:])
    new = w.new_zeros(n_heads, padded, *w.shape[2:])
    new[:, :orig] = w
    return new.reshape(n_heads * padded, *w.shape[2:])


def _pad_input_end(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """End-pad each head on the input dim (dim 1). Used for O."""
    hidden = w.shape[0]
    w = w.view(hidden, n_heads, orig)
    new = w.new_zeros(hidden, n_heads, padded)
    new[:, :, :orig] = w
    return new.reshape(hidden, n_heads * padded)


def _pad_fused_qkv(
    w: torch.Tensor, n_heads: int, n_kv_heads: int, orig: int, padded: int
) -> torch.Tensor:
    """Pad a fused ``[q | k | v]`` checkpoint tensor (Phi-3 ships one, not three)."""
    q, k, v = w.split([n_heads * orig, n_kv_heads * orig, n_kv_heads * orig])
    return torch.cat(
        [
            _pad_qk_interleaved(q, n_heads, orig, padded),
            _pad_qk_interleaved(k, n_kv_heads, orig, padded),
            _pad_output_end(v, n_kv_heads, orig, padded),
        ]
    )


def _pad_weight(
    name: str, w: torch.Tensor, n_heads: int, n_kv_heads: int, orig: int, padded: int
) -> torch.Tensor:
    """Dispatch a single checkpoint tensor to the right padding by its name."""
    # Must precede the v_proj test: "qkv_proj.weight" also ends with "v_proj.weight".
    if name.endswith(("qkv_proj.weight", "qkv_proj.bias")):
        return _pad_fused_qkv(w, n_heads, n_kv_heads, orig, padded)
    if name.endswith(("q_proj.weight", "q_proj.bias")):
        return _pad_qk_interleaved(w, n_heads, orig, padded)
    if name.endswith(("k_proj.weight", "k_proj.bias")):
        return _pad_qk_interleaved(w, n_kv_heads, orig, padded)
    if name.endswith(("v_proj.weight", "v_proj.bias")):
        return _pad_output_end(w, n_kv_heads, orig, padded)
    if name.endswith("o_proj.weight"):
        return _pad_input_end(w, n_heads, orig, padded)
    return w


_SHIM_ATTR = "_spyre_head_dim"


def install_padded_head_dim(model_config) -> None:
    """Force attention modules to build at the padded head_dim, however they derive it.

    Overriding ``config.head_dim`` only widens a model that reads it. vLLM's
    ``Qwen2Attention`` computes ``self.head_dim = hidden_size // total_num_heads``
    and never consults the config, so the override left it 64-wide while the weight
    pass emitted 128-wide tensors — which ``QKVParallelLinear.weight_loader``
    narrows back to the param width without complaint, loading truncated weights
    and no error.

    Everything a module sizes from ``head_dim`` is read back off ``self`` after the
    assignment (q/kv sizes, the QKV and o_proj shapes, RoPE, the Attention layer's
    head_size, and the per-head views in ``forward``), so replacing ``head_dim``
    with a property whose setter substitutes the padded width makes the whole
    module consistent regardless of how it computed the value.

    Skipped on the Transformers backend: HF attention sizes itself from
    ``config.head_dim``, so the override already lands there.
    """
    if not head_padding_active(model_config.hf_config):
        return
    if model_config.using_transformers_backend():
        return
    orig = getattr(model_config.hf_config, _ORIG_ATTR)
    padded = model_config.hf_config.head_dim

    architectures = getattr(model_config.hf_config, "architectures", None) or []
    model_cls, _ = model_config.registry.resolve_model_cls(architectures, model_config=model_config)
    module = sys.modules.get(model_cls.__module__)
    if module is None:
        logger.warning("Cannot locate module for %s; head_dim not shimmed.", model_cls)
        return

    def _make_head_dim_property(orig: int, padded: int) -> property:
        def getter(self):
            try:
                return self.__dict__[_SHIM_ATTR]
            except KeyError:
                raise AttributeError("head_dim") from None

        def setter(self, value):
            # Only the native width is substituted; a model that already read the
            # padded config.head_dim (Llama, Granite) assigns `padded` and is
            # untouched, as is any unrelated head width in the same module.
            self.__dict__[_SHIM_ATTR] = padded if value == orig else value

        prop = property(getter, setter)
        prop.fget._spyre_shim = (orig, padded)
        return prop

    patched = []
    for name, obj in vars(module).items():
        # Model-level attention classes only. The shared vLLM attention *layers* are
        # imported into the same namespace but are handed an already-padded
        # head_size, and patching them would mutate a class the whole process uses.
        if (
            not isinstance(obj, type)
            or not name.endswith("Attention")
            or obj.__module__.startswith("vllm.model_executor.layers")
        ):
            continue
        existing = vars(obj).get("head_dim")
        # Replace a shim left by an earlier model in this process (its widths may
        # differ); leave anything the model itself defines alone.
        if (
            existing is not None
            and getattr(getattr(existing, "fget", None), "_spyre_shim", None) is None
        ):
            continue
        obj.head_dim = _make_head_dim_property(orig, padded)
        patched.append(name)
    logger.info("Shimmed head_dim %d -> %d on: %s", orig, padded, ", ".join(patched))


def _attention_layers(model) -> list[tuple[str, torch.nn.Module]]:
    """``named_modules()`` plus ``attention_instances``, a plain dict nn.Module never
    registers (the Transformers backend keeps its Attention layers there)."""
    instances = getattr(model, "attention_instances", None) or {}
    return list(model.named_modules()) + [(f"attn.{i}", m) for i, m in instances.items()]


def verify_padded_head_dim(model, hf_config) -> None:
    """Fail loudly if any attention layer was still built at the unpadded width.

    Guards the silent-corruption path: the weight pass emits padded tensors and the
    linear weight loader narrows an over-wide tensor to the param width without
    raising, so a model the override failed to reach loads truncated weights and
    produces plausible-looking garbage instead of an error.
    """
    if not head_padding_active(hf_config):
        return
    padded = hf_config.head_dim
    bad = sorted(
        {
            f"{name}(head_size={module.head_size})"
            for name, module in _attention_layers(model)
            if getattr(module, "impl", None) is not None
            and getattr(module, "head_size", padded) != padded
        }
    )
    if bad:
        raise RuntimeError(
            f"Spyre padded head_dim to {padded}, but these attention layers were "
            f"built at a different width, so their weights would load truncated: "
            f"{', '.join(bad)}"
        )


def reject_padded_qk_norm(model, hf_config) -> None:
    """Reject models that normalize over head_dim (QK-norm).

    Padding fills the extra dims with zeros, which is invisible to the QK dot
    product and to V/O, but an RMSNorm taken over the padded head_dim divides by
    the RMS of 128 values of which half are zero — silently rescaling Q/K.
    """
    if not head_padding_active(hf_config):
        return
    padded = hf_config.head_dim
    for name, module in model.named_modules():
        for norm_attr in ("q_norm", "k_norm"):
            norm = getattr(module, norm_attr, None)
            weight = getattr(norm, "weight", None)
            if weight is not None and weight.numel() == padded:
                raise NotImplementedError(
                    f"Spyre must pad attention head_dim to {padded} for stick "
                    f"alignment, but {name}.{norm_attr} normalizes over head_dim; "
                    "zero-padded dims would change the RMS and rescale Q/K."
                )


def install_head_pad_weight_loader(model_loader, hf_config) -> None:
    """Wrap ``model_loader.get_all_weights`` to pad q/k/v/o head_dim 64->128.

    The transform runs on the raw ``(name, tensor)`` stream before vLLM's
    ``WeightsMapper`` and ``weight_loader`` (which ``.narrow`` and assert exact
    shapes against the now-128-wide params). Full unsharded tensors are padded
    per-head, so TP narrowing downstream still selects whole padded heads.
    """
    if not head_padding_active(hf_config):
        return
    if not hasattr(model_loader, "get_all_weights"):
        logger.warning(
            "Head padding active but %s has no get_all_weights; weights not padded.",
            type(model_loader).__name__,
        )
        return

    orig = getattr(hf_config, _ORIG_ATTR)
    padded = hf_config.head_dim
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", None) or n_heads

    original_get_all_weights = model_loader.get_all_weights

    def padded_get_all_weights(model_config, model) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in original_get_all_weights(model_config, model):
            yield name, _pad_weight(name, weight, n_heads, n_kv_heads, orig, padded)

    model_loader.get_all_weights = padded_get_all_weights


def fix_padded_attention_scale(model, hf_config) -> None:
    """Restore the attention scale to ``1/sqrt(orig_head_dim)`` for head_dim-derived scales.

    A model that computes ``scale = head_dim**-0.5`` (Llama, Mistral) picks up
    ``padded**-0.5`` from the widened head_dim, but the real dot product is still
    over the original dims (padded dims are zero), so it must be divided by
    ``sqrt(orig_head_dim)`` or softmax flattens. Models with a head_dim-independent
    scale (Granite's ``attention_multiplier``) were never corrupted by padding, so
    their scale must be left untouched — detected by comparing the built scale
    against the padded head_dim default.

    HF's ``module.scaling`` is reset too: ``vllm_attention_forward`` copies it onto
    ``impl.scale`` on every forward, so fixing only the vLLM layer would not stick.
    """
    if not head_padding_active(hf_config):
        return
    orig = getattr(hf_config, _ORIG_ATTR)
    padded_default = float(hf_config.head_dim**-0.5)
    orig_default = float(orig**-0.5)

    def is_padded_default(scale) -> bool:
        return isinstance(scale, (int, float)) and math.isclose(
            float(scale), padded_default, rel_tol=1e-3
        )

    n = 0
    for _, module in _attention_layers(model):
        impl = getattr(module, "impl", None)
        if impl is not None and is_padded_default(getattr(impl, "scale", None)):
            impl.scale = orig_default
            n += 1
        if is_padded_default(getattr(module, "scaling", None)):
            module.scaling = orig_default
            n += 1
    logger.info("Reset attention scale to 1/sqrt(%d) on %d head_dim-derived layers.", orig, n)


def fix_padded_rope(model, hf_config) -> None:
    """Inject the original-frequency cos/sin cache into each padded RoPE.

    ``get_rope(padded)`` built frequencies at the padded spacing (wrong); rebuild
    a reference rope at the original head_dim (reusing vLLM's rope-scaling dispatch
    for correct Llama3/YaRN frequencies) and swap its narrower cos_sin_cache in.
    ``SpyreRotaryEmbedding._get_rotation_cache`` then derives the real rotations
    from it and zero-pads the trailing dims (harmless — the matching x pair dims
    are zero from weight padding).
    """
    if not head_padding_active(hf_config):
        return
    orig = getattr(hf_config, _ORIG_ATTR)
    max_position = hf_config.max_position_embeddings
    rope_parameters = getattr(hf_config, "rope_parameters", None)

    seen: set[int] = set()
    n = 0
    for module in model.modules():
        # Duck-type on an attribute only _SpyreRotaryMixin sets, not isinstance: keeps
        # `module` typed as nn.Module so the RotaryEmbedding attribute reads below type-check.
        if not hasattr(module, "_rotation_cache") or id(module) in seen:
            continue
        seen.add(id(module))
        ref = get_rope(
            orig,
            max_position=max_position,
            is_neox_style=module.is_neox_style,
            rope_parameters=rope_parameters,
            dtype=module.dtype,
        )
        module.cos_sin_cache = ref.cos_sin_cache.to(module.cos_sin_cache.dtype)
        module._rotation_cache = None
        module._device_rotation_cache = None
        # Narrowed frequencies make this instance model-specific; unshare it so
        # get_rope cannot hand it to a later model with a real head_dim of orig*2.
        for cache_key, cached in list(_ROPE_DICT.items()):
            if cached is module:
                del _ROPE_DICT[cache_key]
        n += 1
    logger.info("Injected original head_dim=%d RoPE frequencies into %d modules.", orig, n)
