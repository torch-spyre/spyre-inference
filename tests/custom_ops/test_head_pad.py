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

"""Head-padding passes that make the platform's head_dim override effective.

CPU-only: these cover the width shim and its two guards, none of which touch the
device. The end-to-end check that a padded model produces correct logits is the
upstream ``test_models`` matrix (Llama, Qwen2 and Granite are all head_dim=64).
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.head_pad import (
    _pad_weight,
    fix_padded_attention_scale,
    install_padded_head_dim,
    verify_padded_head_dim,
)

_ORIG, _PADDED = 64, 128


class _DerivesOwnHeadDim(torch.nn.Module):
    """Attention that computes its own width and never reads config.head_dim.

    This is vLLM's Qwen2Attention shape: everything downstream is sized off
    ``self.head_dim`` after the assignment, so substituting at assignment time is
    what makes the override land.
    """

    def __init__(self, hidden_size=896, num_heads=14):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim


class _ReadsConfigHeadDim(torch.nn.Module):
    """Attention that already honours config.head_dim (Llama/Granite shape)."""

    def __init__(self, config_head_dim=_PADDED):
        super().__init__()
        self.head_dim = config_head_dim


def _fake_model_config(monkeypatch, *, padded=True, classes=None, transformers_backend=False):
    """A model_config whose registry resolves to a throwaway module of attention classes."""
    module_name = "spyre_test_fake_model_module"
    module = ModuleType(module_name)
    for name, cls in (classes or {}).items():
        cls.__module__ = module_name
        setattr(module, name, cls)
    monkeypatch.setitem(sys.modules, module_name, module)

    model_cls = type("FakeForCausalLM", (), {})
    model_cls.__module__ = module_name

    hf_config = SimpleNamespace(head_dim=_PADDED, architectures=["FakeForCausalLM"])
    if padded:
        hf_config._spyre_orig_head_dim = _ORIG
    return SimpleNamespace(
        hf_config=hf_config,
        using_transformers_backend=lambda: transformers_backend,
        registry=SimpleNamespace(
            resolve_model_cls=lambda archs, model_config: (model_cls, archs[0])
        ),
    )


def test_shim_widens_a_model_that_derives_its_own_head_dim(monkeypatch):
    """The whole point: a model ignoring config.head_dim still builds padded."""
    cls = type("FakeAttention", (_DerivesOwnHeadDim,), {})
    model_config = _fake_model_config(monkeypatch, classes={"FakeAttention": cls})

    install_padded_head_dim(model_config)
    attn = cls()

    assert attn.head_dim == _PADDED
    # Sizes derived after the assignment follow the padded width.
    assert attn.q_size == 14 * _PADDED


def test_shim_leaves_a_model_that_already_reads_config_head_dim(monkeypatch):
    """Llama/Granite assign the padded width themselves; the setter must not double it."""
    cls = type("FakeAttention", (_ReadsConfigHeadDim,), {})
    model_config = _fake_model_config(monkeypatch, classes={"FakeAttention": cls})

    install_padded_head_dim(model_config)

    assert cls().head_dim == _PADDED


def test_shim_leaves_unrelated_widths_alone(monkeypatch):
    """Only the native width is substituted, so other head sizes pass through."""
    cls = type("FakeAttention", (_ReadsConfigHeadDim,), {})
    model_config = _fake_model_config(monkeypatch, classes={"FakeAttention": cls})

    install_padded_head_dim(model_config)

    assert cls(config_head_dim=96).head_dim == 96


def test_shim_noop_without_padding(monkeypatch):
    """No platform override -> no shim, model keeps its native width."""
    cls = type("FakeAttention", (_DerivesOwnHeadDim,), {})
    model_config = _fake_model_config(monkeypatch, padded=False, classes={"FakeAttention": cls})

    install_padded_head_dim(model_config)

    assert cls().head_dim == _ORIG


def test_shim_skips_the_shared_vllm_attention_layers(monkeypatch):
    """Patching a class the whole process shares would leak across models."""
    cls = type("Attention", (_ReadsConfigHeadDim,), {})
    cls.__module__ = "vllm.model_executor.layers.attention.attention"
    module_name = "spyre_test_fake_model_module"
    module = ModuleType(module_name)
    module.Attention = cls
    monkeypatch.setitem(sys.modules, module_name, module)

    model_cls = type("FakeForCausalLM", (), {})
    model_cls.__module__ = module_name
    hf_config = SimpleNamespace(
        head_dim=_PADDED, architectures=["FakeForCausalLM"], _spyre_orig_head_dim=_ORIG
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        using_transformers_backend=lambda: False,
        registry=SimpleNamespace(
            resolve_model_cls=lambda archs, model_config: (model_cls, archs[0])
        ),
    )

    install_padded_head_dim(model_config)

    assert "head_dim" not in vars(cls)


def test_shim_skipped_on_the_transformers_backend(monkeypatch):
    """HF attention reads config.head_dim directly, so there is nothing to shim."""
    cls = type("FakeAttention", (_DerivesOwnHeadDim,), {})
    model_config = _fake_model_config(
        monkeypatch, classes={"FakeAttention": cls}, transformers_backend=True
    )

    install_padded_head_dim(model_config)

    assert "head_dim" not in vars(cls)


def _model_with_attention(head_size):
    """A model exposing one vLLM-shaped Attention layer (has .impl and .head_size)."""
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.impl = object()
    layer.head_size = head_size
    model.add_module("attn", layer)
    return model


def test_verify_rejects_a_layer_left_at_the_native_width():
    """The silent-corruption guard: the weight loader truncates without raising."""
    hf_config = SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG)

    with pytest.raises(RuntimeError, match="would load truncated"):
        verify_padded_head_dim(_model_with_attention(_ORIG), hf_config)


def test_verify_accepts_a_padded_layer():
    hf_config = SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG)

    verify_padded_head_dim(_model_with_attention(_PADDED), hf_config)


def test_verify_noop_without_padding():
    """Unpadded models keep their native width; the guard must not fire."""
    verify_padded_head_dim(_model_with_attention(_ORIG), SimpleNamespace(head_dim=_ORIG))


def test_pad_weight_lays_out_qk_norm_interleaved_and_zeros_the_pad():
    """The norm weight must match the interleaved Q/K layout it multiplies."""
    w = torch.arange(1.0, _ORIG + 1)

    out = _pad_weight("layers.0.self_attn.q_norm.weight", w, 4, 2, _ORIG, _PADDED)

    assert out.shape == (_PADDED,)
    half, padded_half = _ORIG // 2, _PADDED // 2
    scale = (_ORIG / _PADDED) ** 0.5
    assert torch.allclose(out[:half], w[:half] * scale)
    assert torch.allclose(out[padded_half : padded_half + half], w[half:] * scale)
    assert not out[half:padded_half].any()
    assert not out[padded_half + half :].any()


def test_pad_weight_qk_norm_reproduces_the_original_rmsnorm():
    """End-to-end: RMSNorm over the padded, interleaved head equals the original."""
    torch.manual_seed(0)
    q = torch.randn(_ORIG)
    w = torch.randn(_ORIG)

    ref = F.rms_norm(q, (_ORIG,), w, eps=1e-6)

    # q and its norm weight are padded exactly as the loader pads q_proj / q_norm.
    q_padded = _pad_weight("q_proj.weight", q.view(_ORIG, 1), 1, 1, _ORIG, _PADDED).view(_PADDED)
    w_padded = _pad_weight("q_norm.weight", w, 1, 1, _ORIG, _PADDED)
    out = F.rms_norm(q_padded, (_PADDED,), w_padded, eps=1e-6)

    half, padded_half = _ORIG // 2, _PADDED // 2
    assert torch.allclose(out[:half], ref[:half], atol=1e-5)
    assert torch.allclose(out[padded_half : padded_half + half], ref[half:], atol=1e-5)
    # Padded dims stay zero, so they never reach the QK dot product or RoPE.
    assert not out[half:padded_half].any()
    assert not out[padded_half + half :].any()


def test_pad_weight_leaves_a_norm_of_another_width_alone():
    """A q_norm taken over a non-head_dim width is not a QK-norm and is untouched."""
    w = torch.arange(float(_ORIG * 7))

    out = _pad_weight("layers.0.self_attn.q_norm.weight", w, 4, 2, _ORIG, _PADDED)

    assert torch.equal(out, w)


def test_verify_checks_the_transformers_backend_attention_dict():
    """The backend's Attention layers live in a plain dict, invisible to named_modules()."""
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.impl = object()
    layer.head_size = _ORIG
    model.attention_instances = {0: layer}
    hf_config = SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG)

    assert list(model.named_modules()) == [("", model)]
    with pytest.raises(RuntimeError, match="would load truncated"):
        verify_padded_head_dim(model, hf_config)


def _attention_with_scale(*, impl_scale=None, hf_scaling=None):
    """One attention layer, optionally carrying a vLLM impl scale and/or HF's scaling."""
    model = torch.nn.Module()
    layer = torch.nn.Module()
    if impl_scale is not None:
        layer.impl = SimpleNamespace(scale=impl_scale)
    if hf_scaling is not None:
        layer.scaling = hf_scaling
    model.add_module("self_attn", layer)
    return model, layer


def test_fix_scale_resets_the_padded_default_on_the_vllm_layer():
    model, layer = _attention_with_scale(impl_scale=_PADDED**-0.5)

    fix_padded_attention_scale(model, SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG))

    assert layer.impl.scale == pytest.approx(_ORIG**-0.5)


def test_fix_scale_resets_the_hf_module_scaling():
    """``vllm_attention_forward`` copies it onto impl.scale every forward."""
    model, layer = _attention_with_scale(impl_scale=_PADDED**-0.5, hf_scaling=_PADDED**-0.5)

    fix_padded_attention_scale(model, SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG))

    assert layer.scaling == pytest.approx(_ORIG**-0.5)


def test_fix_scale_leaves_a_head_dim_independent_scale_alone():
    """Granite's ``attention_multiplier`` was never corrupted by the width override."""
    model, layer = _attention_with_scale(impl_scale=0.5, hf_scaling=0.5)

    fix_padded_attention_scale(model, SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG))

    assert layer.impl.scale == 0.5
    assert layer.scaling == 0.5


def test_pad_weight_splits_a_fused_qkv_projection():
    """Each fused slice needs its own rule, and "qkv_proj" must not hit the v branch."""
    n_heads, n_kv, hidden = 4, 2, 8
    rows = (n_heads + 2 * n_kv) * _ORIG
    w = torch.arange(float(rows * hidden)).reshape(rows, hidden)

    out = _pad_weight("layers.0.self_attn.qkv_proj.weight", w, n_heads, n_kv, _ORIG, _PADDED)

    assert out.shape == ((n_heads + 2 * n_kv) * _PADDED, hidden)
    q, _, v = out.split([n_heads * _PADDED, n_kv * _PADDED, n_kv * _PADDED])

    # Q is interleaved so the RoPE half-split still pairs the real dims.
    half, padded_half = _ORIG // 2, _PADDED // 2
    q_src = w[: n_heads * _ORIG].view(n_heads, _ORIG, hidden)
    q_out = q.view(n_heads, _PADDED, hidden)
    assert torch.equal(q_out[:, :half], q_src[:, :half])
    assert torch.equal(q_out[:, padded_half : padded_half + half], q_src[:, half:])

    # V carries no RoPE, so it is end-padded with zeros.
    v_src = w[(n_heads + n_kv) * _ORIG :].view(n_kv, _ORIG, hidden)
    v_out = v.view(n_kv, _PADDED, hidden)
    assert torch.equal(v_out[:, :_ORIG], v_src)
    assert not v_out[:, _ORIG:].any()
