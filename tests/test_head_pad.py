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

from spyre_inference.custom_ops.head_pad import (
    _pad_weight,
    install_padded_head_dim,
    reject_padded_qk_norm,
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


def _fake_model_config(monkeypatch, *, padded=True, classes=None):
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
        registry=SimpleNamespace(
            resolve_model_cls=lambda archs, model_config: (model_cls, archs[0])
        ),
    )

    install_padded_head_dim(model_config)

    assert "head_dim" not in vars(cls)


def test_shim_follows_the_mro_into_a_base_class_module(monkeypatch):
    """vLLM's Phi3ForCausalLM subclasses LlamaForCausalLM and declares no attention."""
    base_module_name = "spyre_test_fake_base_module"
    base_module = ModuleType(base_module_name)
    attn_cls = type("FakeAttention", (_DerivesOwnHeadDim,), {})
    attn_cls.__module__ = base_module_name
    base_cls = type("FakeBaseForCausalLM", (torch.nn.Module,), {})
    base_cls.__module__ = base_module_name
    base_module.FakeAttention = attn_cls
    base_module.FakeBaseForCausalLM = base_cls
    monkeypatch.setitem(sys.modules, base_module_name, base_module)

    # The thin subclass's own module holds no attention class at all.
    thin_module_name = "spyre_test_fake_thin_module"
    thin_module = ModuleType(thin_module_name)
    model_cls = type("FakeForCausalLM", (base_cls,), {})
    model_cls.__module__ = thin_module_name
    thin_module.FakeForCausalLM = model_cls
    monkeypatch.setitem(sys.modules, thin_module_name, thin_module)

    hf_config = SimpleNamespace(
        head_dim=_PADDED, architectures=["FakeForCausalLM"], _spyre_orig_head_dim=_ORIG
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        registry=SimpleNamespace(
            resolve_model_cls=lambda archs, model_config: (model_cls, archs[0])
        ),
    )

    install_padded_head_dim(model_config)

    assert attn_cls().head_dim == _PADDED


def test_shim_warns_when_it_matches_nothing(monkeypatch, caplog):
    """An empty match used to log as a success; it means the width override missed."""
    model_config = _fake_model_config(monkeypatch, classes={})

    with caplog.at_level("WARNING"):
        install_padded_head_dim(model_config)

    assert "no attention class" in caplog.text.lower()


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


def test_reject_qk_norm_over_padded_head_dim():
    """Zero-padded dims change an RMS taken over head_dim, rescaling Q/K."""
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.add_module("q_norm", torch.nn.LayerNorm(_PADDED))
    model.add_module("attn", layer)
    hf_config = SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG)

    with pytest.raises(NotImplementedError, match="normalizes over head_dim"):
        reject_padded_qk_norm(model, hf_config)


def test_reject_qk_norm_allows_norms_of_other_widths():
    """An RMSNorm over the hidden size (not head_dim) is untouched by padding."""
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.add_module("q_norm", torch.nn.LayerNorm(_PADDED * 7))
    model.add_module("attn", layer)
    hf_config = SimpleNamespace(head_dim=_PADDED, _spyre_orig_head_dim=_ORIG)

    reject_padded_qk_norm(model, hf_config)


_HIDDEN, _N_HEADS, _N_KV = 32, 4, 2


def _qkv_parts(orig=_ORIG, n_heads=_N_HEADS, n_kv=_N_KV, hidden=_HIDDEN):
    sizes = (n_heads * orig, n_kv * orig, n_kv * orig)
    return tuple(torch.arange(n * hidden).float().view(n, hidden) + i for i, n in enumerate(sizes))


def _pad(name, w, orig=_ORIG):
    return _pad_weight(f"model.layers.0.self_attn.{name}", w, _N_HEADS, _N_KV, orig, _PADDED)


def test_pad_q_places_the_halves_for_rope():
    """RoPE pairs dim i with dim i + padded/2, so each half keeps its own pair partner."""
    q, _, _ = _qkv_parts()

    out = _pad("q_proj.weight", q).view(_N_HEADS, _PADDED, _HIDDEN)
    src = q.view(_N_HEADS, _ORIG, _HIDDEN)
    half, padded_half = _ORIG // 2, _PADDED // 2

    assert torch.equal(out[:, :half], src[:, :half])
    assert torch.equal(out[:, padded_half : padded_half + half], src[:, half:])
    assert out[:, half:padded_half].eq(0).all()
    assert out[:, padded_half + half :].eq(0).all()


def test_pad_v_and_o_end_pad():
    """V and O see no RoPE, so the padded dims go at the end of each head."""
    _, _, v = _qkv_parts()

    out = _pad("v_proj.weight", v).view(_N_KV, _PADDED, _HIDDEN)
    assert torch.equal(out[:, :_ORIG], v.view(_N_KV, _ORIG, _HIDDEN))
    assert out[:, _ORIG:].eq(0).all()

    o = torch.arange(_HIDDEN * _N_HEADS * _ORIG).float().view(_HIDDEN, _N_HEADS * _ORIG)
    out = _pad("o_proj.weight", o).view(_HIDDEN, _N_HEADS, _PADDED)
    assert torch.equal(out[:, :, :_ORIG], o.view(_HIDDEN, _N_HEADS, _ORIG))
    assert out[:, :, _ORIG:].eq(0).all()


@pytest.mark.parametrize("orig", [2, _ORIG, 96])
def test_pad_fused_qkv_matches_the_per_projection_padding(orig):
    """Each padded shard must sit where a separate q/k/v checkpoint would have put it."""
    q, k, v = _qkv_parts(orig)

    out = _pad("qkv_proj.weight", torch.cat([q, k, v]), orig)

    assert out.shape == ((_N_HEADS + 2 * _N_KV) * _PADDED, _HIDDEN)
    offsets = (0, _N_HEADS * _PADDED, (_N_HEADS + _N_KV) * _PADDED)
    sizes = (_N_HEADS * _PADDED, _N_KV * _PADDED, _N_KV * _PADDED)
    for name, part, offset, size in zip(
        ("q_proj.weight", "k_proj.weight", "v_proj.weight"), (q, k, v), offsets, sizes
    ):
        assert torch.equal(out.narrow(0, offset, size), _pad(name, part, orig))


def test_pad_fused_qkv_is_not_mistaken_for_v():
    """`qkv_proj.weight` also ends with `v_proj.weight` (issue #596)."""
    q, k, v = _qkv_parts()
    fused = torch.cat([q, k, v])

    assert not torch.equal(_pad("qkv_proj.weight", fused), _pad("v_proj.weight", v))
    assert _pad("qkv_proj.weight", fused).shape[0] == (_N_HEADS + 2 * _N_KV) * _PADDED


def test_pad_rejects_a_tensor_it_cannot_reshape():
    """Naming the tensor beats a bare view error from deep in the padding helper."""
    with pytest.raises(ValueError, match="cannot reshape"):
        _pad("v_proj.weight", torch.zeros(_N_KV * _ORIG + 1, _HIDDEN))


@pytest.mark.parametrize(
    "name",
    ["mlp.gate_up_proj.weight", "self_attn.o_proj.bias", "self_attn.qkv_proj.weight_scale"],
)
def test_pad_leaves_non_head_dim_tensors_untouched(name):
    """MLP fusions, o_proj's hidden-size bias and scales carry no head_dim axis."""
    w = torch.zeros(7, 5)

    assert _pad_weight(f"model.layers.0.{name}", w, _N_HEADS, _N_KV, _ORIG, _PADDED) is w


def _qkv_layer(n_kv):
    from vllm.model_executor.layers.linear import QKVParallelLinear

    return QKVParallelLinear(
        hidden_size=_HIDDEN,
        head_size=_PADDED,
        total_num_heads=_N_HEADS,
        total_num_kv_heads=n_kv,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )


@pytest.mark.parametrize("n_kv", [_N_KV, _N_HEADS])
def test_padded_fused_qkv_loads_like_separate_projections(tp_group, n_kv):
    """A fused load through vLLM's own loader must match loading q, k and v separately."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(n * _ORIG, _HIDDEN, dtype=torch.float16) for n in (_N_HEADS, n_kv, n_kv))

    def pad(proj, w):
        return _pad_weight(f"model.layers.0.self_attn.{proj}", w, _N_HEADS, n_kv, _ORIG, _PADDED)

    fused_layer, split_layer = _qkv_layer(n_kv), _qkv_layer(n_kv)
    fused_layer.weight.weight_loader(
        fused_layer.weight, pad("qkv_proj.weight", torch.cat([q, k, v]))
    )
    for shard, w in (("q", q), ("k", k), ("v", v)):
        split_layer.weight.weight_loader(split_layer.weight, pad(f"{shard}_proj.weight", w), shard)

    assert torch.equal(fused_layer.weight.data, split_layer.weight.data)
    assert fused_layer.weight.data.view(-1, _PADDED, _HIDDEN)[:, _ORIG:].eq(0).any()
