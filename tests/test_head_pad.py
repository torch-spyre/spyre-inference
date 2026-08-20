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
