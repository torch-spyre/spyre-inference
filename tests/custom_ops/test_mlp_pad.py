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

"""MLP intermediate_size padding: end-pad layout, numerical inertness, guard.

CPU-only: these cover the per-tensor padding and the post-build guard, neither of
which touches the device. The end-to-end check that a padded model decodes
correctly is test_padded_head_dim_and_intermediate_size_generate in
test_vllm_spyre_next.py (qwrt/Swedish0.1M), which runs on Spyre hardware.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.mlp_pad import (
    _pad_weight,
    verify_padded_intermediate_size,
)

_ORIG, _PADDED, _HIDDEN = 160, 192, 64


@pytest.mark.parametrize("proj", ["gate_proj", "up_proj"])
def test_pad_weight_end_pads_rows_with_zeros(proj):
    w = torch.arange(1.0, _ORIG * _HIDDEN + 1).reshape(_ORIG, _HIDDEN)

    out = _pad_weight(f"layers.0.mlp.{proj}.weight", w, _ORIG, _PADDED)

    assert out.shape == (_PADDED, _HIDDEN)
    assert torch.equal(out[:_ORIG], w)
    assert not out[_ORIG:].any()


def test_pad_weight_end_pads_down_cols_with_zeros():
    w = torch.arange(1.0, _HIDDEN * _ORIG + 1).reshape(_HIDDEN, _ORIG)

    out = _pad_weight("layers.0.mlp.down_proj.weight", w, _ORIG, _PADDED)

    assert out.shape == (_HIDDEN, _PADDED)
    assert torch.equal(out[:, :_ORIG], w)
    assert not out[:, _ORIG:].any()


def test_pad_weight_splits_and_pads_a_fused_gate_up_projection():
    w = torch.arange(1.0, 2 * _ORIG * _HIDDEN + 1).reshape(2 * _ORIG, _HIDDEN)

    out = _pad_weight("layers.0.mlp.gate_up_proj.weight", w, _ORIG, _PADDED)

    assert out.shape == (2 * _PADDED, _HIDDEN)
    gate, up = out.split([_PADDED, _PADDED])
    gate_src, up_src = w.split([_ORIG, _ORIG])
    assert torch.equal(gate[:_ORIG], gate_src)
    assert torch.equal(up[:_ORIG], up_src)
    assert not gate[_ORIG:].any()
    assert not up[_ORIG:].any()


def test_pad_weight_splits_and_pads_a_fused_gate_up_bias():
    """A fused bias must split and end-pad like the weight, not fall through unpadded."""
    b = torch.arange(1.0, 2 * _ORIG + 1)

    out = _pad_weight("layers.0.mlp.gate_up_proj.bias", b, _ORIG, _PADDED)

    assert out.shape == (2 * _PADDED,)
    gate, up = out.split([_PADDED, _PADDED])
    assert torch.equal(gate[:_ORIG], b[:_ORIG])
    assert torch.equal(up[:_ORIG], b[_ORIG:])
    assert not gate[_ORIG:].any()
    assert not up[_ORIG:].any()


def test_pad_weight_leaves_an_already_aligned_width_untouched():
    """A 64-aligned intermediate never gets stashed, so padded==orig is a no-op."""
    w = torch.arange(1.0, _PADDED * _HIDDEN + 1).reshape(_PADDED, _HIDDEN)

    out = _pad_weight("layers.0.mlp.gate_proj.weight", w, _PADDED, _PADDED)

    assert torch.equal(out, w)


def test_pad_weight_leaves_unrelated_tensors_alone():
    w = torch.arange(1.0, _HIDDEN + 1)

    assert torch.equal(_pad_weight("layers.0.self_attn.q_proj.weight", w, _ORIG, _PADDED), w)
    assert torch.equal(_pad_weight("model.embed_tokens.weight", w, _ORIG, _PADDED), w)


def _swiglu(x, gate_w, up_w, down_w):
    """down(silu(gate) * up), the dense SwiGLU MLP forward."""
    return (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T


def test_padding_is_numerically_inert_for_swiglu():
    """Zero-padded intermediate is arithmetically inert for SwiGLU on CPU."""
    torch.manual_seed(0)
    x = torch.randn(16, _HIDDEN)
    gate_w = torch.randn(_ORIG, _HIDDEN)
    up_w = torch.randn(_ORIG, _HIDDEN)
    down_w = torch.randn(_HIDDEN, _ORIG)

    ref = _swiglu(x, gate_w, up_w, down_w)

    gate_p = _pad_weight("mlp.gate_proj.weight", gate_w, _ORIG, _PADDED)
    up_p = _pad_weight("mlp.up_proj.weight", up_w, _ORIG, _PADDED)
    down_p = _pad_weight("mlp.down_proj.weight", down_w, _ORIG, _PADDED)
    out = _swiglu(x, gate_p, up_p, down_p)

    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def _model_with_down_proj(input_size):
    """A model exposing one MLP whose down_proj carries a vLLM linear .input_size."""
    model = torch.nn.Module()
    mlp = torch.nn.Module()
    down = torch.nn.Module()
    down.input_size = input_size
    mlp.add_module("down_proj", down)
    model.add_module("mlp", mlp)
    return model


def test_verify_rejects_a_down_proj_left_at_the_native_width():
    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    with pytest.raises(RuntimeError, match="would load truncated"):
        verify_padded_intermediate_size(_model_with_down_proj(_ORIG), hf_config)


def test_verify_accepts_a_padded_down_proj():
    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    verify_padded_intermediate_size(_model_with_down_proj(_PADDED), hf_config)


def test_verify_noop_without_padding():
    verify_padded_intermediate_size(
        _model_with_down_proj(_ORIG), SimpleNamespace(intermediate_size=_ORIG)
    )
