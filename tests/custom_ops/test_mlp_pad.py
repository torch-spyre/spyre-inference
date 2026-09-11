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
    install_mlp_pad_weight_loader,
    original_intermediate_size,
    verify_padded_intermediate_size,
    width_multipliers,
)

_ORIG, _PADDED, _HIDDEN = 160, 192, 64
# The multipliers a use_double_wide_mlp config yields: its KV-shared layers are 2x wide.
_DOUBLE_WIDE = (1, 2)


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


@pytest.mark.parametrize("proj", ["gate_proj", "up_proj"])
def test_pad_weight_end_pads_double_wide_rows_with_zeros(proj):
    width = 2 * _ORIG
    padded_width = 2 * _PADDED
    w = torch.arange(1.0, width * _HIDDEN + 1).reshape(width, _HIDDEN)

    out = _pad_weight(f"layers.0.mlp.{proj}.weight", w, _ORIG, _PADDED, _DOUBLE_WIDE)

    assert out.shape == (padded_width, _HIDDEN)
    assert torch.equal(out[:width], w)
    assert not out[width:].any()


def test_pad_weight_splits_and_pads_a_double_wide_fused_projection():
    width = 2 * _ORIG
    padded_width = 2 * _PADDED
    w = torch.arange(1.0, 2 * width * _HIDDEN + 1).reshape(2 * width, _HIDDEN)

    out = _pad_weight("layers.0.mlp.gate_up_proj.weight", w, _ORIG, _PADDED, _DOUBLE_WIDE)

    assert out.shape == (2 * padded_width, _HIDDEN)
    gate, up = out.split([padded_width, padded_width])
    gate_src, up_src = w.split([width, width])
    assert torch.equal(gate[:width], gate_src)
    assert torch.equal(up[:width], up_src)
    assert not gate[width:].any()
    assert not up[width:].any()


def test_pad_weight_end_pads_double_wide_down_cols_with_zeros():
    width = 2 * _ORIG
    padded_width = 2 * _PADDED
    w = torch.arange(1.0, _HIDDEN * width + 1).reshape(_HIDDEN, width)

    out = _pad_weight("layers.0.mlp.down_proj.weight", w, _ORIG, _PADDED, _DOUBLE_WIDE)

    assert out.shape == (_HIDDEN, padded_width)
    assert torch.equal(out[:, :width], w)
    assert not out[:, width:].any()


def test_pad_weight_leaves_a_double_wide_shape_alone_without_the_config_flag():
    """Only a use_double_wide_mlp model has 2x MLPs; elsewhere that shape is not an MLP."""
    w = torch.arange(1.0, 2 * _ORIG * _HIDDEN + 1).reshape(2 * _ORIG, _HIDDEN)

    assert torch.equal(_pad_weight("layers.0.mlp.experts.3.gate_proj.weight", w, _ORIG, _PADDED), w)


def test_width_multipliers_follow_the_double_wide_mlp_flag():
    assert width_multipliers(SimpleNamespace()) == (1,)
    assert width_multipliers(SimpleNamespace(use_double_wide_mlp=False)) == (1,)
    assert width_multipliers(SimpleNamespace(use_double_wide_mlp=True)) == _DOUBLE_WIDE


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


def test_verify_accepts_a_double_wide_padded_down_proj():
    hf_config = SimpleNamespace(
        intermediate_size=_PADDED,
        _spyre_orig_intermediate_size=_ORIG,
        use_double_wide_mlp=True,
    )

    verify_padded_intermediate_size(_model_with_down_proj(2 * _PADDED), hf_config)


def test_verify_rejects_a_double_wide_down_proj_without_the_config_flag():
    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    with pytest.raises(RuntimeError, match="would load truncated"):
        verify_padded_intermediate_size(_model_with_down_proj(2 * _PADDED), hf_config)


def test_verify_rejects_a_model_with_no_down_proj_at_all():
    """An fc1/fc2 MLP is never widened, so finding no down_proj cannot read as success."""
    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    with pytest.raises(RuntimeError, match="no down_proj module"):
        verify_padded_intermediate_size(torch.nn.Linear(_HIDDEN, _ORIG), hf_config)


def test_verify_noop_without_padding():
    verify_padded_intermediate_size(
        _model_with_down_proj(_ORIG), SimpleNamespace(intermediate_size=_ORIG)
    )


def test_install_rejects_a_loader_that_cannot_pad_weights():
    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    with pytest.raises(NotImplementedError, match="get_all_weights.*object is unsupported"):
        install_mlp_pad_weight_loader(object(), hf_config)


def test_install_allows_an_unpadded_config_with_any_loader():
    install_mlp_pad_weight_loader(object(), SimpleNamespace(intermediate_size=_ORIG))


def test_install_allows_dummy_weights_with_padding():
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    hf_config = SimpleNamespace(intermediate_size=_PADDED, _spyre_orig_intermediate_size=_ORIG)

    install_mlp_pad_weight_loader(DummyModelLoader(LoadConfig(load_format="dummy")), hf_config)


def _config_stub(*, tp=1, **fields):
    """Just what ``_maybe_pad_intermediate_size`` reads off a VllmConfig."""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**fields)),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.mark.parametrize(
    ("tp", "fields", "expected"),
    [
        # Aligned at TP=1, but a TP=2 shard of 2112 lands mid-stick, so 2176 it is.
        (1, {"intermediate_size": 2112, "hidden_activation": "gelu_pytorch_tanh"}, None),
        (2, {"intermediate_size": 2112, "hidden_activation": "gelu_pytorch_tanh"}, 2176),
        (1, {"intermediate_size": 160, "hidden_act": "silu"}, 192),
        (2, {"intermediate_size": 160, "hidden_act": "silu"}, 256),
        # A MoE with its own expert width: only the dense MLP is widened here.
        (
            2,
            {
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_experts": 128,
                "hidden_activation": "gelu_pytorch_tanh",
            },
            2176,
        ),
        # A MoE that sizes its experts from intermediate_size would load them truncated.
        (2, {"intermediate_size": 2112, "num_experts": 8, "hidden_act": "silu"}, None),
        # Padding is only provably inert for a gated MLP.
        (2, {"intermediate_size": 2112, "hidden_act": "relu"}, None),
    ],
)
def test_platform_aligns_intermediate_size_to_the_per_rank_shard(tp, fields, expected):
    from spyre_inference.platform import TorchSpyrePlatform

    config = _config_stub(tp=tp, **fields)
    text_config = config.model_config.hf_text_config
    TorchSpyrePlatform._maybe_pad_intermediate_size(config)

    assert text_config.intermediate_size == (expected or fields["intermediate_size"])
    original = fields["intermediate_size"] if expected else None
    assert original_intermediate_size(text_config) == original


def test_platform_rejects_per_layer_intermediate_sizes():
    from spyre_inference.platform import TorchSpyrePlatform

    config = _config_stub(
        tp=2,
        intermediate_size=[160, 192],
        hidden_activation="gelu_pytorch_tanh",
    )

    with pytest.raises(NotImplementedError, match="per-layer intermediate_size values"):
        TorchSpyrePlatform._maybe_pad_intermediate_size(config)
