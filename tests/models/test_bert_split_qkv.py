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

"""CPU-only."""

import importlib

import pytest
import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.models.bert import BertAttention, BertModel, BertSelfAttention
from vllm.utils.torch_utils import set_default_torch_dtype

from spyre_inference.models import _ADAPTED_MODULES, bert, spyre_models
from spyre_inference.models.bert import SpyreSplitQKVMixin, grouped_rectangles, split_bert_qkv
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_key_pad_mask,
)

WIDTH, EXTENT, HIDDEN, HEADS = 2, 64, 128, 4


class _Capture(nn.Module):
    def forward(self, q, k, v):
        return q, k, v


class _RectAttention(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.impl = type("Impl", (), {"scale": head_dim**-0.5})()

    def forward(self, q, k, v):
        h, d = self.heads, self.head_dim
        a = _encoder_rect_kernel(
            q.view(-1, h, d),
            k.view(-1, h, d),
            v.view(-1, h, d),
            encoder_grid().mask,
            d**-0.5,
            WIDTH,
            EXTENT,
            h,
            h,
            d,
        )
        return a.reshape(-1, h * d)


def _self_attention(hidden_size, num_heads, bias):
    attn = BertSelfAttention(hidden_size=hidden_size, num_attention_heads=num_heads)
    proj = attn.qkv_proj
    g = torch.Generator().manual_seed(0)
    proj.weight.data.copy_(torch.randn(proj.weight.shape, generator=g))
    if bias:
        proj.bias.data.copy_(torch.randn(proj.bias.shape, generator=g))
    else:
        proj.register_parameter("bias", None)
    proj.quant_method.process_weights_after_loading(proj)
    attn.attn = _Capture()
    return attn


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_split_matches_the_fused_projection(default_vllm_config, tp_group, bias):
    attn = _self_attention(hidden_size=128, num_heads=2, bias=bias)
    x = torch.randn(5, 128, generator=torch.Generator().manual_seed(1))
    proj = attn.qkv_proj
    qkv, _ = proj(x)
    fused = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    split_bert_qkv(nn.Sequential(attn))
    split_bert_qkv(nn.Sequential(attn))

    assert attn.qkv_proj is proj
    assert not hasattr(proj, "weight")
    assert proj.q_weight_t.shape == (128, attn.q_size)
    assert all(getattr(proj, f"{n}_weight_t").is_contiguous() for n in ("q", "k", "v"))
    assert (proj.bias is not None) == bias
    assert all((getattr(proj, f"{n}_bias") is not None) == bias for n in ("q", "k", "v"))
    for got, want in zip(attn(x), fused):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class _BertCheckpointModel(nn.Module):
    hf_to_vllm_mapper = BertModel.hf_to_vllm_mapper
    load_weights = BertModel.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = BertSelfAttention(
            hidden_size=128, num_attention_heads=2, prefix=f"{prefix}.attention.self"
        )
        self.attention.self.attn = _Capture()

    def forward(self, hidden_states):
        return self.attention.self(hidden_states)


class _ReloadableBert(SpyreSplitQKVMixin, _BertCheckpointModel):
    pass


@torch.no_grad()
def test_checkpoint_reload_updates_split_weights_and_bias_in_place(
    default_vllm_config, tp_group, caplog
):
    config = get_current_vllm_config()
    g = torch.Generator().manual_seed(2)
    original = {
        f"attention.self.{name}.{suffix}": torch.randn(shape, generator=g, dtype=torch.float16)
        for name in ("query", "key", "value")
        for suffix, shape in (("weight", (128, 128)), ("bias", (128,)))
    }
    changed = {name: -value + 0.125 for name, value in original.items()}
    x = torch.randn(5, 128, generator=g, dtype=torch.float16)

    with set_default_torch_dtype(torch.float16):
        model = initialize_model(config, model_class=_ReloadableBert)
    proj = model.attention.self.qkv_proj
    model.load_weights((name, value.clone()) for name, value in original.items())
    # Exercise the real model-level hook, after the per-layer transpose.
    process_weights_after_loading(model, config.model_config, torch.device("cpu"))
    parameters = dict(proj.named_parameters(recurse=False))
    assert parameters.keys() == {
        "q_weight_t",
        "k_weight_t",
        "v_weight_t",
        "q_bias",
        "k_bias",
        "v_bias",
        "bias",
    }
    pointers = {name: param.data_ptr() for name, param in parameters.items()}

    for iteration, checkpoint in enumerate((original, changed, original)):
        if iteration:
            reload_weights(
                config,
                model,
                weights_iterator=((name, value.clone()) for name, value in checkpoint.items()),
                weights_path=None,
                is_checkpoint_format=True,
            )
        assert model.attention.self.qkv_proj is proj
        assert not hasattr(proj, "weight")
        assert dict(proj.named_parameters(recurse=False)).keys() == parameters.keys()
        for name, param in parameters.items():
            assert getattr(proj, name) is param
            assert param.data_ptr() == pointers[name]
            assert param.device.type == "cpu"

        outputs = model(x)
        assert len(outputs) == 3
        for part, name, bias, output in zip(
            ("q", "k", "v"), ("query", "key", "value"), proj.bias.split(128), outputs
        ):
            weight_t = checkpoint[f"attention.self.{name}.weight"].t().contiguous()
            expected_bias = checkpoint[f"attention.self.{name}.bias"]
            torch.testing.assert_close(getattr(proj, f"{part}_weight_t"), weight_t, rtol=0, atol=0)
            torch.testing.assert_close(getattr(proj, f"{part}_bias"), expected_bias, rtol=0, atol=0)
            torch.testing.assert_close(bias, expected_bias, rtol=0, atol=0)
            torch.testing.assert_close(output, x @ weight_t + expected_bias, rtol=0, atol=0)
    assert "Following weights were not loaded from checkpoint" not in caplog.text


@pytest.mark.parametrize(
    "target",
    sorted(
        target
        for target in set(spyre_models().values())
        if target.partition(":")[0].rpartition(".")[2] in _ADAPTED_MODULES
    ),
)
def test_every_bert_family_model_splits_qkv(target):
    module, _, name = target.partition(":")
    assert issubclass(getattr(importlib.import_module(module), name), SpyreSplitQKVMixin)


def _attention(prefix, rects):
    attn = BertAttention(
        hidden_size=HIDDEN, num_attention_heads=HEADS, layer_norm_eps=1e-5, prefix=prefix
    )
    g = torch.Generator().manual_seed(0)
    for proj in (attn.self.qkv_proj, attn.output.dense):
        proj.weight.data.copy_(torch.randn(proj.weight.shape, generator=g) * 0.1)
        proj.bias.data.copy_(torch.randn(proj.bias.shape, generator=g) * 0.1)
        proj.quant_method.process_weights_after_loading(proj)
    attn.self.attn = _RectAttention(HEADS, HIDDEN // HEADS)
    attn.spyre_group_rects = rects
    split_bert_qkv(nn.Sequential(attn))
    return attn


@pytest.mark.parametrize("inline", [False, True], ids=["fallback", "rectangle"])
def test_head_groups_match_one_group(default_vllm_config, tp_group, monkeypatch, inline):
    x = torch.randn(WIDTH * EXTENT, HIDDEN, generator=torch.Generator().manual_seed(1))
    mask = encoder_key_pad_mask(EXTENT, [EXTENT, EXTENT // 2], torch.float32, False)
    encoder_grid().publish(mask if inline else None)
    monkeypatch.setattr(bert, "encoder_inline_active", lambda: inline)
    try:
        one = _attention("one", frozenset())
        two = _attention("two", frozenset({(WIDTH, EXTENT)}))

        assert hasattr(two.self.qkv_proj, "q_weight_t_1")
        assert not hasattr(one.self.qkv_proj, "q_weight_t_1")
        torch.testing.assert_close(two(x), one(x), rtol=1e-5, atol=1e-5)
    finally:
        encoder_grid().publish(None)


class _BertAttentionCheckpointModel(nn.Module):
    hf_to_vllm_mapper = BertModel.hf_to_vllm_mapper
    load_weights = BertModel.load_weights

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.attention = BertAttention(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            layer_norm_eps=1e-5,
            prefix=f"{prefix}.attention",
        )


class _GroupedBert(SpyreSplitQKVMixin, _BertAttentionCheckpointModel):
    pass


@torch.no_grad()
def test_checkpoint_reload_updates_head_groups_in_place(default_vllm_config, tp_group, caplog):
    config = get_current_vllm_config()
    g = torch.Generator().manual_seed(3)
    names = [f"attention.self.{n}" for n in ("query", "key", "value")] + ["attention.output.dense"]
    original = {
        f"{name}.{suffix}": torch.randn(shape, generator=g, dtype=torch.float16)
        for name in names
        for suffix, shape in (("weight", (HIDDEN, HIDDEN)), ("bias", (HIDDEN,)))
    }
    original |= {
        f"attention.output.LayerNorm.{n}": torch.randn(HIDDEN, generator=g, dtype=torch.float16)
        for n in ("weight", "bias")
    }
    changed = {name: -value + 0.125 for name, value in original.items()}

    with set_default_torch_dtype(torch.float16):
        model = initialize_model(config, model_class=_GroupedBert)
    model.attention.spyre_group_rects = frozenset({(WIDTH, EXTENT)})
    model.load_weights((name, value.clone()) for name, value in original.items())
    process_weights_after_loading(model, config.model_config, torch.device("cpu"))
    proj, dense = model.attention.self.qkv_proj, model.attention.output.dense
    grouped = {
        name: param
        for layer in (proj, dense)
        for name, param in layer.named_parameters(recurse=False)
        if name[-2:] in ("_0", "_1")
    }
    assert len(grouped) == 3 * 2 * 2 + 2
    pointers = {name: param.data_ptr() for name, param in grouped.items()}

    reload_weights(
        config,
        model,
        weights_iterator=((name, value.clone()) for name, value in changed.items()),
        weights_path=None,
        is_checkpoint_format=True,
    )
    half = HIDDEN // 2
    for part, name in zip(("q", "k", "v"), ("query", "key", "value")):
        weight_t = changed[f"attention.self.{name}.weight"].t()
        bias = changed[f"attention.self.{name}.bias"]
        for gi in range(2):
            cols = slice(gi * half, (gi + 1) * half)
            torch.testing.assert_close(getattr(proj, f"{part}_weight_t_{gi}"), weight_t[:, cols])
            torch.testing.assert_close(getattr(proj, f"{part}_bias_{gi}"), bias[cols])
    wo_t = changed["attention.output.dense.weight"].t()
    for gi in range(2):
        torch.testing.assert_close(
            getattr(dense, f"weight_t_{gi}"), wo_t[gi * half : (gi + 1) * half]
        )
    for name, param in grouped.items():
        layer = proj if hasattr(proj, name) else dense
        assert getattr(layer, name) is param and param.data_ptr() == pointers[name]
    assert "Following weights were not loaded from checkpoint" not in caplog.text


def test_planner_groups_only_kv_tiled_rectangles():
    rects = [(64, 32), (128, 16), (256, 8), (512, 4)]
    assert grouped_rectangles(rects, 16, 16, 64) == {(4, 512)}  # e5-large, bge-reranker-large
    assert grouped_rectangles(rects, 12, 12, 64) == set()  # granite-embedding
    assert grouped_rectangles([(64, 8), (128, 4)], 16, 16, 64) == set()  # all-roberta, L 128
