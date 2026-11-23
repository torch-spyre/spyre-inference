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
from vllm.model_executor.models.bert import BertModel, BertSelfAttention
from vllm.utils.torch_utils import set_default_torch_dtype

from spyre_inference.models import _ADAPTED_MODULES, spyre_models
from spyre_inference.models.bert import SpyreSplitQKVMixin, split_bert_qkv


class _Capture(nn.Module):
    def forward(self, q, k, v):
        return q, k, v


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
