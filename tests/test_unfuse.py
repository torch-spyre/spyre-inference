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

"""Tests for the model-agnostic weight-unfusing pass (custom_ops/unfuse.py).

These run on CPU (no Spyre device needed): the pass and the QKVSplit
container are pure host-side transformations, and the torch.compile probe
uses the inductor backend on CPU tensors.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_attention_module(num_heads, num_kv_heads, head_size, bias=False):
    """A minimal attention-like module: qkv_proj + the verbatim split idiom."""
    from vllm.model_executor.layers.linear import QKVParallelLinear

    hidden = num_heads * head_size

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = QKVParallelLinear(
                hidden_size=hidden,
                head_size=head_size,
                total_num_heads=num_heads,
                total_num_kv_heads=num_kv_heads,
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="qkv_proj",
            )
            self.q_size = num_heads * head_size
            self.kv_size = num_kv_heads * head_size

        def forward(self, x):
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            return q, k, v

    return Attn()


def _make_mlp_module(hidden, inter, bias=False, with_silu=True):
    """A minimal MLP: gate_up_proj + (optional) SiluAndMul act_fn."""
    from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=hidden,
                output_sizes=[inter, inter],
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.act_fn = SiluAndMul() if with_silu else GeluAndMul()

        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            return self.act_fn(gate_up)

    return MLP()


@pytest.mark.mlp
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size",
    [(8, 8, 64), (8, 2, 64), (8, 1, 64)],
)
def test_qkv_split_returns_correct_slabs(tp_group, num_heads, num_kv_heads, head_size):
    """QKVSplit.split() returns q/k/v that concatenate to the fused output."""
    from spyre_inference.custom_ops.unfuse import QKVSplit, analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(num_heads, num_kv_heads, head_size)
    attn.qkv_proj.weight.data.normal_(std=0.02)

    x = torch.randn(5, num_heads * head_size, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight)

    analyze_and_unfuse(attn)
    assert not hasattr(attn.qkv_proj, "weight")

    qkv, bias = attn.qkv_proj(x)
    assert bias is None
    assert isinstance(qkv, QKVSplit)
    q, k, v = qkv.split([0, 0, 0], dim=-1)  # sizes ignored by design
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_qkv_split_ignores_bias_folding(tp_group):
    """Biases are folded into q/k/v; QKVSplit stays bias-free."""
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(8, 2, 64, bias=True)
    attn.qkv_proj.weight.data.normal_(std=0.02)
    attn.qkv_proj.bias.data.normal_(std=0.02)

    x = torch.randn(3, 8 * 64, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight, attn.qkv_proj.bias)

    analyze_and_unfuse(attn)
    assert not hasattr(attn.qkv_proj, "bias")
    q, k, v = attn.qkv_proj(x)[0].split([0, 0, 0])
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size",
    [(8, 8, 64), (12, 12, 64)],  # chunk(3) only maps onto symmetric (non-GQA) qkv
)
def test_qkv_chunk_returns_correct_slabs(tp_group, num_heads, num_kv_heads, head_size):
    """QKVSplit.chunk(3) returns q/k/v that concatenate to the fused output.

    Mirrors the OPT idiom `q, k, v = qkv.chunk(chunks=3, dim=-1)`.
    """
    from spyre_inference.custom_ops.unfuse import QKVSplit, analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(num_heads, num_kv_heads, head_size)
    attn.qkv_proj.weight.data.normal_(std=0.02)

    x = torch.randn(5, num_heads * head_size, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight)

    analyze_and_unfuse(attn)
    assert not hasattr(attn.qkv_proj, "weight")

    qkv, bias = attn.qkv_proj(x)
    assert bias is None
    assert isinstance(qkv, QKVSplit)
    q, k, v = qkv.chunk(chunks=3, dim=-1)  # chunks/dim ignored by design
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_qkv_split_fails_closed_on_other_access(tp_group):
    """QKVSplit exposes only .split()/.chunk(); other access raises (fail-closed)."""
    from spyre_inference.custom_ops.unfuse import QKVSplit

    c = QKVSplit(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert len(c.split([0, 0, 0])) == 3
    assert len(c.chunk(chunks=3, dim=-1)) == 3
    # A non-3-way chunk is not the qkv idiom we assume: fail closed.
    with pytest.raises(AssertionError):
        c.chunk(chunks=2, dim=-1)
    with pytest.raises(AttributeError):
        c.view(-1)
    with pytest.raises(AttributeError):
        _ = c.shape


@pytest.mark.mlp
def test_merged_unfused_only_with_silu_sibling(tp_group):
    """gate_up_proj is un-fused only when a SiluAndMul sibling is present."""
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    with_silu = _make_mlp_module(64, 128, with_silu=True)
    without_silu = _make_mlp_module(64, 128, with_silu=False)
    with_silu.gate_up_proj.weight.data.normal_(std=0.02)
    without_silu.gate_up_proj.weight.data.normal_(std=0.02)

    analyze_and_unfuse(with_silu)
    analyze_and_unfuse(without_silu)

    # SiluAndMul sibling → un-fused (list return, no fused weight).
    assert not hasattr(with_silu.gate_up_proj, "weight")
    assert hasattr(with_silu.gate_up_proj, "gate_weight")
    # GeluAndMul sibling → left fused (out of scope).
    assert hasattr(without_silu.gate_up_proj, "weight")
    assert not hasattr(without_silu.gate_up_proj, "gate_weight")


@pytest.mark.mlp
def test_merged_list_feeds_silu(tp_group):
    """The un-fused MLP end-to-end matches the fused reference on CPU."""
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    mlp = _make_mlp_module(64, 128, with_silu=True)
    mlp.gate_up_proj.weight.data.normal_(std=0.02)

    x = torch.randn(4, 64, dtype=torch.float16)
    # Fused reference: gate_up then SiluAndMul native.
    fused = F.linear(x, mlp.gate_up_proj.weight)
    d = fused.shape[-1] // 2
    expected = F.silu(fused[..., :d]) * fused[..., d:]

    analyze_and_unfuse(mlp)
    actual = mlp(x)  # gate_up returns [gate, up]; SpyreSiluAndMul consumes it
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_fullgraph_traces_through_unfused(tp_group):
    """torch.compile(fullgraph=True) traces the unmodified split/act idioms
    after un-fusing — the container .split() and list return do not break
    Dynamo. This mirrors the Spyre runtime, which compiles the whole model
    with fullgraph=True.
    """
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _make_attention_module(4, 2, 16)
            self.mlp = _make_mlp_module(4 * 16, 128, with_silu=True)

        def forward(self, x):
            q, k, v = self.attn(x)
            h = torch.cat([q, k, v], dim=-1)[:, : x.shape[-1]]
            return self.mlp(x + h)

    blk = Block()
    blk.attn.qkv_proj.weight.data.normal_(std=0.02)
    blk.mlp.gate_up_proj.weight.data.normal_(std=0.02)
    analyze_and_unfuse(blk)

    x = torch.randn(3, 4 * 16, dtype=torch.float16)
    eager = blk(x)
    compiled = torch.compile(blk, backend="inductor", fullgraph=True, dynamic=False)
    out = compiled(x)
    torch.testing.assert_close(out.float(), eager.float(), atol=1e-2, rtol=1e-2)
