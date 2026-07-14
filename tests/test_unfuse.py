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

These run on CPU (no Spyre device needed): the pass and the SplitQKV
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
def test_qkv_split_returns_correct_parts(tp_group, num_heads, num_kv_heads, head_size):
    """SplitQKV.split() returns q/k/v that concatenate to the fused output."""
    from spyre_inference.custom_ops.unfuse import SplitQKV, analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(num_heads, num_kv_heads, head_size)
    attn.qkv_proj.weight.data.normal_(std=0.02)

    x = torch.randn(5, num_heads * head_size, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight)

    analyze_and_unfuse(attn)
    assert attn.qkv_proj.weight is None

    qkv, bias = attn.qkv_proj(x)
    assert bias is None
    assert isinstance(qkv, SplitQKV)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_qkv_split_ignores_bias_folding(tp_group):
    """Biases are folded into q/k/v; SplitQKV stays bias-free."""
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(8, 2, 64, bias=True)
    attn.qkv_proj.weight.data.normal_(std=0.02)
    attn.qkv_proj.bias.data.normal_(std=0.02)

    x = torch.randn(3, 8 * 64, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight, attn.qkv_proj.bias)

    analyze_and_unfuse(attn)
    assert attn.qkv_proj.bias is None
    q, k, v = attn.qkv_proj(x)[0].split([attn.q_size, attn.kv_size, attn.kv_size])
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size",
    [(8, 8, 64), (12, 12, 64)],  # chunk(3) only maps onto symmetric (non-GQA) qkv
)
def test_qkv_chunk_returns_correct_parts(tp_group, num_heads, num_kv_heads, head_size):
    """SplitQKV.chunk(3) returns q/k/v that concatenate to the fused output.

    Mirrors the OPT idiom `q, k, v = qkv.chunk(chunks=3, dim=-1)`.
    """
    from spyre_inference.custom_ops.unfuse import SplitQKV, analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(num_heads, num_kv_heads, head_size)
    attn.qkv_proj.weight.data.normal_(std=0.02)

    x = torch.randn(5, num_heads * head_size, dtype=torch.float16)
    expected = F.linear(x, attn.qkv_proj.weight)

    analyze_and_unfuse(attn)
    assert attn.qkv_proj.weight is None

    qkv, bias = attn.qkv_proj(x)
    assert bias is None
    assert isinstance(qkv, SplitQKV)
    q, k, v = qkv.chunk(chunks=3, dim=-1)  # chunks/dim ignored by design
    actual = torch.cat([q, k, v], dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_qkv_split_fails_closed_on_other_access(tp_group):
    """SplitQKV exposes only .split()/.chunk(); every other or mismatched
    access raises (fail-closed) so a wrong idiom can never silently return the
    wrong tensors."""
    from spyre_inference.custom_ops.unfuse import SplitQKV

    # q/k/v feature sizes 4/2/2 (GQA-like) on a 2-D [rows, features] tensor.
    q, k, v = torch.zeros(3, 4), torch.zeros(3, 2), torch.zeros(3, 2)
    c = SplitQKV(q, k, v)

    # Correct idioms return the three parts.
    assert len(c.split([4, 2, 2], dim=-1)) == 3
    assert len(c.split([4, 2, 2])) == 3  # default dim=-1
    sym = SplitQKV(torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(3, 2))
    assert len(sym.chunk(chunks=3, dim=-1)) == 3

    # Mismatched split sizes are not our (q, k, v) partition: fail closed.
    with pytest.raises(AssertionError):
        c.split([8, 0, 0])
    # Splitting on a non-last dim is not the qkv idiom.
    with pytest.raises(AssertionError):
        c.split([4, 2, 2], dim=0)
    # A non-3-way chunk is not the qkv idiom.
    with pytest.raises(AssertionError):
        c.chunk(chunks=2, dim=-1)
    # chunk(3) on unequal q/k/v (GQA) cannot map onto equal chunks.
    with pytest.raises(AssertionError):
        c.chunk(chunks=3, dim=-1)
    # __slots__ base: no split/chunk-free attribute access, no stray writes.
    with pytest.raises(AttributeError):
        c.view(-1)
    with pytest.raises(AttributeError):
        _ = c.shape
    with pytest.raises(AttributeError):
        c.extra = 1


@pytest.mark.mlp
def test_split_silu_and_mul_unpacks_to_gate_up(tp_group):
    """SplitSiluAndMul iterates as exactly (gate, up) and exposes nothing else."""
    from spyre_inference.custom_ops.unfuse import SplitSiluAndMul

    gate, up = torch.zeros(2, 4), torch.ones(2, 4)
    proj = SplitSiluAndMul(gate, up)

    g, u = proj  # the SpyreSiluAndMul idiom: `x1, x2 = x`
    assert g is gate and u is up
    assert [t.shape for t in proj] == [gate.shape, up.shape]
    # It is not a plain sequence: no indexing, and __slots__ blocks writes.
    with pytest.raises(TypeError):
        _ = proj[0]
    with pytest.raises(AttributeError):
        proj.extra = 1


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

    # SiluAndMul sibling → un-fused (split parts, no fused weight).
    assert with_silu.gate_up_proj.weight is None
    assert hasattr(with_silu.gate_up_proj, "gate_weight")
    # GeluAndMul sibling → left fused (out of scope).
    assert without_silu.gate_up_proj.weight is not None
    assert not hasattr(without_silu.gate_up_proj, "gate_weight")


@pytest.mark.mlp
def test_quantized_layers_are_left_fused(tp_group):
    """A non-UnquantizedLinearMethod quant_method makes the pass skip the layer.

    Spyre only supports the unquantized path; a quantized QKV/gate-up must be
    left fused (weight untouched, forward unchanged) rather than split apart.
    """
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    attn = _make_attention_module(8, 2, 64)
    mlp = _make_mlp_module(64, 128, with_silu=True)

    # Simulate a quantized layer: any object that is not an
    # UnquantizedLinearMethod trips the `_is_unquantized` guard.
    attn.qkv_proj.quant_method = object()
    mlp.gate_up_proj.quant_method = object()

    analyze_and_unfuse(attn)
    analyze_and_unfuse(mlp)

    # Left fully fused: original weight kept, no per-part params, forward intact.
    assert attn.qkv_proj.weight is not None
    assert not hasattr(attn.qkv_proj, "q_weight")
    assert mlp.gate_up_proj.weight is not None
    assert not hasattr(mlp.gate_up_proj, "gate_weight")


@pytest.mark.mlp
def test_merged_with_non_two_parts_is_left_fused(tp_group):
    """A MergedColumnParallelLinear with != 2 output parts is out of scope.

    The gate/up un-fuse only handles the 2-part (gate, up) case; a 3-part
    merged projection must be left fused.
    """
    import torch.nn as nn

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=64,
                output_sizes=[128, 128, 128],  # three parts, not (gate, up)
                bias=False,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.act_fn = SiluAndMul()

    mlp = MLP()
    assert len(mlp.gate_up_proj.output_partition_sizes) == 3

    analyze_and_unfuse(mlp)

    # Left fused: 3-part projection is not the (gate, up) idiom.
    assert mlp.gate_up_proj.weight is not None
    assert not hasattr(mlp.gate_up_proj, "gate_weight")


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
    actual = mlp(x)  # gate_up returns SplitSiluAndMul; SpyreSiluAndMul consumes it
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_forward_honors_return_bias_false(tp_group):
    """return_bias=False makes the rebound forward return a bare output.

    Mirrors upstream LinearBase.forward, which returns the output alone (not a
    (output, bias) tuple) when return_bias is False.
    """
    import torch.nn as nn

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from spyre_inference.custom_ops.unfuse import SplitSiluAndMul, analyze_and_unfuse

    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=64,
                output_sizes=[128, 128],
                bias=False,
                return_bias=False,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.act_fn = SiluAndMul()

    mlp = MLP()
    mlp.gate_up_proj.weight.data.normal_(std=0.02)
    analyze_and_unfuse(mlp)

    out = mlp.gate_up_proj(torch.randn(4, 64, dtype=torch.float16))
    assert isinstance(out, SplitSiluAndMul)  # bare output, not a (output, bias) tuple


@pytest.mark.mlp
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_qkv_forward_honors_skip_bias_add(tp_group, skip_bias_add):
    """skip_bias_add controls whether bias is folded into the matmul.

    With skip_bias_add=True the bias is NOT added to q/k/v and is instead
    returned separately (the fused bias), matching upstream LinearBase.forward.
    With skip_bias_add=False it is folded and the returned bias is None.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.unfuse import SplitQKV, analyze_and_unfuse

    torch.manual_seed(0)
    layer = QKVParallelLinear(
        hidden_size=8 * 64,
        head_size=64,
        total_num_heads=8,
        total_num_kv_heads=2,
        bias=True,
        skip_bias_add=skip_bias_add,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )
    layer.weight.data.normal_(std=0.02)
    layer.bias.data.normal_(std=0.02)
    saved_bias = layer.bias.data.clone()

    x = torch.randn(5, 8 * 64, dtype=torch.float16)
    # Reference concatenated q/k/v with the bias folded in.
    expected_folded = F.linear(x, layer.weight.data, layer.bias.data)

    analyze_and_unfuse(layer)
    out, out_bias = layer(x)
    assert isinstance(out, SplitQKV)
    q, k, v = out.split([8 * 64, 2 * 64, 2 * 64])
    actual = torch.cat([q, k, v], dim=-1)

    if skip_bias_add:
        # Bias returned separately; matmul is unbiased, so add it back to compare.
        assert out_bias is not None
        torch.testing.assert_close(out_bias.float(), saved_bias.float(), atol=1e-3, rtol=1e-3)
        actual = actual + out_bias
    else:
        assert out_bias is None
    torch.testing.assert_close(actual.float(), expected_folded.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_repr_after_unfuse_does_not_crash(tp_group):
    """repr() on an un-fused layer must not raise.

    The pass clears the fused weight/bias to None (not `del`), so the
    registered-parameter entries survive and LinearBase.extra_repr's
    `self.bias is not None` keeps working; a `del` would raise AttributeError.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

    torch.manual_seed(0)
    layer = QKVParallelLinear(
        hidden_size=8 * 64,
        head_size=64,
        total_num_heads=8,
        total_num_kv_heads=2,
        bias=True,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )
    layer.weight.data.normal_(std=0.02)
    analyze_and_unfuse(layer)

    assert layer.weight is None and layer.bias is None
    repr(layer)  # must not raise


@pytest.mark.mlp
def test_fullgraph_traces_through_unfused(tp_group):
    """torch.compile(fullgraph=True) traces the unmodified split/act idioms
    after un-fusing — the SplitQKV.split() and SplitSiluAndMul unpack do not
    break Dynamo. This mirrors the Spyre runtime, which compiles the whole
    model with fullgraph=True.
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
