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

"""
Test MLP linear layer correctness against upstream CPU reference implementations.
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("hidden_size,intermediate_size", [(64, 128), (128, 256), (512, 1024)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_merged_column_matches_reference(
    tp_group, num_tokens, hidden_size, intermediate_size, use_bias
):
    """MergedColumnParallelLinear output on Spyre matches upstream CPU F.linear."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    dtype = torch.float16
    torch.manual_seed(0)
    layer = MergedColumnParallelLinear(
        input_size=hidden_size,
        output_sizes=[intermediate_size, intermediate_size],
        bias=use_bias,
        params_dtype=dtype,
        quant_config=None,
        disable_tp=True,
        prefix="gate_up_proj",
    )

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    layer = layer.to("spyre")
    actual, _ = layer(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size",
    [
        (8, 8, 64),  # MHA
        (8, 2, 64),  # GQA
        (8, 1, 64),  # MQA
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_qkv_matches_reference(tp_group, num_tokens, num_heads, num_kv_heads, head_size, use_bias):
    """SpyreQKVParallelLinear output matches upstream CPU F.linear.

    SpyreQKVParallelLinear.forward() does a D2H convert on the result before
    returning so downstream .split() doesn't hit strided-tensor issues on
    Spyre.  This test exercises that path.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

    dtype = torch.float16
    hidden_size = num_heads * head_size
    torch.manual_seed(0)
    layer = QKVParallelLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=num_heads,
        total_num_kv_heads=num_kv_heads,
        bias=use_bias,
        params_dtype=dtype,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )
    assert isinstance(layer, SpyreQKVParallelLinear)

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    layer = layer.to("spyre")
    actual, _ = layer(x.to("spyre"))

    # SpyreQKVParallelLinear.forward() does a D2H on the result before returning.
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("input_size,output_size", [(128, 64), (256, 128), (1024, 512)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_row_parallel_matches_reference(tp_group, num_tokens, input_size, output_size, use_bias):
    """SpyreRowParallelLinear output matches upstream CPU F.linear.

    SpyreRowParallelLinear.forward() converts input_ to the weight device
    before the GEMM so CPU inputs (e.g. from GraniteAttention) are moved to
    Spyre automatically.  This test exercises that H2D path on CPU.
    """
    from vllm.model_executor.layers.linear import RowParallelLinear
    from spyre_inference.custom_ops.linear import SpyreRowParallelLinear

    dtype = torch.float16
    torch.manual_seed(0)
    layer = RowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=use_bias,
        params_dtype=dtype,
        quant_config=None,
        reduce_results=True,
        disable_tp=True,
        prefix="down_proj",
    )
    assert isinstance(layer, SpyreRowParallelLinear)

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, input_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    # CPU input + Spyre weight: SpyreRowParallelLinear.forward() H2Ds the input
    # to match the weight device.
    layer = layer.to("spyre")
    actual, _ = layer(x)

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre cannot use a strided tensor as the source of an indexed scatter. "
        "After qkv.split(), v is a strided view; Attention.forward() then calls "
        "v.view(-1, num_kv_heads, head_size) which produces a non-contiguous 3D "
        "tensor (strides [Q+2K, head_size, 1] instead of [K, head_size, 1]). "
        "SpyreAttentionImpl._write_to_kv_cache then does "
        "kv_cache[block_indices, 1, block_offsets] = value with that strided source. "
        "SpyreQKVParallelLinear.forward() works around this with a D2H before "
        "returning so the split and view run on CPU. "
        "When this flips to passing, remove the D2H workaround from "
        "SpyreQKVParallelLinear.forward()."
    ),
)
def test_spyre_strided_scatter_source():
    """Probe: Spyre accepts a non-contiguous tensor as a scatter-write source.

    The failure path when D2H is removed from SpyreQKVParallelLinear:
      1. qkv.split()        → strided 2D Spyre views
      2. v.view(-1, H, D)   → non-contiguous 3D Spyre tensor (Attention.forward)
      3. kv_cache[idx] = v  → scatter write with strided source (_write_to_kv_cache)
    """
    device = torch.device("spyre")
    dtype = torch.float16
    num_tokens = 16
    num_heads, num_kv_heads, head_size = 8, 2, 64
    q_size, kv_size = num_heads * head_size, num_kv_heads * head_size

    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device=device)
    _, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # Replicate what Attention.forward() does before calling impl.forward()
    v = v.view(-1, num_kv_heads, head_size)

    # Replicate _write_to_kv_cache's scatter write
    num_blocks, block_size = 4, 8
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    block_indices = torch.zeros(num_tokens, dtype=torch.long, device=device)
    block_offsets = torch.arange(num_tokens, dtype=torch.long, device=device) % block_size
    kv_cache[block_indices, 1, block_offsets] = v


@pytest.mark.mlp
def test_linear_oot_registration(tp_group):
    """Verify OOT class swaps for the QKV and Row parallel linear layers."""
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from spyre_inference.custom_ops.linear import (
        SpyreQKVParallelLinear,
        SpyreRowParallelLinear,
    )

    gate_up = MergedColumnParallelLinear(
        input_size=64,
        output_sizes=[128, 128],
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="gate_up_proj",
    )
    qkv = QKVParallelLinear(
        hidden_size=64,
        head_size=8,
        total_num_heads=8,
        total_num_kv_heads=8,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="qkv_proj",
    )
    down = RowParallelLinear(
        input_size=128,
        output_size=64,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        reduce_results=True,
        disable_tp=True,
        prefix="down_proj",
    )

    assert isinstance(qkv, SpyreQKVParallelLinear)
    assert isinstance(down, SpyreRowParallelLinear)

    torch.manual_seed(0)
    x_col = torch.randn(4, 64, dtype=torch.float16)
    out_gate_up, _ = gate_up(x_col)
    assert out_gate_up.shape == (4, 256)

    out_qkv, _ = qkv(x_col)
    # MHA: q_size=num_heads*head_size=64, k_size=kv_size=64, v_size=kv_size=64
    q_size = 8 * 8
    kv_size = 8 * 8
    assert out_qkv.shape == (4, q_size + kv_size + kv_size)

    x_row = torch.randn(4, 128, dtype=torch.float16)
    out_down, _ = down(x_row)
    assert out_down.shape == (4, 64)
