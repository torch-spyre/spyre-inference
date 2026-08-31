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
    """A fused gate_up_proj on Spyre matches the upstream CPU F.linear."""
    import torch.nn as nn
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    dtype = torch.float16
    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[intermediate_size, intermediate_size],
                bias=use_bias,
                params_dtype=dtype,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.act_fn = SiluAndMul()

    mlp = MLP()
    layer = mlp.gate_up_proj

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    # Store the transposed weight, as the loader would, before moving to device.
    layer.quant_method.process_weights_after_loading(layer)

    mlp = mlp.to("spyre")
    gate_up, bias = layer(x.to("spyre"))
    assert bias is None
    assert gate_up.shape == (num_tokens, 2 * intermediate_size)

    torch.testing.assert_close(gate_up.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


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
    """A fused qkv_proj on Spyre matches the upstream F.linear, and the plain
    `qkv.split(...)` idiom yields correct q/k/v on-device.

    The fused weight is stored transposed (Wᵀ) by SpyreUnquantizedLinearMethod;
    the layer returns the whole fused output, which the model splits downstream.
    The on-device split works directly (torch-spyre storage-offset fix) — no
    CPU-side unfusing.
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

    # Transpose the weight to [in, out], as the loader would.
    layer.quant_method.process_weights_after_loading(layer)

    layer = layer.to("spyre")
    qkv, bias = layer(x.to("spyre"))
    assert bias is None

    # Exercise the unmodified downstream idiom on-device.
    q_size = num_heads * head_size
    kv_size = num_kv_heads * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    actual = torch.cat([q, k, v], dim=-1)

    assert q.shape == (num_tokens, q_size)
    assert k.shape == (num_tokens, kv_size)
    assert v.shape == (num_tokens, kv_size)

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("input_size,output_size", [(128, 64), (256, 128), (1024, 512)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_row_parallel_matches_reference(tp_group, num_tokens, input_size, output_size, use_bias):
    """RowParallelLinear (down_proj) output on Spyre matches upstream CPU F.linear.

    RowParallel is swapped for its OOT subclass, which stores the weight
    transposed (Wᵀ) so the forward GEMM is the Spyre-fast `x @ A`.
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

    # Store the transposed weight, as the loader would, before moving to device.
    layer.quant_method.process_weights_after_loading(layer)

    layer = layer.to("spyre")
    actual, _ = layer(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize(
    "input_size,output_size",
    [
        pytest.param(
            128,
            1,
            marks=pytest.mark.xfail(
                reason="torch-spyre batchmatmul cannot restickify an out=1 output dim "
                "('cannot restickify any input layout of x to carry x_var=d1'); fails "
                "in both eager and compile. Padding out=1->2 works, but the fix belongs "
                "in torch-spyre. Tracked upstream.",
                strict=True,
            ),
        ),
        (256, 8),
        (1024, 512),
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_replicated_matches_reference(tp_group, num_tokens, input_size, output_size, use_bias):
    """ReplicatedLinear output on Spyre matches upstream CPU F.linear.

    Reaches Spyre as the `score` head of a sequence-classification model
    (`as_seq_cls_model` in vLLM's model adapters), which is why the tiny
    `output_size=1` case is covered: a single-column weight still has to survive
    the `[out, in]` → `[in, out]` transpose. That out=1 case is currently xfail —
    torch-spyre's batchmatmul cannot stickify a size-1 output dimension.
    """
    from vllm.model_executor.layers.linear import ReplicatedLinear

    from spyre_inference.custom_ops.linear import SpyreReplicatedLinear

    dtype = torch.float16
    torch.manual_seed(0)
    layer = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=use_bias,
        params_dtype=dtype,
        quant_config=None,
        prefix="score",
    )
    assert isinstance(layer, SpyreReplicatedLinear)

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, input_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    # Store the transposed weight, as the loader would, before moving to device.
    layer.quant_method.process_weights_after_loading(layer)

    layer = layer.to("spyre")
    actual, _ = layer(x.to("spyre"))

    assert actual.shape == (num_tokens, output_size)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_linear_oot_registration(tp_group):
    """Each unquantized parallel-linear class is swapped for its Spyre OOT
    subclass, all sharing the transposed-weight method. QKV additionally
    asserts the gather_output=False invariant.
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    from spyre_inference.custom_ops.linear import (
        SpyreColumnParallelLinear,
        SpyreMergedColumnParallelLinear,
        SpyreQKVParallelLinear,
        SpyreReplicatedLinear,
        SpyreRowParallelLinear,
        SpyreUnquantizedLinearMethod,
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
    assert isinstance(qkv, SpyreQKVParallelLinear)

    gate_up = MergedColumnParallelLinear(
        input_size=64,
        output_sizes=[128, 128],
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="gate_up_proj",
    )
    assert isinstance(gate_up, SpyreMergedColumnParallelLinear)

    down = RowParallelLinear(
        input_size=128,
        output_size=64,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="down_proj",
    )
    assert isinstance(down, SpyreRowParallelLinear)

    col = ColumnParallelLinear(
        input_size=64,
        output_size=128,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        disable_tp=True,
        prefix="col",
    )
    assert isinstance(col, SpyreColumnParallelLinear)

    # ReplicatedLinear takes no disable_tp (it is replicated by definition).
    score = ReplicatedLinear(
        input_size=64,
        output_size=2,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        prefix="score",
    )
    assert isinstance(score, SpyreReplicatedLinear)

    for layer in (qkv, gate_up, down, col, score):
        assert isinstance(layer.quant_method, SpyreUnquantizedLinearMethod)
