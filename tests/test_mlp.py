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
    """MergedColumnParallelLinear (gate_up_proj) fused output on Spyre
    matches upstream CPU F.linear.

    MergedColumnParallelLinear runs the upstream class unchanged: the fused
    ``[..., 2*d]`` output feeds straight into ``SpyreSiluAndMul``, which
    slices gate/up on-device via indirect access under torch.compile.
    """

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

    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    layer = layer.to("spyre")
    gate_up, bias = layer(x.to("spyre"))
    assert bias is None
    assert isinstance(gate_up, torch.Tensor)
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
    """Fused qkv_proj on Spyre produces a contiguous fused tensor matching
    the CPU reference.  The Q/K/V slice+clone is compiled separately by the
    model runner (``_patch_attention_qkv_splits``) — this test only validates
    the projection itself.
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

    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    layer = layer.to("spyre")
    result, bias = layer(x.to("spyre"))
    assert bias is None
    assert isinstance(result, torch.Tensor)

    torch.testing.assert_close(result.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("input_size,output_size", [(128, 64), (256, 128), (1024, 512)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_row_parallel_matches_reference(tp_group, num_tokens, input_size, output_size, use_bias):
    """RowParallelLinear (down_proj) output on Spyre matches upstream CPU F.linear.

    RowParallel is not un-fused and needs no Spyre subclass: its unquantized
    apply() is already plain F.linear on Spyre.
    """
    from vllm.model_executor.layers.linear import RowParallelLinear

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

    # torch.empty() leaves memory uninitialised (may contain NaN in float16);
    # fill with small random values so the comparison is meaningful.
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.zero_()

    torch.manual_seed(1)
    x = torch.randn(num_tokens, input_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    layer = layer.to("spyre")
    actual, _ = layer(x.to("spyre"))

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_qkv_oot_registration(tp_group):
    """QKVParallelLinear is swapped for the Spyre OOT subclass.

    Merged/Row parallel linears are intentionally NOT subclassed: unquantized
    apply() on Spyre is already plain F.linear.  QKV keeps a subclass to
    assert the ``gather_output=False`` invariant; the compiled slice+clone
    that replaces ``qkv.split()`` is applied at the model-runner level
    (``_patch_attention_qkv_splits``).
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear

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
