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
    """An un-fused gate_up_proj returns [gate, up] whose concatenation
    matches the fused upstream F.linear.

    The model-agnostic pass (analyze_and_unfuse) splits the fused weight on
    CPU and rebinds forward to return the two slabs as a list; a
    MergedColumnParallelLinear is only un-fused when it has a SiluAndMul
    sibling, so we wrap it in a minimal MLP parent.
    """
    import torch.nn as nn

    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from spyre_inference.custom_ops.unfuse import analyze_and_unfuse

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

    # Capture the fused reference BEFORE the pass destructively un-fuses.
    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    # Run the model-agnostic un-fusing pass (weights still on CPU).
    analyze_and_unfuse(mlp)
    assert not hasattr(layer, "weight"), "fused weight should be removed"
    assert hasattr(layer, "gate_weight") and hasattr(layer, "up_weight")

    mlp = mlp.to("spyre")
    gate_up, bias = layer(x.to("spyre"))
    assert bias is None
    assert isinstance(gate_up, list) and len(gate_up) == 2
    actual = torch.cat(gate_up, dim=-1)
    assert actual.shape == (num_tokens, 2 * intermediate_size)

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
    """An un-fused qkv_proj returns a QKVSplit whose (q, k, v) match the
    fused upstream F.linear.

    analyze_and_unfuse splits the fused weight on CPU and rebinds forward to
    return a QKVSplit container; the unmodified `qkv.split(...)` idiom then
    yields three contiguous tensors — no slice on a Spyre tensor.
    """
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear
    from spyre_inference.custom_ops.unfuse import QKVSplit, analyze_and_unfuse

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

    # Capture the fused reference BEFORE the pass destructively un-fuses.
    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    expected = F.linear(x, layer.weight, layer.bias)

    # A bare QKV layer (no parent) still gets un-fused — QKV detection does
    # not require a sibling.
    analyze_and_unfuse(layer)
    assert not hasattr(layer, "weight"), "fused weight should be removed"
    for attr in ("q_weight", "k_weight", "v_weight"):
        assert hasattr(layer, attr), f"missing unfused param {attr}"

    layer = layer.to("spyre")
    qkv, bias = layer(x.to("spyre"))
    assert bias is None
    assert isinstance(qkv, QKVSplit)
    # Exercise the unmodified downstream idiom.
    q_size = num_heads * head_size
    kv_size = num_kv_heads * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    actual = torch.cat([q, k, v], dim=-1)

    assert q.shape == (num_tokens, q_size)
    assert k.shape == (num_tokens, kv_size)
    assert v.shape == (num_tokens, kv_size)
    # Each slab is contiguous on Spyre — no view, no D2H workaround needed.
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre cannot use a strided tensor as the source of an indexed scatter. "
        "Historically this is what forced SpyreQKVParallelLinear to D2H its "
        "result before returning. The current Spyre path side-steps it by "
        "un-fusing the QKV weight after load so q/k/v never arise from a "
        "split() of a Spyre tensor — see analyze_and_unfuse in "
        "custom_ops/unfuse.py. "
        "The probe is kept because the underlying torch-spyre limitation "
        "(strided source for a scatter write) is still real and gates other "
        "rework (e.g. attention's per-token KV scatter)."
    ),
)
def test_spyre_strided_scatter_source():
    """Probe: Spyre accepts a non-contiguous tensor as a scatter-write source.

    The failure path this isolates:
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
def test_qkv_oot_registration(tp_group):
    """QKVParallelLinear is swapped for the Spyre OOT subclass.

    Merged/Row parallel linears are intentionally NOT subclassed: unquantized
    apply() on Spyre is already plain F.linear, and the gate/up + qkv weights
    are handled by analyze_and_unfuse. Only QKV keeps a subclass, to assert
    the gather_output=False invariant.
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
