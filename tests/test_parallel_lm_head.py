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
Test SpyreParallelLMHead custom op correctness against a reference implementation.
"""

import sys

import pytest
import torch
import torch.nn.functional as F


def reference_lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Golden reference: standard F.linear as used by upstream ParallelLMHead."""
    return F.linear(x, weight, bias)


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("vocab_size", [64, 128, 49216, 51200])
@pytest.mark.parametrize("embedding_dim", [64, 128])
def test_spyre_parallel_lm_head_matches_reference(tp_group, num_tokens, vocab_size, embedding_dim):
    """SpyreParallelLMHead.forward_oot output matches a plain F.linear reference.

    Exercises the full padded-weight path: checkpoint values are written into
    layer.weight, padded_weight is materialized in process_weights_after_loading,
    forward_oot runs the compiled Spyre matmul and unpads the logits.
    """
    from spyre_inference.custom_ops.parallel_lm_head import SpyreParallelLMHead
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    torch.manual_seed(42)

    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)
    assert isinstance(layer, SpyreParallelLMHead)

    # Simulate checkpoint loading: copy known values into the existing Parameter.
    loaded = torch.randn(layer.weight.shape, dtype=torch.float16)
    layer.weight.data.copy_(loaded)

    # Materialize padded_weight from the now-populated weight, as the loader would.
    layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, embedding_dim, dtype=torch.float16)
    expected = reference_lm_head(x, layer.weight.data)

    # In production weights live on Spyre after `model.to(spyre_device)`;
    # mirror that here so forward_oot's H2D + Spyre F.linear actually run.
    layer = layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"))

    assert actual.shape == (num_tokens, layer.weight.shape[0])
    # Spyre matmul accumulation order diverges from the CPU reference in fp16;
    # see the "expect numerical differences" warning in
    # SpyreUnquantizedLMHeadMethod.process_weights_after_loading.
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-1, rtol=5e-2)


# ---------------------------------------------------------------------------
# Padding-workaround tests
#
# These tests cover a temporary workaround for a torch-spyre work-division
# limitation: matmul shapes must be a multiple of 64 * (k * 32), where k is
# an integer. Once torch-spyre lifts that restriction, the workaround in
# SpyreUnquantizedLMHeadMethod.process_weights_after_loading and the tests
# below (marked `padding_workaround`) can be removed.
# ---------------------------------------------------------------------------


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
@pytest.mark.parametrize(
    "vocab_size, expect_padding, expect_padded_shape",
    [
        (49216, True, 51200),  # 49216 = 64 * (24.03125 * 32) → needs padding to 51200
        (51200, False, 51200),  # 51200 = 64 * (25 * 32) → already aligned, no padding
    ],
)
def test_padded_weight_reflects_loaded_weight(
    tp_group, vocab_size, expect_padding, expect_padded_shape
):
    """padded_weight must hold the loaded checkpoint values, not uninitialized data.

    Regression guard: padded_weight was previously snapshotted in __init__,
    before load_weights ran, so it held whatever torch.empty produced. It is
    now materialized in process_weights_after_loading instead.

    Also asserts the no-padding path: when the weight row count is already a
    multiple of 64 * 32, process_weights_after_loading must leave padded_weight
    identical to the weight Parameter (no F.pad, no extra allocation).
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    embedding_dim = 64
    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)

    loaded = torch.randn(layer.weight.shape, dtype=torch.float16)
    layer.weight.data.copy_(loaded)

    layer.quant_method.process_weights_after_loading(layer)

    if expect_padding:
        assert layer.padding > 0
        assert layer.padded_weight.shape == (
            expect_padded_shape,
            embedding_dim,
        )
        # Top slice mirrors the loaded weight bit-for-bit.
        torch.testing.assert_close(
            layer.padded_weight[: layer.weight.shape[0]],
            layer.weight,
            atol=0.0,
            rtol=0.0,
        )
        # Padding rows are zeros (F.pad default), so they contribute 0 to logits.
        assert torch.all(layer.padded_weight[layer.weight.shape[0] :] == 0)
    else:
        # Aligned shape: no padding applied, padded_weight aliases the weight
        # Parameter so we don't allocate or copy a second vocab-sized tensor.
        assert layer.padding == 0
        assert layer.padded_weight is layer.weight
        torch.testing.assert_close(
            layer.padded_weight,
            layer.weight,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parallel_lm_head
def test_lm_head_oot_dispatch(tp_group):
    """Verify ParallelLMHead OOT registration: class swap + quant_method swap."""
    from spyre_inference.custom_ops.parallel_lm_head import (
        SpyreParallelLMHead,
        SpyreUnquantizedLMHeadMethod,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)

    # OOT class swap: ParallelLMHead.__new__ should produce SpyreParallelLMHead.
    assert isinstance(layer, SpyreParallelLMHead)
    # quant_method swap: unquantized method is replaced with the Spyre-routing one.
    assert isinstance(layer.quant_method, SpyreUnquantizedLMHeadMethod)


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
def test_non_aligned_weight_is_padded(tp_group):
    """process_weights_after_loading pads weight rows not divisible by ALIGN.

    Part of the padding workaround — remove together with the other
    `padding_workaround` tests once torch-spyre lifts the shape restriction.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    ALIGN = 64 * 32

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)

    original = torch.randn(63, 64, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(original.clone(), requires_grad=False)

    layer.quant_method.process_weights_after_loading(layer)

    expected_padded_rows = ALIGN  # ceil(63 / ALIGN) * ALIGN
    assert layer.padded_weight.shape[0] == expected_padded_rows
    assert layer.padding == expected_padded_rows - 63
    # Original values preserved in the top rows
    torch.testing.assert_close(layer.padded_weight[:63], original, atol=0.0, rtol=0.0)
    # Padding rows are zeros
    assert torch.all(layer.padded_weight[63:] == 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
