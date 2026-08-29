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
    """SpyreUnquantizedLMHeadMethod.apply output matches a plain F.linear reference.

    Exercises the full padded-weight path: checkpoint values are written into
    layer.weight, padded_weight_t is materialized in process_weights_after_loading,
    and quant_method.apply runs the Spyre matmul and eagerly unpads the logits.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import SpyreParallelLMHead

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
    # mirror that here so the H2D + Spyre matmul actually run.
    layer = layer.to("spyre")
    actual = layer.quant_method.apply(layer, x.to("spyre"))

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
    """padded_weight_t must hold the loaded checkpoint values, not uninitialized data.

    Regression guard: the padded weight was previously snapshotted in __init__,
    before load_weights ran, so it held whatever torch.empty produced. It is
    now materialized in process_weights_after_loading instead.

    padded_weight_t is stored transposed ([embedding_dim, padded_vocab]) so the
    forward GEMM is the Spyre-fast `x @ A`; the vocab padding lands on the
    trailing columns.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    embedding_dim = 64
    layer = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)

    loaded = torch.randn(layer.weight.shape, dtype=torch.float16)
    layer.weight.data.copy_(loaded)

    layer.quant_method.process_weights_after_loading(layer)

    vocab = layer.weight.shape[0]
    if expect_padding:
        assert layer.spyre_row_padding > 0
        assert layer.padded_weight_t.shape == (
            embedding_dim,
            expect_padded_shape,
        )
        # Leading columns mirror the loaded weight (transposed) bit-for-bit.
        torch.testing.assert_close(
            layer.padded_weight_t[:, :vocab],
            layer.weight.t(),
            atol=0.0,
            rtol=0.0,
        )
        # Padding columns are zeros (F.pad default), so they contribute 0 to logits.
        assert torch.all(layer.padded_weight_t[:, vocab:] == 0)
    else:
        # Aligned shape: no padding applied, padded_weight_t is just weightᵀ.
        assert layer.spyre_row_padding == 0
        torch.testing.assert_close(
            layer.padded_weight_t,
            layer.weight.t(),
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.parallel_lm_head
def test_lm_head_oot_dispatch(tp_group):
    """Verify ParallelLMHead OOT registration: class swap + quant_method swap."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import (
        SpyreParallelLMHead,
        SpyreUnquantizedLMHeadMethod,
    )

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)

    # OOT class swap: ParallelLMHead.__new__ should produce SpyreParallelLMHead.
    assert isinstance(layer, SpyreParallelLMHead)
    # quant_method swap: unquantized method is replaced with the Spyre-routing one.
    assert isinstance(layer.quant_method, SpyreUnquantizedLMHeadMethod)


@pytest.mark.parallel_lm_head
def test_lm_head_fp8_config_accepted(tp_group):
    """SpyreParallelLMHead accepts Fp8Config without raising.

    Fp8Config.get_quant_method returns None for ParallelLMHead (it only
    handles LinearBase/Attention), so upstream falls back to
    UnquantizedEmbeddingMethod, which we then replace with
    SpyreUnquantizedLMHeadMethod. The LM head always runs FP16 regardless
    of the checkpoint's quantization config.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from spyre_inference.custom_ops.parallel_lm_head import (
        SpyreParallelLMHead,
        SpyreUnquantizedLMHeadMethod,
    )

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16, quant_config=Fp8Config())

    assert isinstance(layer, SpyreParallelLMHead)
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
    # padded_weight_t is transposed: [embedding_dim, padded_vocab].
    assert layer.padded_weight_t.shape[1] == expected_padded_rows
    assert layer.spyre_row_padding == expected_padded_rows - 63
    # Original values preserved in the leading columns (transposed)
    torch.testing.assert_close(layer.padded_weight_t[:, :63], original.t(), atol=0.0, rtol=0.0)
    # Padding columns are zeros
    assert torch.all(layer.padded_weight_t[:, 63:] == 0)


@pytest.mark.parallel_lm_head
@pytest.mark.padding_workaround
def test_padded_matmul_and_unpad_slice_run_on_device(spyre_or_cpu_device):
    """The transposed matmul and the un-pad slice run on-device eagerly.

    SpyreUnquantizedLMHeadMethod.apply does `x @ weight_t` on a padding-aligned
    output dim then slices off the trailing pad columns. Post torch-spyre #3578
    the un-pad slice lowers on-device in eager mode (the storage offset is
    honored), so no torch.compile is needed. This isolates that primitive pair
    (matmul + trailing-slice) from the full layer path.
    """
    ALIGN = 64 * 32
    vocab = 32000
    padding = (-vocab) % ALIGN
    hidden = torch.randn(32, 4096, dtype=torch.float16) * 0.01
    weight_t = torch.randn(4096, vocab + padding, dtype=torch.float16) * 0.01

    def project(x, w):
        out = torch.matmul(x, w)
        return out[:, :-padding]

    expected = torch.matmul(hidden, weight_t)[:, :-padding]
    actual = project(hidden.to(spyre_or_cpu_device), weight_t.to(spyre_or_cpu_device))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("scale", [1.0, 1.0 / 6.0, 2.0])
def test_spyre_logits_processor_scaling(tp_group, spyre_or_cpu_device, scale):
    """SpyreLogitsProcessor matches upstream reference for logits_scaling.

    Granite 3.3 sets logits_scaling, so LogitsProcessor.forward runs an in-place
    `logits *= self.scale` — on the host, as SpyreLogitsProcessor returns CPU logits.
    """

    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    torch.manual_seed(42)

    vocab_size = 32000
    embedding_dim = 4096
    num_tokens = 8

    torch.manual_seed(43)
    # Small random values keep logits in a range where fp16 accumulation-order
    # differences between CPU and Spyre matmuls do not dominate the tolerance.
    weight = torch.randn(vocab_size, embedding_dim, dtype=torch.float16) * 0.01

    # Minimal fake LM head: just a linear weight with the right interface.
    class FakeLMHead:
        def __init__(self, weight_tensor):
            self.weight = weight_tensor
            # Upstream _get_logits reads lm_head.tp_size to decide whether to
            # gather across TP ranks; single-rank test, so no gather.
            self.tp_size = 1
            self.shard_indices = type(
                "SI", (), {"num_org_vocab_padding": 0, "org_vocab_start_index": 0}
            )()
            self.quant_method = type(
                "QM",
                (),
                {"apply": lambda self, layer, x, bias=None: F.linear(x, layer.weight, bias)},
            )()

    weight_device = weight.to(spyre_or_cpu_device)
    fake_head = FakeLMHead(weight_device)

    processor = LogitsProcessor(
        vocab_size=vocab_size,
        org_vocab_size=vocab_size,
        scale=scale,
    )
    assert isinstance(processor, SpyreLogitsProcessor)

    torch.manual_seed(44)
    hidden = torch.randn(num_tokens, embedding_dim, dtype=torch.float16) * 0.01
    hidden_spyre = hidden.to(spyre_or_cpu_device)

    # Reference: upstream logic on CPU.
    logits_ref = F.linear(hidden, weight)
    logits_ref = logits_ref[..., :vocab_size]
    logits_ref = logits_ref * scale

    # Spyre path.
    logits_out = processor(fake_head, hidden_spyre, embedding_bias=None)
    assert logits_out is not None

    torch.testing.assert_close(logits_out.cpu().float(), logits_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.fixture
def spyre_or_cpu_device():
    """Use Spyre if available, otherwise CPU."""
    try:
        torch.randn(1, device=torch.device("spyre"))
        return torch.device("spyre")
    except Exception:
        return torch.device("cpu")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
