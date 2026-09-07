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
from spyre_testing_plugin.pytest_plugin import spyre_available


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


def _apply_tracked(layer):
    """Run `layer._apply` with an identity fn; return the tensors handed to it."""
    seen: list[torch.Tensor] = []
    layer._apply(lambda t: (seen.append(t), t)[1])
    return seen


@pytest.mark.parallel_lm_head
def test_lm_head_apply_moves_weight_before_process(tp_group):
    """Before `process_weights_after_loading`, `padded_weight_t` is absent and `weight`
    is still live, so `_apply` must move it rather than strand it on host."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    assert not hasattr(layer, "padded_weight_t")
    assert any(t is layer.weight for t in _apply_tracked(layer))


@pytest.mark.parallel_lm_head
def test_lm_head_apply_skips_dead_weight(tp_group):
    """Once `padded_weight_t` exists, `_apply` moves it and skips the dead `weight`,
    restoring `weight` as the same registered Parameter."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    layer.quant_method.process_weights_after_loading(layer)

    original = layer.weight
    seen = _apply_tracked(layer)

    assert any(t is layer.padded_weight_t for t in seen), "padded_weight_t was not moved"
    assert not any(t is original for t in seen), "dead weight should be skipped"
    assert layer.weight is original and "weight" in layer._parameters


@pytest.mark.parallel_lm_head
def test_lm_head_weight_stays_on_cpu_after_to_spyre(tp_group):
    """Real device move: the dead `weight` stays on CPU; `padded_weight_t` goes to Spyre.
    Skipped on CPU-only hosts, where off-device placement is not observable."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = ParallelLMHead(128, 64, params_dtype=torch.float16)
    layer.quant_method.process_weights_after_loading(layer)

    layer.to("spyre")

    assert layer.weight.device.type == "cpu"
    assert layer.padded_weight_t.device.type == "spyre"


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


@pytest.mark.parallel_lm_head
@pytest.mark.parametrize("vocab_size", [128, 49216])
def test_tied_head_promoted_on_first_logits_call(tp_group, vocab_size):
    """An embedding used as the head gains a padded `Wᵀ`, leaving `weight` untouched."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod
    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embedding_dim = 64
    torch.manual_seed(42)

    embed = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    assert not isinstance(embed.quant_method, SpyreUnquantizedLMHeadMethod)

    loaded = torch.randn(embed.weight.shape, dtype=torch.float16)
    embed.weight.data.copy_(loaded)
    embed.quant_method.process_weights_after_loading(embed)
    assert not hasattr(embed, "padded_weight_t")

    weight_before = embed.weight
    promote_tied_lm_head(embed)

    assert isinstance(embed.quant_method, SpyreUnquantizedLMHeadMethod)
    assert embed.weight is weight_before
    torch.testing.assert_close(embed.weight.data, loaded, atol=0.0, rtol=0.0)

    padded_vocab = vocab_size + embed.spyre_row_padding
    assert padded_vocab % (64 * 32) == 0
    assert embed.padded_weight_t.shape == (embedding_dim, padded_vocab)
    torch.testing.assert_close(
        embed.padded_weight_t[:, :vocab_size], loaded.t(), atol=0.0, rtol=0.0
    )


@pytest.mark.parallel_lm_head
def test_promotion_is_idempotent(tp_group):
    """`_apply_head` runs every decode step, so a second promotion must be a no-op."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embed = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    embed.quant_method.process_weights_after_loading(embed)

    promote_tied_lm_head(embed)
    method, weight_t = embed.quant_method, embed.padded_weight_t
    promote_tied_lm_head(embed)

    assert embed.quant_method is method
    assert embed.padded_weight_t is weight_t


@pytest.mark.parallel_lm_head
def test_tied_head_projection_matches_reference(tp_group):
    """The promoted head's logits match F.linear, and its gather still works."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    vocab_size, embedding_dim, num_tokens = 49216, 64, 7
    torch.manual_seed(42)

    embed = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    embed.weight.data.normal_(std=0.02)
    embed.quant_method.process_weights_after_loading(embed)

    x = torch.randn(num_tokens, embedding_dim, dtype=torch.float16)
    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    logits_ref = reference_lm_head(x, embed.weight.data)
    gather_ref = F.embedding(input_ids, embed.weight)

    # Promote after the device move, as the first logits call does.
    embed = embed.to("spyre")
    promote_tied_lm_head(embed)
    assert embed.padded_weight_t.device.type == "spyre"

    logits = embed.quant_method.apply(embed, x.to("spyre"))
    gather = embed(input_ids.to("spyre"))

    assert logits.shape == (num_tokens, vocab_size)
    # Spyre matmul accumulation order diverges from the CPU reference in fp16.
    torch.testing.assert_close(logits.cpu().float(), logits_ref.float(), atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(gather.cpu().float(), gather_ref.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parallel_lm_head
def test_real_head_is_left_alone(tp_group):
    """A real ParallelLMHead already owns the projection; promotion must not touch it."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import promote_tied_lm_head

    embed = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    head = ParallelLMHead(49216, 64, params_dtype=torch.float16).tie_weights(embed)
    assert head.weight is embed.weight

    embed.weight.data.normal_(std=0.02)
    head.quant_method.process_weights_after_loading(head)
    method, weight_t = head.quant_method, head.padded_weight_t

    promote_tied_lm_head(head)

    assert head.quant_method is method
    assert head.padded_weight_t is weight_t
    # The tied table is only ever gathered from, so it never grows a projection.
    assert not hasattr(embed, "padded_weight_t")


@pytest.mark.parallel_lm_head
def test_gather_only_tables_are_never_promoted(tp_group):
    """Only the table handed to the logits processor is promoted.

    Gemma 3n builds a second, gather-only `embed_tokens_per_layer` under the same
    tied config; duplicating that table transposed would be pure waste.
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod

    torch.manual_seed(42)
    embed_tokens = VocabParallelEmbedding(49216, 64, params_dtype=torch.float16)
    per_layer = VocabParallelEmbedding(2048, 64, params_dtype=torch.float16)
    for table in (embed_tokens, per_layer):
        table.weight.data.normal_(std=0.02)
        table.quant_method.process_weights_after_loading(table)

    processor = LogitsProcessor(vocab_size=49216, org_vocab_size=49216)
    processor(embed_tokens, torch.randn(4, 64, dtype=torch.float16), embedding_bias=None)

    assert isinstance(embed_tokens.quant_method, SpyreUnquantizedLMHeadMethod)
    assert not isinstance(per_layer.quant_method, SpyreUnquantizedLMHeadMethod)
    assert not hasattr(per_layer, "padded_weight_t")


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
