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

"""Tests for SpyreVocabParallelEmbedding (issue #135).

Coverage:
  1. OOT class swap.
  2. TP=1 forward matches `F.embedding`.
  3. "Fake TP=2" forward (patched rank/world, all_reduce stubbed) sums
     to the full-vocab `F.embedding` reference — i.e. masking +
     per-rank re-indexing compute the right thing.
  4. Strict-xfail tripwire on the int64 `tensor >= int_const` Spyre
     compile failure that motivates the CPU-bounce in
     SpyreVocabParallelEmbedding.forward. When the tripwire flips to
     passing, this custom op can likely be deleted.

Real TP=2 collective correctness on hardware lives in
`tests/e2e/test_distributed_tp2.py`.
"""

import sys
import warnings

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.vocab_parallel_embedding
def test_vocab_parallel_embedding_oot_dispatch(tp_group):
    """VocabParallelEmbedding(...) instantiates SpyreVocabParallelEmbedding."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    layer = VocabParallelEmbedding(128, 64, params_dtype=torch.float16)
    assert isinstance(layer, SpyreVocabParallelEmbedding)


@pytest.mark.vocab_parallel_embedding
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("vocab_size", [128, 1024, 32000])
@pytest.mark.parametrize("embedding_dim", [64, 128])
def test_tp1_forward_matches_reference(tp_group, num_tokens, vocab_size, embedding_dim):
    """At TP=1, forward(input_ids) matches F.embedding(input_ids, weight)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    torch.manual_seed(0)
    layer = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    layer.weight.data.normal_(std=0.02)

    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    expected = F.embedding(input_ids, layer.weight)

    layer = layer.to("spyre")
    actual = layer(input_ids.to("spyre"))

    assert actual.shape == (num_tokens, embedding_dim)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.vocab_parallel_embedding
@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize("vocab_size", [1024, 32000])
@pytest.mark.parametrize("embedding_dim", [64, 128])
def test_fake_tp2_forward_matches_reference(
    tp_group, monkeypatch, num_tokens, vocab_size, embedding_dim
):
    """Two patched-rank layers + masked sum reproduce F.embedding.

    Bypasses the real all-reduce (the `tp_group` fixture is TP=1) by
    swapping `tensor_model_parallel_all_reduce` for a passthrough and
    summing per-rank outputs in the test. Each rank zeroes tokens
    outside its shard, so the sum over ranks equals the full-vocab
    reference. Real-collective correctness is in e2e/test_distributed_tp2.py.
    """
    from vllm.model_executor.layers import vocab_parallel_embedding as upstream

    import spyre_inference.custom_ops.vocab_parallel_embedding as svpe

    monkeypatch.setattr(svpe, "tensor_model_parallel_all_reduce", lambda x: x)

    torch.manual_seed(42)
    full_weight = torch.randn(vocab_size, embedding_dim, dtype=torch.float16) * 0.02

    torch.manual_seed(7)
    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    expected = F.embedding(input_ids, full_weight)

    def _build_rank(rank: int, world_size: int):
        monkeypatch.setattr(upstream, "get_tensor_model_parallel_rank", lambda r=rank: r)
        monkeypatch.setattr(
            upstream,
            "get_tensor_model_parallel_world_size",
            lambda ws=world_size: ws,
        )
        layer = upstream.VocabParallelEmbedding(
            vocab_size, embedding_dim, params_dtype=torch.float16
        )
        assert layer.tp_size == world_size
        start = layer.shard_indices.org_vocab_start_index
        end = layer.shard_indices.org_vocab_end_index
        layer.weight.data.zero_()
        layer.weight.data[: end - start].copy_(full_weight[start:end])
        return layer.to("spyre")

    rank0 = _build_rank(0, 2)
    rank1 = _build_rank(1, 2)

    spyre_input_ids = input_ids.to("spyre")
    summed = rank0(spyre_input_ids) + rank1(spyre_input_ids)
    torch.testing.assert_close(summed.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


# --- int64 comparison tripwire ---------------------------------------------
#
# SpyreVocabParallelEmbedding.forward currently bounces TP-mask compute to
# CPU because the upstream `get_masked_input_and_mask` does
# `input_ >= org_vocab_start_index` under @torch.compile, and Spyre's
# inductor backend rejects the int64 Python-int constant:
#
#     Spyre backend does not support: unexpected argument
#     Constant(value=N, dtype=torch.int64) to greaterequal
#
# A 0-D tensor workaround compiles but produces silently-wrong values, so
# CPU bounce is the only correct path today. This tripwire is
# xfail(strict=True): when it flips to passing, delete the custom
# SpyreVocabParallelEmbedding and check that the upstream code correctly runs
# on TP > 1.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre's inductor backend rejects int64 comparisons: `Spyre backend "
        "does not support: torch.bool result of operand with device format "
        "DataFormats.IEEE_INT32`. This is the load-bearing limitation behind "
        "SpyreVocabParallelEmbedding's CPU bounce — upstream "
        "`get_masked_input_and_mask` runs `input_ >= org_vocab_start_index` "
        "under @torch.compile. A 0-D-tensor workaround compiles but produces "
        "silently-wrong values, so CPU bounce is the only correct path. When "
        "this flips to passing, delete the CPU bounce in "
        "SpyreVocabParallelEmbedding.forward and let the upstream forward path "
        "run on-device."
    ),
)
def test_int64_compiled_compare_against_python_int(tp_group) -> None:
    @torch.compile
    def cmp_ge(x, c):
        return x >= c

    cpu = torch.arange(16, dtype=torch.int64)
    on_spyre = cpu.to(torch.device("spyre:0"))

    out = cmp_ge(on_spyre, 8)
    expected = cpu >= 8
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre's inductor backend rejects int32 subtraction: `Spyre backend "
        "does not support: sub on DataFormats.IEEE_INT32`. Upstream "
        "`get_masked_input_and_mask` also runs `input_ - valid_offset` under "
        "@torch.compile, so this is a second load-bearing limitation behind "
        "SpyreVocabParallelEmbedding's lookup-table workaround. When this "
        "flips to passing, the on-device arithmetic path can be reconsidered."
    ),
)
def test_int32_compiled_subtract_against_python_int(tp_group) -> None:
    @torch.compile
    def sub(x, c):
        return x - c

    cpu = torch.arange(16, dtype=torch.int32)
    on_spyre = cpu.to(torch.device("spyre:0"))

    out = sub(on_spyre, 8)
    expected = cpu - 8
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.vocab_parallel_embedding
def test_embedding_does_not_fall_back_to_cpu() -> None:
    """torch-spyre handles aten.embedding.default on-device (no CPU fallback), so a
    multi-row F.embedding on-device must not emit a FallbackWarning.

    The single-row gather works too (torch-spyre#3418; see
    test_single_token_embedding_on_device), so SpyreVocabParallelEmbedding.forward
    gathers on-device."""
    from torch_spyre.ops.fallbacks import FallbackWarning

    weight = torch.randn(128, 64, dtype=torch.float16, device="spyre")
    input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="spyre")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        F.embedding(input_ids, weight)

    fallback_msgs = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not any("embedding" in m for m in fallback_msgs), fallback_msgs


@pytest.mark.vocab_parallel_embedding
def test_single_token_embedding_on_device() -> None:
    """Single-row embedding gather (single-token decode). Fixed by torch-spyre#3418;
    guards the on-device gather in SpyreVocabParallelEmbedding.forward."""
    weight = torch.randn(128, 64, dtype=torch.float16, device="spyre")
    input_ids = torch.tensor([5], dtype=torch.int64, device="spyre")

    out = F.embedding(input_ids, weight)
    torch.testing.assert_close(out.cpu().float(), weight[5:6].cpu().float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.vocab_parallel_embedding
@pytest.mark.parametrize("vocab_size,embedding_dim", [(32000, 128), (1024, 64)])
def test_gather_layout_applied_on_move(tp_group, vocab_size, embedding_dim):
    """Moving to Spyre places the table rows-outermost and the gather still matches
    F.embedding, whose result is what the layout must preserve."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    torch.manual_seed(3)
    layer = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    layer.weight.data.normal_(std=0.02)

    input_ids = torch.randint(0, vocab_size, (16,), dtype=torch.int64)
    expected = F.embedding(input_ids, layer.weight)

    layer = layer.to("spyre")

    from torch_spyre._C import get_elem_in_stick, get_spyre_tensor_layout

    eps = get_elem_in_stick(layer.weight.data.dtype)
    device_size = list(get_spyre_tensor_layout(layer.weight.data).device_size)
    assert device_size == [vocab_size, embedding_dim // eps, eps], (
        "vocab dim must be outermost in the device layout"
    )
    assert layer.weight.data.shape == (vocab_size, embedding_dim)
    actual = layer(input_ids.to("spyre"))
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.vocab_parallel_embedding
def test_lm_head_weight_not_relaid_out(tp_group):
    """The LM head subclasses VocabParallelEmbedding but uses its weight as a matmul
    operand, so it must not pick up the gather layout."""
    from spyre_inference.custom_ops.parallel_lm_head import SpyreParallelLMHead
    from spyre_inference.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    assert not issubclass(SpyreParallelLMHead, SpyreVocabParallelEmbedding)


@pytest.mark.vocab_parallel_embedding
def test_vocab_table_crosses_to_device_once(tp_group):
    """Only the one-row probe goes through fn; the table itself is placed directly."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    vocab_size, embedding_dim = 1024, 128
    layer = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)

    moved = []

    def fn(tensor):
        moved.append(tuple(tensor.shape))
        return tensor.to("spyre")

    layer._apply(fn)

    assert moved == [(1, embedding_dim)], (
        f"expected only the one-row probe to go through fn, got {moved}"
    )


@pytest.mark.vocab_parallel_embedding
def test_unaligned_row_width_keeps_default_device_layout(tp_group, monkeypatch):
    """Unaligned rows keep the default layout: still on device, still correct."""
    from torch_spyre._C import get_elem_in_stick, get_spyre_tensor_layout
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops import utils

    vocab_size, embedding_dim = 256, 96
    assert embedding_dim % get_elem_in_stick(torch.float16), "dim must be stick-unaligned"

    torch.manual_seed(7)
    layer = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    layer.weight.data.normal_(std=0.02)

    input_ids = torch.randint(0, vocab_size, (8,), dtype=torch.int64)
    expected = F.embedding(input_ids, layer.weight)

    warned: list[str] = []
    monkeypatch.setattr(utils.logger, "warning_once", lambda msg, *a: warned.append(msg % a))

    layer = layer.to("spyre")

    assert layer.weight.device.type == "spyre", "the table must still move to device"
    device_size = list(get_spyre_tensor_layout(layer.weight.data).device_size)
    assert device_size[0] != vocab_size, (
        f"unaligned rows cannot be outermost, got device_size={device_size}"
    )
    assert len(warned) == 1 and "vocab table" in warned[0], warned

    actual = layer(input_ids.to("spyre"))
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.vocab_parallel_embedding
def test_tied_lm_head_table_is_row_gathered(tp_group):
    """A tied LM head shares the table, which must still be row-gathered for the lookup."""
    from torch_spyre._C import get_elem_in_stick, get_spyre_tensor_layout
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    torch.manual_seed(5)
    vocab_size, embedding_dim, num_tokens = 1024, 128, 8

    embed = VocabParallelEmbedding(vocab_size, embedding_dim, params_dtype=torch.float16)
    embed.weight.data.normal_(std=0.02)
    head = ParallelLMHead(vocab_size, embedding_dim, params_dtype=torch.float16)
    head = head.tie_weights(embed)
    assert head.weight is embed.weight, "tie_weights should share the Parameter"

    head.quant_method.process_weights_after_loading(head)

    input_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64)
    hidden = torch.randn(num_tokens, embedding_dim, dtype=torch.float16)
    table = embed.weight.data.clone()
    expected_embedding = F.embedding(input_ids, table)
    expected_logits = hidden.float() @ table.float().t()

    model = torch.nn.Module()
    model.embed = embed
    model.lm_head = head
    model.to("spyre")

    eps = get_elem_in_stick(torch.float16)
    assert list(get_spyre_tensor_layout(model.embed.weight.data).device_size) == [
        vocab_size,
        embedding_dim // eps,
        eps,
    ], "tied table must stay rows-outermost"

    weight_t = model.lm_head.padded_weight_t.data
    rows, width = weight_t.shape
    assert list(get_spyre_tensor_layout(weight_t).device_size) != [rows, width // eps, eps], (
        "padded_weight_t is a matmul operand, not a gather source"
    )

    actual_embedding = model.embed(input_ids.to("spyre"))
    torch.testing.assert_close(
        actual_embedding.cpu().float(), expected_embedding.float(), atol=1e-3, rtol=1e-3
    )

    actual_logits = model.lm_head.quant_method.apply(model.lm_head, hidden.to("spyre"))
    assert actual_logits.shape == (num_tokens, vocab_size)
    # Spyre matmul accumulation order diverges from the CPU reference in fp16.
    torch.testing.assert_close(actual_logits.cpu().float(), expected_logits, atol=1e-1, rtol=5e-2)
