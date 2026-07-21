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
`tests/test_distributed_tp2.py`.
"""

import sys
import warnings

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.vocab_parallel_embedding
def test_vocab_parallel_embedding_oot_dispatch(tp_group):
    """VocabParallelEmbedding(...) instantiates SpyreVocabParallelEmbedding."""
    from spyre_inference.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
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
    reference. Real-collective correctness is in test_distributed_tp2.py.
    """
    import spyre_inference.custom_ops.vocab_parallel_embedding as svpe
    from vllm.model_executor.layers import vocab_parallel_embedding as upstream

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
        "Spyre's inductor backend rejects int64 Python-int constants in "
        "comparisons: `Constant(value=N, dtype=torch.int64) to greaterequal`. "
        "This is the load-bearing limitation behind SpyreVocabParallelEmbedding's "
        "CPU bounce — upstream `get_masked_input_and_mask` runs `input_ >= "
        "org_vocab_start_index` under @torch.compile. A 0-D-tensor workaround "
        "compiles but produces silently-wrong values, so CPU bounce is the only "
        "correct path. When this flips to passing, delete the CPU bounce in "
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
        "torch-spyre routes aten.embedding.default to CPU. When this flips to "
        "passing, remove the CPU pin in TorchSpyreModelRunner.load_model and "
        "the forward-path CPU bounce in SpyreVocabParallelEmbedding."
    ),
)
def test_embedding_does_not_fall_back_to_cpu() -> None:
    from torch_spyre.ops.fallbacks import FallbackWarning

    weight = torch.randn(128, 64, dtype=torch.float16, device="spyre")
    input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="spyre")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        F.embedding(input_ids, weight)

    fallback_msgs = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not any("embedding" in m for m in fallback_msgs), fallback_msgs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
