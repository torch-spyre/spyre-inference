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

"""Cheap unit tests for ``configure_pooling_for_spyre`` patching.

No Spyre hardware: builds minimal ``SequencePooler`` / ``DispatchPooler`` /
``TokenPooler`` graphs and checks CLS/LAST/MEAN/AllPool become Spyre forms.
FP32 linear heads are downcast to fp16 so the GEMM can run on Spyre.
An embed projector with no classifier still falls back to CPU.

Host MEAN crop lives in ``tests/pool/test_spyre_mean_pool.py``. Destagger
of a device fp32 sum is ``test_spyre_fp32_reduce_d2h_with_destagger``
(xfail). FP32 heads are ``test_spyre_fp32_linear_for_pooling_heads``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, LastPool, MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool, StepPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler

from spyre_inference.custom_ops.utils import register as register_spyre_convert
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreAllPool,
    SpyreClassifierLinear,
    SpyreCLSPool,
    SpyreCpuClassifier,
    SpyreDispatchPooler,
    SpyreEmbeddingPoolerHead,
    SpyreLastPool,
    SpyreMeanPool,
    SpyreNormalize,
    SpyreTokenPooler,
    configure_pooling_for_spyre,
    patch_pooler_for_spyre,
    run_pooling_tail_on_cpu,
)

# Downcast goes through spyre_convert. Production registers it at plugin load;
# these tests pass a CPU device and never import the plugin.
register_spyre_convert()

_SPYRE = torch.device("cpu")  # configure only needs a device label for logging


def _embed_pooler(pooling) -> SequencePooler:
    return SequencePooler(
        pooling=pooling,
        head=EmbeddingPoolerHead(activation=PoolerNormalize()),
    )


def _model_with_pooler(pooler: nn.Module) -> nn.Module:
    model = nn.Module()
    model.pooler = pooler
    return model


def test_configure_pooling_patches_cls_to_spyre_cls_pool():
    model = _model_with_pooler(_embed_pooler(CLSPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreCLSPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)
    assert isinstance(model.pooler.head.activation, SpyreNormalize)


def test_configure_pooling_patches_last_to_spyre_last_pool():
    model = _model_with_pooler(_embed_pooler(LastPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreLastPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)


def test_configure_pooling_patches_mean_to_spyre_mean_pool():
    model = _model_with_pooler(_embed_pooler(MeanPool()))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreMeanPool)
    assert isinstance(model.pooler.head, SpyreEmbeddingPoolerHead)


def test_configure_pooling_dispatch_patches_embed_mean():
    pooler = DispatchPooler({"embed": _embed_pooler(MeanPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreMeanPool)


def test_configure_pooling_dispatch_patches_embed_cls():
    """DispatchPooler (real embed models) must still install SpyreCLSPool."""
    pooler = DispatchPooler({"embed": _embed_pooler(CLSPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreCLSPool)


def test_configure_pooling_dispatch_patches_embed_last():
    pooler = DispatchPooler({"embed": _embed_pooler(LastPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.pooling, SpyreLastPool)


def test_configure_pooling_fp32_classifier_downcasts_to_fp16():
    classifier = nn.Linear(8, 2)
    pooler = SequencePooler(
        pooling=CLSPool(),
        head=ClassifierPoolerHead(classifier=classifier, head_dtype=torch.float32),
    )
    model = _model_with_pooler(pooler)
    model.classifier = classifier
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(model.pooler.pooling, SpyreCLSPool)
    assert isinstance(model.classifier, SpyreClassifierLinear)
    assert model.classifier is model.pooler.head.classifier
    assert model.classifier.weight.dtype == torch.float16
    assert model.classifier.bias is not None
    assert tuple(model.classifier.bias.shape) == (2,)
    assert model.classifier.weight.shape == (8, 2)
    assert isinstance(model.pooler.head, ClassifierPoolerHead)
    assert model.pooler.head.head_dtype == torch.float16


def test_configure_pooling_dispatch_classify_sets_head_dtype():
    """``poolers_by_task`` is a plain dict; ``modules()`` never sees ``head_dtype``."""
    classifier = nn.Linear(8, 2)
    pooler = DispatchPooler(
        {
            "classify": SequencePooler(
                pooling=CLSPool(),
                head=ClassifierPoolerHead(classifier=classifier, head_dtype=torch.float32),
            )
        }
    )
    model = _model_with_pooler(pooler)
    model.classifier = classifier
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    classify = model.pooler.poolers_by_task["classify"]
    assert isinstance(classify.head, ClassifierPoolerHead)
    assert classify.head.head_dtype == torch.float16
    assert isinstance(classify.head.classifier, SpyreClassifierLinear)
    assert classify.head.classifier.weight.dtype == torch.float16


def test_configure_pooling_dispatch_classify_leaves_embed_head_fp32():
    """Classifier fp16 must not downcast an embed projector in the same dispatcher."""
    classifier = nn.Linear(8, 2)
    projector = nn.Linear(8, 8)
    pooler = DispatchPooler(
        {
            "classify": SequencePooler(
                pooling=CLSPool(),
                head=ClassifierPoolerHead(classifier=classifier, head_dtype=torch.float32),
            ),
            "embed": SequencePooler(
                pooling=CLSPool(),
                head=EmbeddingPoolerHead(
                    projector=projector,
                    head_dtype=torch.float32,
                    activation=PoolerNormalize(),
                ),
            ),
        }
    )
    model = _model_with_pooler(pooler)
    model.classifier = classifier
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    embed = model.pooler.poolers_by_task["embed"]
    assert isinstance(embed.head, SpyreEmbeddingPoolerHead)
    assert embed.head.head_dtype == torch.float32
    assert projector.weight.dtype == torch.float32


def test_configure_pooling_roberta_head_dense_and_out_proj_downcast():
    class _RobertaHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(8, 8)
            self.out_proj = nn.Linear(8, 1)

    head = _RobertaHead()
    pooler = SequencePooler(
        pooling=CLSPool(),
        head=ClassifierPoolerHead(classifier=head, head_dtype=torch.float32),
    )
    model = _model_with_pooler(pooler)
    model.classifier = head
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert isinstance(head.dense, SpyreClassifierLinear)
    assert isinstance(head.out_proj, SpyreClassifierLinear)
    assert head.dense is model.pooler.head.classifier.dense
    assert head.dense.weight.dtype == torch.float16
    assert head.out_proj.weight.dtype == torch.float16
    assert head.dense.bias is not None
    assert tuple(head.dense.bias.shape) == (8,)
    assert isinstance(model.pooler.head, ClassifierPoolerHead)
    assert model.pooler.head.head_dtype == torch.float16


def test_spyre_classifier_linear_matches_nn_linear_on_cpu():
    """Class-swapped Linear must match F.linear, including short-row CLS."""
    torch.manual_seed(0)
    dense = nn.Linear(8, 8)
    x = torch.randn(1, 8)
    dense_ref = dense(x).detach()
    x_batch = torch.randn(3, 8)
    dense_batch_ref = dense(x_batch).detach()
    wrapped_dense = SpyreClassifierLinear.convert(dense)
    assert wrapped_dense is dense
    assert wrapped_dense.weight.shape == (8, 8)
    assert tuple(wrapped_dense.bias.shape) == (8,)
    torch.testing.assert_close(wrapped_dense(x), dense_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(wrapped_dense(x_batch), dense_batch_ref, atol=1e-5, rtol=1e-5)

    out_proj = nn.Linear(8, 1)
    out_ref = out_proj(x).detach()
    wrapped_out = SpyreClassifierLinear.convert(out_proj)
    torch.testing.assert_close(wrapped_out(x), out_ref, atol=1e-5, rtol=1e-5)

    ner = nn.Linear(768, 9)
    x_ner = torch.randn(16, 768)
    ner_ref = ner(x_ner).detach()
    wrapped_ner = SpyreClassifierLinear.convert(ner)
    assert wrapped_ner.weight.shape == (768, 9)
    torch.testing.assert_close(wrapped_ner(x_ner), ner_ref, atol=1e-5, rtol=1e-5)


def test_spyre_classifier_linear_cls_warmup_shape_compiles_on_spyre():
    """B=1, H=1024 + bias stays on Spyre through dense and out_proj."""
    if not spyre_available():
        pytest.skip("needs Spyre")

    torch.manual_seed(0)
    src = nn.Linear(1024, 1024, dtype=torch.float16)
    x_cpu = torch.randn(1, 1024, dtype=torch.float16)
    ref = src(x_cpu).detach()
    wrapped = SpyreClassifierLinear.convert(src).to("spyre")
    x = x_cpu.to("spyre")
    out = wrapped(x)
    assert out.shape == (1, 1024)
    assert out.device.type == "spyre"
    torch.testing.assert_close(out.cpu().float(), ref.float(), atol=2e-2, rtol=2e-2)

    out_proj_src = nn.Linear(1024, 1, dtype=torch.float16)
    out_proj = SpyreClassifierLinear.convert(out_proj_src).to("spyre")
    logits = out_proj(out)
    assert logits.shape == (1, 1)
    assert logits.device.type == "spyre"

    ner_src = nn.Linear(768, 9, dtype=torch.float16)
    x_ner_cpu = torch.randn(16, 768, dtype=torch.float16)
    ner_ref = ner_src(x_ner_cpu).detach()
    ner = SpyreClassifierLinear.convert(ner_src).to("spyre")
    ner_out = ner(x_ner_cpu.to("spyre"))
    assert ner_out.shape == (16, 9)
    assert ner_out.device.type == "spyre"
    torch.testing.assert_close(ner_out.cpu().float(), ner_ref.float(), atol=2e-2, rtol=2e-2)


def test_configure_pooling_fp32_embed_projector_falls_back_to_cpu():
    """No classifier → embed projector stays fp32 and the pooler stays on CPU."""
    projector = nn.Linear(8, 8)
    pooler = SequencePooler(
        pooling=CLSPool(),
        head=EmbeddingPoolerHead(
            projector=projector, head_dtype=torch.float32, activation=PoolerNormalize()
        ),
    )
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE) is False
    assert projector.weight.dtype == torch.float32


def test_configure_pooling_no_pooler_returns_false():
    assert configure_pooling_for_spyre(nn.Module(), _SPYRE) is False


def test_model_applied_classifier_is_wrapped_for_cpu() -> None:
    """A classifier the pooler does not own is applied by the model: wrap it."""
    model = nn.Module()
    model.classifier = nn.Linear(4, 2, dtype=torch.float32)
    model.head_dtype = torch.float32
    pooler = SequencePooler(pooling=MeanPool(), head=None)

    run_pooling_tail_on_cpu(model, pooler)

    assert isinstance(model.classifier, SpyreCpuClassifier)
    assert model.head_dtype == torch.float16


def test_pooler_owned_classifier_is_not_wrapped() -> None:
    """A reranker head owns the classifier, so moving it to CPU is enough."""
    classifier = nn.Linear(4, 2, dtype=torch.float32)
    model = nn.Module()
    model.classifier = classifier
    pooler = SequencePooler(pooling=MeanPool(), head=None)
    pooler.head = nn.Module()
    pooler.head.classifier = classifier

    run_pooling_tail_on_cpu(model, pooler)

    assert model.classifier is classifier


def _token_pooler(cls) -> TokenPooler:
    """``AllPool.__init__`` reads the vLLM config; bypass it for a unit test."""
    pooling = cls.__new__(cls)
    nn.Module.__init__(pooling)
    pooling.enable_chunked_prefill = False
    return TokenPooler(pooling=pooling, head=None)


def test_token_pooler_all_pool_is_patched():
    pooler = _token_pooler(AllPool)
    num_patched, unsupported = patch_pooler_for_spyre(pooler)
    assert (num_patched, unsupported) == (1, [])
    assert isinstance(pooler.pooling, SpyreAllPool)


def test_token_pooler_promoted_so_bucketed_rows_are_always_trimmed():
    """``defer_trim`` may only be set together with the pooler that trims.

    A bucketed gather without a trim would ship padded rows to the client, so the
    two are switched on as a pair and never independently.
    """
    pooler = _token_pooler(AllPool)
    patch_pooler_for_spyre(pooler)
    assert isinstance(pooler, SpyreTokenPooler)
    assert pooler.pooling.defer_trim is True


def test_spyre_all_pool_defers_trim_only_when_asked():
    """Standalone use keeps AllPool's real-length contract; deferring buckets."""
    counts = [3, 1, 4]
    hidden_states = torch.arange(sum(counts) * 9, dtype=torch.float16).reshape(-1, 9)
    meta = _counts_metadata(counts)

    plain = SpyreAllPool(enable_chunked_prefill=False)(hidden_states, meta)
    assert [c.shape[0] for c in plain] == counts

    deferred = SpyreAllPool(enable_chunked_prefill=False, defer_trim=True)(hidden_states, meta)
    # Every chunk is padded up to the same bucket, so the gather's shape no longer
    # tracks the request's token count.
    assert {c.shape[0] for c in deferred} == {64}
    for chunk, n, expected in zip(deferred, counts, torch.split(hidden_states, counts)):
        assert torch.equal(chunk[:n], expected)
        # Padded rows duplicate the last real row, so a trim is all that is needed.
        assert torch.equal(chunk[n:], expected[-1].expand(chunk.shape[0] - n, -1))


def test_spyre_token_pooler_trims_bucketed_rows_to_real_lengths():
    counts = [3, 1, 4]
    hidden_states = torch.arange(sum(counts) * 9, dtype=torch.float16).reshape(-1, 9)
    pooler = _token_pooler(AllPool)
    patch_pooler_for_spyre(pooler)

    got = pooler(hidden_states, _counts_metadata(counts))
    for chunk, expected in zip(got, torch.split(hidden_states, counts)):
        assert torch.equal(chunk, expected)


def test_token_pooler_step_pool_is_unsupported():
    """StepPool subclasses AllPool but indexes by step tag; keep it on CPU."""
    pooler = _token_pooler(StepPool)
    num_patched, unsupported = patch_pooler_for_spyre(pooler)
    assert (num_patched, unsupported) == (0, ["StepPool"])


def _counts_metadata(counts: list[int]):
    class _Meta:
        def get_pooling_cursor(self):
            return type("C", (), {"num_scheduled_tokens_cpu": torch.tensor(counts)})()

    return _Meta()


def test_spyre_all_pool_matches_torch_split():
    counts = [3, 1, 4]
    hidden_states = torch.arange(sum(counts) * 9, dtype=torch.float16).reshape(-1, 9)

    got = SpyreAllPool(enable_chunked_prefill=False)(hidden_states, _counts_metadata(counts))
    for chunk, expected in zip(got, torch.split(hidden_states, counts)):
        assert torch.equal(chunk, expected)


# ---------------------------------------------------------------------------
# Length ladder: the bucketed gather only bounds compiled shapes if the ladder
# it rounds onto actually arrives. Resolving it lazily from
# get_current_vllm_config() did not work -- forward runs outside that context --
# so it is passed in at construction and these pin that it is used.
# ---------------------------------------------------------------------------

_LADDER = [64, 128, 256, 512]


def test_spyre_all_pool_rounds_onto_the_length_ladder():
    """A ladder must produce power-of-two buckets, not every 64-multiple.

    300 discriminates: the ladder rounds it to 512, plain stick alignment to 320.
    """
    counts = [300]
    hidden_states = torch.zeros(sum(counts), 9, dtype=torch.float16)
    meta = _counts_metadata(counts)

    with_ladder = SpyreAllPool(False, defer_trim=True, len_ladder=_LADDER)(hidden_states, meta)
    assert [c.shape[0] for c in with_ladder] == [512]

    # No ladder: still bounded, but at twice as many distinct shapes.
    without = SpyreAllPool(False, defer_trim=True)(hidden_states, meta)
    assert [c.shape[0] for c in without] == [320]


def test_configure_pooling_threads_the_declared_lengths_into_the_ladder():
    """The ladder reaches SpyreAllPool from configure, not from a contextvar."""
    model = _model_with_pooler(_token_pooler(AllPool))
    assert configure_pooling_for_spyre(model, _SPYRE, _LADDER) is True
    assert model.pooler.pooling.len_ladder == _LADDER


def test_configure_pooling_without_a_ladder_leaves_it_empty():
    """Degrades to stick alignment rather than raising; configure warns."""
    model = _model_with_pooler(_token_pooler(AllPool))
    assert configure_pooling_for_spyre(model, _SPYRE) is True
    assert model.pooler.pooling.len_ladder == []


def test_spyre_all_pool_handles_a_zero_token_request():
    """A zero count has no last real row to clamp onto; it must stay empty."""
    counts = [0, 3]
    hidden_states = torch.arange(sum(counts) * 9, dtype=torch.float16).reshape(-1, 9)
    meta = _counts_metadata(counts)

    deferred = SpyreAllPool(False, defer_trim=True, len_ladder=_LADDER)(hidden_states, meta)
    assert [c.shape[0] for c in deferred] == [0, 64]
    plain = SpyreAllPool(False)(hidden_states, meta)
    assert [c.shape[0] for c in plain] == [0, 3]


# ---------------------------------------------------------------------------
# Every pooled item must come back on one device. An item whose token count
# happens to land on a bucket used to skip the D2H, leaving it on Spyre while
# its neighbours were converted -- and late-interaction scoring rejects a
# query/document device mismatch.
# ---------------------------------------------------------------------------


def test_spyre_token_pooler_converts_every_item_including_exact_bucket(monkeypatch):
    from spyre_inference.v1.pool import spyre_pooler

    seen: list[tuple[int, ...]] = []
    real_convert = spyre_pooler.convert

    def recording_convert(tensor, device=None, dtype=None):
        seen.append(tuple(tensor.shape))
        return real_convert(tensor, device, dtype)

    monkeypatch.setattr(spyre_pooler, "convert", recording_convert)

    # 64 lands exactly on a bucket (no trim needed), 30 does not.
    counts = [64, 30]
    hidden_states = torch.zeros(sum(counts), 9, dtype=torch.float16)
    pooler = _token_pooler(AllPool)
    patch_pooler_for_spyre(pooler, _LADDER)

    got = pooler(hidden_states, _counts_metadata(counts))

    assert [c.shape[0] for c in got] == counts
    # Both items, not just the one that needed trimming.
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# SpyreDispatchPooler: everything else leans on it not slicing, so pin that it
# is installed, that the single-group path hands the sub-pooler the full
# bucketed tensor, and that anything else defers to upstream.
# ---------------------------------------------------------------------------


_ANY_TASK = ("embed", "encode", "token_embed", "classify", "score")


class _RecordingPooler(nn.Module):
    """Stands in for a sub-pooler; records the row count it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_rows: int | None = None

    def get_supported_tasks(self):
        return _ANY_TASK

    def forward(self, hidden_states, pooling_metadata):
        self.seen_rows = hidden_states.shape[0]
        return [hidden_states]


def _dispatch_metadata(counts: list[int], tasks: list[str]):
    cursor = type("C", (), {"num_scheduled_tokens_cpu": torch.tensor(counts)})()

    class _Meta:
        def __init__(self) -> None:
            self.tasks = tasks
            self.pooling_cursor = cursor

        def get_pooling_cursor(self):
            return cursor

    return _Meta()


def test_configure_pooling_installs_spyre_dispatch_pooler():
    pooler = DispatchPooler({"embed": _embed_pooler(MeanPool())})
    model = _model_with_pooler(pooler)
    assert configure_pooling_for_spyre(model, _SPYRE, _LADDER) is True
    assert type(model.pooler) is SpyreDispatchPooler


def test_spyre_dispatch_pooler_keeps_hidden_states_bucketed():
    """Upstream slices to the real token count; the whole point is not to."""
    if not spyre_available():
        pytest.skip("needs Spyre: the bypass is device-gated")

    sub = _RecordingPooler()
    pooler = DispatchPooler({"embed": sub})
    pooler.__class__ = SpyreDispatchPooler
    # 5 real tokens padded up to a 64-row bucket.
    hidden_states = torch.zeros(64, 9, dtype=torch.float16, device="spyre")

    out = pooler(hidden_states, _dispatch_metadata([5], ["embed"]))

    assert sub.seen_rows == 64, "sub-pooler must see the bucketed length, not 5"
    assert len(out) == 1


def test_spyre_dispatch_pooler_defers_to_upstream_for_mixed_tasks(monkeypatch):
    """Several groups need upstream's per-group offsets, so do not bypass."""
    if not spyre_available():
        pytest.skip("needs Spyre: the bypass is device-gated")

    called: list[bool] = []

    def fake_super_forward(self, hidden_states, pooling_metadata):
        called.append(True)
        return []

    monkeypatch.setattr(DispatchPooler, "forward", fake_super_forward)

    pooler = DispatchPooler({"embed": _RecordingPooler(), "encode": _RecordingPooler()})
    pooler.__class__ = SpyreDispatchPooler
    hidden_states = torch.zeros(64, 9, dtype=torch.float16, device="spyre")

    pooler(hidden_states, _dispatch_metadata([5, 5], ["embed", "encode"]))

    assert called == [True], "mixed-task batch must defer to upstream"


# ---------------------------------------------------------------------------
# The swap gate: an unrecognised pooler used to return (0, []), invisible to
# both of configure's gates, so a mixed DispatchPooler passed and the swap then
# handed the unrecognised sub-pooler padded hidden_states.
# ---------------------------------------------------------------------------


class _UnrecognisedPooler(nn.Module):
    """A Pooler subclass patch_pooler_for_spyre knows nothing about."""

    def get_supported_tasks(self):
        return _ANY_TASK

    def forward(self, hidden_states, pooling_metadata):
        return [hidden_states]


def test_unrecognised_pooler_is_reported_as_unsupported():
    num_patched, unsupported = patch_pooler_for_spyre(_UnrecognisedPooler())
    assert (num_patched, unsupported) == (0, ["_UnrecognisedPooler"])


def test_mixed_dispatch_pooler_is_not_swapped():
    """One patched sub-pooler is not enough; the unrecognised one would break."""
    pooler = DispatchPooler({"embed": _embed_pooler(MeanPool()), "encode": _UnrecognisedPooler()})
    model = _model_with_pooler(pooler)

    assert configure_pooling_for_spyre(model, _SPYRE, _LADDER) is False
    assert type(model.pooler) is DispatchPooler
