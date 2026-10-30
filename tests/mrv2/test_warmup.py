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

"""warmup.plan() and the dummy-run wrapper, without executing a model."""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.worker.spyre import warmup


def _runner(*, pooling=False, eager=False, buckets=(1, 2, 4, 64, 512), max_reqs=8, max_tokens=512):
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=list(buckets)),
        model_config=SimpleNamespace(runner_type="pooling" if pooling else "generate"),
    )
    return SimpleNamespace(
        vllm_config=vllm_config,
        model_config=SimpleNamespace(enforce_eager=eager),
        is_pooling_model=pooling,
        max_num_reqs=max_reqs,
        max_num_tokens=max_tokens,
    )


@pytest.fixture(autouse=True)
def _bucketer(monkeypatch):
    """SpyreShapeBucketer.for_config with only what plan() reads: the bucket list, or
    None for an eager decoder (which runs no bucketer)."""

    def for_config(config):
        sizes = config.compilation_config.compile_sizes
        return SimpleNamespace(bucket_sizes=sorted(sizes)) if sizes else None

    monkeypatch.setattr(warmup.SpyreShapeBucketer, "for_config", staticmethod(for_config))


def test_decoder_dummies_cover_every_bucket_largest_first_and_skip_attention() -> None:
    calls = warmup.plan(_runner())
    assert [c.num_tokens for c in calls] == [512, 64, 4, 2, 1]
    assert all(c.skip_attn for c in calls)


def test_num_reqs_is_the_token_count_capped_at_max_num_seqs() -> None:
    """Not 1: a one-request dummy would trace a prefill-shaped graph for every bucket."""
    calls = warmup.plan(_runner(max_reqs=8))
    assert [c.num_reqs for c in calls] == [8, 8, 4, 2, 1]


def test_pooling_dummies_keep_attention() -> None:
    calls = warmup.plan(_runner(pooling=True, buckets=(2048,)))
    assert [(c.num_tokens, c.skip_attn) for c in calls] == [(2048, False)]


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(_runner(buckets=()), id="no-bucketer"),
        pytest.param(_runner(pooling=True, eager=True), id="eager-pooling"),
    ],
)
def test_without_buckets_one_short_dummy_runs(runner) -> None:
    calls = warmup.plan(runner)
    assert [c.num_tokens for c in calls] == [16]


def test_a_decoder_dummy_publishes_null_slots_and_a_pooling_one_does_not(monkeypatch) -> None:
    published = []
    monkeypatch.setattr(warmup.attn_layer, "publish_null_slots", published.append)
    seen = []
    wrapper = SimpleNamespace(_keep_outputs_on_device=False)

    def upstream(num_tokens, *, skip_attn, is_profile):
        seen.append((num_tokens, skip_attn, is_profile))
        return torch.zeros(num_tokens, 4), torch.zeros(1, 4)

    runner = SimpleNamespace(model=wrapper, _dummy_run=upstream, spyre_device="cpu")

    warmup._dummy_run(runner, 8, skip_attn=True)
    warmup._dummy_run(runner, 16, skip_attn=False)

    assert published == [8]
    assert seen == [(8, True, True), (16, False, True)]


def test_a_pooling_dummy_hands_back_device_hidden_states(monkeypatch) -> None:
    """Upstream gathers logits_indices on the host, so the wrapper is told to D2H for
    the call; the pooler then reads the hidden states where it expects them."""
    flags = []
    wrapper = SimpleNamespace(_keep_outputs_on_device=True)

    def upstream(num_tokens, *, skip_attn, is_profile):
        flags.append(wrapper._keep_outputs_on_device)
        return torch.zeros(num_tokens, 4), None

    moved = []
    monkeypatch.setattr(warmup, "convert", lambda t, device: moved.append(device) or t)
    runner = SimpleNamespace(model=wrapper, _dummy_run=upstream, spyre_device="spyre")

    warmup._dummy_run(runner, 4, skip_attn=False)

    assert flags == [False]
    assert wrapper._keep_outputs_on_device is True
    assert moved == ["spyre"]
