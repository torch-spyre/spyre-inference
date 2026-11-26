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

"""suffix_decode.py: the V2 suffix speculator against fake request state, and the runner's
swap of upstream's init_speculator."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import model_runner as upstream_model_runner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre import model_runner
from spyre_inference.v1.worker.spyre.suffix_decode import (
    SpyreSuffixSpeculator,
    building_suffix_speculator,
)

K = 3
MAX_MODEL_LEN = 64
MAX_REQS = 4


class _Cache:
    """Records what the speculator feeds arctic-inference's SuffixDecodingCache."""

    def __init__(self, max_tree_depth, max_cached_requests):
        self.active_requests: set[str] = set()
        self.cached_requests: set[str] = set()
        self.calls: list[tuple] = []
        self.next_draft: list[int] = []

    def start_request(self, req_id, prompt):
        self.calls.append(("start", req_id, list(prompt)))
        self.active_requests.add(req_id)

    def add_active_response(self, req_id, tokens):
        self.calls.append(("add", req_id, list(tokens)))

    def stop_request(self, req_id):
        self.calls.append(("stop", req_id))
        self.active_requests.discard(req_id)
        self.cached_requests.add(req_id)

    def evict_cached_response(self, req_id):
        self.calls.append(("evict", req_id))
        self.cached_requests.discard(req_id)

    def speculate(self, req_id, pattern, **kwargs):
        self.calls.append(("speculate", req_id, list(pattern), kwargs["max_spec_tokens"]))
        return SimpleNamespace(token_ids=self.next_draft)


def _config(max_tree_depth=24):
    spec = SimpleNamespace(
        method="suffix",
        num_speculative_tokens=K,
        suffix_decoding_max_tree_depth=max_tree_depth,
        suffix_decoding_max_spec_factor=1.0,
        suffix_decoding_min_token_prob=0.1,
        suffix_decoding_max_cached_requests=8,
    )
    return SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_REQS),
    )


class _Requests:
    """The RequestState fields the speculator reads, on the host as the Spyre runner has them."""

    def __init__(self):
        self.all_token_ids = SimpleNamespace(
            gpu=torch.zeros(MAX_REQS, MAX_MODEL_LEN, dtype=torch.int32)
        )
        self.total_len = SimpleNamespace(gpu=torch.zeros(MAX_REQS, dtype=torch.int32))
        self.prompt_len = SimpleNamespace(np=np.zeros(MAX_REQS, dtype=np.int32))
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}

    def add(self, req_id, idx, prompt, outputs=()):
        tokens = [*prompt, *outputs]
        self.all_token_ids.gpu[idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        self.total_len.gpu[idx] = len(tokens)
        self.prompt_len.np[idx] = len(prompt)
        self.req_id_to_index[req_id] = idx
        self.index_to_req_id[idx] = req_id

    def append(self, idx, tokens):
        n = int(self.total_len.gpu[idx])
        self.all_token_ids.gpu[idx, n : n + len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        self.total_len.gpu[idx] = n + len(tokens)

    def remove(self, req_id):
        self.index_to_req_id.pop(self.req_id_to_index.pop(req_id))


@pytest.fixture()
def fake_cache(monkeypatch):
    # The parent too: a from-import loads it first, and arctic-inference may be absent.
    package = ModuleType("arctic_inference")
    module = ModuleType("arctic_inference.suffix_decoding")
    module.SuffixDecodingCache = _Cache  # ty: ignore[unresolved-attribute]
    package.suffix_decoding = module  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "arctic_inference", package)
    monkeypatch.setitem(sys.modules, "arctic_inference.suffix_decoding", module)


@pytest.fixture()
def requests():
    return _Requests()


@pytest.fixture()
def speculator(fake_cache, requests):
    return SpyreSuffixSpeculator(_config(), requests)  # ty: ignore[invalid-argument-type]


def _propose(speculator, requests, rows, num_sampled=None):
    """Propose for the requests at slots ``rows``, each with its last token as last sampled."""
    idx = torch.tensor(rows, dtype=torch.int64)
    last = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
    for r in rows:
        last[r, 0] = requests.all_token_ids.gpu[r, int(requests.total_len.gpu[r]) - 1]
    sampled = torch.ones(len(rows), dtype=torch.int32) if num_sampled is None else num_sampled
    unused = torch.empty(0)
    return speculator.propose(
        SimpleNamespace(num_reqs=len(rows), idx_mapping=idx),
        None,
        None,
        unused,
        None,
        sampled,
        unused,
        last,
        unused,
        unused,
        unused,
    ).tolist()


def test_a_new_request_starts_from_its_prompt_and_adds_its_outputs(speculator, requests):
    requests.add("a", 2, prompt=[5, 6, 7], outputs=[8])

    _propose(speculator, requests, [2])

    calls = speculator.suffix_cache.calls
    assert calls[:2] == [("start", "a", [5, 6, 7]), ("add", "a", [8])]
    assert calls[2] == ("speculate", "a", [5, 6, 7, 8], K)


def test_a_later_step_feeds_only_the_new_tokens(speculator, requests):
    requests.add("a", 0, prompt=[5, 6, 7], outputs=[8])
    _propose(speculator, requests, [0])
    speculator.suffix_cache.calls.clear()

    requests.append(0, [9, 10])
    _propose(speculator, requests, [0])

    assert speculator.suffix_cache.calls[0] == ("add", "a", [9, 10])


def test_a_short_speculation_is_filled_with_the_last_sampled_token(speculator, requests):
    requests.add("a", 1, prompt=[5, 6, 7], outputs=[8])
    speculator.suffix_cache.next_draft = [11]

    assert _propose(speculator, requests, [1]) == [[11, 8, 8]]


def test_no_speculation_verifies_the_last_sampled_token(speculator, requests):
    requests.add("a", 1, prompt=[5, 6, 7], outputs=[8])

    assert _propose(speculator, requests, [1]) == [[8, 8, 8]]


def test_a_partial_prefill_is_not_fed(speculator, requests):
    requests.add("a", 0, prompt=[5, 6, 7])

    drafts = _propose(speculator, requests, [0], num_sampled=torch.zeros(1, dtype=torch.int32))

    assert speculator.suffix_cache.calls == []
    assert drafts == [[7, 7, 7]]


def test_the_pattern_is_the_last_max_tree_depth_tokens(fake_cache, requests):
    speculator = SpyreSuffixSpeculator(_config(max_tree_depth=2), requests)  # ty: ignore[invalid-argument-type]
    requests.add("a", 0, prompt=[5, 6, 7], outputs=[8])

    _propose(speculator, requests, [0])

    assert speculator.suffix_cache.calls[-1] == ("speculate", "a", [7, 8], K)


def test_speculation_stops_short_of_max_model_len(speculator, requests):
    requests.add("a", 0, prompt=list(range(MAX_MODEL_LEN - 2)), outputs=[1])

    _propose(speculator, requests, [0])

    assert speculator.suffix_cache.calls[-1][-1] == 0


def test_a_request_that_left_the_state_is_stopped(speculator, requests):
    requests.add("a", 0, prompt=[5, 6], outputs=[7])
    requests.add("b", 1, prompt=[1, 2], outputs=[3])
    _propose(speculator, requests, [0, 1])

    requests.remove("a")
    _propose(speculator, requests, [1])

    assert ("stop", "a") in speculator.suffix_cache.calls
    assert speculator.suffix_cache.active_requests == {"b"}


def test_a_resumed_request_restarts_from_its_prompt(speculator, requests):
    requests.add("a", 0, prompt=[5, 6], outputs=[7])
    _propose(speculator, requests, [0])
    requests.remove("a")
    _propose(speculator, requests, [])
    speculator.suffix_cache.calls.clear()

    # Re-added after preemption with its outputs so far, in another slot.
    requests.add("a", 3, prompt=[5, 6], outputs=[7, 9])
    _propose(speculator, requests, [3])

    assert speculator.suffix_cache.calls[:3] == [
        ("evict", "a"),
        ("start", "a", [5, 6]),
        ("add", "a", [7, 9]),
    ]


def test_a_dummy_run_reads_no_request_state(speculator):
    drafts = speculator.propose(SimpleNamespace(num_reqs=2), *[None] * 10, dummy_run=True)  # ty: ignore[invalid-argument-type]
    assert drafts.shape == (2, K)


def test_the_real_cache_drafts_a_repeated_continuation(requests):
    pytest.importorskip("arctic_inference.suffix_decoding")
    speculator = SpyreSuffixSpeculator(_config(), requests)  # ty: ignore[invalid-argument-type]
    # The prompt repeats 1 2 3 4, so after "... 4 1 2" the cache continues with 3 4.
    requests.add("a", 0, prompt=[1, 2, 3, 4] * 4, outputs=[1, 2])

    (drafts,) = _propose(speculator, requests, [0])

    assert drafts[:2] == [3, 4]


def _runner_with(monkeypatch, method):
    """Build the runner over an upstream __init__ that, like the real one, asks
    init_speculator for its speculator."""

    def upstream_init(self, vllm_config, device):
        self.device = device
        self.speculator = upstream_model_runner.init_speculator(vllm_config, device, "states")

    monkeypatch.setattr(GPUModelRunner, "__init__", upstream_init)
    monkeypatch.setattr(model_runner, "validate_dflash_config", lambda config: None)
    config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
    return model_runner.SpyreModelRunnerV2(config, torch.device("meta"))  # ty: ignore[invalid-argument-type]


def test_the_runner_builds_the_suffix_speculator(monkeypatch):
    built = []
    monkeypatch.setattr(
        SpyreSuffixSpeculator,
        "__init__",
        lambda self, config, req_states: built.append(req_states),
    )

    original = upstream_model_runner.init_speculator

    runner = _runner_with(monkeypatch, "suffix")

    assert isinstance(runner.speculator, SpyreSuffixSpeculator)
    assert built == ["states"]
    # The swap is scoped to __init__.
    assert upstream_model_runner.init_speculator is original


def test_the_runner_refuses_a_bypassed_swap(monkeypatch):
    def upstream_init(self, vllm_config, device):
        self.speculator = object()

    monkeypatch.setattr(GPUModelRunner, "__init__", upstream_init)
    monkeypatch.setattr(model_runner, "validate_dflash_config", lambda config: None)
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="suffix"))

    with pytest.raises(RuntimeError, match="bypassed Spyre's swap of init_speculator"):
        model_runner.SpyreModelRunnerV2(config, torch.device("meta"))  # ty: ignore[invalid-argument-type]


def test_the_swap_is_undone_when_init_fails():
    original = upstream_model_runner.init_speculator
    with pytest.raises(ValueError), building_suffix_speculator():
        raise ValueError
    assert upstream_model_runner.init_speculator is original
