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

"""The PyTorch path the samplers take when the csrc/ kernels are not built.

Runs whether or not the extension is built: every test forces the fallback."""

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.v1.sample import sampler, topk_topp_sampler
from spyre_inference.v1.sample.sampler import greedy_sample, install_rank0_sampling
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.fixture(autouse=True)
def _no_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampler, "has_sampling_kernels", lambda: False)
    monkeypatch.setattr(topk_topp_sampler, "has_sampling_kernels", lambda: False)


def _meta(rows: int, k: int | None, p: float | None, greedy: bool = False) -> SamplingMetadata:
    z = torch.zeros(rows)
    return SamplingMetadata(
        temperature=None if greedy else torch.full((rows,), 0.8),
        all_greedy=greedy,
        all_random=not greedy,
        top_p=None if p is None else torch.full((rows,), p),
        top_k=None if k is None else torch.full((rows,), k, dtype=torch.long),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=z.clone(),
        presence_penalties=z.clone(),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.mark.parametrize("rows", [1, 8])
def test_greedy_falls_back_to_argmax(rows: int) -> None:
    logits = torch.randn(rows, 32000)
    logits[0, [10, 20]] = logits.max() + 1  # tie: the lowest index wins
    assert torch.equal(greedy_sample(logits), logits.argmax(dim=-1))


# The fallback draws the same fp32 Exp(1) noise from the same RNG calls as stock
# vLLM, in log space, so with tie-free fp32 logits it picks the same tokens.
@pytest.mark.parametrize(
    "k,p,greedy",
    [(None, None, True), (None, None, False), (50, None, False), (50, 0.8, False)],
    ids=["greedy", "random", "topk", "topk_topp"],
)
@pytest.mark.parametrize("rows", [1, 8])
def test_patched_sampler_matches_stock_tokens(
    rows: int, k: int | None, p: float | None, greedy: bool
) -> None:
    meta = _meta(rows, k, p, greedy)
    torch.manual_seed(rows)
    logits = torch.randn(rows, 32000, dtype=torch.float32)

    stock = Sampler()
    torch.manual_seed(1234)
    expected = stock(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    # The runner's in-place patch (TorchSpyreModelRunner.__init__).
    patched = Sampler()
    patched.topk_topp_sampler = SpyreTopKTopPSampler(patched.logprobs_mode, patched.use_fp64_gumbel)
    patched.greedy_sample = greedy_sample  # ty: ignore[invalid-assignment]
    torch.manual_seed(1234)
    actual = patched(logits=logits.clone(), sampling_metadata=meta).sampled_token_ids

    assert torch.equal(actual, expected)


def test_random_draw_matches_softmax_distribution() -> None:
    vocab, draws = 8, 200_000
    logp = torch.log_softmax(torch.randn(vocab) * 1.5, dim=-1)
    torch.manual_seed(0)
    out = SpyreTopKTopPSampler("raw_logprobs", False).forward_native(
        logp.expand(draws, vocab).clone(), {}, None, None
    )[0]
    freq = torch.bincount(out, minlength=vocab).float() / draws
    assert (freq - logp.exp()).abs().max() < 0.01


def test_topk_draw_stays_in_kept_set() -> None:
    logits = torch.randn(64, 32000)
    k = torch.full((64,), 5)
    out = SpyreTopKTopPSampler("raw_logprobs", False).forward_native(logits.clone(), {}, k, None)[0]
    assert (out.unsqueeze(1) == logits.topk(5).indices).any(dim=1).all()


class _TpGroup:
    def __init__(self, first: bool, world_size: int = 2) -> None:
        self.world_size = world_size
        self.is_first_rank = first
        self.ranks = [3, 4]
        self.cpu_group = object()
        self.broadcasts: list[tuple[Any, int]] = []

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        self.broadcasts.append((obj, src))
        # A real broadcast delivers rank 0's object to every rank.
        return "lp"


def _install_tp(monkeypatch: pytest.MonkeyPatch, group: _TpGroup | None) -> None:
    monkeypatch.setattr(parallel_state, "_TP", object() if group is not None else None)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)


def test_single_rank_samples_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.Tensor] = []

    def forward(
        self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None
    ):
        calls.append(logits)
        return SamplerOutput(sampled_token_ids=torch.zeros(logits.shape[0], 1, dtype=torch.int32))

    def fail_broadcast(*args: object, **kwargs: object) -> None:
        raise AssertionError("broadcast")

    monkeypatch.setattr(Sampler, "forward", forward)
    monkeypatch.setattr(torch.distributed, "broadcast", fail_broadcast)

    layer = Sampler()
    install_rank0_sampling(layer)
    logits = torch.zeros(2, 4)
    _install_tp(monkeypatch, None)
    out = layer(logits, sampling_metadata=None)  # ty: ignore[invalid-argument-type]
    assert out.sampled_token_ids.shape == (2, 1)
    _install_tp(monkeypatch, _TpGroup(first=True, world_size=1))
    layer(logits, sampling_metadata=None)  # ty: ignore[invalid-argument-type]
    assert len(calls) == 2


@pytest.mark.parametrize("first", [True, False], ids=["rank0", "other"])
def test_tp_samples_on_rank0_and_broadcasts_over_cpu_group(
    monkeypatch: pytest.MonkeyPatch, first: bool
) -> None:
    tokens = torch.tensor([[4], [9]], dtype=torch.int32)
    sampled: list[int] = []
    tp = _TpGroup(first)

    def forward(
        self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None
    ):
        sampled.append(1)
        return SamplerOutput(
            sampled_token_ids=tokens.clone(),
            logprobs_tensors="lp",  # ty: ignore[invalid-argument-type]
        )

    def broadcast(tensor: torch.Tensor, src: int, group: object) -> None:
        assert src == tp.ranks[0]
        assert group is tp.cpu_group
        if not tp.is_first_rank:
            tensor.copy_(tokens)

    monkeypatch.setattr(Sampler, "forward", forward)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    _install_tp(monkeypatch, tp)

    layer = Sampler()
    install_rank0_sampling(layer)
    out = layer(
        torch.zeros(2, 4),
        sampling_metadata=None,  # ty: ignore[invalid-argument-type]
    )

    assert torch.equal(out.sampled_token_ids, tokens)
    assert out.logprobs_tensors == "lp"
    assert tp.broadcasts == [("lp" if first else None, 0)]
    assert sampled == ([1] if first else [])


def test_fallback_seeded_requests_are_reproducible() -> None:
    topk = SpyreTopKTopPSampler("raw_logprobs", False)
    logits = torch.randn(4, 32000)

    def draw() -> torch.Tensor:
        gens = {i: torch.Generator().manual_seed(i) for i in (0, 2)}
        return topk.forward_native(logits.clone(), gens, None, None)[0]

    a, b = draw(), draw()
    assert torch.equal(a[[0, 2]], b[[0, 2]])


def _runner_with_discarded_prefill() -> TorchSpyreModelRunner:
    # Request 0 is a seeded unfinished prefill, 1 a seeded decode.
    runner = object.__new__(TorchSpyreModelRunner)
    generators = {i: torch.Generator().manual_seed(37 + i) for i in range(2)}
    runner.input_batch = SimpleNamespace(
        num_reqs=2, generators=generators, sampling_metadata=SimpleNamespace(generators=generators)
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False, False]))
    return runner


def test_upstream_rewind_still_needs_the_cpu_generator_port() -> None:
    """Fails once vLLM restores CPU generators itself (vllm#62688): then drop the port (#1420)."""
    bookkeeping = inspect.getsource(GPUModelRunner._bookkeeping_sync)
    assert "gen.set_offset(gen.get_offset() - 4)" in bookkeeping
    # The port hides discarded generators from this method's only read of the dict.
    assert bookkeeping.count("generators") == 1
    assert "get_state" not in inspect.getsource(GPUModelRunner._sample)
    with pytest.raises(RuntimeError, match="does not use offset"):
        torch.Generator().set_offset(0)


def test_sample_restores_discarded_generator_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner_with_discarded_prefill()
    generators = runner.input_batch.generators
    prefill_before, decode_before = (gen.get_state() for gen in generators.values())

    def fail(self, *args):
        for gen in generators.values():
            torch.rand(1, generator=gen)
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(GPUModelRunner, "_sample", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        runner._sample(None, None)
    assert torch.equal(generators[0].get_state(), prefill_before)
    assert not torch.equal(generators[1].get_state(), decode_before)


def test_bookkeeping_restores_generator_mapping_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner_with_discarded_prefill()
    generators = runner.input_batch.generators

    def fail(self, *args):
        assert self.input_batch.generators == {1: generators[1]}
        assert self.input_batch.sampling_metadata.generators is generators
        raise RuntimeError("bookkeeping failed")

    monkeypatch.setattr(GPUModelRunner, "_bookkeeping_sync", fail)
    with pytest.raises(RuntimeError, match="bookkeeping failed"):
        runner._bookkeeping_sync(None, None, None, None, 0)
    assert runner.input_batch.generators is generators
