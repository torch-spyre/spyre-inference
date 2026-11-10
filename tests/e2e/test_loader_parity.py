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

"""Loader parity: the direct-to-device loader and the stock loader agree on greedy output."""

from types import SimpleNamespace

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
_PROMPTS = [
    "The capital of France is",
    "Write one sentence about the sea:",
    "def fibonacci(n):",
    "Q: What is 12 times 12?\nA:",
]
# Two tokens cover prefill (every weight is used) and the first decode step. More tokens
# add no loader coverage but cost minutes: eager generation has a ~85 s per-engine floor.
_MAX_TOKENS = 2

# The loaders are not bit-identical: measured logprob drift between them is 0.03-0.08, the
# same size as the APC-vs-plain noise on a single loader, and exact fp16 ties flip the
# greedy pick. So a divergence must be such a tie (each engine's pick within
# _TIE_TOLERANCE of the other's best candidate), and the logprobs up to it must stay within
# _MAX_LOGPROB_DRIFT (~2.6x the largest drift seen). A miss-loaded weight moves logprobs by
# tenths or more.
_TIE_TOLERANCE = 0.15
_MAX_LOGPROB_DRIFT = 0.2


def _within_tie(logprobs: dict, token_id: int) -> bool:
    """Whether ``token_id`` is within ``_TIE_TOLERANCE`` of the best candidate."""
    if token_id not in logprobs:
        return False
    best = max(lp.logprob for lp in logprobs.values())
    return best - logprobs[token_id].logprob <= _TIE_TOLERANCE


def _assert_same_greedy_output(i: int, spyre, stock) -> None:
    s, r = spyre.outputs[0], stock.outputs[0]
    n = min(len(s.token_ids), len(r.token_ids))
    k = next((j for j in range(n) if s.token_ids[j] != r.token_ids[j]), None)
    # Both engines saw the same context up to and including position k, so logprobs compare.
    upto = n if k is None else k + 1
    drift = max(
        (
            abs(s.logprobs[j][t].logprob - r.logprobs[j][t].logprob)
            for j in range(upto)
            for t in s.logprobs[j].keys() & r.logprobs[j].keys()
        ),
        default=0.0,
    )
    outputs = f"  spyre : {list(s.token_ids)}\n  stock : {list(r.token_ids)}"

    assert drift <= _MAX_LOGPROB_DRIFT, (
        f"Prompt {i}: logprobs drift by {drift:.3f} (> {_MAX_LOGPROB_DRIFT}) between loaders.\n"
        f"{outputs}"
    )
    if k is None:
        assert len(s.token_ids) == len(r.token_ids), (
            f"Prompt {i}: identical prefix but one output stopped early "
            f"({len(s.token_ids)} vs {len(r.token_ids)} tokens).\n{outputs}"
        )
        return
    spyre_pick_is_close = _within_tie(r.logprobs[k], s.token_ids[k])
    stock_pick_is_close = _within_tie(s.logprobs[k], r.token_ids[k])
    assert spyre_pick_is_close and stock_pick_is_close, (
        f"Prompt {i}: outputs diverge at token {k} and it is not a near-tie.\n{outputs}"
    )


def _generate(load_format: str, tp_size: int):
    """Greedy outputs with logprobs, plus the load format the engine actually resolved."""
    llm = LLM(
        model=_MODEL,
        dtype="float16",
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        load_format=load_format,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )
    resolved = llm.llm_engine.vllm_config.load_config.load_format
    sp = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, logprobs=5)
    outputs = llm.generate(_PROMPTS, sp, use_tqdm=False)
    del llm
    return resolved, outputs


def _check_loader_parity(tp_size: int) -> None:
    """``auto`` (resolved to the spyre loader) and ``safetensors`` give the same greedy output."""
    resolved, spyre = _generate("auto", tp_size)
    assert resolved == "spyre", "the default load format should resolve to the direct loader"
    resolved, stock = _generate("safetensors", tp_size)
    assert resolved == "safetensors"

    assert len(spyre) == len(stock) == len(_PROMPTS)
    for i, (s, r) in enumerate(zip(spyre, stock)):
        _assert_same_greedy_output(i, s, r)


@pytest.mark.uses_subprocess
def test_spyre_loader_matches_stock_loader_tp1() -> None:
    _check_loader_parity(tp_size=1)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_spyre_loader_matches_stock_loader_tp2() -> None:
    """At TP=2 the spyre loader shards linear weights on read (``_iter_tp_sharded_weights``);
    the stock loader narrows on the host. Needs two cards."""
    _check_loader_parity(tp_size=2)


def _fake(ids, per_position):
    """A RequestOutput stand-in: token ids plus {token_id: logprob} per position."""
    logprobs = [{t: SimpleNamespace(logprob=v) for t, v in p.items()} for p in per_position]
    return SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, logprobs=logprobs)])


_PLAIN = ([1, 2], [{1: -0.1}, {2: -0.2}])

_CHECK_CASES = [
    pytest.param(_PLAIN, _PLAIN, True, id="identical"),
    pytest.param(_PLAIN, ([1, 2], [{1: -0.14}, {2: -0.25}]), True, id="small-drift"),
    pytest.param(
        ([1, 3], [{1: -0.1}, {3: -1.807, 4: -1.807}]),
        ([1, 4], [{1: -0.1}, {4: -1.782, 3: -1.829}]),
        True,
        id="tie-flip",
    ),
    pytest.param(
        ([1, 3], [{1: -0.1}, {3: -0.2, 4: -3.0}]),
        ([1, 4], [{1: -0.1}, {4: -0.25, 3: -3.1}]),
        False,
        id="clear-preference",
    ),
    pytest.param(_PLAIN, ([1, 2], [{1: -0.1}, {2: -1.4}]), False, id="bad-weights"),
    pytest.param(_PLAIN, ([1], [{1: -0.1}]), False, id="early-stop"),
    # Drift is small (0.16 < 0.2), but stock prefers token 4 by 0.16 > 0.15: not a tie.
    pytest.param(
        ([1, 3], [{1: -0.1}, {3: -0.5, 4: -0.5}]),
        ([1, 4], [{1: -0.1}, {4: -0.5, 3: -0.66}]),
        False,
        id="one-sided-preference",
    ),
]


@pytest.mark.parametrize(("spyre", "stock", "ok"), _CHECK_CASES)
def test_parity_check_logic(spyre, stock, ok) -> None:
    if ok:
        _assert_same_greedy_output(0, _fake(*spyre), _fake(*stock))
    else:
        with pytest.raises(AssertionError):
            _assert_same_greedy_output(0, _fake(*spyre), _fake(*stock))
