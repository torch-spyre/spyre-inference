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

"""Compare two greedy runs by logprob, so an fp16 near-tie flip between them is not a failure."""

from __future__ import annotations

from collections.abc import Sequence

# Per prompt: the sampled token ids, and per step the run's top-k {token id: logprob}.
Trace = tuple[list[list[int]], list[list[dict[int, float]]]]


def logprob_trace(outputs: Sequence) -> Trace:
    """The trace of ``LLM.generate`` outputs; the request needs ``logprobs=k``."""
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    steps = [
        [{tid: lp.logprob for tid, lp in step.items()} for step in o.outputs[0].logprobs]
        for o in outputs
    ]
    return tokens, steps


def assert_greedy_runs_agree(
    ref: Trace, run: Trace, tolerance: float, names: tuple[str, str] = ("ref", "run")
) -> None:
    """Up to the first divergence, which must be a near tie, the picked tokens' logprobs agree
    within ``tolerance``; low-ranked tokens drift further and are not bounded."""
    a, b = names
    assert len(ref[0]) == len(run[0]), f"prompt count: {a} {len(ref[0])} vs {b} {len(run[0])}"
    for name, (tokens, steps) in ((a, ref), (b, run)):
        for i, (seq, rows) in enumerate(zip(tokens, steps, strict=True)):
            assert len(seq) == len(rows), (
                f"prompt {i}: {name} has {len(seq)} tokens but {len(rows)} logprob rows"
            )
    for i, (ref_ids, ids, ref_rows, rows) in enumerate(
        zip(ref[0], run[0], ref[1], run[1], strict=True)
    ):
        # strict: without a near-tie flip both runs must generate the same number of tokens.
        for step, (ref_pick, pick, ref_lp, lp) in enumerate(
            zip(ref_ids, ids, ref_rows, rows, strict=True)
        ):
            where = f"prompt {i} step {step}"
            assert ref_pick in lp, f"{where}: {a} token {ref_pick} not in {b} top {lp}"
            assert pick in ref_lp, f"{where}: {b} token {pick} not in {a} top {ref_lp}"
            for tid in {ref_pick, pick}:
                gap = abs(ref_lp[tid] - lp[tid])
                assert gap <= tolerance, (
                    f"{where}: token {tid} logprob {a} {ref_lp[tid]:.3f} vs {b} {lp[tid]:.3f} "
                    f"(gap {gap:.3f} > {tolerance})"
                )
            if pick != ref_pick:
                margins = (ref_lp[ref_pick] - ref_lp[pick], lp[pick] - lp[ref_pick])
                assert max(margins) <= tolerance, (
                    f"{where}: {a} picked {ref_pick}, {b} picked {pick}; not a near tie "
                    f"(margins {a} {margins[0]:.3f}, {b} {margins[1]:.3f} > {tolerance})"
                )
                # Past a flip the two runs condition on different tokens.
                break
