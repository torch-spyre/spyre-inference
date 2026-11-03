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

"""Real Qwen3 batched verification, request isolation and changing active batches."""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from test_speculative_decoding import (
    DRAFT,
    DRAFT_REVISION,
    TARGET,
    TARGET_CACHE_NAMES,
    TARGET_REVISION,
    _acceptance_metrics,
    _check_acceptance_metrics,
    _compare_cache,
)
from vllm import LLM, SamplingParams

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]


@pytest.mark.parametrize("tuned", [False, True], ids=["defaults", "tuned"])
def test_qwen3_xpress_batch_four(monkeypatch, tmp_path, record_property, tuned):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", "head_major")
    if tuned:
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16,64,128")
        monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "4")
    else:
        monkeypatch.delenv("SPYRE_ATTN_QUERY_BUCKETS", raising=False)
        monkeypatch.delenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", raising=False)
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    llm = LLM(
        model=TARGET,
        revision=TARGET_REVISION,
        tokenizer_revision=TARGET_REVISION,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        enable_prefix_caching=False,
        disable_log_stats=False,
        compilation_config={"compile_sizes": [1, 16, 32, 64, 128]} if tuned else None,
        worker_cls="xpress_batch_validation_worker.BatchValidationWorker",
        speculative_config=dict(
            method="dflash",
            model=DRAFT,
            revision=DRAFT_REVISION,
            num_speculative_tokens=15,
            xpress_num_passes=6,
            xpress_topc=int(os.environ.get("SPYRE_XPRESS_TEST_TOPC", "128")),
        ),
    )
    report = {
        "complete": False,
        "tuned": tuned,
        "topc": int(os.environ.get("SPYRE_XPRESS_TEST_TOPC", "128")),
        "attention_record": os.environ.get("SPYRE_ATTN_RECORD", "1"),
    }
    report_path = tmp_path / "xpress-batch-validation.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        repeated = llm.get_tokenizer().encode(
            "IBM develops computers, software, and consulting services. " * 64
        )
        cases = [
            dict(prompt_ids=llm.get_tokenizer().encode(prompt), max_tokens=limit, reject_at=reject)
            for prompt, limit, reject in (
                ("1, 2, 3, 4,", 24, 0),
                ("2, 3, 4, 5, 6,", 32, 2),
                ("3, 4, 5, 6, 7, 8,", 40, 5),
                ("3, 4, 5, 6, 7, 8, 9,", 48, -1),
            )
        ]
        assert sum(len(case["prompt_ids"]) for case in cases) <= 128 - 4 * 15
        engine = llm.llm_engine

        def run(
            label, selected, mode, references=None, salt=0, capture=False, overrides=None, passes=6
        ):
            llm.sleep(level=0, mode="keep")
            # The pause reply follows prior engine outputs on the same socket.
            # Drain cancelled-wave statistics before resetting worker counters.
            while not engine.engine_core.outputs_queue.empty():
                engine.step()
            llm.collective_rpc(
                "configure_proposer",
                args=(
                    mode,
                    references,
                    -1,
                    passes,
                    salt,
                    str(tmp_path / label) if capture else None,
                ),
            )
            before = _acceptance_metrics(llm)
            requested = []
            internal_ids = {}
            for index, case in enumerate(selected):
                request_id = f"{label}_{index}"
                internal_ids[request_id] = engine.add_request(
                    request_id,
                    {"prompt_token_ids": case["prompt_ids"]},
                    SamplingParams(
                        **(
                            dict(
                                temperature=0.0,
                                max_tokens=case["max_tokens"],
                                ignore_eos=True,
                                logprobs=5,
                            )
                            | case.get("sampling", {})
                            | (overrides or {})
                        )
                    ),
                )
                requested.append(request_id)
            llm.wake_up(tags=["scheduling"])
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = output
            assert set(finished) == set(requested)
            state = llm.collective_rpc("validation_state")[0]
            state["request_ids"] = internal_ids
            metrics = _check_acceptance_metrics(
                before, _acceptance_metrics(llm), state["verification_steps"]
            )
            result = [finished[request_id] for request_id in requested]
            for output, case in zip(result, selected):
                completion = output.outputs[0]
                if not overrides:
                    assert len(completion.token_ids) == case["max_tokens"]
                assert len(completion.logprobs) == len(completion.token_ids)
                temperature = (case.get("sampling", {}) | (overrides or {})).get("temperature", 0)
                for token, row in zip(completion.token_ids, completion.logprobs, strict=True):
                    assert token in row and len(row) >= 5
                    assert all(math.isfinite(value.logprob) for value in row.values())
                    if temperature == 0:
                        assert row[token].logprob == max(value.logprob for value in row.values())
                if capture:
                    assert internal_ids[output.request_id] in state["snapshots"]
            report[label] = dict(
                outputs=[list(output.outputs[0].token_ids) for output in result],
                logprobs=[
                    [
                        {token: value.logprob for token, value in row.items()}
                        for row in output.outputs[0].logprobs
                    ]
                    for output in result
                ],
                state=state,
                acceptance=metrics,
            )
            save()
            return result, state

        baseline, baseline_state = run("baseline", cases, "none", capture=True)
        references = [
            dict(
                prompt_ids=case["prompt_ids"],
                token_ids=list(output.outputs[0].token_ids),
                reject_at=case["reject_at"],
            )
            for case, output in zip(cases, baseline)
        ]
        snapshots = {}
        for label, mode, salt in (
            ("controlled", "real_scripted", 0),
            ("changed_rejected_tail", "real_scripted", 101),
            ("real", "real", 0),
        ):
            outputs, state = run(label, cases, mode, references, salt, capture=True)
            caches = []
            report[label]["cache_errors"] = {}
            for index, output in enumerate(outputs):
                assert list(output.outputs[0].token_ids) == list(
                    baseline[index].outputs[0].token_ids
                )
                cache = torch.load(
                    state["snapshots"][state["request_ids"][output.request_id]]["path"],
                    weights_only=True,
                )
                caches.append(cache)
                if label == "changed_rejected_tail":
                    assert (
                        list(output.outputs[0].token_ids) == report["controlled"]["outputs"][index]
                    )
                    errors = _compare_cache(cache, snapshots["controlled"][index], exact=True)
                else:
                    base = torch.load(
                        baseline_state["snapshots"][
                            baseline_state["request_ids"][baseline[index].request_id]
                        ]["path"],
                        weights_only=True,
                    )
                    errors = _compare_cache(
                        {k: v for k, v in cache.items() if k in TARGET_CACHE_NAMES},
                        base,
                        prompt_length=len(cases[index]["prompt_ids"]),
                    )
                report[label]["cache_errors"][index] = errors
            snapshots[label] = caches
            save()
            assert any(
                len(batch) == 4 and all(row["proposals"] for row in batch)
                for batch in state["batch_steps"]
            )
            assert any(len(batch) < 4 for batch in state["batch_steps"])
            for request_id in {row["request"] for row in state["steps"]}:
                rows = [row for row in state["steps"] if row["request"] == request_id]
                assert all(b["position"] == a["committed_end"] for a, b in zip(rows, rows[1:]))
            assert state["drafter"]["proposal_batch_sizes"][4] > 0

        mixed = [dict(prompt_ids=repeated[:n], max_tokens=64) for n in (24, 48, 80, 127)]
        _, mixed_state = run("partial_prefill", mixed, "real")
        assert any(
            any(not row["sampled"] for row in batch) and any(row["drafts"] for row in batch)
            for batch in mixed_state["batch_steps"]
        )
        assert any(len(batch) == 4 for batch in mixed_state["batch_steps"])

        near_limit = [dict(prompt_ids=repeated[:503], max_tokens=8), *mixed[:3]]
        _, limited_state = run("context_limit", near_limit, "real")
        assert any(
            any(row["committed_end"] > 496 and not row["proposals"] for row in batch)
            and any(row["proposals"] for row in batch)
            for batch in limited_state["batch_steps"]
        )

        llm.collective_rpc("configure_proposer", args=("real",))
        llm.sleep(level=0, mode="keep")
        cancelled = [
            engine.add_request(
                f"cancel_{i}",
                {"prompt_token_ids": case["prompt_ids"]},
                SamplingParams(temperature=0.0, max_tokens=192, ignore_eos=True),
            )
            for i, case in enumerate(cases)
        ]
        llm.wake_up(tags=["scheduling"])
        for _ in range(3):
            engine.step()
        cancel_state = llm.collective_rpc("validation_state")[0]
        engine.abort_request(cancelled[:1], internal=True)
        # Frontend step() can consume outputs queued before the abort; wait
        # for an actual worker batch without the cancelled request.
        for _ in range(16):
            engine.step()
            shrunk_state = llm.collective_rpc("validation_state")[0]
            if any(len(batch) == 3 for batch in shrunk_state["batch_steps"]):
                break
            if not engine.has_unfinished_requests():
                break
        report["cancel_before"] = cancel_state
        report["cancel_after_one"] = shrunk_state
        save()
        assert any(len(batch) == 3 for batch in shrunk_state["batch_steps"])
        assert cancelled[0] not in shrunk_state["runner_requests"]
        engine.abort_request(cancelled[1:], internal=True)
        assert not engine.has_unfinished_requests()
        clean, clean_state = run("after_cancel", cases, "real_scripted", references, capture=True)
        cancelled_blocks = {b for row in cancel_state["steps"] for b in row["blocks"] if b}
        reused_blocks = {b for row in clean_state["steps"] for b in row["blocks"] if b}
        assert cancelled_blocks & reused_blocks
        assert not set(cancelled) & set(clean_state["runner_requests"])
        for index, output in enumerate(clean):
            assert list(output.outputs[0].token_ids) == report["controlled"]["outputs"][index]
            cache = torch.load(
                clean_state["snapshots"][clean_state["request_ids"][output.request_id]]["path"],
                weights_only=True,
            )
            _compare_cache(cache, snapshots["controlled"][index], exact=True)
        report["cancellation"] = dict(
            cancelled=cancelled, reused_blocks=sorted(cancelled_blocks & reused_blocks)
        )

        single, _ = run("batch_one_regression", cases[:1], "real")
        assert list(single[0].outputs[0].token_ids) == list(baseline[0].outputs[0].token_ids)
        stop_id = references[0]["token_ids"][3]
        stopped, _ = run(
            "token_stop", cases, "scripted", references, overrides={"stop_token_ids": [stop_id]}
        )
        assert any(output.outputs[0].finish_reason == "stop" for output in stopped)

        stochastic = [
            dict(case, sampling=dict(options, seed=37 + index))
            for index, (case, options) in enumerate(
                zip(
                    cases,
                    (
                        dict(temperature=0.7),
                        dict(temperature=0.7, top_k=32),
                        dict(temperature=0.7, top_p=0.8),
                        dict(temperature=0.7, top_k=32, top_p=0.8),
                    ),
                    strict=True,
                )
            )
        ]
        for label, mode, passes in (
            ("ordinary", "none", 6),
            ("dflash", "real", 0),
            ("xpress", "real", 6),
        ):
            original, original_state = run(
                f"{label}_seeded", stochastic, mode, capture=True, passes=passes
            )
            replay, replay_state = run(
                f"{label}_replay", stochastic, mode, capture=True, passes=passes
            )
            assert report[f"{label}_seeded"]["logprobs"] == report[f"{label}_replay"]["logprobs"]
            for first, second in zip(original, replay, strict=True):
                assert first.outputs[0].token_ids == second.outputs[0].token_ids
                caches = [
                    torch.load(
                        state["snapshots"][state["request_ids"][output.request_id]]["path"],
                        weights_only=True,
                    )
                    for output, state in ((first, original_state), (second, replay_state))
                ]
                _compare_cache(*caches, exact=True)
            if mode == "real":
                assert original_state["drafter"]["proposal_batch_sizes"][4] > 0
        report["complete"] = True
        save()
        record_property("validation_report", str(report_path))
    finally:
        llm.llm_engine.engine_core.shutdown()
