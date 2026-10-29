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

"""Qwen3-8B: target verification, cache rollback, DFlash and XPress on one card."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from vllm import LLM, SamplingParams

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

TARGET = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT = "UIUC-SSAIL/Qwen3-8B-XPress-b16"
DRAFT_REVISION = "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5"
TARGET_CACHE_NAMES = {f"model.layers.{i}.self_attn.attn" for i in range(36)}
DRAFT_CACHE_NAMES = {f"model.layers.{i}.self_attn.attn" for i in range(36, 41)}


def _cached_model(model, revision):
    try:
        path = Path(
            snapshot_download(
                model,
                revision=revision,
                local_files_only=True,
                allow_patterns=["config.json", "*.safetensors"],
            )
        )
    except FileNotFoundError:
        pytest.skip(f"Cache {model}@{revision} before running this hardware test")
    if not list(path.glob("*.safetensors")):
        pytest.skip(f"Missing safetensors weights in {path}")


def _compare_cache(actual, expected, *, prompt_length=0, exact=False):
    assert actual.keys() == expected.keys()
    errors = {}
    for name, pair in actual.items():
        assert len(pair) == len(expected[name]) == 2
        for label, a, b in zip(("key", "value"), pair, expected[name]):
            assert a.shape == b.shape, (name, label, a.shape, b.shape)
            a, b = a.double(), b.double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.equal(a[:prompt_length], b[:prompt_length]), (name, label, "prompt")
            relative = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
            if exact:
                assert torch.equal(a, b), (name, label, relative)
            elif name == "model.layers.0.self_attn.attn":
                assert relative < 0.005, (name, label, relative)
            # FP16 differences can grow across layers when token shapes change.
            # Exact comparisons below hold the schedule fixed and vary rejected tails.
            errors[f"{name}.{label}"] = relative
    assert errors
    return errors


def _acceptance_metrics(engine):
    names = {
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "num_accepted_tokens_per_pos",
    }
    prefix = "vllm:spec_decode_"
    result = {}
    for metric in engine.get_metrics():
        name = metric.name.removeprefix(prefix)
        if name in names:
            assert name not in result
            result[name] = metric.values if name.endswith("_per_pos") else metric.value
    assert result.keys() == names
    return result


def _check_acceptance_metrics(before, after, steps):
    verified = [s for s in steps if s["drafts"] and s["sampled"]]
    accepted = [len(s["sampled"]) - 1 for s in verified]
    expected = dict(
        num_drafts=len(verified),
        num_draft_tokens=sum(s["drafts"] for s in verified),
        num_accepted_tokens=sum(accepted),
        num_accepted_tokens_per_pos=[sum(n > i for n in accepted) for i in range(15)],
    )
    delta = {
        name: [a - b for a, b in zip(value, before[name])]
        if isinstance(value, list)
        else value - before[name]
        for name, value in after.items()
    }
    assert delta == expected, (delta, expected)
    return delta


def test_qwen3_xpress_verification_and_generation(monkeypatch, tmp_path, record_property):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    _cached_model(TARGET, TARGET_REVISION)
    _cached_model(DRAFT, DRAFT_REVISION)
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", "head_major")
    monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16,64")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    engine = LLM(
        model=TARGET,
        revision=TARGET_REVISION,
        tokenizer_revision=TARGET_REVISION,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        disable_log_stats=False,
        compilation_config={"compile_sizes": [1, 16, 64]},
        worker_cls="xpress_validation_worker.ValidationWorker",
        speculative_config=dict(
            method="dflash",
            model=DRAFT,
            revision=DRAFT_REVISION,
            num_speculative_tokens=15,
            xpress_num_passes=6,
            xpress_topc=int(os.environ.get("SPYRE_XPRESS_TEST_TOPC", "128")),
        ),
    )
    tokenizer = engine.get_tokenizer()
    context = (
        "Read the following context. "
        + "IBM develops computers, software, and consulting services. " * 16
        + "\nName the company described above:"
    )
    prompts = [
        "The capital of France is",
        "What are the main businesses of IBM? Answer in one sentence.",
        context,
    ]
    cases = [
        dict(name=f"prompt_{i}", prompt=prompt, max_tokens=24) for i, prompt in enumerate(prompts)
    ]
    repeated = tokenizer.encode("IBM develops computers, software, and consulting services. " * 64)
    cases += [
        dict(name="page_boundary", prompt={"prompt_token_ids": repeated[:127]}, max_tokens=24),
        dict(name="context_limit", prompt={"prompt_token_ids": repeated[:503]}, max_tokens=8),
    ]
    chat = tokenizer.apply_chat_template(
        [dict(role="user", content="Say only OK.")],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    cases.append(dict(name="eos", prompt=chat, max_tokens=32))
    references, baseline_caches, report = [], {}, {}
    path = tmp_path / "xpress-validation.json"

    def save_report():
        path.write_text(json.dumps(report, indent=2))

    def generate(case, **overrides):
        params = dict(temperature=0.0, max_tokens=case["max_tokens"], logprobs=5) | overrides
        result = engine.generate(case["prompt"], SamplingParams(**params), use_tqdm=False)[0]
        output = result.outputs[0]
        return result, output

    def snapshot(name, result, output, include_draft=False):
        path = tmp_path / f"{name}.pt"
        length = len(result.prompt_token_ids) + len(output.token_ids) - 1
        metadata = engine.collective_rpc(
            "save_cache_prefix",
            args=(str(path), length, include_draft),
        )[0]
        cache = torch.load(path, weights_only=True)
        expected_lengths = dict.fromkeys(TARGET_CACHE_NAMES, length)
        if include_draft:
            expected_lengths.update(
                dict.fromkeys(DRAFT_CACHE_NAMES, min(length, metadata["draft_context_end"]))
            )
        assert cache.keys() == metadata["lengths"].keys() == expected_lengths.keys()
        assert metadata["lengths"] == expected_lengths
        for layer, pair in cache.items():
            assert len(pair) == 2
            assert all(t.shape == (expected_lengths[layer], 8, 128) for t in pair), layer
        return cache

    for case in cases:
        engine.collective_rpc("configure_proposer", args=("none",))
        started = time.perf_counter()
        result, output = generate(case)
        reference = dict(
            prompt_ids=result.prompt_token_ids,
            token_ids=list(output.token_ids),
            text=output.text,
            logprobs=[
                {token: value.logprob for token, value in row.items()} for row in output.logprobs
            ],
            seconds=time.perf_counter() - started,
        )
        references.append(reference)
        baseline_caches[case["name"]] = snapshot("base_" + case["name"], result, output)
        if case["name"] == "eos":
            assert output.finish_reason == "stop"
            assert output.token_ids[-1] == tokenizer.eos_token_id
    report["references"] = references
    save_report()
    for label, mode, rejection, passes, salt, exact_reference in (
        ("reject_first", "scripted", 0, None, 0, None),
        ("reject_first_tail", "scripted", 0, None, 101, "reject_first"),
        ("reject_third", "scripted", 2, None, 0, None),
        ("reject_third_tail", "scripted", 2, None, 101, "reject_third"),
        ("accept_all", "scripted", -1, None, 0, None),
        ("dflash_controlled", "real_scripted", 2, 0, 0, "reject_third"),
        ("xpress_controlled", "real_scripted", 2, 6, 101, "dflash_controlled"),
        ("dflash", "real", -1, 0, 0, None),
        ("xpress", "real", -1, 6, 0, None),
    ):
        rows = []
        report[label] = rows
        for case, reference in zip(cases, references):
            engine.collective_rpc(
                "configure_proposer", args=(mode, references, rejection, passes, salt)
            )
            before = engine.collective_rpc("validation_state")[0]
            metrics_before = _acceptance_metrics(engine)
            started = time.perf_counter()
            result, output = generate(case)
            elapsed = time.perf_counter() - started
            report["last_output"] = dict(
                label=label,
                case=case["name"],
                token_ids=list(output.token_ids),
                text=output.text,
                logprobs=[
                    {token: value.logprob for token, value in row.items()}
                    for row in output.logprobs
                ],
            )
            save_report()
            assert list(output.token_ids) == reference["token_ids"], (label, case["name"], output)
            actual = snapshot(label + "_" + case["name"], result, output, mode.startswith("real"))
            errors = _compare_cache(
                {name: actual[name] for name in TARGET_CACHE_NAMES},
                baseline_caches[case["name"]],
                prompt_length=len(result.prompt_token_ids),
            )
            exact_errors = {}
            if exact_reference is not None:
                expected = torch.load(
                    tmp_path / f"{exact_reference}_{case['name']}.pt", weights_only=True
                )
                exact_actual = (
                    {name: actual[name] for name in TARGET_CACHE_NAMES}
                    if expected.keys() == TARGET_CACHE_NAMES
                    else actual
                )
                exact_errors = _compare_cache(exact_actual, expected, exact=True)
            state = engine.collective_rpc("validation_state")[0]
            public_metrics = _check_acceptance_metrics(
                metrics_before, _acceptance_metrics(engine), state["verification_steps"]
            )
            for previous, current in zip(state["steps"], state["steps"][1:]):
                assert current["position"] == previous["committed_end"]
            verified = [s for s in state["steps"] if s["drafts"]]
            if label == "reject_first":
                assert all(len(s["sampled"]) == 1 for s in verified)
            if label == "accept_all" and case["name"] == "prompt_0":
                assert any(len(s["sampled"]) == 16 for s in verified)
            metrics = {
                group: {k: v - before[group][k] for k, v in state[group].items()}
                for group in ("drafter", "model")
            }
            rows.append(
                dict(
                    case=case["name"],
                    seconds=elapsed,
                    cache_relative_errors=errors,
                    cache_exact_errors=exact_errors,
                    steps=state["steps"],
                    verification_steps=state["verification_steps"],
                    public_acceptance_metrics=public_metrics,
                    metrics=metrics,
                )
            )
            save_report()

    stop_index = next(
        i
        for i in range(2, 10)
        if references[0]["token_ids"][i] not in references[0]["token_ids"][:i]
    )
    engine.collective_rpc("configure_proposer", args=("scripted", references, -1))
    _, stopped = generate(cases[0], stop_token_ids=[references[0]["token_ids"][stop_index]])
    assert list(stopped.token_ids) == references[0]["token_ids"][: stop_index + 1]
    stop_text = tokenizer.decode(references[0]["token_ids"][2:5])
    assert stop_text and stop_text in references[0]["text"]
    engine.collective_rpc("configure_proposer", args=("scripted", references, -1))
    _, stopped = generate(cases[0], stop=[stop_text])
    assert stopped.finish_reason == "stop" and stop_text not in stopped.text
    engine.collective_rpc("configure_proposer", args=("scripted", references, -1))
    _, limited = generate(cases[0], max_tokens=4)
    assert list(limited.token_ids) == references[0]["token_ids"][:4]
    report["stopping"] = dict(token_stop=True, text_stop=True, max_tokens=True)
    save_report()

    # The cancelled request fills all four usable pages. The next request must
    # reuse them, and its target/draft cache must match the prior clean run.
    engine.collective_rpc("configure_proposer", args=("real", None, -1, 6))
    core = engine.llm_engine
    cancelled = core.add_request(
        "cancel_xpress",
        {"prompt_token_ids": repeated[:383]},
        SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
    )
    emitted = 0
    while emitted < 2:
        for result in core.step():
            assert not result.finished
            emitted = len(result.outputs[0].token_ids)
    core.abort_request([cancelled], internal=True)
    assert not core.has_unfinished_requests()
    cancelled_state = engine.collective_rpc("validation_state")[0]
    cancelled_blocks = {b for s in cancelled_state["steps"] for b in s["blocks"] if b}
    assert len(cancelled_blocks) == 4
    assert any(s["drafts"] for s in cancelled_state["verification_steps"])

    engine.collective_rpc("configure_proposer", args=("real_scripted", references, 2, 6, 101))
    case = next(c for c in cases if c["name"] == "page_boundary")
    reference = references[cases.index(case)]
    result, output = generate(case)
    assert list(output.token_ids) == reference["token_ids"]
    reused = snapshot("after_cancellation", result, output, include_draft=True)
    clean = torch.load(tmp_path / "xpress_controlled_page_boundary.pt", weights_only=True)
    cache_errors = _compare_cache(reused, clean, exact=True)
    reuse_state = engine.collective_rpc("validation_state")[0]
    assert cancelled not in reuse_state["runner_requests"]
    reused_blocks = {b for s in reuse_state["steps"] for b in s["blocks"] if b}
    assert reused_blocks & cancelled_blocks
    report["cancellation"] = dict(
        request=cancelled,
        emitted_before_abort=emitted,
        cancelled_blocks=sorted(cancelled_blocks),
        reused_blocks=sorted(reused_blocks),
        cache_exact_errors=cache_errors,
        worker_request_removed=True,
    )
    report["memory"] = reuse_state["memory"]
    save_report()
    record_property("validation_report", str(path))
