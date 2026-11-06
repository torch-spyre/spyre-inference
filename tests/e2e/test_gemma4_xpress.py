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

import json
import math
import os
import time
from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from test_speculative_decoding import _acceptance_metrics, _check_acceptance_metrics
from vllm import LLM, SamplingParams

from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

TARGET = "google/gemma-4-26B-A4B-it"
TARGET_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
DRAFT = "UIUC-SSAIL/gemma4-26b-a4b-XPress-b16"
DRAFT_REVISION = "e69e3580df7a7cd76b77e82ef739803fd88ee445"


# Allow cold target/draft compilation and attention warmup in CI.
@pytest.mark.timeout(3600)
def test_gemma4_xpress_generation(monkeypatch, tmp_path, record_property):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    monkeypatch.setenv("SPYRE_ATTN_KV_LAYOUT", "head_major")
    monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16,64")
    monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,256,2304")
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    engine = LLM(
        model=TARGET,
        revision=TARGET_REVISION,
        tokenizer_revision=TARGET_REVISION,
        hf_overrides=GEMMA4_TEXT_BACKBONE_OVERRIDE,
        dtype="float16",
        max_model_len=2304,
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
            xpress_topc=128,
        ),
    )
    report = {"cases": []}
    report_path = tmp_path / "gemma4-xpress-validation.json"
    try:
        prompts = [
            ("short", "Continue counting by ones, separated by commas:\n1, 2, 3, 4, 5,"),
            ("chunked", ", ".join(str(i) for i in range(1, 37)) + ","),
            ("past_draft_window", ", ".join(str(i) for i in range(1, 451)) + ","),
        ]
        for name, prompt in prompts:
            case = {"name": name, "prompt": prompt}
            report["cases"].append(case)
            for mode in ("none", "real"):
                engine.collective_rpc("configure_proposer", args=(mode,))
                before = _acceptance_metrics(engine)
                started = time.perf_counter()
                result = engine.generate(
                    prompt,
                    SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, logprobs=5),
                    use_tqdm=False,
                )[0]
                seconds = time.perf_counter() - started
                output = result.outputs[0]
                state = engine.collective_rpc("validation_state")[0]
                metrics = _check_acceptance_metrics(
                    before, _acceptance_metrics(engine), state["verification_steps"]
                )
                case[mode] = dict(
                    prompt_ids=result.prompt_token_ids,
                    token_ids=list(output.token_ids),
                    text=output.text,
                    logprobs=[
                        {token: value.logprob for token, value in row.items()}
                        for row in output.logprobs
                    ],
                    seconds=seconds,
                    metrics=metrics,
                    state=state,
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                if name == "past_draft_window":
                    assert 2048 < len(result.prompt_token_ids) <= 2304 - 32
                assert len(output.token_ids) == 32
                assert all(
                    math.isfinite(value.logprob)
                    for row in output.logprobs
                    for value in row.values()
                )
                if mode == "real":
                    assert case[mode]["token_ids"] == case["none"]["token_ids"], case
                    assert metrics["num_draft_tokens"] > 0
                    # These pinned counting prompts accept over 80%; allow numerical headroom.
                    assert metrics["num_accepted_tokens"] >= 0.5 * metrics["num_draft_tokens"], case
                    assert any(step["native_proposals"] for step in state["steps"])
        record_property("gemma4_xpress_validation", json.dumps(report))
    finally:
        engine.llm_engine.engine_core.shutdown()
