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

"""XPress on the V2 runner: greedy output matches the target alone, with no compile after
warmup and no CPU fallback."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

QWEN3_PROMPTS = [
    "Continue counting by ones, separated by commas:\n1, 2, 3, 4, 5,",
    "What are the main businesses of IBM? Answer in one sentence.",
    "Read the following context. "
    + "IBM develops computers, software, and consulting services. " * 16
    + "\nName the company described above:",
    "2, 4, 6, 8,",
]

CASES = {
    # Different lengths, so the active draft batch shrinks through 4, 2 and 1 as they finish.
    "qwen3": dict(
        target="Qwen/Qwen3-8B",
        target_revision="b968826d9c46dd6066d109eabc6255188de91218",
        draft="UIUC-SSAIL/Qwen3-8B-XPress-b16",
        draft_revision="f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5",
        engine=dict(
            max_model_len=512,
            max_num_seqs=4,
            max_num_batched_tokens=128,
            compilation_config={"compile_sizes": [1, 16, 32, 64, 128]},
        ),
        env={"SPYRE_ATTN_QUERY_BUCKETS": "1,16,64,128", "SPYRE_MAX_NUM_PARTIAL_PREFILLS": "4"},
        batches={"batch": (QWEN3_PROMPTS, [48, 40, 24, 32])},
    ),
    # A draft of four 2048-token sliding-window layers and one full-attention layer; the
    # last prompt passes the window.
    "gemma4": dict(
        target="google/gemma-4-26B-A4B-it",
        target_revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        draft="UIUC-SSAIL/gemma4-26b-a4b-XPress-b16",
        draft_revision="e69e3580df7a7cd76b77e82ef739803fd88ee445",
        engine=dict(
            max_model_len=2304,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            compilation_config={"compile_sizes": [1, 16, 64]},
            gemma4_text_backbone=True,
        ),
        env={"SPYRE_ATTN_QUERY_BUCKETS": "1,16,64", "SPYRE_ATTN_KV_BUCKETS": "128,256,2304"},
        batches={
            name: ([prompt], [32])
            for name, prompt in (
                ("short", "Continue counting by ones, separated by commas:\n1, 2, 3, 4, 5,"),
                ("past_draft_window", ", ".join(str(i) for i in range(1, 451)) + ","),
            )
        },
    ),
}


def _engine_outputs(case_name: str, speculative: bool) -> dict:
    from vllm import LLM, SamplingParams

    from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

    case = CASES[case_name]
    engine = dict(case["engine"])
    if engine.pop("gemma4_text_backbone", False):
        engine["hf_overrides"] = GEMMA4_TEXT_BACKBONE_OVERRIDE
    spec = dict(
        method="dflash",
        model=case["draft"],
        revision=case["draft_revision"],
        num_speculative_tokens=15,
        xpress_topc=128,
    )
    llm = LLM(
        model=case["target"],
        revision=case["target_revision"],
        tokenizer_revision=case["target_revision"],
        dtype="float16",
        enable_prefix_caching=False,
        disable_log_stats=False,
        speculative_config=spec if speculative else None,
        **engine,
    )
    outputs = {}
    for name, (prompts, limits) in case["batches"].items():
        params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in limits]
        results = llm.generate(prompts, params)
        outputs[name] = [list(r.outputs[0].token_ids) for r in results]
        outputs[name + "_prompt_len"] = [len(r.prompt_token_ids) for r in results]
    metrics = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    outputs["drafted"] = metrics.get("vllm:spec_decode_num_draft_tokens")
    outputs["accepted"] = metrics.get("vllm:spec_decode_num_accepted_tokens")
    # Frees the card for the next engine in this process.
    llm.llm_engine.engine_core.shutdown()
    return outputs


def _both_engines(case_name: str) -> dict:
    """Runs in the child: the Triton gate binds the runner at vLLM import."""
    import warnings

    import torch  # noqa: F401  (autoloads torch_spyre)
    from torch_spyre.ops.fallbacks import FallbackWarning

    warnings.simplefilter("error", FallbackWarning)
    reference = _engine_outputs(case_name, speculative=False)
    return {"reference": reference, "spec": _engine_outputs(case_name, speculative=True)}


def _generate(case_name: str) -> dict:
    env = {
        **os.environ,
        "SPYRE_MODEL_RUNNER": "v2",
        "SPYRE_ATTN_KV_LAYOUT": "head_major",
        "SPYRE_COMPILE_GUARD": "error",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "36000",
        **CASES[case_name]["env"],
    }
    result = subprocess.run(
        [sys.executable, __file__, case_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert result.returncode == 0 and lines, result.stdout[-4000:] + result.stderr[-6000:]
    return json.loads(lines[-1].removeprefix("RESULT "))


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("case_name", list(CASES))
def test_xpress_on_v2_matches_the_target_alone(case_name):
    from spyre_testing_plugin.pytest_plugin import spyre_hardware_present

    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    result = _generate(case_name)
    reference, speculative = result["reference"], result["spec"]
    for name in CASES[case_name]["batches"]:
        assert speculative[name] == reference[name], name
    assert speculative["drafted"] and speculative["accepted"] > 0, speculative
    if case_name == "gemma4":
        assert speculative["past_draft_window_prompt_len"][0] > 2048
        # These counting prompts accept over 80% on V1; allow numerical headroom.
        assert speculative["accepted"] >= 0.5 * speculative["drafted"], speculative


if __name__ == "__main__":
    print("RESULT " + json.dumps(_both_engines(sys.argv[1])))
