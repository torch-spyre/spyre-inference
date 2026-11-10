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

"""Qwen3-8B with the XPress draft on the V2 runner: greedy output matches the target
alone, with no compile after warmup and no CPU fallback."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.model_quality, pytest.mark.uses_subprocess]

TARGET = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT = "UIUC-SSAIL/Qwen3-8B-XPress-b16"
DRAFT_REVISION = "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5"
PROMPTS = [
    "Continue counting by ones, separated by commas:\n1, 2, 3, 4, 5,",
    "What are the main businesses of IBM? Answer in one sentence.",
    "Read the following context. "
    + "IBM develops computers, software, and consulting services. " * 16
    + "\nName the company described above:",
    "2, 4, 6, 8,",
]
# Different lengths, so the active draft batch shrinks through 4, 2 and 1 as they finish.
MAX_TOKENS = [48, 40, 24, 32]


def _engine_outputs(speculative: bool) -> dict:
    """Runs in the child: the Triton gate binds the runner at vLLM import."""
    import warnings

    import torch  # noqa: F401  (autoloads torch_spyre)
    from torch_spyre.ops.fallbacks import FallbackWarning
    from vllm import LLM, SamplingParams

    warnings.simplefilter("error", FallbackWarning)
    spec = dict(
        method="dflash",
        model=DRAFT,
        revision=DRAFT_REVISION,
        num_speculative_tokens=15,
        xpress_topc=128,
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
        compilation_config={"compile_sizes": [1, 16, 32, 64, 128]},
        speculative_config=spec if speculative else None,
    )
    outputs = {}
    for name, count in (("single", 1), ("batch", len(PROMPTS))):
        params = [SamplingParams(temperature=0.0, max_tokens=n) for n in MAX_TOKENS[:count]]
        results = llm.generate(PROMPTS[:count], params)
        outputs[name] = [list(r.outputs[0].token_ids) for r in results]
    metrics = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
    outputs["drafted"] = metrics.get("vllm:spec_decode_num_draft_tokens")
    outputs["accepted"] = metrics.get("vllm:spec_decode_num_accepted_tokens")
    return outputs


def _generate(speculative: bool) -> dict:
    env = {
        **os.environ,
        "SPYRE_MODEL_RUNNER": "v2",
        "SPYRE_ATTN_KV_LAYOUT": "head_major",
        "SPYRE_ATTN_QUERY_BUCKETS": "1,16,64,128",
        "SPYRE_MAX_NUM_PARTIAL_PREFILLS": "4",
        "SPYRE_COMPILE_GUARD": "error",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "36000",
    }
    mode = "spec" if speculative else "reference"
    result = subprocess.run(
        [sys.executable, __file__, mode], env=env, capture_output=True, text=True, timeout=7200
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    assert result.returncode == 0 and lines, result.stdout[-4000:] + result.stderr[-6000:]
    return json.loads(lines[-1].removeprefix("RESULT "))


def test_qwen3_xpress_on_v2_matches_the_target_alone():
    from spyre_testing_plugin.pytest_plugin import spyre_hardware_present

    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    reference = _generate(speculative=False)
    speculative = _generate(speculative=True)
    assert speculative["single"] == reference["single"]
    assert speculative["batch"] == reference["batch"]
    assert speculative["drafted"] and speculative["accepted"] > 0, speculative


if __name__ == "__main__":
    print("RESULT " + json.dumps(_engine_outputs(sys.argv[1] == "spec")))
