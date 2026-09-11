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

"""Regression probe for native FP32 RMSNorm at the S=64 prefill.

Disables the Spyre OOT RMSNorm registration so the model dispatches directly to
vLLM's upstream ``RMSNorm.forward_native``. The native implementation upcasts
fp16->fp32 and must produce the same greedy prefix as the registered Spyre path.

The comparison uses the shared-prefix criterion from
``tests/e2e/test_distributed_tp2.py``. It guards both whole-block compilation
and the FP32 reduction while allowing a later sampling tie-break to differ.

Runs against the real Spyre device when available; otherwise skips silently.
"""

import gc

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

# Guard on spyre_device_count (reads AIU_WORLD_SIZE), never spyre_available: the latter
# does torch.randn on the spyre device, which opens the card in the main pytest process
# and then blocks this test's own EngineCore subprocess -- and every later subprocess
# probe -- from opening it ("Device or resource busy"). uses_subprocess keeps this off
# the shared card by running it before any in-process device test.
pytestmark = [pytest.mark.probe, pytest.mark.uses_subprocess]

_PROMPT = "What are IBMs main businesses?"
_MAX_TOKENS = 8
_MIN_MATCHING_TOKENS = 2


def _generate_greedy(native: bool) -> list[int]:
    """Greedy token ids for the prompt, with native or custom RMSNorm."""
    from vllm import LLM, SamplingParams

    # register_oot renames the op to its class name, so disabling the Spyre op needs
    # -RMSNorm/-TPAwareRMSNorm (the in-tree -rms_norm is a no-op here).
    custom_ops = ["all", "-RMSNorm", "-TPAwareRMSNorm"] if native else ["all"]
    llm = LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=1,
        # Buckets the 8-token prefill up to the S=64 query graph.
        max_num_batched_tokens=64,
        compilation_config={"custom_ops": custom_ops, "compile_sizes": [64, 1]},
    )
    output = llm.generate(
        _PROMPT,
        SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS),
        use_tqdm=False,
    )
    token_ids = list(output[0].outputs[0].token_ids)
    # vllm has no explicit LLM.shutdown(); rely on GC + child-process reaping.
    del llm
    gc.collect()
    return token_ids


def test_native_rmsnorm_prefill_s64_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native RMSNorm must open on the same tokens as the registered OOT path."""
    if spyre_device_count() < 1:
        pytest.skip("Spyre device not available")

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    reference = _generate_greedy(native=False)
    native = _generate_greedy(native=True)

    matching = next(
        (i for i, (a, b) in enumerate(zip(native, reference)) if a != b),
        min(len(native), len(reference)),
    )
    assert matching >= _MIN_MATCHING_TOKENS, (
        f"native and custom RMSNorm diverged at token {matching} "
        f"(expected >={_MIN_MATCHING_TOKENS} matching tokens). "
        f"native={native} custom={reference}"
    )
