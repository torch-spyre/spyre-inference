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

"""Strict-xfail probe for native RMSNorm at the S=64 prefill.

Disables the Spyre custom RMSNorm op so the model dispatches to vLLM's upstream
``RMSNorm.forward_native``, which upcasts fp16->fp32. torch-spyre does lower that
upcast, but hands back the fp32 reduction as an eager result whose device layout the
compiled graph does not assume (a ``RetileWarning`` per (64, 1) and (1, 1) reduction),
so the graph runs and silently produces wrong values: the model answers "+" and stops
instead of continuing the prompt. When torch-spyre keeps the reduction in the graph the
probe flips to XPASS, the strict xfail fails CI, and that's the signal to delete the
custom ``SpyreRMSNorm`` op.

The comparison is the shared-prefix one ``tests/e2e/test_distributed_tp2.py`` uses against
its TP=1 twin: native accumulates the norm in fp32 where the custom op stays in fp16, so a
working native path may tie-break away later but not disagree from the first token.
Asserting the whole continuation would conflate "native works" with "native is bit-equal
to fp16"; asserting non-empty output is weaker still, since the broken path does emit a
token and which one it picks turns on fp16 rounding anywhere in the model.

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream RMSNorm.forward_native upcasts fp16->fp32. torch-spyre lowers the upcast "
        "but returns the fp32 reduction as an eager result whose device layout the compiled "
        "graph does not assume, so the S=64 prefill graph runs and silently produces wrong "
        "values. SpyreRMSNorm (fp16, no upcast) works around it. When this passes, "
        "torch-spyre keeps the reduction in the graph and the custom op can be removed."
    ),
)
def test_native_rmsnorm_prefill_s64_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native RMSNorm must open on the same tokens as the shipped custom op.

    Whole-model (block-graph) phenomenon: a lone torch.compile of forward_native does
    not reproduce it, and S=1 decode does not either -- only the S=64 prefill compile.
    """
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
