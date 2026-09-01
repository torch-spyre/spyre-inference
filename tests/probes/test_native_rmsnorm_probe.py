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

"""Strict-xfail probe for native RMSNorm lowering at the S=64 prefill.

Disables the Spyre custom RMSNorm op so the model dispatches to vLLM's upstream
``RMSNorm.forward_native``, which upcasts fp16->fp32. torch-spyre cannot yet lower
that upcast at the S=64 prefill (the STANDARD reduction factor cannot broadcast
against the staggered EA of the upcast tensor -- "Multi-arg pointwise with mixed
EA"). When torch-spyre gains that support the probe flips to XPASS, the strict xfail
fails CI, and that's the signal to delete the custom ``SpyreRMSNorm`` op.

Runs against the real Spyre device when available; otherwise skips silently.
"""

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_available

pytestmark = pytest.mark.compile


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream RMSNorm.forward_native upcasts fp16->fp32; torch-spyre rejects the "
        "resulting mixed-EA broadcast at the S=64 prefill ('Multi-arg pointwise with "
        "mixed EA'). SpyreRMSNorm (fp16, no upcast) works around it. When this passes, "
        "torch-spyre supports native RMSNorm and the custom op can be removed."
    ),
)
def test_native_rmsnorm_prefill_s64_lowers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream native RMSNorm must lower at the S=64 prefill.

    Whole-model (block-graph) phenomenon: a lone torch.compile of forward_native does
    not reproduce it, and S=1 decode does not either -- only the S=64 prefill compile.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    # register_oot renames the op to its class name, so disabling the Spyre op needs
    # -RMSNorm/-TPAwareRMSNorm (the in-tree -rms_norm is a no-op here).
    engine = LLM(
        model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        enforce_eager=False,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        compilation_config={
            "custom_ops": ["all", "-RMSNorm", "-TPAwareRMSNorm"],
            "compile_sizes": [64, 1],
        },
    )

    # The mixed-EA rejection fires during engine construction (prefill warmup compile),
    # before generate; generation just confirms the graph runs.
    output = engine.generate(
        {"prompt_token_ids": list(range(1, 65))},
        SamplingParams(temperature=0.0, max_tokens=4),
        use_tqdm=False,
    )

    assert output[0].outputs[0].text, "S=64 prefill graph did not produce output"
