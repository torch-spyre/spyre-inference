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

"""Probes for upstream vLLM changes that retire a spyre-inference workaround.

Each test checks the pinned vLLM for one upstream fix. Blocked probes are
xfail(strict=True): when a vLLM bump brings the fix in, they flip to passing, the
strict-xfail fails the probe job, and that is the signal to delete the workaround.
"""

import pytest

pytestmark = pytest.mark.probe


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream Muse Glimmer builds its norms from the plain nn.Module "
        "MuseGlimmerRMSNorm, which the platform cannot override, so "
        "spyre_inference.models.muse_glimmer swaps it at construction. "
        "vllm-project/vllm#64920 replaces it with the RMSNorm / GemmaRMSNorm CustomOps. "
        "When this XPASS-es, delete _spyre_rms_norm, _spyre_norms and "
        "SpyreMuseGlimmerForCausalLM, and register upstream's classes directly."
    ),
)
def test_muse_glimmer_uses_custom_op_norms():
    from vllm.model_executor.models import muse_glimmer as mg

    assert not hasattr(mg, "MuseGlimmerRMSNorm")
