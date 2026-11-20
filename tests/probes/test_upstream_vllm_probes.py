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
    raises=AssertionError,
    reason=(
        "Upstream Gemma4ForConditionalGeneration keeps its PLE cache as a plain attribute "
        "that stays on the device it was built on, so spyre_inference.models.gemma4_mm "
        "re-registers it as a buffer. vllm-project/vllm#62659 reallocates it on the inputs' "
        "device. When this XPASS-es, merge torch-spyre/spyre-inference#1669."
    ),
)
def test_gemma4_ple_cache_follows_inputs_device():
    from types import SimpleNamespace

    import torch
    from torch import nn
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

    model = object.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.per_layer_embeddings = torch.zeros(8, 2, 4, device="meta")
    model._full_attn_layer_idxs = ()
    seen = []

    def backbone(*args, per_layer_inputs, **kwargs):
        seen.append(per_layer_inputs)

    model.language_model = SimpleNamespace(model=backbone)
    model(None, None, inputs_embeds=torch.zeros(3, 1))

    assert seen[0].device.type == "cpu"
