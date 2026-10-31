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

import importlib.util

from vllm.model_executor.models.registry import _VLLM_MODELS


def test_vllm_does_not_ship_xpress_yet():
    """Force migration of the temporary XPress port when upgrading vLLM."""
    cleanup = (
        "vLLM now ships XPress: replace the vendored head and registrations with Spyre "
        "overrides of upstream XPress, then remove this canary. "
        "See https://github.com/torch-spyre/spyre-inference/issues/1417"
    )
    assert importlib.util.find_spec("vllm.model_executor.models.qwen3_xpress") is None, cleanup
    assert "Qwen3XPressModel" not in _VLLM_MODELS, cleanup
    assert "DFlashQwen3XPressModel" not in _VLLM_MODELS, cleanup
