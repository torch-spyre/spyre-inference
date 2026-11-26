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

"""The EngineArgs overrides apply_prelaunch_overrides makes for every model."""

from types import SimpleNamespace

import pytest
from vllm.engine.arg_utils import EngineArgs

from spyre_inference.models import _prelaunch, apply_prelaunch_overrides


@pytest.mark.parametrize("model_type", ["gemma4", "mistral3", "llama"])
def test_the_processor_normalizes_images_for_every_spyre_tower(
    monkeypatch: pytest.MonkeyPatch, tmp_path, model_type: str
) -> None:
    """vLLM 0.34 defaults to raw pixels for towers that normalize on the device, which
    the Spyre towers' own patch embedding does not do."""
    monkeypatch.setattr(
        _prelaunch, "load_hf_config", lambda args: SimpleNamespace(model_type=model_type)
    )
    engine_args = EngineArgs(model=str(tmp_path))  # a local path: no hub lookup
    assert engine_args.mm_device_do_normalize
    apply_prelaunch_overrides(engine_args)
    assert engine_args.mm_device_do_normalize is False
