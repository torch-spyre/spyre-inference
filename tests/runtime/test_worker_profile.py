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

"""PrivateUse1 set through --profiler-config.torch_profiler_activities reaches the profiler.

Importing the worker module lifts vLLM's "cpu" validation, so a worker on Spyre's "cpu"
device type accepts the activity and vLLM's own profiler path forwards it to torch.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.profiler import ProfilerConfig
from vllm.profiler import wrapper

from spyre_inference.v1.worker import spyre_worker


@pytest.fixture
def cpu_device_type(monkeypatch):
    """Spyre reports device_type "cpu"; pin it so the tests do not depend on the host."""
    monkeypatch.setattr(wrapper, "current_platform", SimpleNamespace(device_type="cpu"))


def _torch_profiler_config(trace_dir, activities) -> ProfilerConfig:
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(trace_dir),
        torch_profiler_activities=activities,
    )


@pytest.mark.parametrize("activities", [["CPU", "PrivateUse1"], ["PrivateUse1"]])
def test_worker_validation_accepts_private_use1_on_cpu(tmp_path, cpu_device_type, activities):
    wrapper.validate_worker_profiler_config(_torch_profiler_config(tmp_path, activities))


def test_allow_is_idempotent():
    before = dict(wrapper._SUPPORTED_TORCH_PROFILER_ACTIVITIES)

    spyre_worker.allow_private_use1_profiler_activity()
    spyre_worker.allow_private_use1_profiler_activity()

    assert before == wrapper._SUPPORTED_TORCH_PROFILER_ACTIVITIES


def test_upstream_has_not_drifted_from_the_supported_activities_table():
    """allow_private_use1_profiler_activity patches this table; fail here if it moves."""
    assert "cpu" in wrapper._SUPPORTED_TORCH_PROFILER_ACTIVITIES


def test_configured_activities_reach_the_torch_profiler(tmp_path, cpu_device_type):
    config = _torch_profiler_config(tmp_path, ["CPU", "PrivateUse1"])

    profiler = wrapper.create_worker_profiler(config, worker_name="w", local_rank=0)

    assert profiler._profiler_kwargs["activities"] == [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]
