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

"""SpyreModelRunnerV2's own overrides."""

import contextlib
from types import SimpleNamespace

import pytest
import torch.nn as nn
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1GPUModelRunner

from spyre_inference.v1.worker import model_load
from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

from .fakes import make_fake_scheduler_output


def test_prefix_cache_block_copies_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream would copy over the empty self.kv_caches, silently dropping the copy."""
    upstream = []
    monkeypatch.setattr(GPUModelRunner, "update_requests", lambda self, out: upstream.append(out))
    runner = SpyreModelRunnerV2.__new__(SpyreModelRunnerV2)
    output = make_fake_scheduler_output({"req": 4})
    output.kv_cache_block_copies = [(1, 2)]

    with pytest.raises(NotImplementedError, match="copy-on-write block copies"):
        runner.update_requests(output)
    assert upstream == []


def test_a_step_without_block_copies_reaches_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = []
    monkeypatch.setattr(GPUModelRunner, "update_requests", lambda self, out: upstream.append(out))
    runner = SpyreModelRunnerV2.__new__(SpyreModelRunnerV2)
    output = make_fake_scheduler_output({"req": 4})

    runner.update_requests(output)

    assert upstream == [output]


@pytest.mark.parametrize("scales_changed", [True, False])
@pytest.mark.parametrize(
    "runner_cls, upstream_cls",
    [
        pytest.param(SpyreModelRunnerV2, GPUModelRunner, id="v2"),
        pytest.param(TorchSpyreModelRunner, V1GPUModelRunner, id="v1"),
    ],
)
def test_reload_rewarms_only_when_the_kv_scales_change(
    monkeypatch: pytest.MonkeyPatch, scales_changed: bool, runner_cls, upstream_cls
) -> None:
    """The block graphs guard on the KV scales a reload rewrites; V1 and V2 share this."""
    events = []
    monkeypatch.setattr(upstream_cls, "reload_weights", lambda self: events.append("reload"))
    monkeypatch.setattr(model_load, "resolve_kv_scales", lambda model, ctx: scales_changed)
    monkeypatch.setattr(
        model_load.compile_guard,
        "allow_compile",
        lambda reason: contextlib.nullcontext(events.append("allow")),
    )
    runner = runner_cls.__new__(runner_cls)
    runner.model = nn.Identity()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    monkeypatch.setattr(runner, "warming_up_model", lambda: events.append("warm"))

    runner.reload_weights()

    assert events == (["reload", "allow", "warm"] if scales_changed else ["reload"])
