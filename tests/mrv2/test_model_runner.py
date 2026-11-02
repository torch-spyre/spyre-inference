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
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre import model_runner
from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

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
def test_reload_rewarms_only_when_the_kv_scales_change(
    monkeypatch: pytest.MonkeyPatch, scales_changed: bool
) -> None:
    """The block graphs guard on the KV scales a reload rewrites, as in V1."""
    events = []
    monkeypatch.setattr(GPUModelRunner, "reload_weights", lambda self: events.append("reload"))
    monkeypatch.setattr(model_runner, "resolve_kv_scales", lambda model, ctx: scales_changed)
    monkeypatch.setattr(
        model_runner.compile_guard,
        "allow_compile",
        lambda reason: contextlib.nullcontext(events.append("allow")),
    )
    runner = SpyreModelRunnerV2.__new__(SpyreModelRunnerV2)
    runner.model = object()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    monkeypatch.setattr(runner, "warming_up_model", lambda: events.append("warm"))

    runner.reload_weights()

    assert events == (["reload", "allow", "warm"] if scales_changed else ["reload"])
