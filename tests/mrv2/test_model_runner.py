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

import pytest
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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
