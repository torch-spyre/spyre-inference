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

"""TorchSpyreWorker.profile keeps upstream's signature and forwards every argument."""

import inspect

from vllm.v1.worker.gpu_worker import Worker

from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker


def test_signature_matches_upstream():
    assert inspect.signature(TorchSpyreWorker.profile) == inspect.signature(Worker.profile)


def test_forwards_iteration_bounds(monkeypatch):
    calls = []

    def record(self, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(Worker, "profile", record)
    worker = TorchSpyreWorker.__new__(TorchSpyreWorker)

    worker.profile(is_start=False, profile_prefix="p", delay_iterations=1, max_iterations=2)

    assert calls == [((False, "p"), {"delay_iterations": 1, "max_iterations": 2})]
