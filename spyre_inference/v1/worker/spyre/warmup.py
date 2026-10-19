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

"""MRV2: V2 warmup driver (vllm/v1/worker/gpu_worker.py:Worker.compile_or_warm_up_model,
TorchSpyreModelRunner.warming_up_model).

Decoder dummies skip attention and publish null KV slots; pooling dummies
keep attention on; a separate pass then records attention over attn_groups
and runs the sampler dummy. plan() is kept apart from run() so the
dummy-call list can be asserted without executing a model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


@dataclass(frozen=True)
class DummyCall:
    """One _dummy_run() invocation in the warmup plan."""

    num_reqs: int
    num_tokens: int
    skip_attn: bool


def plan(runner: "SpyreModelRunnerV2") -> list[DummyCall]:
    """Build the ordered list of dummy calls warmup makes, without running any."""
    raise NotImplementedError


def run(runner: "SpyreModelRunnerV2") -> None:
    """Execute plan(runner), then record attention and the sampler dummy."""
    raise NotImplementedError
