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

"""MRV2: synchronous pooler output, replacing
vllm/v1/worker/gpu/async_utils.py:AsyncPoolingOutput.

AsyncPoolingOutput defers the device-to-host copy behind a torch.cuda.Event
for later polling; Spyre has no async copy stream worth deferring to, so
pool_sync() does the copy inline and returns the result directly.
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def pool_sync(runner: "SpyreModelRunnerV2", *args: Any, **kwargs: Any) -> ModelRunnerOutput | None:
    """Pool synchronously and return the host-side result directly."""
    raise NotImplementedError
