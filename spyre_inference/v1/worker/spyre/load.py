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

"""MRV2: SpyreModelRunnerV2.load_model (vllm/v1/worker/gpu/model_runner.py:
GPUModelRunner.load_model).

CPU load, .to(spyre), wrap with the shared _SpyreModelWrapper, then
compile.py's per-block compile.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def load_and_wrap(runner: "SpyreModelRunnerV2", *args: Any, **kwargs: Any) -> None:
    """Load weights, move the model to Spyre, wrap, and compile per block.

    Postcondition: ``runner.model`` is the wrapper, and every other holder of
    the model (``model_state.model``, ``encoder_runner.model``,
    ``pooling_runner.model``) points at that same wrapper. ``runner.model_state``
    comes from ``model_states.resolve_spyre_model_state``, and
    ``runner.decode_query_len`` is set: upstream sets it in the ``load_model`` this
    replaces, and ``SpyreBucketManager.from_runner`` reads it.
    """
    raise NotImplementedError
