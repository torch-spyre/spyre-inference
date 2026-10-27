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

"""MRV2: decoder model state (vllm/v1/worker/gpu/model_states/default.py:DefaultModelState).

Upstream's prepare_inputs / prepare_attn are used as-is; bucketing.py pads the body.
"""

from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin


class SpyreDefaultModelState(SpyreModelStateMixin, DefaultModelState):
    """MRV2: DefaultModelState, constructed with the Spyre device."""
