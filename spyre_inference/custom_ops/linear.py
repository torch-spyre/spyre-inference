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

"""Spyre OOT registration for the fused QKV projection."""

from vllm.logger import init_logger

from vllm.model_executor.layers.linear import QKVParallelLinear

logger = init_logger(__name__)


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(QKVParallelLinear):
    """Out-of-tree (OOT) QKVParallelLinear implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )
