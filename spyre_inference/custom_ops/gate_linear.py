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

"""Spyre OOT ``GateLinear``: keep router logits in fp16.

Upstream forces ``out_dtype=torch.float32`` for CUDA's top-k, but Spyre cannot
restickify fp32 (``spyre::ReStickifyOpHBM`` unsupported for IEEE_FP32), which
the routing softmax's reduction would trigger. Match hf-adapters#293 and route
in fp16 throughout.
"""

import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot(name="GateLinear")
class SpyreGateLinear(GateLinear):
    def __init__(self, *args, **kwargs):
        kwargs["out_dtype"] = torch.float16
        super().__init__(*args, **kwargs)
