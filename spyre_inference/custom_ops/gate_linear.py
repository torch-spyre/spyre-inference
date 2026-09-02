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

"""Spyre OOT ``GateLinear``: MoE router logits stay in the weight dtype.

Models ask this layer for fp32 logits because CUDA's top-k kernels are more
stable that way. Spyre cannot restickify fp32 (``spyre::ReStickifyOpHBM`` is
unsupported for IEEE_FP32), so the routing softmax's reduction over fp32 logits
fails to lower. Clearing ``out_dtype`` drops both of upstream's casts and leaves
the logits in the weight dtype, which the platform already forces to float16.
"""

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot(name="GateLinear")
class SpyreGateLinear(GateLinear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cleared after ``super().__init__`` rather than through the argument, so
        # it also covers callers that pass ``out_dtype`` positionally. The kernel
        # eligibility flags it feeds are all CUDA-gated, so they do not matter here.
        self.out_dtype = None
