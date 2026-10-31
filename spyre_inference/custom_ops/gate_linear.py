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

Spyre cannot restickify fp32 (``spyre::ReStickifyOpHBM`` is unsupported for
IEEE_FP32); the routing reduction over them is promoted in ``moe`` instead.

The models in ``HOST_FP32_ROUTER_MODEL_TYPES`` instead keep their router GEMM in fp32 on
the host: see ``spyre_inference.host_router``, a TEMPORARY CPU FALLBACK.
"""

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

# GLM's noaux_tc routers decide their top-k below fp16 resolution of the logits, and Spyre
# has no fp32-accumulating matmul; every other model keeps its router on Spyre.
HOST_FP32_ROUTER_MODEL_TYPES = frozenset({"glm4_moe_lite", "glm_moe_dsa"})


def host_fp32_router_requested() -> bool:
    """Whether the model being built takes the host fp32 router (gate and MoE alike)."""
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    model_config = None if vllm_config is None else vllm_config.model_config
    if model_config is None:
        return False
    return model_config.hf_text_config.model_type in HOST_FP32_ROUTER_MODEL_TYPES


@GateLinear.register_oot(name="GateLinear")
class SpyreGateLinear(GateLinear):
    def __init__(self, *args, **kwargs) -> None:
        # vLLM's OOT registry instantiates this class directly, so the host variant is
        # picked here, before any state exists.
        if type(self) is SpyreGateLinear and host_fp32_router_requested():
            from spyre_inference.host_router import HostFp32GateLinear

            self.__class__ = HostFp32GateLinear
            HostFp32GateLinear.__init__(self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)
        # Cleared here rather than through the argument, to also catch positional callers.
        self.out_dtype = None
