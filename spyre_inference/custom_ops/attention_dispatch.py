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

"""CPU-side dispatcher for vllm::unified_attention_with_output.

SpyreQKVParallelLinear D2Hs q/k/v (downstream .split()/scatter need CPU),
so the opaque attention op dispatches to CPU — but the real kernel is only
registered for PrivateUse1. Move inputs back to Spyre, re-dispatch, and
mirror the mutated outputs to the caller's CPU buffers.
"""

import torch

# Force vllm::unified_attention_with_output to be defined before we add an impl.
import vllm.model_executor.layers.attention.attention  # noqa: F401
from vllm.logger import init_logger
from vllm.utils.torch_utils import vllm_lib

from .utils import convert

logger = init_logger(__name__)

_SPYRE = "spyre"


def _cpu_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    q = convert(query, device=_SPYRE)
    k = convert(key, device=_SPYRE)
    v = convert(value, device=_SPYRE)
    # `output` arrives uninitialized; allocate fresh on Spyre.
    out = torch.empty_like(output, device=_SPYRE)
    os_ = convert(output_scale, device=_SPYRE)
    obs_ = convert(output_block_scale, device=_SPYRE)

    torch.ops.vllm.unified_attention_with_output(
        q,
        k,
        v,
        out,
        layer_name,
        output_scale=os_,
        output_block_scale=obs_,
        kv_cache_dummy_dep=kv_cache_dummy_dep,
    )

    # mutates_args=["output", "output_block_scale"] in vllm's schema.
    output.copy_(out)
    if output_block_scale is not None:
        output_block_scale.copy_(obs_)


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    vllm_lib.impl(
        "unified_attention_with_output",
        _cpu_unified_attention_with_output,
        dispatch_key="CPU",
    )
    _REGISTERED = True
    logger.info("Registered CPU dispatch impl for vllm::unified_attention_with_output")
