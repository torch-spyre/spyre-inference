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

"""CPU dispatch glue for vLLM's opaque attention op.

PR #246 enabled the opaque attention wrapper so torch.compile sees attention
as one opaque op (rather than tracing into SpyreAttentionImpl.forward). With
opaque mode, vLLM dispatches attention via
``torch.ops.vllm.unified_attention_with_output(...)``, registered against
``current_platform.dispatch_key`` (``PrivateUse1``).

But ``SpyreQKVParallelLinear.forward`` D2Hs the QKV tensors so downstream
``.split()``, ``.view()``, and the scatter into the KV cache run on CPU —
Spyre rejects non-contiguous tensors as scatter sources and has no advanced
indexing. The dispatcher therefore sees CPU tensors with no CPU kernel
registered, and the model fails to boot.

This module adds a CPU impl that H2Ds the inputs to Spyre, re-dispatches the
op (which then picks the existing PrivateUse1 kernel), and mirrors the
mutated outputs back to the caller's CPU buffers. The attention output ends
up on Spyre because ``vllm.Attention.forward`` allocates ``output`` on
``query.device``, so downstream layers consume Spyre tensors directly and
torch.compile sees a clean Spyre→Spyre opaque-op boundary.
"""

import torch

# Importing the attention module registers torch.ops.vllm.unified_attention_with_output;
# we need that to exist before we can add a dispatch impl for it.
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
    # `output` arrives uninitialized (torch.empty in Attention.forward), so
    # allocate fresh on Spyre rather than paying an H2D for empty bytes.
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
