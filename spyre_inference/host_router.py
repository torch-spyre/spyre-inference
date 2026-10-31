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

"""TEMPORARY CPU FALLBACK: fp32 MoE router logits and expert selection on the host.

This is a deliberate exception to spyre-inference's rule of no CPU fallbacks. Spyre has no
matmul that accumulates and returns fp32 (torch-spyre: ``batchmatmul on
DataFormats.IEEE_FP32`` is unsupported, and ``mm(..., out_dtype=float32)`` does not lower),
and an fp32 top-k does not lower either. Remove this once torch-spyre has both;
``tests/probes/test_spyre_fallback_probes.py::test_spyre_fp32_accum_matmul`` flips when the
matmul lands.

Only the models ``custom_ops.gate_linear.HOST_FP32_ROUTER_MODEL_TYPES`` names take it: GLM's
noaux_tc routers (sigmoid scores, a selection-only correction bias, renormalized top-k)
are trained with an fp32 router GEMM, and their k-th and (k+1)-th biased scores can be
closer than one fp16 ulp of the logits.

It follows upstream's split, with the Spyre pieces swapped in where Spyre falls short:

- ``HostFp32GateLinear`` (what ``SpyreGateLinear`` builds for those models) computes the
  logits as upstream's ``GateLinear`` does with ``out_dtype=float32``, in fp32 on the host,
  and returns them there.
- The Spyre MoE method selects experts from them with the layer's own upstream router
  config, through vLLM's CPU router, on the host (``host_routing_table``).
- The experts run on Spyre from the resulting dense routing table.

Cost per MoE layer per step: the [tokens, hidden] activation goes to the host and a
[tokens, experts] routing table comes back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.cpu_router import CPURouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import get_layer_from_name
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from spyre_inference.custom_ops.gate_linear import SpyreGateLinear
from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)


def _host_fp32_logits(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    # .cpu() before the upcast: an fp32 tensor read back from Spyre may be staggered.
    return x.cpu().float() @ weight_t


def _host_fp32_logits_fake(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], weight_t.shape[1]), dtype=torch.float32)


direct_register_custom_op(
    op_name="spyre_host_fp32_logits",
    op_func=_host_fp32_logits,
    fake_impl=_host_fp32_logits_fake,
    # The activation comes from Spyre or, in tests, the host.
    dispatch_key="CompositeExplicitAutograd",
)


class HostFp32GateLinear(SpyreGateLinear):
    """``GateLinear`` whose logits are fp32, computed on the host from an fp32 weight copy.

    The opaque op keeps the host GEMM out of the block graph. The logits it returns live on
    the host: only ``host_routing_table`` reads them.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_dtype = torch.float32
        # Not a parameter or buffer: it must not follow the model to Spyre.
        self.__dict__["host_weight_t"] = None
        # Captured before the Spyre linear method stores the weight transposed, Wᵀ [in, out].
        process = self.quant_method.process_weights_after_loading

        def capture_then_process(layer: torch.nn.Module) -> None:
            self.capture_host_weight()
            process(layer)

        self.quant_method.process_weights_after_loading = capture_then_process
        logger.warning_once(
            "TEMPORARY CPU FALLBACK: MoE router logits and expert selection run on the "
            "host in fp32, because Spyre's matmul has no fp32 accumulation; the experts "
            "stay on Spyre. See spyre_inference/host_router.py."
        )

    def capture_host_weight(self) -> None:
        """Take the fp32 host copy, from the checkpoint's [experts, hidden] weight."""
        weight = self.weight.data
        assert weight.shape == (self.output_size, self.input_size), tuple(weight.shape)
        self.__dict__["host_weight_t"] = weight.t().to("cpu", torch.float32).contiguous()

    def forward(self, x: torch.Tensor):
        weight_t = self.__dict__["host_weight_t"]
        assert weight_t is not None, "capture_host_weight() not run"
        logits = torch.ops.vllm.spyre_host_fp32_logits(x, weight_t)  # ty: ignore[invalid-argument-type]
        return logits if not self.return_bias else (logits, None)


def _prepare_layer(layer: RoutedExperts) -> None:
    """The host counterpart of ``moe._prepare_layer``: the layer's router state, in fp32.

    vLLM's CPU router, built from the layer's own upstream routing config. Runs from the
    MoE method's ``process_weights_after_loading``, once the correction bias is loaded and
    before the device move, which the router then does not take part in (it is not a
    submodule).
    """
    bias = layer.e_score_correction_bias
    layer.__dict__["spyre_host_router"] = CPURouter(
        top_k=layer.top_k,
        global_num_experts=layer.global_num_experts,
        use_grouped_topk=layer.use_grouped_topk,
        num_expert_group=layer.num_expert_group,
        topk_group=layer.topk_group,
        renormalize=layer.renormalize,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=layer.routed_scaling_factor,
        e_score_correction_bias=(
            None if bias is None else bias.detach().to("cpu", torch.float32).clone()
        ),
        custom_routing_function=layer.custom_routing_function,
    )


def host_routing_table(
    layer: RoutedExperts, router_logits: torch.Tensor, like: torch.Tensor
) -> torch.Tensor:
    """The dense [tokens, experts] routing weights for fp32 host logits, zero off the top-k,
    on ``like``'s device and in its dtype."""
    return torch.ops.vllm.spyre_host_routing_table(
        router_logits,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(layer.layer_name),
        like,  # ty: ignore[invalid-argument-type]
    )


def _routing_table(
    logits: torch.Tensor, layer_name: LayerNameType, like: torch.Tensor
) -> torch.Tensor:
    # The routed experts share their runner's name, under which vLLM registers the runner.
    experts = get_layer_from_name(_resolve_layer_name(layer_name)).routed_experts
    logits = logits.cpu().float()
    weights, ids = experts.spyre_host_router.select_experts(logits, logits)
    dense = torch.zeros_like(logits).scatter(1, ids.long(), weights)
    return convert(dense.to(like.dtype), device=like.device)


def _routing_table_fake(
    logits: torch.Tensor, layer_name: LayerNameType, like: torch.Tensor
) -> torch.Tensor:
    return like.new_empty(logits.shape)


direct_register_custom_op(
    op_name="spyre_host_routing_table",
    op_func=_routing_table,
    fake_impl=_routing_table_fake,
    dispatch_key="CompositeExplicitAutograd",
)
