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

"""Spyre OOT expert compute for unquantized FusedMoE.

The modular MoE kernels do not lower on Spyre (the OOT backend builds no
``moe_kernel``, so ``forward_native`` asserts). This evaluates every expert
densely with plain matmuls -- gather-free, so the whole path stays in matmul +
elementwise ops that lower -- mirroring hf-adapters#293. Routing is already done
by the Gemma4 custom routing function, so ``topk_weights`` is the final combine.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

logger = init_logger(__name__)


@UnquantizedFusedMoEMethod.register_oot(name="UnquantizedFusedMoEMethod")
class SpyreUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    @property
    def is_monolithic(self) -> bool:
        # Modular path: base is_monolithic would deref a None kernel.
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Split stacked ``w13_weight`` [E, 2I, H] into contiguous gate/up halves.

        An offset slice ``w13[:, I:, :]`` is a stick-unaligned partial-stick
        start, and the full transposed ``[E, H, 2I]`` blows the ~484 MiB
        per-core span at E=128. Runs on CPU before ``model.to("spyre")``; the
        stacked param is de-registered, not emptied -- a ``[0]`` param cannot be
        stickified and would fail the transfer.
        """
        super().process_weights_after_loading(layer)

        w13 = layer.w13_weight.data  # [E, 2I, H], on CPU (pre-transfer)
        inter = w13.shape[1] // 2

        w1 = w13[:, :inter, :].contiguous()  # gate, [E, I, H]
        w3 = w13[:, inter:, :].contiguous()  # up,   [E, I, H]

        layer.register_parameter("w1_weight", torch.nn.Parameter(w1, requires_grad=False))
        layer.register_parameter("w3_weight", torch.nn.Parameter(w3, requires_grad=False))
        del layer._parameters["w13_weight"]

    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: torch.nn.Module | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense-over-all-experts SwiGLU MoE for Spyre.

        ``topk_weights`` is the **dense** [T, E] combine from the Spyre routing
        patch, not the usual [T, K] -- rebuilding [T, E] from a [T, K] selection
        needs a scatter Spyre cannot lower. ``topk_ids`` is unused here.
        """
        if shared_experts is not None:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod does not support fused shared experts."
            )
        if not self.moe.is_act_and_mul:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod requires a gated (act-and-mul) expert MLP."
            )

        w1 = layer.w1_weight  # gate, [E, I, H]  (split in process_weights)
        w3 = layer.w3_weight  # up,   [E, I, H]
        w2 = layer.w2_weight  # down, [E, H, I]
        num_experts = w1.shape[0]

        # Fail loudly if the routing patch did not run: we cannot rebuild the
        # dense combine from a [T, K] selection here.
        if topk_weights.shape[-1] != num_experts:
            raise NotImplementedError(
                "SpyreUnquantizedFusedMoEMethod expects a dense [T, E] combine "
                f"from the Spyre routing patch, got shape {tuple(topk_weights.shape)}."
            )
        combine = topk_weights.to(x.dtype)  # [T, E]
        num_tokens = x.shape[0]

        # xb must be materialized contiguous: an expanded view into the batched
        # matmul does not lower ("expected exactly 1 generated variable").
        xb = x.unsqueeze(0).expand(num_experts, num_tokens, -1).contiguous()
        gate_out = torch.matmul(xb, w1.transpose(1, 2))  # [E, T, I]
        up_out = torch.matmul(xb, w3.transpose(1, 2))  # [E, T, I]
        activated = self._activate(gate_out, layer.activation) * up_out
        expert_out = torch.matmul(activated, w2.transpose(1, 2))  # [E, T, H]

        # bmm over contiguous tensors lowers; the size-1 broadcast-and-sum does not.
        expert_out = expert_out.permute(1, 0, 2).contiguous()  # [T, E, H]
        combined = torch.bmm(combine.unsqueeze(1), expert_out)  # [T, 1, H]
        return combined.squeeze(1)  # [T, H]

    @staticmethod
    def _activate(x: torch.Tensor, activation) -> torch.Tensor:
        # layer.activation may be a MoEActivation enum or a plain string.
        activation = getattr(activation, "value", activation)
        if activation == "gelu_tanh":
            return F.gelu(x, approximate="tanh")
        if activation == "silu":
            return F.silu(x)
        if activation == "gelu":
            return F.gelu(x)
        raise NotImplementedError(
            f"SpyreUnquantizedFusedMoEMethod: unsupported activation {activation!r}"
        )
