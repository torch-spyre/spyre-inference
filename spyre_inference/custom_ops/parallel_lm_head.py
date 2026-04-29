# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific ParallelLMHead implementation using out-of-tree (OOT) registration.

Follows the same architecture as SpyreRMSNorm:
    - OOT Registration replaces upstream layer
    - forward_oot uses torch.ops custom-op boundary
    - _op_func executes outside torch.compile
    - _forward_spyre_impl contains actual execution (CPU/Spyre)
"""

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree

from functools import lru_cache

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead for Spyre following RMSNorm pattern."""

    _dynamic_arg_dims = {"hidden_states": [], "embedding_bias": []}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre ParallelLMHead")

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        # Compile Spyre kernel (future use)
        self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

        # Register layer for custom op lookup
        self._layer_name = register_layer(self, "spyre_parallel_lm_head")

    def _apply(self, fn, recurse=True):
        """Keep LMHead weights on CPU."""
        return self

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """OOT forward pass using custom-op boundary."""

        logger.debug(
            f"LMHead forward: shape={hidden_states.shape}, "
            f"dtype={hidden_states.dtype}, device={hidden_states.device}"
        )

        # Direct execution if already on Spyre
        if hidden_states.device.type == "spyre":
            return self._forward_spyre_impl(hidden_states, embedding_bias)

        # Support both [B, H] and [B, S, H]
        output_shape = hidden_states.shape[:-1] + (self.weight.shape[0],)

        output = torch.empty(
            output_shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Custom op call (outside torch.compile graph)
        torch.ops.vllm.spyre_parallel_lm_head(
            hidden_states,
            output,
            self._layer_name,
            embedding_bias,
        )

        return output

    @staticmethod
    def forward_spyre(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Spyre-compiled LMHead kernel."""
        return F.linear(hidden_states, weight, embedding_bias)

    def _forward_spyre_impl(
        self,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execution path (CPU fallback / Spyre-ready)."""

        # Optional safety check
        if hidden_states.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"Hidden size mismatch: got {hidden_states.shape[-1]}, "
                f"expected {self.weight.shape[1]}"
            )

        hs_dtype = hidden_states.dtype
        hs_device = hidden_states.device

        logits = self.maybe_compiled_forward_spyre(
            convert(hidden_states, self._target_device, self._target_dtype),
            convert(self.weight.data, self._target_device, self._target_dtype),
            convert(embedding_bias, self._target_device, self._target_dtype)
            if embedding_bias is not None else None,
        )

        return pytree.tree_map(
            lambda x: convert(x, dtype=hs_dtype, device=hs_device),
            logits,
        )


def _op_func(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    embedding_bias: torch.Tensor | None = None,
) -> None:
    """Custom op execution (outside torch.compile)."""

    layer = get_layer(layer_name)
    result = layer._forward_spyre_impl(hidden_states, embedding_bias)

    output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """Register custom op."""
    direct_register_custom_op(
        op_name="spyre_parallel_lm_head",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreParallelLMHead")
