# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific SiluAndMul implementation using out-of-tree (OOT) registration.

This module provides a custom SiluAndMul (SwiGLU) activation layer for
IBM's Spyre device, replacing the upstream vLLM implementation from
vllm/model_executor/layers/activation.py when instantiated.

Architecture:
    - OOT Registration: @SiluAndMul.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, fully transparent to the outer
      torch.compile graph (no opaque custom-op boundary)
    - CPU slicing workaround: Slice on CPU, transfer to Spyre separately to avoid
      memory corruption from slicing Spyre tensors directly

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.

Output Shape Note:
    input shape: [..., 2*d] -> output shape: [..., d]

References:
    - Upstream SiluAndMul: vllm/model_executor/layers/activation.py
    - Pattern reference:   spyre_inference/custom_ops/linear.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul

from .utils import convert

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device.

    This replaces the upstream vLLM SiluAndMul (vllm/model_executor/layers/activation.py)
    when instantiated, providing Spyre-specific optimizations and device handling.

    Computes: x -> silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2

    Fully transparent to the outer torch.compile graph — no opaque custom-op
    boundary. Slices input on CPU before transferring to Spyre to avoid memory
    corruption from slicing Spyre tensors directly.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer.

        Sets up the target device (Spyre) and dtype (float16) for computation.
        The simplified implementation computes directly in forward_oot() without
        requiring layer registry or custom op registration.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom SiluAndMul")

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        logger.warning_once(
            "SpyreSiluAndMul: no dtype promotion (torch-spyre limitation), "
            "expect numerical differences to upstream vLLM.",
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre-optimized SiLU and multiply activation (SwiGLU).

        Computes silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.
        
        Implementation:
            1. Convert to CPU and target dtype (avoids Spyre dtype conversion issues)
            2. Slice on CPU (avoids Spyre slicing memory corruption bug)
            3. Transfer each slice to Spyre (spyre_copy_from handles contiguity)
            4. Compute F.silu(x1) * x2 on Spyre
        
        NOTE: Slicing directly on Spyre tensors causes memory corruption and
        crashes with "free(): invalid size". This workaround ensures correctness
        at the cost of a D2H transfer for the input tensor.

        Forward pass is transparent to torch.compile - no custom op boundary.

        Args:
            x: Input tensor of shape [..., 2*d] containing concatenated gate halves.

        Returns:
            Activated output tensor of shape [..., d] on the original device
            with the original dtype.
        """
        x_dtype = x.dtype
        x_device = x.device

        # Convert dtype on CPU, slice there, then transfer each half to Spyre.
        # spyre_copy_from makes non-contiguous slices contiguous during H2D.
        x_cpu = x.to(device="cpu", dtype=self._target_dtype)
        d = x_cpu.shape[-1] // 2
        x1 = x_cpu[..., :d].to(self._target_device)
        x2 = x_cpu[..., d:].to(self._target_device)

        out = F.silu(x1) * x2
        return convert(out, device=x_device, dtype=x_dtype)


def register():
    """No-op: the custom-op barrier has been removed."""
    logger.debug("SpyreSiluAndMul: no custom ops to register")