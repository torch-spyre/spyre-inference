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

"""Spyre-specific SiluAndMul implementation"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul

from .utils import convert

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer."""
        super().__init__(*args, **kwargs)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre-optimized SiLU and multiply activation (SwiGLU).

        Args:
            x: Input tensor of shape [..., 2*d] containing concatenated gate halves.

        Returns:
            Activated output tensor of shape [..., d] with same device and dtype as input.
        """
        original_device = x.device

        # Slicing must run on CPU (slicing Spyre tensors causes corruption);
        # convert is a no-op when x is already on CPU.
        x = convert(x, device="cpu")

        # Slice and make contiguous before transferring to Spyre.
        # Non-contiguous slices get corrupted during transfer to Spyre!
        d = x.shape[-1] // 2
        x1 = x[..., :d].contiguous()
        x2 = x[..., d:].contiguous()

        # Transfer contiguous slices back to original device.
        x1 = convert(x1, device=original_device)
        x2 = convert(x2, device=original_device)

        return F.silu(x1) * x2
