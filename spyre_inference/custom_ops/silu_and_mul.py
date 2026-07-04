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

    def forward_oot(self, x) -> torch.Tensor:
        """SwiGLU: silu(gate) * up, output shape [..., d].

        `x` is either a `[gate, up]` list of already-split tensors (from an
        un-fused gate_up_proj, see unfuse.py) or a fused [..., 2*d] tensor.
        The fused path only runs for layers unfuse left alone (e.g. quantized).
        """
        if isinstance(x, (list, tuple)):
            # Fast path: halves already split, both contiguous on device.
            x1, x2 = x
        else:
            # Fused path: slice on CPU — slicing a Spyre tensor corrupts memory,
            # and non-contiguous slices corrupt again on transfer back.
            original_device = x.device
            x = convert(x, device="cpu")
            d = x.shape[-1] // 2
            x1 = convert(x[..., :d].contiguous(), device=original_device)
            x2 = convert(x[..., d:].contiguous(), device=original_device)
        return F.silu(x1) * x2
