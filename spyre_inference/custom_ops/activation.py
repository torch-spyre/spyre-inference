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

"""Spyre OOT replacement for the `gelu_new` activation.

References:
    - Upstream NewGELU: vllm/model_executor/layers/activation.py
"""

import math

import torch
from vllm.model_executor.layers.activation import NewGELU


@NewGELU.register_oot(name="NewGELU")
class SpyreNewGELU(NewGELU):
    """`gelu_new` for Spyre, cubing by multiplication instead of `torch.pow`."""

    # `torch.pow(x, 3.0)` returns `|x| ** 4` on Spyre (torch-spyre#4009).
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))
