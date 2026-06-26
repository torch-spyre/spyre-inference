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

"""Spyre patch for OPT's positional embedding.

`OPTLearnedPositionalEmbedding.forward` does `positions + self.offset` on
int64 tensors; torch-spyre has no integer aten.add kernel. Do the add on
CPU. Remove once torch-spyre supports integer aten.add.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models import opt

logger = init_logger(__name__)


def _spyre_forward(self, positions: torch.Tensor) -> torch.Tensor:
    shifted = (positions.cpu() + self.offset).to(positions.device)
    return nn.Embedding.forward(self, shifted)


def register() -> None:
    opt.OPTLearnedPositionalEmbedding.forward = _spyre_forward
    logger.info("Patched OPTLearnedPositionalEmbedding.forward for Spyre")
