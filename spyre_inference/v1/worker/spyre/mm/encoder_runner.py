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

"""MRV2: mm-embed staging (vllm/v1/worker/gpu/mm/encoder_runner.py:EncoderRunner).

A Spyre-resident staging buffer for multimodal embeddings, ported from
spyre_inference's existing mm-embed staging work (commit 2623972). Needs
the model wrapper (load.load_and_wrap) installed first: the buffer must
allocate on Spyre, not on the CPU device the base runner constructs on.
"""

from typing import Any

import torch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner


class SpyreEncoderRunner(EncoderRunner):
    """MRV2: EncoderRunner, staging inputs_embeds on Spyre instead of the GPU."""

    def __init__(
        self,
        model: Any,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError
