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

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .utils import convert


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        """Project through the lm_head, then D2H the logits on the single-card path.

        SpyreParallelLMHead.forward_oot returns logits on Spyre so the TP
        all_gather in ``_gather_logits`` (which also D2Hs) can run on-device.
        Upstream only calls ``_gather_logits`` when ``tp_size > 1``, so with a
        single card the logits would otherwise stay on Spyre and the sampler's
        ``logits.to(torch.float32)`` crashes torch-spyre's ``copy_from_d2d``.
        Move them to CPU here so downstream sampling always gets CPU logits.
        """
        logits = super()._apply_head(lm_head, hidden_states, embedding_bias)
        if lm_head.tp_size <= 1:
            logits = convert(logits, device="cpu")
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Gather TP-sharded logits on Spyre, then move the result to CPU."""
        return convert(super()._gather_logits(logits), device="cpu")
