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
import torch.distributed as dist
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from spyre_inference import envs
from spyre_inference.v1.sample import device_greedy

from .lazy_compile import CompileOutermost, maybe_compile
from .utils import convert
from .vocab_parallel_embedding import promote_tied_lm_head


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(CompileOutermost, LogitsProcessor):
    def forward(self, lm_head, hidden_states, embedding_bias=None, skip_gather=False):
        """All-greedy batches return ``[B]`` token ids from the card.

        Everything else keeps the host path: full logits, copied in ``_apply_head``.
        Soft-cap and a non-unit scale stay on the host too. Both are monotonic for
        the usual positive scale, but they run after ``_get_logits`` upstream, and
        applying them again to token ids would corrupt the ids.
        """
        if (
            device_greedy.sampling_active()
            and not skip_gather
            and not self.logits_as_input
            and self.soft_cap is None
            and self.scale == 1.0
            and lm_head.tp_size <= 1
        ):
            promote_tied_lm_head(lm_head)
            logits = LogitsProcessor._apply_head(self, lm_head, hidden_states, embedding_bias)
            mask = self._vocab_padding_mask(logits.shape[-1], logits.device)
            return convert(self._greedy_argmax(logits, mask), device="cpu")
        return super().forward(lm_head, hidden_states, embedding_bias, skip_gather)

    def _vocab_padding_mask(self, padded_vocab: int, device: torch.device) -> torch.Tensor:
        cached = getattr(self, "_greedy_pad_mask", None)
        if cached is None or cached.shape[0] != padded_vocab or cached.device != device:
            self._greedy_pad_mask = device_greedy.padding_mask(
                padded_vocab, self.org_vocab_size, device
            )
        return self._greedy_pad_mask

    @maybe_compile
    def _greedy_argmax(self, logits: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return device_greedy.greedy_token_ids(logits, pad_mask)

    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        """Project through the lm_head, then D2H the logits on the single-card path.

        The head's ``SpyreUnquantizedLMHeadMethod.apply`` returns logits on Spyre so the
        TP all_gather in ``_gather_logits`` (which also D2Hs) can run on-device.
        Upstream only calls ``_gather_logits`` when ``tp_size > 1``, so with a
        single card the logits would otherwise stay on Spyre and the sampler's
        ``logits.to(torch.float32)`` crashes torch-spyre's ``copy_from_d2d``.
        Move them to CPU here so downstream sampling always gets CPU logits.
        """
        # A model may hand us its embedding table as the head; give it a padded `Wᵀ`
        # on the first call, where the head's identity is finally unambiguous.
        promote_tied_lm_head(lm_head)

        logits = super()._apply_head(lm_head, hidden_states, embedding_bias)
        if lm_head.tp_size <= 1:
            logits = convert(logits, device="cpu")
        return logits

    @maybe_compile
    def _all_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_gather(logits)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # `Platform.use_all_gather()` is constant-True on Spyre, so the gather is always the
        # all-gather form; compiled it lowers to `spyre::all_gather_async`, then D2H for sampling.
        if envs.SPYRE_LOGITS_CPU_GATHER:
            return self._cpu_all_gather_logits(logits)
        return convert(self._all_gather_logits(logits), device="cpu")

    @staticmethod
    def _cpu_all_gather_logits(logits: torch.Tensor) -> torch.Tensor:
        # The device-to-host copy converts every fp16 element on the host, so copying one
        # vocab shard per rank is cheaper than a full-vocab copy on every rank.
        tp = get_tp_group()
        shard = convert(logits, device="cpu").contiguous()
        shards = [torch.empty_like(shard) for _ in range(tp.world_size)]
        dist.all_gather(shards, shard, group=tp.cpu_group)  # ty: ignore[possibly-missing-attribute]
        return torch.cat(shards, dim=-1)
