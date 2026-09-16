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
from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
    empty_exponential_noise_like,
)

from spyre_inference.v1.sample.async_ring_buffer import AsyncExponential_Log_RingBuffer

logger = init_logger(__name__)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """Force the sort-free top-k path. Upstream only takes it under
    ``allow_cpu_sync`` (CPU platform only); Spyre D2Hs logits before sampling, so
    that host-device sync is free and the full-vocab sort it otherwise runs is
    pure waste. Top-p still sorts (unaffected)."""

    def __init__(self, logprobs_mode: LogprobsMode, use_fp64_gumbel: bool, vllm_config: VllmConfig):
        super().__init__(logprobs_mode=logprobs_mode, use_fp64_gumbel=use_fp64_gumbel)

        if use_fp64_gumbel:
            logger.warning(
                "use_fp64=True is not compatible with async Exp(1) precomputation. "
                "Falling back to default noise generation with reduced performance on Spyre platform."
            )
        concurrency = SpyreTopKTopPSampler._try_get_concurrency(vllm_config)
        if concurrency is None:
            logger.warning(
                "The provided vllm_config does not provide max_num_seqs. "
                "Falling back to default noise generation with reduced performance on Spyre platform."
            )
        vocab_size = SpyreTopKTopPSampler._try_get_vocab_size(vllm_config)
        if vocab_size is None:
            logger.warning(
                "The provided vllm_config does not provide vocab_size. "
                "Falling back to default noise generation with reduced performance on Spyre platform."
            )

        if concurrency is not None and vocab_size is not None and not use_fp64_gumbel:
            self._noise_buffer = AsyncExponential_Log_RingBuffer(
                vocab_size=vocab_size,
                max_batch_size=concurrency,
            )
        else:
            self._noise_buffer = None

    @staticmethod
    def _try_get_concurrency(vllm_config: VllmConfig) -> int | None:
        """Try to extract the max_num_seqs parameter from the VllmConfig.

        Returns:
            The max_num_seqs value if present, otherwise None.
        """
        return getattr(vllm_config.scheduler_config, "max_num_seqs", None)

    @staticmethod
    def _try_get_vocab_size(vllm_config: VllmConfig) -> int | None:
        """Try to extract the vocab_size parameter from the VllmConfig.

        Returns:
            The vocab_size value if present, otherwise None.
        """
        if hasattr(vllm_config, "model_config") and hasattr(vllm_config.model_config, "hf_config"):
            hf_cfg = vllm_config.model_config.hf_config
            if hasattr(hf_cfg, "vocab_size"):
                # convention: HuggingFace model configs have a vocab_size attribute
                return hf_cfg.vocab_size
            elif hasattr(hf_cfg, "text_config") and hasattr(hf_cfg.text_config, "vocab_size"):
                # fallback: some multi-modal HuggingFace model configs have a text_config
                # with a vocab_size attribute
                return hf_cfg.text_config.vocab_size
        return None

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Mirrors upstream TopKTopPSampler.forward_native (vLLM 0.28.0) with two
        # Spyre changes: the sort-free top-k path (allow_cpu_sync=True) and a
        # log-space Gumbel draw. Re-sync with upstream on a vLLM bump.
        logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        # argmax(softmax(x)/q) == argmax(x - log q) for q~Exp(1): the softmax's
        # per-row normalization is a constant the argmax ignores, so skip it.
        # Noise generation mirrors upstream random_sample.

        if generators and self._noise_buffer is not None:
            logger.warning(
                "Generators are not supported by async Exp(1) precomputation. Falling back to sync hot-path computation.",
            )

        if generators or self._noise_buffer is None:
            q = empty_exponential_noise_like(logits, self.use_fp64_gumbel)
            if len(generators) != logits.shape[0]:
                q.exponential_()
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
            sampled = SpyreTopKTopPSampler._sample_with_predrawn_log_noise(logits, q)
        else:
            with self._noise_buffer.borrow_rows(n=logits.shape[0]) as log_noise:
                sampled = SpyreTopKTopPSampler._sample_with_predrawn_log_noise(logits, log_noise)

        return sampled, logits_to_return

    def shutdown(self) -> None:
        if self._noise_buffer is not None:
            self._noise_buffer.shutdown()

    @staticmethod
    def _sample_with_predrawn_log_noise(
        logits: torch.Tensor, log_noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample using pre-drawn exponential log noise (no exponential_() call)."""
        return (logits - log_noise).argmax(dim=-1).view(-1)
