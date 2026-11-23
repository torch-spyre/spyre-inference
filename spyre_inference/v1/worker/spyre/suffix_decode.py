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

"""MRV2: suffix decoding speculator, which upstream's V2 runner does not have.

Upstream's ``init_speculator`` (vllm/v1/worker/gpu/spec_decode/__init__.py) raises for
``method="suffix"``. ``building_suffix_speculator`` swaps in one that builds
``SpyreSuffixSpeculator`` while ``GPUModelRunner.__init__`` runs, as load.py swaps the
loader. The speculator follows V1's ``SuffixDecodingProposer``
(vllm/v1/spec_decode/suffix_decoding.py) on arctic-inference's suffix cache. It reads each
request's tokens from the runner's ``RequestState``, which the Spyre runner keeps on the
host, and returns drafts in the shape upstream's ngram speculator does.
"""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as upstream_model_runner
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.states import RequestState


def uses_suffix_decoding(vllm_config: VllmConfig) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    return spec is not None and spec.method == "suffix"


class SpyreSuffixSpeculator(BaseSpeculator):
    """Suffix decoding (https://arxiv.org/pdf/2411.04975) on the V2 runner.

    V2 verifies every request's ``num_speculative_tokens`` drafts, so a shorter
    speculation is filled with the request's last sampled token, as upstream's ngram
    speculator fills a missing match. A filled slot is an ordinary draft the target can
    reject. Like ngram, it has no draft probabilities.
    """

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, req_states: "RequestState") -> None:
        spec = vllm_config.speculative_config
        assert spec is not None and spec.method == "suffix"
        self.req_states = req_states
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.max_tree_depth = spec.suffix_decoding_max_tree_depth
        self.max_spec_factor = spec.suffix_decoding_max_spec_factor
        self.min_token_prob = spec.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy, as upstream's proposer: arctic-inference is needed only for suffix.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=spec.suffix_decoding_max_tree_depth,
            max_cached_requests=spec.suffix_decoding_max_cached_requests,
        )
        # How many of each active request's tokens the cache holds.
        self._fed: dict[str, int] = {}
        self.drafts = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, self.num_speculative_steps),
            dtype=torch.int64,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def _sync(self, req_id: str, req_idx: int, num_tokens: int) -> None:
        """Feed the cache the tokens of ``req_id`` it has not seen yet."""
        token_ids = self.req_states.all_token_ids.gpu[req_idx]
        fed = self._fed.get(req_id)
        if fed is None or fed > num_tokens:
            # New, or re-added with a different history: start over from its prompt.
            cache = self.suffix_cache
            if req_id in cache.active_requests:
                cache.stop_request(req_id)
            if req_id in cache.cached_requests:
                cache.evict_cached_response(req_id)
            fed = int(self.req_states.prompt_len.np[req_idx])
            cache.start_request(req_id, token_ids[:fed].numpy())
        if num_tokens > fed:
            # A resumed request's earlier outputs come with its first step here.
            self.suffix_cache.add_active_response(req_id, token_ids[fed:num_tokens].tolist())
        self._fed[req_id] = num_tokens

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: "DPSyncState | None" = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        drafts = self.drafts[:num_reqs]
        if dummy_run:
            return drafts

        req_states = self.req_states
        max_spec = (
            self.num_speculative_steps if num_speculative_tokens is None else num_speculative_tokens
        )
        req_indices = input_batch.idx_mapping[:num_reqs].tolist()
        total_len = req_states.total_len.gpu
        last = last_sampled.view(-1)
        drafts.copy_(last[input_batch.idx_mapping[:num_reqs]].unsqueeze(1).expand_as(drafts))
        for row, (req_idx, sampled) in enumerate(
            zip(req_indices, num_sampled[:num_reqs].tolist(), strict=True)
        ):
            if not sampled:
                # A partial prefill sampled nothing to continue from.
                continue
            req_id = req_states.index_to_req_id[req_idx]
            num_tokens = int(total_len[req_idx])
            self._sync(req_id, req_idx, num_tokens)
            if num_tokens >= self.max_model_len:
                continue
            # The cache matches only the last max_tree_depth tokens.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = req_states.all_token_ids.gpu[req_idx, start:num_tokens].numpy()
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(max_spec, self.max_model_len - num_tokens - 1),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            if draft.token_ids:
                n = len(draft.token_ids)
                drafts[row, :n] = torch.tensor(draft.token_ids, dtype=drafts.dtype)

        # Finished and preempted requests have left the request state.
        for req_id in self.suffix_cache.active_requests - req_states.req_id_to_index.keys():
            self.suffix_cache.stop_request(req_id)
            self._fed.pop(req_id, None)
        return drafts


@contextlib.contextmanager
def building_suffix_speculator() -> Iterator[None]:
    """Swap upstream's ``init_speculator`` for one that builds ``SpyreSuffixSpeculator``.

    ``GPUModelRunner.__init__`` looks it up as a module global of
    vllm/v1/worker/gpu/model_runner.py; the runner checks afterwards that the swap ran.
    """
    original = upstream_model_runner.init_speculator

    def init_speculator(
        vllm_config: VllmConfig, device: torch.device, req_states: "RequestState"
    ) -> SpyreSuffixSpeculator:
        return SpyreSuffixSpeculator(vllm_config, req_states)

    # ty reads upstream's return type as Unknown and rejects any declared one here.
    upstream_model_runner.init_speculator = init_speculator  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        upstream_model_runner.init_speculator = original
