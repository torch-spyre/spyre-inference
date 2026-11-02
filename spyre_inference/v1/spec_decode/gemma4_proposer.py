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

"""Gemma-4 MTP drafting on Spyre."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod
from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.spec_decode import kernels

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


def create_spyre_drafter(vllm_config: VllmConfig, runner: Any) -> SpyreGemma4Proposer:
    spec = vllm_config.speculative_config
    assert spec is not None
    if not spec.use_gemma4_mtp():
        raise NotImplementedError(
            f"Speculative decoding method {spec.method!r} is not supported on Spyre; "
            "only a Gemma-4 MTP assistant checkpoint is."
        )
    kernels.install()
    return SpyreGemma4Proposer(vllm_config, runner.device, runner)


def _all_gather_host(shard: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
    world_size = torch.distributed.get_world_size(group)
    # gloo's half support varies by build; the weights are small, so gather in fp32.
    parts = [torch.empty_like(shard, dtype=torch.float32) for _ in range(world_size)]
    torch.distributed.all_gather(parts, shard.float().contiguous(), group=group)
    return torch.cat(parts, dim=dim).to(shard.dtype)


def _set_weight(layer: nn.Module, weight: torch.Tensor) -> None:
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)


class SpyreGemma4Proposer(Gemma4Proposer):
    """Drafts one row per request, at every step.

    The assistant's layers are Q-only and read the target's KV cache, so the drafter keeps
    no state of its own: a position's draft depends only on that position's inputs.
    Upstream still runs the first draft step over every scheduled token -- the whole
    prompt at prefill -- and keeps one row per request. Running just those rows gives the
    same drafts, makes a draft step cost the same at prefill as at decode, and keeps every
    draft forward on the decode buckets warmup compiles.

    Greedy drafts keep the draft hidden state on the card between passes: it feeds the
    next pass and the lm_head where it is, and only the logits come to the host.
    """

    # Upstream types it as a bare nn.Module; once loaded it is the runner's draft wrapper.
    model: Any

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner: Any) -> None:
        super().__init__(vllm_config, device, runner)
        self._runner = runner
        self._backbone_hidden_size: int = 0

    def get_model_name(self, model: nn.Module) -> str:
        # Spyre adaptations subclass the vLLM model as "Spyre<Name>"; upstream's per-model
        # branches (e.g. the Gemma-4 image-token id) match on the vLLM name.
        return super().get_model_name(model).removeprefix("Spyre")

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._backbone_hidden_size = self.model.model.backbone_hidden_size
        if getattr(self.model, "masked_embedding", None) is not None:
            # Its top-k and vocabulary gather have no Spyre lowering; the full draft-dim
            # lm_head is cheap at a few rows, and its argmax is the unmasked one.
            logger.info("Gemma-4 MTP: centroid logit masking disabled on Spyre.")
            self.model.masked_embedding = None
        self._replicate_projections()

    def _replicate_projections(self) -> None:
        """Hold the full pre/post projections on every rank, so drafting runs no eager collective.

        Both run between the compiled blocks, and an eager Spyre collective sets up its comms
        bundle on every call: about 90 ms for the pre-projection's gather and 150 ms for the
        post-projection's reduce, per draft step, against about 15 ms for the whole draft
        forward. Together the two weights are about 17 MB at fp16, gathered once here while
        they are still on the host.
        """
        tp_group = get_tp_group()
        if tp_group.world_size == 1:
            return
        predictor = self.model.model
        pre, post = predictor.pre_projection, predictor.post_projection
        assert pre.bias is None and post.bias is None
        # Spyre's linear methods have already stored each weight transposed, as [in, out].
        transposed = isinstance(getattr(pre, "quant_method", None), SpyreTransposedWeightMethod)
        assert transposed == isinstance(
            getattr(post, "quant_method", None), SpyreTransposedWeightMethod
        )
        assert not getattr(pre, "spyre_row_padding", 0), "padding would interleave the shards"
        out_dim, in_dim = (1, 0) if transposed else (0, 1)
        _set_weight(pre, _all_gather_host(pre.weight.data, dim=out_dim, group=tp_group.cpu_group))
        pre.gather_output = False
        _set_weight(post, _all_gather_host(post.weight.data, dim=in_dim, group=tp_group.cpu_group))
        # The full input in, the full output out: no split, no reduce.
        post.input_is_parallel = True
        post.reduce_results = False

    def attention_layers(self) -> list[Attention]:
        return [m for m in self.model.modules() if isinstance(m, Attention)]

    def _setup_centroids_cuda_graphs(self) -> None:
        """No CUDA graphs here; ``load_model`` drops the masking they would capture."""

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return greedy_sample(self.model.compute_logits(hidden_states).float())

    def _drafts_on_card(self, batch_size: int, sampling_metadata: SamplingMetadata) -> bool:
        """Whether this step's draft passes can keep their hidden state on the card.

        Greedy only: a probabilistic draft samples per request, not per padded row. The
        padded draft batch must also be a width the lm_head was compiled at.
        """
        greedy = not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy
        return greedy and self.model.lm_head_takes_padded_rows(batch_size)

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Draft on the runner's own builders.

        The draft layers already sit in the runner's attention groups, split by sliding
        window, and warmup records their kernels through those builders. Reusing them keeps
        drafting on exactly the recorded set.
        """
        draft_groups: list[AttentionGroup] = []
        for groups in self._runner.attn_groups:
            for group in groups:
                names = [n for n in group.layer_names if n in self._draft_attn_layer_names]
                if not names:
                    continue
                draft_group = AttentionGroup(
                    group.backend, names, group.kv_cache_spec, group.kv_cache_group_id
                )
                draft_group.metadata_builders = group.metadata_builders
                draft_groups.append(draft_group)
        missing = self._draft_attn_layer_names - {n for g in draft_groups for n in g.layer_names}
        assert not missing, f"draft attention layers with no runner attention group: {missing}"
        self.draft_attn_groups = draft_groups
        self.kv_cache_gid = draft_groups[0].kv_cache_group_id
        self.block_size = draft_groups[0].get_metadata_builder().kv_cache_spec.block_size

    def _draft_attn_metadata(
        self,
        cad: CommonAttentionMetadata,
        rows: torch.Tensor,
        num_rejected_tokens: torch.Tensor | None,
    ) -> CommonAttentionMetadata:
        """One query per request, at its last accepted position.

        Rejected drafts' KV entries are still in the cache, so the context ends before
        them; past that, causality is the context length itself.
        """
        batch_size = cad.batch_size()
        seq_lens = cad.seq_lens[:batch_size]
        if num_rejected_tokens is not None:
            seq_lens = seq_lens - num_rejected_tokens[:batch_size].to(seq_lens.dtype)
        query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=batch_size,
            num_actual_tokens=batch_size,
            max_query_len=1,
            max_seq_len=int(seq_lens.max()) if batch_size else 0,
            block_table_tensor=cad.block_table_tensor[:batch_size],
            # Never written: the draft layers share the target's KV.
            slot_mapping=cad.slot_mapping[rows],
            causal=True,
        )

    def _draft_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        on_device: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.shape[-1]
        inputs_embeds = None
        if self.supports_mm_inputs:
            inputs_embeds = self.model.embed_input_ids(input_ids)
            input_ids = None
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            return self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                on_device=on_device,
            )

    def propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: Any = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: Any = None,
    ) -> torch.Tensor:
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        cad = common_attn_metadata
        batch_size = cad.batch_size()
        if num_speculative_tokens == 0:
            return torch.empty(batch_size, 0, dtype=torch.int64)
        if self._runner._is_all_reqs_chunked_prefill():
            # Every request is mid-prefill, and the scheduler drops a prefill chunk's
            # drafts, so the k draft forwards would be thrown away on every chunk but a
            # prompt's last.
            return torch.zeros(batch_size, num_speculative_tokens, dtype=torch.int64)

        if token_indices_to_sample is None:
            token_indices_to_sample = cad.query_start_loc[1 : batch_size + 1] - 1
        rows = token_indices_to_sample[:batch_size].long()
        draft_cad = self._draft_attn_metadata(cad, rows, num_rejected_tokens_gpu)
        # Positions stay fixed across steps (constant_draft_positions), so one build serves all.
        _, attn_metadata = self.build_per_group_and_layer_attn_metadata(draft_cad)

        input_ids = next_token_ids[:batch_size].int()
        positions = target_positions[..., rows]
        hidden_states = target_hidden_states[rows]
        on_card = self._drafts_on_card(batch_size, sampling_metadata)
        draft_ids: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] = []
        for _ in range(num_speculative_tokens):
            last_hidden_states, hidden_states = self._draft_forward(
                input_ids, positions, hidden_states, attn_metadata, on_device=on_card
            )
            if on_card:
                # Every padded row is projected; the padding's tokens are dropped here.
                tokens, probs = self._greedy_sample(last_hidden_states)[:batch_size], None
            else:
                tokens, probs = self._sample_draft_tokens(last_hidden_states, sampling_metadata)
            draft_ids.append(tokens)
            if probs is not None:
                draft_probs.append(probs)
            input_ids = tokens.int()

        if draft_probs:
            self._last_draft_probs = torch.stack(draft_probs, dim=1).contiguous()
        return torch.stack(draft_ids, dim=1)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int, *args, **kwargs) -> None:
        """Compile the draft blocks and the draft lm_head at one decode bucket.

        Upstream calls this at every body bucket. Drafting never exceeds one row per
        request, so a wider bucket would compile a graph nothing reaches.
        """
        if num_tokens > self.max_batch_size:
            return
        dtype = cast(torch.dtype, self.dtype)
        last_hidden_states, _ = self._draft_forward(
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.int64),
            torch.zeros(num_tokens, self._backbone_hidden_size, dtype=dtype),
            None,
        )
        self.model.compute_logits(last_hidden_states)
