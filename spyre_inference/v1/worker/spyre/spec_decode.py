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

"""MRV2 speculators: upstream's DFlash/LiLiCorr bookkeeping, draft on Spyre.

Upstream's propose() and its input-prep kernel run as-is; the overrides keep the
draft on Spyre, on shapes warmup compiled, and the LiLiCorr candidate walk on the
host.
"""

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.worker.model_load import move_model_to_spyre
from spyre_inference.v1.worker.spyre.attn_groups import split_by_window

if TYPE_CHECKING:
    from spyre_inference.models.lilicorr import SpyreLiLiCorrForCausalLM
    from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM


def padded_draft_reqs(num_reqs: int, max_num_reqs: int) -> int:
    # Capped: max_num_batched_tokens need only cover max_num_reqs blocks.
    return min(1 << (num_reqs - 1).bit_length(), max_num_reqs)


class _DraftBatchPadding:
    """Pads the draft step to a power-of-two request count, the batches warmup compiles."""

    def __init__(self, block: int, max_num_reqs: int) -> None:
        self.block = block
        self.max_num_reqs = max_num_reqs

    def dispatch(self, num_reqs, num_tokens, *args, **kwargs) -> BatchExecutionDescriptor:
        padded_reqs = padded_draft_reqs(num_reqs, self.max_num_reqs)
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=padded_reqs * self.block, num_reqs=num_reqs
        )


class _SpyreSpeculatorMixin:
    """Platform overrides shared by the DFlash and LiLiCorr speculators.

    Upstream's two speculators have different base classes (``DFlashSpeculator``
    and ``LiLiCorrSpeculator``, itself a DFlash2 subclass), so the overrides live
    in this mixin instead of one inheriting from the other. ``super()`` calls
    resolve into whichever upstream speculator is the second base.
    """

    model: nn.Module

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
    ) -> None:
        super().__init__(vllm_config, device)
        self.spyre_device = spyre_device
        self._context: torch.Tensor | None = None

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]):
        model = cast(Any, super().load_draft_model(target_model, target_attn_layer_names))
        move_model_to_spyre(model, self.spyre_device)
        model.prepare_for_spyre()
        return model

    def set_attn(self, model_state, kv_cache_config, block_tables, *args, **kwargs) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables, *args, **kwargs)
        # Splits within each KV cache group, so upstream's group ids hold.
        split_by_window(
            self.attn_groups, self.attn_vllm_config, self.device, block_tables.block_sizes
        )
        causal_by_layer = dict(
            zip(
                self.model.get_draft_kv_cache_layer_names(),
                self.model.get_draft_attn_causal(),
                strict=True,
            )
        )
        for group in (g for groups in self.attn_groups for g in groups):
            (causal, *mixed) = {causal_by_layer[name] for name in group.layer_names}
            if mixed:
                raise ValueError(
                    f"Draft layers {group.layer_names} share an attention group but differ in "
                    "causality; Spyre masks per group"
                )
            for builder in group.metadata_builders:
                cast(SpyreAttentionMetadataBuilder, builder).causal_override = causal

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self.query_cudagraph_manager = _DraftBatchPadding(  # ty: ignore[invalid-assignment]
            self.num_query_per_req, self.max_num_reqs
        )

    def capture(self) -> None:
        pass

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        *args,
        dummy_run: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if not dummy_run:
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                *args,
                **kwargs,
            )
        # Warmup's decoder dummies skip attention, and upstream's skip_attn branch (GPU
        # memory profiling) reads host buffers with no hook; warm the context store at
        # this row count against the null page, and the draft batches in warm_draft_batches.
        self._store_hidden_states(0, last_hidden_states, aux_hidden_states)
        self._precompute_context_kv(0, 0, dummy_run=True)
        return self.draft_tokens[: input_batch.num_reqs]

    def _store_hidden_states(self, num_tokens, last_hidden_states, aux_hidden_states, *args):
        # Stays on Spyre at the target's padded width; upstream copies into a host buffer.
        assert aux_hidden_states, "Spyre speculative decoding requires auxiliary hidden states"
        self._context = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))

    def _precompute_context_kv(self, start: int, end: int, dummy_run: bool = False) -> None:
        # Upstream passes start > 0 only from _prepare_graph_replay (FULL CUDA graphs).
        assert start == 0 and self._context is not None
        rows = self._context.shape[0]
        # Rows past the batch write the null page (upstream's PAD_SLOT_ID, clamped).
        slots = F.pad(self._context_slot_mappings[0][:end], (0, rows - end), value=-1)
        if dummy_run:
            slots.fill_(-1)
        self.model.precompute_and_store_context_kv(
            self._context,
            convert(F.pad(self.context_positions[:end], (0, rows - end)), self.spyre_device),
            slots.clamp(min=0),
        )

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        with set_forward_context(
            attn_metadata, self.vllm_config, num_tokens=num_tokens, slot_mapping=slot_mappings
        ):
            return self.model(
                input_ids=convert(
                    self.input_buffers.input_ids[:num_tokens], self.spyre_device, torch.int64
                ),
                positions=convert(self.input_buffers.positions[:num_tokens], self.spyre_device),
            )

    def _publish_padded_slot_mappings(self, num_tokens: int, num_tokens_padded: int) -> None:
        # Upstream's eager metadata publishes slots for the real rows only; the padded
        # rows' KV write goes to the null page, so the block graph keeps its width.
        if num_tokens >= num_tokens_padded:
            return
        for group in (g for groups in self.attn_groups for g in groups):
            slots = self.block_tables.slot_mappings[group.kv_cache_group_id]
            builder = cast(SpyreAttentionMetadataBuilder, group.get_metadata_builder())
            builder.publish_slot_mapping(
                F.pad(slots[:num_tokens], (0, num_tokens_padded - num_tokens), value=0)
            )


class SpyreDFlashSpeculator(_SpyreSpeculatorMixin, DFlashSpeculator):
    """DFlash/XPress drafts: upstream bookkeeping, proposal generated by the model."""

    model: "SpyreDFlashQwen3ForCausalLM"

    def _generate_draft(self, num_reqs: int, num_tokens_padded: int, *args, **kwargs) -> None:
        num_tokens = num_reqs * self.num_query_per_req
        self._publish_padded_slot_mappings(num_tokens, num_tokens_padded)
        hidden = self._run_model(num_tokens_padded, *args, **kwargs)
        block = self.num_query_per_req
        anchors = self.input_buffers.input_ids[:num_tokens_padded:block].to(torch.int64)
        proposals = self.model.propose_block(hidden, anchors)
        self.draft_tokens[:num_reqs] = torch.tensor(proposals[:num_reqs])

    def warm_draft_batches(self) -> None:
        """The draft forward and proposal at every padded batch _DraftBatchPadding serves.

        Upstream's warmup dummies only reach the draft at their own request counts.
        """
        block = self.num_query_per_req
        widths = {padded_draft_reqs(n, self.max_num_reqs) for n in range(1, self.max_num_reqs + 1)}
        for reqs in sorted(widths, reverse=True):
            num_tokens = reqs * block
            attn_layer.publish_null_slots(num_tokens)
            self.input_buffers.input_ids[:num_tokens] = self.parallel_drafting_token_id
            self.input_buffers.input_ids[:num_tokens:block] = 0
            self.input_buffers.positions[:num_tokens] = torch.arange(block).repeat(reqs)
            self._generate_draft(reqs, num_tokens, None, {}, None)
        self.draft_tokens.zero_()


class SpyreLiLiCorrSpeculator(_SpyreSpeculatorMixin, LiLiCorrSpeculator):
    """LiLiCorr drafts: upstream proposal flow with the lattice scored on Spyre.

    ``_generate_draft`` scores the lattice on Spyre, then hands the host scores to
    upstream's ``CandidateSampler``; its Triton walk and draft-logits cache are
    bound to csrc host ops (``kernels.HOST_KERNELS``).
    """

    model: "SpyreLiLiCorrForCausalLM"

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
    ) -> None:
        super().__init__(vllm_config, device, spyre_device)
        config = self.draft_model_config.hf_config
        # Upstream allocates on the runner device, which is CPU here, as bool. The
        # anchor is consumed by the device lattice, so make it fp16 on Spyre.
        self.anchor_hidden = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=spyre_device
        )
        self.anchor_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.float16, device=spyre_device
        )
        self._sample_row_cache: dict[int, torch.Tensor] = {}
        self._sample_row_device_cache: dict[int, torch.Tensor] = {}
        # Anchor indices and the validity mask are rebuilt every step; keep their
        # device buffers and copy the new values in.
        self._anchor_index_cache: dict[int, torch.Tensor] = {}
        self._anchor_mask_cache: dict[int, torch.Tensor] = {}

    def propose(self, *args, dummy_run: bool = False, **kwargs):
        out = super().propose(*args, dummy_run=dummy_run, **kwargs)
        if dummy_run:
            # The dummy path stops before prepare_context_anchor, so the anchor
            # gather is warmed explicitly, once per target bucket width.
            self._warm_anchor_gather()
        return out

    def _sample_rows(self, padded_reqs: int) -> torch.Tensor:
        """Mask-token rows in the padded draft batch: one anchor plus N mask rows.

        LiLiCorr keeps DFlash's fill-in layout, so the anchor (row 0 of each
        request block) is the bonus token and only rows 1..block-1 predict. The
        indices are static per padded width, which keeps the gather compiled.
        """
        cached = self._sample_row_cache.get(padded_reqs)
        if cached is None:
            block = self.num_query_per_req
            cached = (
                torch.arange(padded_reqs, dtype=torch.long)[:, None] * block
                + torch.arange(1, block, dtype=torch.long)[None, :]
            ).reshape(-1)
            self._sample_row_cache[padded_reqs] = cached
        return cached

    def _sample_rows_device(self, padded_reqs: int) -> torch.Tensor:
        # The row set is static per padded width, so convert it once and reuse.
        cached = self._sample_row_device_cache.get(padded_reqs)
        if cached is None:
            cached = convert(self._sample_rows(padded_reqs), self.spyre_device, torch.int64)
            self._sample_row_device_cache[padded_reqs] = cached
        return cached

    def _anchor_buffer(
        self, cache: dict[int, torch.Tensor], value: torch.Tensor, num_reqs: int
    ) -> torch.Tensor:
        buffer = cache.get(num_reqs)
        if buffer is None:
            buffer = convert(value, self.spyre_device)
            cache[num_reqs] = buffer
        else:
            buffer.copy_(value)
        return buffer

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        """Publish each request's last committed target feature, normalized.

        ``_store_hidden_states`` kept the raw combined context; upstream normalizes
        the gathered row with the draft's ``hidden_norm``. The runner device is CPU,
        so the index arithmetic stays on the host and only the gather runs on Spyre.
        """
        assert num_rejected is not None and self._context is not None
        starts = query_start_loc[:num_reqs].detach().cpu()
        ends = (
            query_start_loc[1 : num_reqs + 1].detach().cpu()
            - num_rejected[:num_reqs].detach().cpu()
        )
        valid = ends > starts
        indices = self._anchor_buffer(
            self._anchor_index_cache, (ends - 1).clamp_min(0).long(), num_reqs
        )
        rows = torch.index_select(self._context, 0, indices)
        anchor = self.model.model.normalize_context(rows)
        mask = self._anchor_buffer(
            self._anchor_mask_cache, valid.to(torch.float16), num_reqs
        )
        self.anchor_hidden.zero_()
        self.anchor_valid.zero_()
        self.anchor_hidden[:num_reqs].copy_(anchor * mask[:, None])
        self.anchor_valid[:num_reqs].copy_(mask)

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # ``compute_candidates`` returns CPU ids/log-probs (the Spyre lm_head D2Hs
        # them); candidates are embedded through the target table on Spyre.
        num_reqs, steps, topk = candidate_ids.shape
        ids = convert(candidate_ids.reshape(-1), self.spyre_device, torch.int64)
        embeddings = self.target_embeddings(ids).view(num_reqs, steps, topk, -1)
        return self.model.model.lilicorr.score(
            token_embeddings=embeddings,
            candidate_log_probs=unary_logits,
            pass_hidden=hidden_states,
            anchor_hidden=self.anchor_hidden[:num_reqs],
            anchor_valid=self.anchor_valid[:num_reqs],
        )

    def _generate_draft(self, num_reqs: int, num_tokens_padded: int, *args, **kwargs) -> None:
        # Shapes use the padded request count so each compiled piece (draft body,
        # lm-head rows, embedding gather, lattice) has one graph per batch width.
        # Proposals for padded requests are computed and discarded, keeping the
        # graphs independent of the live batch size.
        num_tokens = num_reqs * self.num_query_per_req
        self._publish_padded_slot_mappings(num_tokens, num_tokens_padded)
        hidden = self._run_model(num_tokens_padded, *args, **kwargs)
        block = self.num_query_per_req
        padded_reqs = num_tokens_padded // block
        steps = self.num_speculative_steps
        sample_hidden = torch.index_select(hidden, 0, self._sample_rows_device(padded_reqs))
        candidate_ids, log_probs = self.model.compute_candidates(sample_hidden)
        topk = self.model.model.lilicorr.candidate_topk
        candidate_ids = candidate_ids.view(padded_reqs, steps, topk)
        log_probs = log_probs.view_as(candidate_ids)
        pass_hidden = sample_hidden.view(padded_reqs, steps, -1)
        scores = self._score_candidates(candidate_ids, log_probs, pass_hidden)
        # Upstream's walk and realized-proposal cache are csrc host ops here, so
        # this stays the standard CandidateSampler path for greedy and probabilistic.
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            padded_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )

    def _warm_anchor_gather(self) -> None:
        # index_select and the context norm compile per (source width, index
        # length). Each warmup dummy leaves _context at one target bucket width,
        # so warming here covers every width a real step can present.
        if self._context is None:
            return
        for reqs in range(1, min(self.max_num_reqs, self._context.shape[0]) + 1):
            index = convert(
                torch.arange(reqs, dtype=torch.long), self.spyre_device, torch.int64
            )
            self.model.model.normalize_context(
                torch.index_select(self._context, 0, index)
            )

    def warm_draft_batches(self) -> None:
        """Warm the draft body, candidate head and anchor gather at every width."""
        block = self.num_query_per_req
        widths = {padded_draft_reqs(n, self.max_num_reqs) for n in range(1, self.max_num_reqs + 1)}
        for reqs in sorted(widths, reverse=True):
            num_tokens = reqs * block
            attn_layer.publish_null_slots(num_tokens)
            self.input_buffers.input_ids[:num_tokens] = self.parallel_drafting_token_id
            self.input_buffers.input_ids[:num_tokens:block] = 0
            self.input_buffers.positions[:num_tokens] = torch.arange(block).repeat(reqs)
            self._generate_draft(reqs, num_tokens, None, {}, None)
        self.draft_tokens.zero_()
        self._warm_anchor_gather()


def make_spyre_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    spyre_device: torch.device,
    upstream_speculator,
) -> _SpyreSpeculatorMixin:
    """Wrap the speculator upstream selected with its Spyre counterpart.

    ``init_speculator`` is imported by value in the runner, so it cannot be
    patched; the architecture check it performed is reused through the instance's
    type instead of re-reading the draft config.
    """
    cls = (
        SpyreLiLiCorrSpeculator
        if isinstance(upstream_speculator, LiLiCorrSpeculator)
        else SpyreDFlashSpeculator
    )
    return cls(vllm_config, device, spyre_device)
