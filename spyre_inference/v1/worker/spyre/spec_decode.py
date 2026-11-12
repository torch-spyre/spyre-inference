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

"""MRV2: DFlash/XPress speculator (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py).

Upstream's propose() and its input-prep kernel run as-is; the overrides keep the draft
on Spyre and on shapes warmup compiled.
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

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.worker.model_load import move_model_to_spyre
from spyre_inference.v1.worker.spyre.attn_groups import split_by_window

if TYPE_CHECKING:
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


class SpyreDFlashSpeculator(DFlashSpeculator):
    model: "SpyreDFlashQwen3ForCausalLM"

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
    ) -> None:
        super().__init__(vllm_config, device)
        self.spyre_device = spyre_device
        self._context: torch.Tensor | None = None

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]):
        model = cast(
            "SpyreDFlashQwen3ForCausalLM",
            super().load_draft_model(target_model, target_attn_layer_names),
        )
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
        assert aux_hidden_states, "Spyre DFlash requires the target's auxiliary hidden states"
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

    def _generate_draft(self, num_reqs: int, num_tokens_padded: int, *args, **kwargs) -> None:
        # Upstream's eager metadata publishes slots for the real rows only; the padded
        # rows' KV write goes to the null page, so the block graph keeps its width.
        num_tokens = num_reqs * self.num_query_per_req
        if num_tokens < num_tokens_padded:
            for group in (g for groups in self.attn_groups for g in groups):
                slots = self.block_tables.slot_mappings[group.kv_cache_group_id]
                builder = cast(SpyreAttentionMetadataBuilder, group.get_metadata_builder())
                builder.publish_slot_mapping(
                    F.pad(slots[:num_tokens], (0, num_tokens_padded - num_tokens), value=0)
                )
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
