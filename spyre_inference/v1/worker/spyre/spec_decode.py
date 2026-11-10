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

Upstream's runner drives it unchanged; only propose() differs. It stores the accepted
target context and drafts one block per request with the Spyre draft model's
propose_block, all bookkeeping on the host (the runner's device is the CPU), so no
Triton kernel is launched.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.worker.model_load import move_model_to_spyre
from spyre_inference.v1.worker.spyre.attn_groups import split_by_window


def plan_drafts(
    query_start_loc: list[int],
    num_sampled: list[int],
    num_rejected: list[int],
    positions: torch.Tensor,
    block: int,
    max_model_len: int,
) -> tuple[list[range], list[int | None]]:
    """Each request's committed target rows, and the first position of its draft block
    (None when it drafts nothing: a prefill chunk, or a block past max_model_len)."""
    committed: list[range] = []
    draft_starts: list[int | None] = []
    for i, (start, end) in enumerate(zip(query_start_loc, query_start_loc[1:])):
        # The rejected rows and the bonus token's uncomputed state are not context.
        valid = end - start - num_rejected[i]
        if not 0 < valid <= end - start:
            raise ValueError(f"Request {i}: {num_rejected[i]} rejected of {end - start} rows")
        committed.append(range(start, start + valid))
        context_end = int(positions[start + valid - 1]) + 1
        drafts = num_sampled[i] > 0 and context_end + block <= max_model_len
        draft_starts.append(context_end if drafts else None)
    return committed, draft_starts


class SpyreDFlashSpeculator(DFlashSpeculator):
    """DFlash and XPress drafts: the Spyre draft model refines when it has an XPress head."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
    ) -> None:
        super().__init__(vllm_config, device)
        self.spyre_device = spyre_device
        self._warmed_context_rows: set[int] = set()
        self._draft_warmed = False

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        move_model_to_spyre(model, self.spyre_device)
        model.prepare_for_spyre()
        return model

    def set_attn(self, model_state, kv_cache_config, block_tables, *args, **kwargs) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables, *args, **kwargs)
        # As the target's groups (kv_cache.py); the split stays inside each KV cache
        # group, so the group ids and causality upstream derived above still hold.
        split_by_window(
            self.attn_groups, self.attn_vllm_config, self.device, block_tables.block_sizes
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: Any = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        assert aux_hidden_states, "Spyre DFlash requires the target's auxiliary hidden states"
        num_reqs = input_batch.num_reqs
        # On Spyre, at the target's padded (bucketed) row count.
        features = torch.cat(aux_hidden_states, dim=-1)
        if dummy_run:
            self._warm(features)
            return self.draft_tokens[:num_reqs]

        num_tokens = input_batch.num_tokens
        positions = input_batch.positions[:num_tokens]
        committed, draft_starts = plan_drafts(
            input_batch.query_start_loc_np[: num_reqs + 1].tolist(),
            num_sampled[:num_reqs].tolist(),
            num_rejected[:num_reqs].tolist(),
            positions,
            self.num_query_per_req,
            self.max_model_len,
        )
        # The target's own slots for this step; rejected and padded rows write the
        # reserved null page instead.
        target_slots = slot_mappings[next(iter(self.draft_attn_layer_names))]
        slots = torch.zeros(features.shape[0], dtype=torch.int64)
        for rows in committed:
            slots[rows.start : rows.stop] = target_slots[rows.start : rows.stop]
        self._store_context(features, F.pad(positions, (0, features.shape[0] - num_tokens)), slots)

        # Rows that draft nothing still verify whatever the scheduler places there.
        self.draft_tokens[:num_reqs] = 0
        active = [i for i, start in enumerate(draft_starts) if start is not None]
        if active and num_speculative_tokens != 0:
            anchors = last_sampled[input_batch.idx_mapping[active].long(), 0]
            starts = [draft_starts[i] for i in active]
            self.draft_tokens[active] = torch.tensor(self._draft(active, starts, anchors))
        return self.draft_tokens[:num_reqs]

    def _store_context(
        self, features: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor
    ) -> None:
        context = self.model.combine_hidden_states(features)
        self.model.precompute_and_store_context_kv(
            context, convert(positions, self.spyre_device), slots
        )

    def _draft(
        self, active: list[int], starts: list[int], anchors: torch.Tensor
    ) -> list[list[int]]:
        """One non-causal forward over every active request's block, padded to a power of
        two requests, then the draft model's proposal (and XPress refinement)."""
        block = self.num_query_per_req
        batch = 1 << (len(active) - 1).bit_length()
        num_tokens = batch * block
        actual_tokens = len(active) * block
        block_tables = [table[active] for table in self.block_tables.input_block_tables]

        positions = torch.zeros(num_tokens, dtype=torch.int64)
        # Padded rows write the reserved null page.
        slots = torch.zeros(len(block_tables), num_tokens, dtype=torch.int64)
        ids = torch.full((num_tokens,), self.parallel_drafting_token_id, dtype=torch.int64)
        ids[::block] = 0
        for index, (start, anchor) in enumerate(zip(starts, anchors.tolist(), strict=True)):
            rows = slice(index * block, (index + 1) * block)
            pos = torch.arange(start, start + block, dtype=torch.int64)
            positions[rows] = pos
            ids[index * block] = anchor
            for gid in self.draft_kv_cache_group_ids:
                size = self.block_tables.block_sizes[gid]
                pages = block_tables[gid][index, pos // size].long()
                slots[gid, rows] = pages * size + pos % size

        query_start_loc = torch.arange(0, actual_tokens + 1, block, dtype=torch.int32)
        seq_lens = torch.tensor(starts, dtype=torch.int32) + block
        per_layer = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=len(active),
            num_tokens=actual_tokens,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            max_query_len=block,
            seq_lens=seq_lens,
            max_seq_len=int(seq_lens.max()),
            block_tables=block_tables,
            slot_mappings=slots,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens,
            causal=False,
        )
        with set_forward_context(
            per_layer,
            self.vllm_config,
            num_tokens=num_tokens,
            slot_mapping=build_slot_mappings_by_layer(slots, self.kv_cache_config),
        ):
            hidden = self.model(
                input_ids=convert(ids, self.spyre_device),
                positions=convert(positions, self.spyre_device),
            )
        return self.model.propose_block(hidden, ids[::block])[: len(active)]

    def _warm(self, features: torch.Tensor) -> None:
        """Warmup: the context store at this dummy's row count against the null page, and
        the draft forward and proposal at every power-of-two batch, once each."""
        rows = features.shape[0]
        if rows not in self._warmed_context_rows:
            positions = torch.zeros(rows, dtype=torch.int64)
            self._store_context(features, positions, torch.zeros(rows, dtype=torch.int64))
            self._warmed_context_rows.add(rows)
        if self._draft_warmed:
            return
        block = self.num_query_per_req
        for power in range((self.max_num_reqs - 1).bit_length(), -1, -1):
            num_tokens = (1 << power) * block
            # The block graph's KV write needs a target, as for the target's dummies.
            attn_layer.publish_null_slots(num_tokens)
            ids = torch.full((num_tokens,), self.parallel_drafting_token_id, dtype=torch.int64)
            ids[::block] = 0
            with set_forward_context(
                None, self.vllm_config, num_tokens=num_tokens, slot_mapping={}
            ):
                hidden = self.model(
                    input_ids=convert(ids, self.spyre_device),
                    positions=convert(torch.arange(block).repeat(1 << power), self.spyre_device),
                )
            self.model.propose_block(hidden, ids[::block])
        self._draft_warmed = True
