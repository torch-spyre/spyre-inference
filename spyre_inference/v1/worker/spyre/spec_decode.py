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

"""MRV2: the DFlash/XPress speculator (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py)
and the Gemma-4 MTP one (vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py).

Upstream's propose() and its input-prep kernels run as-is; the overrides keep the draft
on Spyre and on shapes warmup compiled.
"""

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, get_shared_kv_cache_layers
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import prepare_prefill_inputs

from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.worker.compile_granularity import compile_for_spyre
from spyre_inference.v1.worker.kv_cache_setup import with_kv_sharing_layer_specs
from spyre_inference.v1.worker.model_load import move_model_to_spyre
from spyre_inference.v1.worker.spyre.attn_groups import split_by_window
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _SpyreModelWrapper,
    model_dtype,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer, logits_row_buckets

if TYPE_CHECKING:
    from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM


def init_speculator(
    vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
) -> BaseSpeculator:
    """The Spyre speculator for this config, in place of upstream's init_speculator."""
    spec = vllm_config.speculative_config
    assert spec is not None
    # The platform refuses every other method on V2 (_reject_unsupported_v2_features).
    if spec.use_gemma4_mtp():
        return SpyreGemma4Speculator(vllm_config, device, spyre_device)
    return SpyreDFlashSpeculator(vllm_config, device, spyre_device)


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


def _all_gather_host(shard: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    parts = [torch.empty_like(shard, dtype=torch.float32) for _ in range(world_size)]
    # gloo's half support varies by build; the weights are small, so gather in fp32.
    dist.all_gather(parts, shard.float().contiguous(), group=group)  # ty: ignore[possibly-missing-attribute]
    return torch.cat(parts, dim=dim).to(shard.dtype)


def _replicate_projections(predictor: Any) -> None:
    """Hold the full pre/post projections on every rank, so drafting runs no eager collective.

    Both run between the compiled blocks, where each eager Spyre collective sets up its
    comms bundle per call and costs more than the whole draft forward.
    """
    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return
    pre, post = predictor.pre_projection, predictor.post_projection
    assert pre.bias is None and post.bias is None
    # Spyre's linear method stores each weight transposed, as [in, out].
    transposed = isinstance(getattr(pre, "quant_method", None), SpyreTransposedWeightMethod)
    assert transposed == isinstance(
        getattr(post, "quant_method", None), SpyreTransposedWeightMethod
    )
    assert not getattr(pre, "spyre_row_padding", 0), "padding would interleave the shards"
    out_dim, in_dim = (1, 0) if transposed else (0, 1)
    pre.weight = nn.Parameter(
        _all_gather_host(pre.weight.data, out_dim, tp_group.cpu_group), requires_grad=False
    )
    pre.gather_output = False
    post.weight = nn.Parameter(
        _all_gather_host(post.weight.data, in_dim, tp_group.cpu_group), requires_grad=False
    )
    # The full input in, the full output out: no split, no reduce.
    post.input_is_parallel = True
    post.reduce_results = False


class SpyreDraftModelWrapper(_SpyreModelWrapper):
    """The Gemma-4 drafter's boundary, which also takes the target's hidden state.

    Every input is padded here onto a compiled body bucket, and outputs come back to the
    host before the trim, avoiding a dim-0 slice on device. With ``on_device`` they stay
    on the card, padded, and a float input already there is the previous pass's output.
    """

    def padded_rows(self, num_tokens: int) -> int:
        bucketer = self._shape_bucketer
        padded = bucketer.find_bucket(num_tokens) if bucketer is not None else None
        return num_tokens if padded is None else padded

    def __call__(self, *, input_ids, positions, hidden_states, inputs_embeds=None, on_device=False):
        num_tokens = positions.shape[-1]
        padded = self.padded_rows(num_tokens)
        pad = padded - num_tokens

        def to_spyre(t: torch.Tensor | None, rows_last: bool) -> torch.Tensor | None:
            if t is None:
                return None
            if t.device.type != "cpu":
                assert t.shape[0] == padded, "an on-card input is already padded"
                return t
            if pad:
                t = F.pad(t, (0, pad) if rows_last else (0, 0, 0, pad))
            if t.is_floating_point():
                return convert(t, device=self._spyre_device, dtype=self._model_dtype)
            return convert(t, device=self._spyre_device, dtype=torch.int64)

        result = self._model(
            input_ids=to_spyre(input_ids, rows_last=True),
            positions=to_spyre(positions, rows_last=True),
            hidden_states=to_spyre(hidden_states, rows_last=False),
            inputs_embeds=to_spyre(inputs_embeds, rows_last=False),
        )
        if on_device:
            return result
        return tuple(convert(t, device="cpu")[:num_tokens] for t in result)


class SpyreGemma4Speculator(Gemma4Speculator):
    """Upstream's Gemma-4 MTP speculator, with the drafter on Spyre.

    Upstream loads the drafter on the host, sharing the target's embedding (already on
    Spyre); load_model then moves it to Spyre and compiles it per block like the target.
    Its layers read the target's KV pages and are causal, so they need no override.
    """

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, spyre_device: torch.device
    ) -> None:
        super().__init__(vllm_config, device)
        self.spyre_device = spyre_device

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # Its centroid top-k and vocabulary gather have no Spyre lowering; the full
        # draft-dim lm_head is cheap at a few rows, and its argmax is the unmasked one.
        model.masked_embedding = None
        _replicate_projections(model.model)
        move_model_to_spyre(model, self.spyre_device)
        compiled, _ = compile_for_spyre(model, self.vllm_config)
        return compiled

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        bucketer = SpyreShapeBucketer.for_config(self.vllm_config)
        self.model = SpyreDraftModelWrapper(
            cast(nn.Module, self.model),
            self.spyre_device,
            logits_row_buckets=(
                []
                if bucketer is None
                else logits_row_buckets(bucketer.bucket_sizes, self.max_num_reqs)
            ),
            shape_bucketer=bucketer,
            model_dtype=model_dtype(self.draft_model_config),
        )

    def set_attn(self, model_state, kv_cache_config, block_tables, *args, **kwargs) -> None:
        # Every draft layer shares a target layer's KV, which upstream's init_attn_backend
        # looks up in the per-layer specs; only attention sees the copy that maps them.
        shared = with_kv_sharing_layer_specs(
            kv_cache_config, get_shared_kv_cache_layers(self.vllm_config)
        )
        super().set_attn(model_state, shared, block_tables, *args, **kwargs)
        self.kv_cache_config = kv_cache_config
        split_by_window(
            self.attn_groups, self.attn_vllm_config, self.device, block_tables.block_sizes
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        """Every draft pass on one row per request.

        The assistant's layers only read the target's KV, at positions that stay fixed
        across steps, so a request's last accepted row drafts the same without the rest
        of its verify rows (or, at prefill, its prompt). Each pass then runs on the
        decode buckets warmup compiles, and one attention build serves every step.
        """
        num_reqs = input_batch.num_reqs
        steps = self.num_speculative_steps
        if num_speculative_tokens is not None:
            steps = num_speculative_tokens
        if steps == 0:
            self.draft_tokens[:num_reqs].fill_(-1)
            return self.draft_tokens[:num_reqs]
        max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())
        self.draft_max_seq_len = min(max_seq_len + steps, self.max_model_len)
        self._copy_request_inputs(
            num_reqs, input_batch.idx_mapping, temperature, seeds, dummy_run=dummy_run
        )
        buffers = self.input_buffers
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )
        rows = self.last_token_indices[:num_reqs]
        input_ids = buffers.input_ids[rows]
        positions = buffers.positions[rows]
        # The target's hidden states stay on Spyre for its sampled-row gather.
        hidden_states = convert(last_hidden_states.as_subclass(torch.Tensor), device="cpu")[rows]

        buffers.positions[:num_reqs] = positions
        buffers.query_start_loc[: num_reqs + 1] = torch.arange(num_reqs + 1)
        buffers.query_start_loc[num_reqs + 1 :] = num_reqs
        # Rejected drafts' KV is still in the cache, so each context ends before it.
        buffers.seq_lens[:num_reqs] = input_batch.seq_lens[:num_reqs] - num_rejected
        buffers.seq_lens[num_reqs:] = 0
        idx_mapping = self.idx_mapping[:num_reqs]
        draft_attn_metadata = draft_slot_mappings = None
        if not (dummy_run and skip_attn_for_dummy_run):
            slots = self.block_tables.compute_slot_mappings(
                idx_mapping, buffers.query_start_loc[: num_reqs + 1], positions, num_reqs
            )
            draft_slot_mappings = build_slot_mappings_by_layer(slots, self.kv_cache_config)
            draft_attn_metadata = self._build_uniform_attn_metadata(
                batch_desc=BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE, num_tokens=num_reqs, num_reqs=num_reqs
                ),
                num_reqs=num_reqs,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=0,
            )

        # Greedy drafts keep the draft hidden state on the card between passes, when the
        # padded batch is a width the lm_head compiled at; only the logits come back.
        model = cast(SpyreDraftModelWrapper, self.model)
        on_card = (
            self.draft_logits is None and model.padded_rows(num_reqs) in model._logits_row_buckets
        )
        sample_src_positions = positions + 1
        for step in range(steps):
            self.current_draft_step.fill_(step)
            with set_forward_context(
                draft_attn_metadata,
                self.vllm_config,
                num_tokens=num_reqs,
                slot_mapping=draft_slot_mappings,
            ):
                draft_hidden_states, hidden_states = model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    on_device=on_card,
                )
            if on_card:
                # Every padded row is projected; the padding's tokens are dropped here.
                logits = model.compute_logits(draft_hidden_states)[:num_reqs]
                self._maybe_predict_acceptance(logits, idx_mapping, self.current_draft_step)
                tokens = logits.argmax(dim=-1)
            else:
                tokens = self.sample_draft(
                    draft_hidden_states,
                    sample_src_positions,
                    idx_mapping,
                    self.temperature,
                    self.seeds,
                    self.current_draft_step,
                    self.draft_logits,
                )
            self.draft_tokens[:num_reqs, step] = tokens
            input_ids = tokens.to(input_ids.dtype)
            sample_src_positions = sample_src_positions + 1
        return self.draft_tokens[:num_reqs]
