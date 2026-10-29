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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

# vLLM 0.32 lacks XPress (vllm#54448); test_xpress_vendored.py flags migration to its subclasses.
# Retain Spyre context-KV, layout and refinement overrides when resolving #1417.

import time
from typing import cast

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl, SpyrePagedKVCache
from spyre_inference.v1.spec_decode.checkpoint import validate_dflash_weights
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


def _project_context(attention, context, positions):
    kv = context @ attention._context_kv_weight_t
    key, value = kv.split(attention.kv_size, dim=-1)
    key = attention.k_norm(key.view(-1, attention.num_kv_heads, attention.head_dim))
    key, _ = attention.rotary_emb(positions, key.flatten(1), None)
    return key.view(-1, attention.num_kv_heads, attention.head_dim), value.view(
        -1, attention.num_kv_heads, attention.head_dim
    )


class SpyreDFlashQwen3Model(DFlashQwen3Model):
    fc: ReplicatedLinear

    def _build_fused_kv_buffers(self) -> None:
        # Spyre projectors are built after final device/dtype placement.
        pass

    def prepare_for_spyre(self) -> None:
        for layer in self.layers:
            attention = cast(DFlashQwen3DecoderLayer, layer).self_attn
            # Spyre stores unquantized weights as [input, Q|K|V].
            weight_t = cast(torch.Tensor, attention.qkv_proj.weight)
            attention.register_buffer(
                "_context_kv_weight_t",
                weight_t.detach().cpu()[:, attention.q_size :].contiguous().to(weight_t.device),
                persistent=False,
            )
        self._context_norm = torch.compile(self.hidden_norm, fullgraph=True, dynamic=False)
        self._context_project = torch.compile(_project_context, fullgraph=True, dynamic=False)

    def precompute_and_store_context_kv(
        self,
        context_states,
        context_positions,
        context_slot_mapping=None,
    ) -> None:
        context = self._context_norm(context_states)
        indices = None
        for layer in self.layers:
            layer = cast(DFlashQwen3DecoderLayer, layer)
            attention = layer.self_attn
            key, value = self._context_project(attention, context, context_positions)
            if context_slot_mapping is not None:
                attn = attention.attn
                impl = cast(SpyreAttentionImpl, attn.impl)
                if indices is None:
                    indices = impl.kv_write_index(context_slot_mapping.clamp(min=0), key.device)
                impl.do_kv_cache_update(
                    None, key, value, cast(SpyrePagedKVCache, attn.kv_cache), indices
                )


class SpyreDFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = SpyreDFlashQwen3Model
    model: SpyreDFlashQwen3Model

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            attention = cast(DFlashQwen3DecoderLayer, layer).self_attn
            attention.attn.spyre_causal = attention.causal  # ty: ignore[invalid-assignment]
        self.xpress_head = None
        self.xpress_num_passes = 0
        self.xpress_topc = 0
        self.device_to_host_seconds = 0.0
        self.host_argmax_seconds = 0.0
        self.host_topk_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.candidate_gather_seconds = 0.0
        self.candidate_transfer_bytes = 0
        self.refiner_seconds = 0.0
        self.logits_seconds = 0.0
        self.selection_calls = 0
        self.logits_transfer_bytes = 0
        if hasattr(self.config, "xpress_rank"):
            self.xpress_topc = getattr(self.config, "xpress_topc", 128)
            self.xpress_head = XPressRefinerHead(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.xpress_block_size,
                self.config.xpress_rank,
                self.config.xpress_mlp_hidden,
                topc=self.xpress_topc,
            )
            self.xpress_num_passes = getattr(self.config, "xpress_num_passes", 6)

    def load_weights(self, weights):
        weights = dict(weights)
        validate_dflash_weights(self.config, weights)
        if self.xpress_head is not None:
            self.xpress_head.load_checkpoint_weights(weights)
            weights = {
                name: weight
                for name, weight in weights.items()
                if not name.startswith("xpress_head.")
            }
        elif any(name.startswith("xpress_head.") for name in weights):
            raise ValueError("XPress weights require an XPress head in the checkpoint config")
        super().load_weights(weights.items())

    def prepare_for_spyre(self) -> None:
        self.model.prepare_for_spyre()
        self._combine = torch.compile(self.model.fc, fullgraph=True, dynamic=False)
        for layer in self.model.layers:
            layer.compile(fullgraph=True, dynamic=False)
        head = self.xpress_head
        if head is not None:
            head.prepare_for_spyre()
            self._hidden_cache = torch.compile(
                head.project_hidden_cache, fullgraph=True, dynamic=False
            )
            self._refine = torch.compile(head.refine_full, fullgraph=True, dynamic=False)
            self._gather_readout = torch.compile(head.gather_readout, fullgraph=True, dynamic=False)
            self._refine_candidates = torch.compile(
                head.refine_candidates, fullgraph=True, dynamic=False
            )

    def combine_hidden_states(self, hidden_states):
        return self._combine(hidden_states)

    def compute_device_logits(self, hidden_states):
        # TP=1 and the shared full vocabulary are checked before model loading.
        method = cast(SpyreUnquantizedLMHeadMethod, self.lm_head.quant_method)
        return method.apply(self.lm_head, hidden_states)

    def _copy_logits_to_cpu(self, logits):
        started = time.perf_counter()
        # Cast during D2H: device widening produces a nonstandard lane arrangement.
        host_logits = logits.to(device="cpu", dtype=torch.float32)
        self.device_to_host_seconds += time.perf_counter() - started
        self.logits_transfer_bytes += host_logits.numel() * host_logits.element_size()
        return host_logits

    def _argmax_on_cpu(self, host_logits, candidates=None):
        started = time.perf_counter()
        result = host_logits.argmax(-1)
        if candidates is not None:
            result = candidates.gather(-1, result.unsqueeze(-1)).squeeze(-1)
        self.host_argmax_seconds += time.perf_counter() - started
        self.selection_calls += 1
        return result

    def propose_block(self, hidden_states, anchor_ids):
        batch = anchor_ids.numel()
        hidden_blocks = hidden_states.view(batch, -1, hidden_states.shape[-1])
        started = time.perf_counter()
        base = self.compute_device_logits(hidden_states).view(batch, hidden_blocks.shape[1], -1)
        self.logits_seconds += time.perf_counter() - started
        host_base = self._copy_logits_to_cpu(base)[:, 1:]
        draft = self._argmax_on_cpu(host_base)
        if self.xpress_head is not None and self.xpress_num_passes:
            started = time.perf_counter()
            cache = self._hidden_cache(hidden_blocks)
            self.refiner_seconds += time.perf_counter() - started
            candidates = readout = None
            base = base[:, 1:]
            if self.xpress_topc:
                started = time.perf_counter()
                base_c, candidates = host_base.topk(
                    min(self.xpress_topc, host_base.shape[-1]), dim=-1
                )
                self.host_topk_seconds += time.perf_counter() - started
                started = time.perf_counter()
                # Flat indices avoid gather page splits across request axes.
                candidate_ids = convert(candidates.flatten(), device=hidden_states.device)
                base = convert(base_c, device=hidden_states.device, dtype=hidden_states.dtype)
                self.host_to_device_seconds += time.perf_counter() - started
                self.candidate_transfer_bytes += candidates.numel() * 4 + base.numel() * 2
                started = time.perf_counter()
                readout = self._gather_readout(candidate_ids)
                self.candidate_gather_seconds += time.perf_counter() - started
            for _ in range(self.xpress_num_passes):
                # The verified anchor is also its own predecessor in merged XPress.
                previous = torch.cat(
                    (anchor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1
                )
                started = time.perf_counter()
                previous = convert(previous, device=hidden_states.device)
                self.host_to_device_seconds += time.perf_counter() - started
                started = time.perf_counter()
                logits = (
                    self._refine(base, previous, cache)
                    if candidates is None
                    else self._refine_candidates(base, previous, cache, readout)
                )
                self.refiner_seconds += time.perf_counter() - started
                draft = self._argmax_on_cpu(self._copy_logits_to_cpu(logits), candidates)
        return draft.tolist()
