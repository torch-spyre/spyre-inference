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

from contextlib import contextmanager
from typing import cast

import torch
from torch.nn import functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.qwen3_xpress import Qwen3XPressForCausalLM, Qwen3XPressModel

from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl, SpyrePagedKVCache
from spyre_inference.v1.spec_decode.checkpoint import validate_dflash_weights
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead, select_feedback
from spyre_inference.v1.worker import compile_guard


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
    xpress_num_passes = 0

    @property
    def xpress_head(self) -> XPressRefinerHead | None:
        return getattr(self.model, "xpress_head", None)

    @property
    def xpress_topc(self) -> int:
        head = self.xpress_head
        return 0 if head is None else head.topc

    def load_weights(self, weights):
        weights = dict(weights)
        validate_dflash_weights(self.config, weights)
        head = self.xpress_head
        names = {
            name.removeprefix("xpress_head.") for name in weights if name.startswith("xpress_head.")
        }
        if head is None and names:
            raise ValueError("XPress weights require an XPress checkpoint architecture")
        if head is not None:
            # Upstream loads partial heads and permits broadcasting the raw mixer.
            expected = head.HYBRID_KEY_MAP.keys()
            if names != expected:
                raise ValueError(
                    f"Invalid XPress head weights: missing={sorted(expected - names)}, "
                    f"unexpected={sorted(names - expected)}"
                )
            raw = weights["xpress_head.mix.L"]
            if raw.shape != head.mix_L.shape:
                raise ValueError(
                    f"Invalid xpress_head.mix.L shape: expected {tuple(head.mix_L.shape)}, "
                    f"got {tuple(raw.shape)}"
                )
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
            # DL16 represents shortlist positions and the sentinel exactly through 1024.
            order = torch.arange(self.xpress_topc, dtype=torch.float16)
            self._feedback_order = F.pad(order, (0, -self.xpress_topc % 64), value=-1).to(
                head.w1.weight.device
            )
            self._prepare_feedback = torch.compile(
                head.prepare_feedback, fullgraph=True, dynamic=False
            )
            self._refine_feedback = torch.compile(
                head.refine_feedback, fullgraph=True, dynamic=False
            )
            self._select_feedback = torch.compile(select_feedback, fullgraph=True, dynamic=False)
            for function in (head.prepare_feedback, head.refine_feedback, select_feedback):
                compile_guard.watch(function, f"XPress {function.__name__}")

    def combine_hidden_states(self, hidden_states):
        return self._combine(hidden_states)

    def compute_device_logits(self, hidden_states):
        # TP=1 and the shared full vocabulary are checked before model loading.
        method = cast(SpyreUnquantizedLMHeadMethod, self.lm_head.quant_method)
        return method.apply(self.lm_head, hidden_states)

    def _copy_logits_to_cpu(self, logits):
        # Cast during D2H: device widening produces a nonstandard lane arrangement.
        return logits.to(device="cpu", dtype=torch.float32)

    def _argmax_on_cpu(self, host_logits):
        return host_logits.argmax(-1)

    def propose_block(self, hidden_states, anchor_ids):
        batch = anchor_ids.numel()
        hidden_blocks = hidden_states.view(batch, -1, hidden_states.shape[-1])
        base = self.compute_device_logits(hidden_states).view(batch, hidden_blocks.shape[1], -1)
        host_base = self._copy_logits_to_cpu(base)[:, 1:, : self.config.vocab_size]
        draft = self._argmax_on_cpu(host_base)
        if self.xpress_head is None or not self.xpress_num_passes:
            return draft.tolist()
        cache = self._hidden_cache(hidden_blocks)
        base_c, candidates = host_base.topk(self.xpress_topc, dim=-1)
        # BMM padding cannot split flattened candidate embedding rows.
        padding = -self.xpress_topc % 64
        candidate_ids = F.pad(candidates, (0, padding))
        base_c = F.pad(base_c, (0, padding), value=-torch.inf)
        # Flat indices avoid gather page splits across request axes.
        candidate_ids = convert(candidate_ids.flatten(), device=hidden_states.device)
        base = convert(base_c, device=hidden_states.device, dtype=hidden_states.dtype)
        # The anchor is its own predecessor, as in upstream's XPressSpeculator.
        previous = torch.cat((anchor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1)
        embeddings, feedback, readout = self._prepare_feedback(
            candidate_ids, convert(previous, device=hidden_states.device)
        )
        for _ in range(self.xpress_num_passes):
            logits = self._refine_feedback(base, feedback, cache, readout)
            indices, feedback = self._select_feedback(
                logits, embeddings, self._feedback_order, feedback
            )
        return candidates.gather(-1, indices.cpu().long().unsqueeze(-1)).squeeze(-1).tolist()


class SpyreQwen3XPressModel(SpyreDFlashQwen3Model, Qwen3XPressModel):
    pass


class SpyreQwen3XPressForCausalLM(SpyreDFlashQwen3ForCausalLM, Qwen3XPressForCausalLM):
    """Upstream's XPress model and loader, built with the Spyre head and context path."""

    def __init__(self, *, vllm_config, prefix=""):
        with _spyre_xpress_classes():
            Qwen3XPressForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        # Upstream wraps refine_bias in torch.compile; Spyre compiles its own helpers.
        self.model.xpress_head.__dict__.pop("refine_bias", None)
        self.xpress_num_passes = self.config.xpress_num_passes


@contextmanager
def _spyre_xpress_classes():
    # Upstream constructs both by module-global name, with no class hook.
    from vllm.model_executor.models import qwen3_xpress

    saved = qwen3_xpress.Qwen3XPressModel, qwen3_xpress.XPressRefinerHead
    qwen3_xpress.Qwen3XPressModel = SpyreQwen3XPressModel  # ty: ignore[invalid-assignment]
    qwen3_xpress.XPressRefinerHead = XPressRefinerHead  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        qwen3_xpress.Qwen3XPressModel, qwen3_xpress.XPressRefinerHead = saved
