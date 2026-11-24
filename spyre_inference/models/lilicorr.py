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

"""Spyre LiLiCorr draft: upstream's lattice head with device-safe splits.

The candidate features and the 5-wide feature MLP run on the host; the
hidden-width lattice math runs on Spyre. Every fused buffer is materialized on
the host by ``load_weights`` (before the OOT linear methods transpose their
weights in place) and only converted to the device afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch
from torch.nn import functional as F
from vllm.distributed import get_tp_group
from vllm.model_executor.models import lilicorr as upstream_lilicorr
from vllm.model_executor.models.lilicorr import (
    LiLiCorr,
    LiLiCorrForCausalLM,
    LiLiCorrHead,
    LiLiCorrLatticeAttention,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models.qwen3_dflash import (
    SpyreDFlashQwen3ForCausalLM,
    SpyreDFlashQwen3Model,
)
from spyre_inference.v1.spec_decode.checkpoint import validate_dflash_weights
from spyre_inference.v1.worker import compile_guard


def _topk_log_probs(
    logits: torch.Tensor,
    *,
    vocab_size: int,
    top_k: int,
    scale: float,
    soft_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host top-k + full-vocabulary log-partition, as get_top_k_tokens does.

    Used for the TP>1 path, where the logits are all-gathered to the host first.
    """
    logits = logits.float()
    if logits.shape[-1] > vocab_size:
        # .float() is a no-op for an fp32 input, so clone before masking.
        logits = logits.clone()
        logits[..., vocab_size:] = -float("inf")
    if soft_cap is not None:
        logits = torch.tanh(logits / soft_cap) * soft_cap
    if scale != 1.0:
        logits = logits * scale
    log_partition = torch.logsumexp(logits, dim=-1, keepdim=True)
    values, ids = torch.topk(logits, top_k, dim=-1)
    return ids.to(torch.int64), values - log_partition


def _gather_full_weight(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather a TP weight shard on the host, so every rank runs the full feature MLP."""
    tensor = tensor.float()
    tp = get_tp_group()
    if tp.world_size == 1:
        return tensor
    shards = [torch.empty_like(tensor) for _ in range(tp.world_size)]
    torch.distributed.all_gather(shards, tensor, group=tp.cpu_group)
    return torch.cat(shards, dim=dim)


class SpyreLiLiCorrLatticeAttention(LiLiCorrLatticeAttention):
    """Lattice attention with contiguous head layouts.

    Upstream's ``view().transpose(1, 2)`` leaves the head axes non-contiguous,
    which Spyre's SDPA path does not handle (see ``models/blip2.py``); the added
    ``contiguous()`` calls are the fix. ``in_proj_weight`` stays the single
    parameter and ``F.linear`` transposes it, so no second copy is kept.
    """

    def forward(self, hidden_states: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2).contiguous()
        k = k.view(shape).transpose(1, 2).contiguous()
        v = v.view(shape).transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        return self.out_proj(out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size))


class SpyreLiLiCorrHead(LiLiCorrHead):
    """LiLiCorrHead with host candidate features and a Spyre lattice body."""

    _spyre_prepared = False
    _attn_bias_cpu: torch.Tensor | None = None

    def materialize_inference_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        # Runs at the end of load_weights, before the OOT linear methods replace
        # .weight with its transpose. Only host features and the attention bias are
        # stashed; the linear weights stay the single parameters and are used in
        # their OOT orientation (transposed in place, applied via their modules).
        if self._spyre_prepared:
            raise NotImplementedError(
                "LiLiCorr draft weight reload is not supported on Spyre: the head's "
                "inference buffers must be materialized before the OOT linear "
                "transpose, and a reload would rebuild them from transposed weights."
            )
        topk = self.candidate_topk
        self._attn_bias_cpu = self._build_cpu_attention_bias()
        if topk > 1:
            rank_frac = torch.arange(topk, dtype=torch.float32).view(1, 1, topk) / float(topk - 1)
        else:
            rank_frac = torch.zeros(1, 1, topk, dtype=torch.float32)
        is_top1 = torch.zeros(1, 1, topk, dtype=torch.float32)
        is_top1[..., 0] = 1.0
        self._rank_frac_col = rank_frac.contiguous()
        self._is_top1_col = is_top1.contiguous()
        # The 5-wide feature normalization and MLP consume a last dim of 5, which is
        # not a whole Spyre stick. Keep their weights on the host and run that path
        # in fp32 there; the hidden-width lattice below stays on the device.
        self._feature_norm_w = self.feature_norm.weight.detach().cpu().float()
        self._feature_norm_b = self.feature_norm.bias.detach().cpu().float()
        # Under TP the feature MLP is column/row sharded; gather the full weights so
        # every rank computes the same host-side term. Down-proj bias is not sharded.
        self._feature_up_w = _gather_full_weight(
            self.feature_mlp.up_proj.weight.detach().cpu(), dim=0
        )
        self._feature_up_b = _gather_full_weight(
            self.feature_mlp.up_proj.bias.detach().cpu(), dim=0
        )
        self._feature_down_w = _gather_full_weight(
            self.feature_mlp.down_proj.weight.detach().cpu(), dim=1
        )
        self._feature_down_b = self.feature_mlp.down_proj.bias.detach().cpu().float()
        if (
            self._feature_up_w.shape[0] != self.hidden_size
            or self._feature_down_w.shape[1] != self.hidden_size
        ):
            raise RuntimeError("Gathered LiLiCorr feature-MLP weights have the wrong width")
        self._attn_bias = None

    def _build_cpu_attention_bias(self) -> torch.Tensor:
        # Built from host copies: on a reload the learned biases are already on
        # the device, and indexing a device tensor with host indices is not a
        # supported Spyre pattern.
        topk = self.candidate_topk
        relative = self.relative_slot_bias.detach().cpu()
        same = self.same_slot_bias.detach().cpu()
        slot_ids = torch.arange(self.num_candidate_slots, dtype=torch.long).repeat_interleave(topk)
        rel = slot_ids.view(-1, 1) - slot_ids.view(1, -1)
        bias = relative[:, rel + self.block_size - 1]
        same_slot = slot_ids.view(-1, 1) == slot_ids.view(1, -1)
        bias = bias + same_slot.unsqueeze(0).to(bias.dtype) * same.view(-1, 1, 1)
        return bias.contiguous()

    def prepare_for_spyre(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move the stashed attention bias to the device and compile the lattice body."""
        if self._attn_bias_cpu is None:
            raise RuntimeError(
                "LiLiCorr head must be materialized exactly once before "
                "prepare_for_spyre; draft reload is unsupported."
            )
        self._attn_bias = convert(self._attn_bias_cpu, device, dtype)
        # Prepare-only host copy; reload is unsupported, so release it.
        self._attn_bias_cpu = None
        self._score_lattice_compiled = torch.compile(
            self._score_lattice, fullgraph=True, dynamic=False
        )
        compile_guard.watch(self._score_lattice, "LiLiCorr lattice head")
        self._spyre_device = device
        self._spyre_prepared = True

    def _feature_term(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, slots, topk, 5] fp32 on the host.
        x = F.layer_norm(
            features,
            (self.num_candidate_features,),
            self._feature_norm_w,
            self._feature_norm_b,
            self.feature_norm.eps,
        )
        x = F.silu(F.linear(x, self._feature_up_w, self._feature_up_b))
        return F.linear(x, self._feature_down_w, self._feature_down_b)

    def _score_lattice(
        self,
        token_embeddings: torch.Tensor,
        pass_hidden: torch.Tensor,
        feature_term: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_slots, topk, _ = token_embeddings.shape
        token_states = self.token_proj(token_embeddings)
        pass_states = self.pass_hidden_proj(pass_hidden).unsqueeze(-2)
        anchor_state = self.context_proj(anchor_hidden)
        anchor_state = anchor_state * anchor_valid[:, None].to(anchor_state.dtype)
        hidden_states = token_states + pass_states + feature_term
        hidden_states = hidden_states + self.slot_embedding[:, 0, :n_slots]
        hidden_states = hidden_states + self.rank_embedding[:, 0]
        hidden_states = hidden_states.reshape(bsz, n_slots * topk, self.hidden_size)
        # Shorter drafts use the learned prefix, as upstream.
        lattice = n_slots * topk
        attention_bias = self._attn_bias[None, :, :lattice, :lattice]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        hidden_states = self.output_norm(hidden_states).reshape(
            bsz, n_slots, topk, self.hidden_size
        )
        anchor_state = self.anchor_norm(anchor_state)
        anchor_row = anchor_state[:, None, None, :]
        # Same input as upstream's factor head, evaluated by the module so its
        # weight stays the single parameter in its OOT orientation.
        factor_input = torch.cat(
            (hidden_states, anchor_row.expand_as(hidden_states), hidden_states * anchor_row),
            dim=-1,
        )
        factor_hidden = F.silu(self.factor_input_proj(factor_input))
        out_vec = F.normalize(self.out_head(factor_hidden), dim=-1, eps=self.vector_eps)
        in_vec = F.normalize(self.in_head(factor_hidden), dim=-1, eps=self.vector_eps)
        anchor_out = F.normalize(self.anchor_out_head(anchor_state), dim=-1, eps=self.vector_eps)
        start_scores = (anchor_out[:, None, :] * in_vec[:, 0, :, :]).sum(dim=-1)
        pair_scores = torch.matmul(out_vec[:, :-1], in_vec[:, 1:].transpose(-1, -2))
        return start_scores, pair_scores

    @torch.no_grad()
    def score(
        self,
        *,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return the lattice logits upstream's ``forward`` produces, on the host.

        ``[batch, steps, topk, topk]``: row 0 is the anchor-to-first-slot score and
        one row follows per draft token. The speculator walks these on the host.
        """
        if not self._spyre_prepared:
            raise RuntimeError("Call prepare_for_spyre() after loading the LiLiCorr head.")
        bsz, n_slots, topk = candidate_log_probs.shape
        if not 1 <= n_slots <= self.num_candidate_slots:
            raise ValueError(
                f"LiLiCorr supports 1..{self.num_candidate_slots} candidate slots, got {n_slots}."
            )
        if topk != self.candidate_topk:
            raise ValueError(f"LiLiCorr expects candidate_topk={self.candidate_topk}, got {topk}.")
        log_probs = candidate_log_probs.float()
        features = torch.stack(
            [
                log_probs,
                log_probs.exp(),
                log_probs - log_probs.max(dim=-1, keepdim=True).values,
                self._rank_frac_col.expand_as(log_probs),
                self._is_top1_col.expand_as(log_probs),
            ],
            dim=-1,
        )
        feature_term = convert(
            self._feature_term(features).to(self.slot_embedding.dtype),
            self._spyre_device,
        )
        start, pairs = self._score_lattice_compiled(
            token_embeddings,
            pass_hidden,
            feature_term,
            anchor_hidden,
            anchor_valid,
        )
        # Upstream scales and widens the scores in ``forward``. Spyre has no fp32
        # device path, so the scores return to the host first and the candidate walk
        # runs there; scaling on the host also matches upstream's fp32 result.
        start = self.logit_scale * start.to("cpu", dtype=torch.float32)
        pairs = self.logit_scale * pairs.to("cpu", dtype=torch.float32)
        first = start[:, None, None, :].expand(-1, 1, topk, -1)
        return torch.cat((first, pairs), dim=1)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        pass_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        # score() already applies logit_scale; upstream's forward would apply it twice.
        return self.score(
            token_embeddings=token_embeddings,
            candidate_log_probs=candidate_log_probs,
            pass_hidden=pass_hidden,
            anchor_hidden=anchor_hidden,
            anchor_valid=anchor_valid,
        )


class SpyreLiLiCorrModel(SpyreDFlashQwen3Model, LiLiCorr):
    """LiLiCorr backbone and lattice head, built with Spyre classes.

    MRO runs upstream ``LiLiCorr.__init__`` (which chooses the decoder layer and
    builds the head) and then ``DFlashQwen3Model.__init__``; the scoped swap is
    how the two Spyre classes reach upstream's module-global constructors.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _spyre_lilicorr_classes():
            super().__init__(*args, **kwargs)

    def normalize_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The draft speculator needs the same normalization for its context anchor.
        return self._context_norm(hidden_states)


class SpyreLiLiCorrForCausalLM(SpyreDFlashQwen3ForCausalLM, LiLiCorrForCausalLM):
    """Upstream's LiLiCorr wrapper, loading through upstream's coverage checks."""

    model_cls = SpyreLiLiCorrModel
    model: SpyreLiLiCorrModel

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Bypass SpyreDFlashQwen3ForCausalLM.load_weights (it rejects lilicorr.*
        # keys), but keep its backbone validation on everything else. Upstream's
        # LiLiCorr loader checks complete lilicorr coverage and materializes the
        # head while .weight still has its checkpoint orientation.
        weights = dict(weights)
        backbone = {
            name.removeprefix("model."): value
            for name, value in weights.items()
            if not name.removeprefix("model.").startswith(
                ("lilicorr.", "attention_conv.", "mlp_conv.")
            )
        }
        validate_dflash_weights(self.config, backbone)
        LiLiCorrForCausalLM.load_weights(self, weights.items())

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TP=1 goes through upstream's local top-k path, which already runs on the
        # host (the Spyre lm-head D2Hs its logits). Under TP each rank holds one
        # vocab shard, and upstream's device logsumexp/.float() path is not a Spyre
        # op, so all-gather the logits and do the full-vocabulary math on the host.
        if self.lm_head.tp_size == 1:
            return super().compute_candidates(hidden_states)
        processor = self.candidate_logits_processor
        logits = processor._gather_logits(processor._apply_head(self.lm_head, hidden_states))
        # The OOT processor already D2Hs; .to() also covers a base processor that
        # returns the gathered device tensor.
        logits = logits.to(device="cpu", dtype=torch.float32)
        return _topk_log_probs(
            logits,
            vocab_size=self.lm_head.org_vocab_size,
            top_k=self.model.lilicorr.candidate_topk,
            scale=processor.scale,
            soft_cap=processor.soft_cap,
        )

    def prepare_for_spyre(self) -> None:
        # Runs after move_model_to_spyre: the DFlash backbone prepares its context
        # path first, then the lattice head converts its stashed host buffers.
        super().prepare_for_spyre()
        head = self.model.lilicorr
        head.prepare_for_spyre(head.slot_embedding.device, head.slot_embedding.dtype)


@contextmanager
def _spyre_lilicorr_classes():
    # Upstream constructs these by module-global name, with no class hook.
    saved = (
        upstream_lilicorr.LiLiCorrHead,
        upstream_lilicorr.LiLiCorrLatticeAttention,
    )
    head_cls = SpyreLiLiCorrHead
    attention_cls = SpyreLiLiCorrLatticeAttention
    upstream_lilicorr.LiLiCorrHead = head_cls  # ty: ignore[invalid-assignment]
    upstream_lilicorr.LiLiCorrLatticeAttention = attention_cls  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        (
            upstream_lilicorr.LiLiCorrHead,
            upstream_lilicorr.LiLiCorrLatticeAttention,
        ) = saved


__all__ = [
    "SpyreLiLiCorrForCausalLM",
    "SpyreLiLiCorrHead",
    "SpyreLiLiCorrLatticeAttention",
    "SpyreLiLiCorrModel",
]
