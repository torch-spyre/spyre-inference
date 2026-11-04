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

# Upstream's Jacobi loop is Triton-only; Spyre compiles these step helpers instead.

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.models.qwen3_xpress import XPressRefinerHead as _UpstreamHead


def select_feedback(scores, embeddings, order, previous):
    batch, slots, count = scores.shape
    valid = order >= 0
    maxima = torch.where(valid, scores, -torch.inf).amax(dim=-1, keepdim=True)
    choices = torch.where(scores == maxima, order, count)
    # Spyre's notequal misses NaNs; negative priorities select the first NaN.
    choices = torch.where(scores == scores, choices, order - count)
    choices = torch.where(valid, choices, count)
    first = choices.amin(dim=-1, keepdim=True)
    first = torch.where(first < 0, first + count, first)
    selected = (order == first).to(scores.dtype)
    # TODO: replace BMM and the candidate cache once torch-spyre supports
    # embedding gathers with device-generated INT32 indices.
    winners = torch.bmm(
        selected.view(batch * slots, 1, count), embeddings.view(batch * slots, count, -1)
    ).view(batch, slots, -1)
    return first.squeeze(-1), torch.cat((previous[:, :2], winners[:, :-1]), dim=1)


class XPressRefinerHead(_UpstreamHead):
    """Upstream's head, construction and loading included, with Spyre step helpers."""

    _mix_kjc: torch.Tensor | None = None
    _hidden_proj_t: torch.Tensor | None = None
    _token_proj_t: torch.Tensor | None = None

    def build_fused_buffers(self) -> None:
        # Only the Triton kernels read these; prepare_for_spyre builds the device layouts.
        pass

    def prepare_for_spyre(self) -> None:
        """Build device layouts after the checkpoint is loaded in its serving dtype."""
        from spyre_inference.custom_ops.utils import convert, row_outermost_layout

        device = self.mix_L.device
        weight = self.w1.weight.detach().cpu()
        self.w1.weight = nn.Parameter(
            convert(weight, device, device_layout=row_outermost_layout(weight.shape, weight.dtype)),
            requires_grad=False,
        )
        self._mix_kjc = self.mix_L.detach().cpu().permute(1, 2, 0).contiguous().to(device)
        projection = self.in_proj.weight.detach().cpu()
        self._hidden_proj_t = projection[:, : 2 * self.rank].t().contiguous().to(device)
        self._token_proj_t = projection[:, 2 * self.rank :].t().contiguous().to(device)
        weight = self.w2.weight.detach().cpu()
        self.w2.weight = nn.Parameter(
            convert(weight, device, device_layout=row_outermost_layout(weight.shape, weight.dtype)),
            requires_grad=False,
        )

    def _mix_and_mlp(self, x):
        # Keep rank as the contiguous dimension for Spyre's stick layout.
        mixer = self.mix_L.permute(1, 2, 0) if self._mix_kjc is None else self._mix_kjc
        x = (x[:, None, :, :] * mixer[None, :, :, :]).sum(dim=2)
        return x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))

    def project_hidden_cache(self, hidden_states):
        batch, block, _ = hidden_states.shape
        if block != self.block_size:
            raise ValueError("XPress input block size must match the checkpoint's learned mixer")
        local = self.down_h(hidden_states.flatten(0, 1)).view(batch, block, self.rank)
        summary = self.down_g(hidden_states.mean(dim=1))[:, None, :].expand_as(local)
        cache = torch.cat((local, summary), dim=-1)
        if self._hidden_proj_t is not None:
            return cache @ self._hidden_proj_t
        return F.linear(cache, self.in_proj.weight[:, : 2 * self.rank])

    def prepare_feedback(self, candidates, previous_ids):
        return self.w1(candidates), self.w1(previous_ids), F.embedding(candidates, self.w2.weight)

    def refine_feedback(self, base, previous_embeddings, projected_cache, readout):
        token_projection = (
            previous_embeddings @ self._token_proj_t
            if self._token_proj_t is not None
            else F.linear(previous_embeddings, self.in_proj.weight[:, 2 * self.rank :])
        )
        latent = self._mix_and_mlp(projected_cache + token_projection)[:, 1:]
        batch, slots, candidates = base.shape
        bias = torch.bmm(
            readout.reshape(batch * slots, candidates, self.rank),
            latent.reshape(batch * slots, self.rank, 1),
        ).view_as(base)
        return base + bias
