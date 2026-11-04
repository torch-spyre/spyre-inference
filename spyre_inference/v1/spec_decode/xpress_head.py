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


class XPressRefinerHead(_UpstreamHead):
    """Upstream's head, construction and loading included, with Spyre step helpers."""

    _mix_kjc: torch.Tensor | None = None
    _readout_t: torch.Tensor | None = None
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
        # Match the target LM head's output tiling; trim only full sticks.
        padded = F.pad(weight, (0, 0, 0, (-weight.shape[0]) % 2048))
        self._readout_t = padded.t().contiguous().to(device)

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

    def refine_latent(self, previous_ids, projected_cache):
        embeddings = self.w1(previous_ids)
        token_projection = (
            embeddings @ self._token_proj_t
            if self._token_proj_t is not None
            else F.linear(embeddings, self.in_proj.weight[:, 2 * self.rank :])
        )
        return self._mix_and_mlp(projected_cache + token_projection)[:, 1:]

    def gather_readout(self, candidates):
        """Candidate IDs must be flat for Spyre's embedding kernel."""
        return F.embedding(candidates, self.w2.weight)

    def refine_full(self, base, previous_ids, projected_cache):
        latent = self.refine_latent(previous_ids, projected_cache)
        if self._readout_t is not None:
            bias = (latent @ self._readout_t)[..., : self.w2.out_features]
        else:
            bias = self.w2(latent)
        return base + bias

    def refine_candidates(self, base, previous_ids, projected_cache, readout):
        latent = self.refine_latent(previous_ids, projected_cache)
        batch, slots, candidates = base.shape
        bias = torch.bmm(
            readout.reshape(batch * slots, candidates, self.rank),
            latent.reshape(batch * slots, self.rank, 1),
        ).view_as(base)
        return base + bias
