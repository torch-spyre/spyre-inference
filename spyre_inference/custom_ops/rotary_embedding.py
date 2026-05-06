# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacement for RotaryEmbedding.

The full cos/sin table is precomputed on CPU in __init__. Each forward
indexes the cache on CPU with positions and transfers only the sliced
cos/sin to the device, where the rotation is applied.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from .utils import convert


logger = init_logger(__name__)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding on spyre with a CPU-resident cos/sin cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre RotaryEmbedding")

        d = self.rotary_dim
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                      # [max_pos, d/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)   # [max_pos, d]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)   # [max_pos, d]

        # Plain attributes (not buffers) so module.to(device) doesn't move them.
        self._cos_cache_cpu = cos.to(self.dtype)
        self._sin_cache_cpu = sin.to(self.dtype)

    def _apply_rotary(self, x, cos, sin):
        """Apply rotary embedding on the original device."""
        T, hidden = x.shape
        H = hidden // self.head_size
        d = self.rotary_dim

        x = x.view(T, H, self.head_size)
        x_rot = x[..., :d]
        x_pass = x[..., d:]

        cos = cos.unsqueeze(1)  # [T, 1, d]
        sin = sin.unsqueeze(1)

        d_half = d // 2
        x1 = x_rot[..., :d_half]
        x2 = x_rot[..., d_half:]
        x_rotated = torch.cat([-x2, x1], dim=-1)

        out = (x_rot * cos) + (x_rotated * sin)
        out = torch.cat([out, x_pass], dim=-1)
        return out.reshape(T, hidden)

    def forward_oot(self, positions, q, k):
        device = q.device

        positions_cpu = convert(positions, device="cpu")
        cos = convert(self._cos_cache_cpu[positions_cpu], device=device)
        sin = convert(self._sin_cache_cpu[positions_cpu], device=device)

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k
