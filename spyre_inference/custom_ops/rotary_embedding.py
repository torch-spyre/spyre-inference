# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacement for RotaryEmbedding (CPU fallback).

Keeps cos_sin_cache on CPU. Inputs are moved to
CPU for computation, and outputs are copied back to the original device.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from functools import lru_cache

from .utils import convert


logger = init_logger(__name__)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding on spyre with CPU-only cos/sin computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre RotaryEmbedding")

        # Precompute inverse frequencies
        dim = self.rotary_dim
        base = self.base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def _compute_cos_sin_cpu(self, positions: torch.Tensor):
        inv_freq = convert(self.inv_freq, device="cpu")
        freqs = torch.outer(positions, inv_freq)  # [seq, d/2]
        
        cos = freqs.cos()
        sin = freqs.sin()

        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        return cos, sin

    def _apply_rotary(self, x, cos, sin):
        """Apply rotary embedding (runs on original device)."""

        T, hidden = x.shape
        H = hidden // self.head_size

        # --- reshape to heads ---
        x = x.view(T, H, self.head_size)
        d = self.rotary_dim
        x_rot = x[..., :d]          # [T, H, d]
        x_pass = x[..., d:]         # [T, H, D-d]

        #--- reshape cos/sin properly ---
        cos = cos.unsqueeze(1)      # [T, 1, d]
        sin = sin.unsqueeze(1)

        # --- split ---
        d_half = d // 2
        x1 = x_rot[..., :d_half]
        x2 = x_rot[..., d_half:]

        # --- rotate---
        x_rotated = torch.cat([-x2, x1], dim=-1)

        # --- apply rope ---
        out = (x_rot * cos) + (x_rotated * sin)
        # --- concat ---
        out = torch.cat([out, x_pass], dim=-1)
        # --- flatten back ---
        return out.reshape(T, hidden)

    def forward_oot(self, positions, q, k):
        original_device = q.device
        dtype = q.dtype

        # --- move positions to CPU ---
        positions_cpu = convert(positions, device="cpu")
        # --- compute cos/sin on CPU ---
        cos_cpu, sin_cpu = self._compute_cos_sin_cpu(positions_cpu)
        
        # --- move back to device ---
        cos = convert(cos_cpu.to(dtype), device=original_device)
        sin = convert(sin_cpu.to(dtype), device=original_device)

        # --- apply rotary ---
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        return q, k
