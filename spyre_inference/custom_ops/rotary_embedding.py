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

"""Spyre OOT replacement for RotaryEmbedding.

Applies rotary position embeddings on the Spyre device using a complex-free
2x2 rotation-matrix formulation (ported from foundation-model-stack, so there
is no dependency on `fms`). The rotation runs eagerly on Spyre (like the other
Spyre custom ops); the frequency cache is indexed on CPU -- Spyre has no eager
index_select -- and only the gathered slice is moved to Spyre.

Set ``SPYRE_INFERENCE_ROPE_DEVICE=cpu`` to fall back to the previous CPU
implementation ``RotaryEmbedding.forward_native``.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from functools import lru_cache

from .utils import convert

logger = init_logger(__name__)

# Spyre stick size is 128 bytes = 64 float16 elements. The 2x2 layout's innermost
# dim is rotary_dim // 2; torch-spyre  can only stick it when it is a multiple of 64, i.e.
# rotary_dim % 128 == 0. Otherwise RoPE cannot run on the device and we fall back to CPU.
_SPYRE_STICK = 64


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full RoPE via 2x2 rotation matrices (neox / split-half pairing).

    Full rotary only (rotary_dim == head_size). Partial rotary needs a slicing,
    so it is routed to the CPU fallback and never reaches this function.

    Args:
        x: query or key, shape ``[T, H*head_size]`` or ``[T, H, head_size]``.
        rot: per-token rotation matrices, shape ``[T, 2, 2, head_size//2]``,
            ``[[cos, -sin], [sin, cos]]`` on the leading 2x2 axes.
        head_size: per-head dim (== rotary_dim).

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    num_tokens = x.shape[0]
    # Split-half pairing: x_pairs[..., 0, :] = first half, [..., 1, :] = second half.
    x_pairs = x.view(num_tokens, -1, 2, head_size // 2)  # [T, H, 2, D/2]
    # out[..., a, :] = sum_b rot[..., a, b, :] * x_pairs[..., b, :]
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)  # [T, H, 2, D/2]
    return out.flatten(-2).view(x.shape)  # [T, H, D] -> original shape


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """OOT RotaryEmbedding that applies the rotation on Spyre.

    The 2x2 rotation-matrix cache is derived from the base ``cos_sin_cache`` (so
    all rope-scaling variants are inherited) and kept on CPU. It is built lazily.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # On-device rotation needs neox (split-half) pairing, FULL rotary
        # (rotary_dim == head_size; partial-rotary slicing has no Spyre kernel),
        # and a stick-aligned inner dim (rotary_dim % 128 == 0). Everything else
        # (gptj/interleaved, partial rotary, unaligned head_size like 64) uses
        # the CPU fallback.
        self._spyre_rope_supported = (
            self.is_neox_style
            and self.rotary_dim == self.head_size
            and (self.rotary_dim // 2) % _SPYRE_STICK == 0
        )
        self._rotation_cache: torch.Tensor | None = None
        # Per-forward-pass caching of the gathered+transferred rotation slice
        self._rotation_cache_slice: torch.Tensor | None = None
        self._rotation_cache_key: tuple | None = None

    def _apply(self, fn, recurse=True):
        # Keep cos_sin_cache on CPU so forward_native can use it directly.
        # rotation_cache (spyre implemenation) is a plain CPU attribute and is
        # not a registered buffer, hence so it is unaffected here.
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache from cos_sin_cache.

        cos_sin_cache is [max_pos, rotary_dim] = cat((cos, sin)); reshape into
        [max_pos, 2, 2, rotary_dim//2] rotation matrices [[cos, -sin], [sin, cos]].
        """
        # ToDO ysc: need to make sure to trigger this in model warmup before serving a model.
        if self._rotation_cache is None:
            cpu_cache = convert(self.cos_sin_cache, device="cpu", dtype=torch.float32)
            assert cpu_cache is not None
            cos, sin = cpu_cache.chunk(2, dim=-1)  # [max_pos, Dr/2]
            self._rotation_cache = (
                torch.stack([cos, -sin, sin, cos], dim=1)
                .view(cpu_cache.shape[0], 2, 2, self.rotary_dim // 2)
                .to(self.dtype)
            )
        return self._rotation_cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on Spyre
        target_device = positions.device
        target_dtype = query.dtype

        # old CPU implementation (set SPYRE_INFERENCE_ROPE_DEVICE="cpu")
        if not (
            self._spyre_rope_supported
            and os.environ.get("SPYRE_INFERENCE_ROPE_DEVICE", "spyre") == "spyre"
        ):
            result_query, result_key = RotaryEmbedding.forward_native(
                self,
                convert(positions, device="cpu"),
                convert(query, device="cpu"),
                convert(key, device="cpu"),
            )
            return (
                convert(result_query, device=target_device, dtype=target_dtype),
                convert(result_key, device=target_device, dtype=target_dtype)
                if result_key is not None
                else None,
            )

        # new implementation: do rotation on Spyre, but keep cache on CPU.
        # Gather the per-token rotation matrices on CPU (Spyre has no eager
        # index_select) and move the slice to Spyre for the first attention layer.
        # Keep the slice on device and reuse for later attention layers as positions
        # are shared accross the layers (detection via cache_key below).
        cpu_positions = convert(positions, device="cpu")
        assert cpu_positions is not None
        cpu_positions = cpu_positions.flatten()
        cache_key = (hash(tuple(cpu_positions.tolist())), target_device, target_dtype)
        if self._rotation_cache_slice is not None and self._rotation_cache_key == cache_key:
            rot = self._rotation_cache_slice
        else:
            selected = self._get_rotation_cache().index_select(0, cpu_positions)
            rot = convert(selected, device=target_device, dtype=target_dtype)
            self._rotation_cache_slice = rot
            self._rotation_cache_key = cache_key
        # move q/k to the device (a real transfer only on the first layer;
        # later layers already have them on Spyre)
        q = convert(query, device=target_device, dtype=target_dtype)
        k = convert(key, device=target_device, dtype=target_dtype) if key is not None else None

        out_query = convert(
            _rotate_neox_2x2(q, rot, self.head_size), device=target_device, dtype=target_dtype
        )
        out_key = (
            convert(
                _rotate_neox_2x2(k, rot, self.head_size), device=target_device, dtype=target_dtype
            )
            if k is not None
            else None
        )
        return out_query, out_key


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(Llama3RotaryEmbedding, SpyreRotaryEmbedding):
    """OOT Llama3RotaryEmbedding that runs rotary computation on Spyre."""

    pass


@lru_cache(maxsize=1)
def register():
    # No-op: RotaryEmbedding doesn't require custom op registration.

    # Unlike other Spyre layers (RMSNorm, SiluAndMul, etc.), RotaryEmbedding
    # only needs a class replacement that overrides _apply() to keep weights on CPU.
    # This replacement happens at import time via @RotaryEmbedding.register_oot().
    pass
