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

Applies neox RoPE on Spyre via a 2x2 rotation-matrix formulation (ported from
foundation-model-stack). The rotation cache is device-resident; ``forward_oot`` gathers
this pass's per-token slice with ``index_select`` and applies it with ``_rotate_neox_2x2``,
both directly in the enclosing compile graph (the transformer block, by default).

The cache must be materialized on-device *before* compile: building it inside the traced
forward (host chunk/stack/view then device transfer) segfaults libsenlib during warmup.
``_apply`` primes it when the module moves to Spyre, ahead of ``torch.compile``.

The cache is flattened to 2D and placed rows-outermost (see ``place_row_gathered``).

A head whose half is not stick-aligned (head_size=64) uses ``_rotate_neox_split_free``
instead: the 2x2 form views each head as two halves, which cannot restickify when a
half is narrower than a stick.

Only neox-style full rotary is supported; other configs raise ``NotImplementedError``.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
    Gemma4RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

from .utils import place_row_gathered

# One Spyre stick of fp16 elements.
_STICK = 64

logger = init_logger(__name__)


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, head_size // 2].
    The inner dim head_size // 2 is stick-aligned (a head whose half isn't takes
    ``_rotate_neox_split_free``), so the split-half pairing is a pure view.
    Returns the rotated tensor with ``x``'s shape.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    return out.flatten(-2).view(x.shape)


def _rotate_half_matrix(head_size: int, dtype: torch.dtype) -> torch.Tensor:
    """``x @ R == cat(-x[half:], x[:half])``, neox ``rotate_half`` as a signed permutation."""
    half = head_size // 2
    idx = torch.arange(half)
    r = torch.zeros(head_size, head_size, dtype=dtype)
    r[idx + half, idx] = -1.0
    r[idx, idx + half] = 1.0
    return r


def _rotate_neox_split_free(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
    rotate_half: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Full neox RoPE as ``x * cos + rotate_half(x) * sin``, viewing only whole heads.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``cos_sin`` is [T, 2, head_size].
    """
    num_tokens = x.shape[0]
    heads = x.view(num_tokens, -1, head_size)
    cos, sin = cos_sin[:, 0].unsqueeze(1), cos_sin[:, 1].unsqueeze(1)
    rotated = (heads.reshape(-1, head_size) @ rotate_half).view(heads.shape)
    return (heads * cos + rotated * sin).view(x.shape)


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the OOT rotary classes.

    Runs the 2x2 (or, for a sub-stick half, split-free) rotation on Spyre for supported
    configs; unsupported configs raise ``NotImplementedError`` at construction. The
    rotation cache is derived lazily on CPU from the base ``cos_sin_cache`` (inheriting
    all rope-scaling variants), and ``_apply`` places a copy on the device when the
    module moves there.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only neox full rotary has a Spyre kernel; gptj/interleaved and partial
        # rotary are rejected here rather than run on CPU.
        if not (self.is_neox_style and self.rotary_dim == self.head_size):
            raise NotImplementedError(
                "SpyreRoPE supports only neox-style full rotary (rotary_dim == "
                f"head_size); got is_neox_style={self.is_neox_style}, "
                f"rotary_dim={self.rotary_dim}, head_size={self.head_size}."
            )
        self._padded_inner = self.rotary_dim // 2
        self._rotation_cache: torch.Tensor | None = None
        self._device_rotation_cache: torch.Tensor | None = None
        # Set only for the split-free path, i.e. when a half head is sub-stick.
        self._rotate_half = (
            _rotate_half_matrix(self.head_size, self.cos_sin_cache.dtype)
            if self._padded_inner % _STICK != 0
            else None
        )

    def _apply(self, fn, recurse=True):
        # Skip super()._apply: cos_sin_cache is intentionally CPU-pinned and this module
        # holds no other movable tensor, so there is nothing to relocate. We instead prime
        # the device rotation cache here (before torch.compile traces forward_oot).
        self._device_rotation_cache = place_row_gathered(
            self._get_device_rotation_cache(), fn, "RoPE rotation cache"
        )
        if self._rotate_half is not None:
            self._rotate_half = fn(self._rotate_half)
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, _padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim up to
        _padded_inner when a padded head injected a narrower original-frequency cache."""
        if self._rotation_cache is None:
            # Derive inner from the cache actually present, not rotary_dim: when a
            # head is padded (e.g. head_size=96 -> 128), fix_padded_rope injects the
            # original narrower cos_sin_cache so the real frequencies survive; the
            # trailing dims are then zero-padded to _padded_inner (harmless because
            # the matching x pair dims are zero from weight padding).
            inner = self.cos_sin_cache.shape[-1] // 2
            cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
            cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self.cos_sin_cache.shape[0], 2, 2, inner
            )
            if self._padded_inner != inner:
                cache = torch.nn.functional.pad(cache, (0, self._padded_inner - inner))
            self._rotation_cache = cache
        return self._rotation_cache

    def _get_device_rotation_cache(self) -> torch.Tensor:
        """Device-resident rotation cache, flattened to 2D ``[max_pos, 4 * padded]`` so
        it can be stickified with the position axis outermost, and gathered on-device via
        ``index_select`` (single-row gather has a kernel since torch-spyre#3418). The
        split-free path stores ``[cos | cos | sin | sin]`` instead."""
        if self._device_rotation_cache is None:
            rot = self._get_rotation_cache()
            if self._rotate_half is not None:
                cos, sin = rot[:, 0, 0], rot[:, 1, 0]
                self._device_rotation_cache = torch.cat([cos, cos, sin, sin], dim=-1)
            else:
                self._device_rotation_cache = rot.flatten(1)
        return self._device_rotation_cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Cache was primed in _apply before compile, so only the index_select is traced.
        cache = self._get_device_rotation_cache()
        rotate_half = self._rotate_half
        if rotate_half is not None:
            cos_sin = cache.index_select(0, positions.flatten()).view(-1, 2, self.head_size)
            out_query = _rotate_neox_split_free(query, cos_sin, rotate_half, self.head_size)
            out_key = (
                _rotate_neox_split_free(key, cos_sin, rotate_half, self.head_size)
                if key is not None
                else None
            )
            return out_query, out_key
        rot = cache.index_select(0, positions.flatten()).view(-1, 2, 2, self._padded_inner)
        out_query = _rotate_neox_2x2(query, rot, self.head_size)
        out_key = _rotate_neox_2x2(key, rot, self.head_size) if key is not None else None
        return out_query, out_key


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(_SpyreRotaryMixin, RotaryEmbedding):
    """OOT RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(_SpyreRotaryMixin, Llama3RotaryEmbedding):
    """OOT Llama3RotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="YaRNScalingRotaryEmbedding")
class SpyreYaRNScalingRotaryEmbedding(_SpyreRotaryMixin, YaRNScalingRotaryEmbedding):
    """OOT YaRNScalingRotaryEmbedding that applies the rotation on Spyre."""

    pass


@RotaryEmbeddingBase.register_oot(name="Gemma4RotaryEmbedding")
class SpyreGemma4RotaryEmbedding(_SpyreRotaryMixin, Gemma4RotaryEmbedding):
    """OOT Gemma4RotaryEmbedding (proportional RoPE) that applies the rotation on Spyre.

    ``partial_rotary_factor < 1`` but ``rotary_dim == head_size`` with the non-rotated
    frequencies identity-padded, so the neox full-rotary path applies unchanged.
    """

    pass
