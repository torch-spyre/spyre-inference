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

Applies rotary position embeddings on the Spyre device via a complex-free 2x2
rotation-matrix formulation (ported from foundation-model-stack). The 2x2 rotation
cache is held device-resident and gathered on Spyre with ``index_select``
(torch-spyre#3418 gave single-row gather a kernel): ``forward_oot`` calls
``gather_rotation`` to index this pass's per-token slice on the fly, then applies the
rotation through the opaque ``spyre_rope_rotate`` op. The rotation is kept opaque (its
body runs eagerly on Spyre) because torch-spyre's compiled lowering of the 2x2 rotation
corrupts when fused into the full-model graph.

Only neox-style full rotary is supported; other configs raise
``NotImplementedError`` at construction instead of silently falling back to CPU.
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert

logger = init_logger(__name__)

# Spyre stick size = 64 float16 elements. The 2x2 layout's inner dim is
# rotary_dim // 2; when that is not a stick multiple the split-half view has a
# sub-stick stride the inductor rejects, so it is padded up on-device.
_SPYRE_STICK = 64


@lru_cache
def _get_expand_matrix(
    inner: int, padded: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Constant ``{0, 1}`` matrix ``E`` [2*inner, 2*padded] that zero-pads each neox
    half up to the stick-aligned ``padded`` on-device via ``x @ E`` (so the sub-stick
    ``[.,2,inner]`` view is never materialized). Cached per ``(inner, padded, device, dtype)``.
    """
    e = torch.zeros(2 * inner, 2 * padded, dtype=dtype)
    idx = torch.arange(inner)
    e[idx, idx] = 1
    e[inner + idx, padded + idx] = 1
    return convert(e, device=device, dtype=dtype)


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, padded]
    with ``padded >= head_size // 2``. When the inner dim head_size//2 is stick-aligned
    the split-half pairing is a pure view; otherwise each half is zero-padded to
    ``padded`` on-device with a constant matmul so the pairing-axis stride is aligned.
    Returns the rotated tensor with ``x``'s shape.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    padded = rot.shape[-1]
    if padded != inner:
        e = _get_expand_matrix(inner, padded, x.device, x.dtype)
        x_pairs = (x.view(num_tokens, -1, head_size) @ e).view(num_tokens, -1, 2, padded)
    else:
        x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    if padded != inner:
        out = out[..., :inner].contiguous()  # non-contiguous slice; copy before reshape
    return out.flatten(-2).view(x.shape)


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Runs the 2x2 rotation on Spyre for supported configs; unsupported configs raise
    ``NotImplementedError`` at construction. The rotation cache is derived lazily from
    the base ``cos_sin_cache`` (inheriting all rope-scaling variants) and kept on CPU.
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
        self._padded_inner = round_up(self.rotary_dim // 2, _SPYRE_STICK)
        self._rotation_cache: torch.Tensor | None = None
        self._device_rotation_cache: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        # cos_sin_cache has no Spyre kernel; keep cos_sin_cache on CPU.
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim to the
        next stick multiple."""
        if self._rotation_cache is None:
            inner = self.rotary_dim // 2
            cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
            cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self.cos_sin_cache.shape[0], 2, 2, inner
            )
            if self._padded_inner != inner:
                cache = torch.nn.functional.pad(cache, (0, self._padded_inner - inner))
            self._rotation_cache = cache
        return self._rotation_cache

    def _get_device_rotation_cache(self, device: torch.device) -> torch.Tensor:
        """Device-resident copy of the 4D rotation cache ``[max_pos, 2, 2, padded]``,
        built once from the CPU cache so the per-pass gather runs on-device via
        ``index_select`` (single-row gather has a kernel since torch-spyre#3418)."""
        if self._device_rotation_cache is None:
            self._device_rotation_cache = convert(
                self._get_rotation_cache().contiguous(), device=device, dtype=self.dtype
            )
        return self._device_rotation_cache

    def gather_rotation(self, positions: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Index the device-resident cache to get this pass's per-token 2x2 rotation
        slice; ``positions`` may arrive on the host or on Spyre.

        In production this is called on the fly inside ``forward_oot`` with Spyre int64
        positions (the model wrapper converts integer inputs to int64 at the model
        boundary); torch-spyre's ``index_select`` downcasts int64 indices to int32
        internally (safe for positions < 2**31), so no explicit cast is needed. The
        host path (``target_device.type != "spyre"``) still handles CPU int64 positions
        for the CPU-reference tests on dev laptops."""
        idx = positions.flatten()
        if target_device.type != "spyre":
            # CPU-reference path (dev laptops, rotation-math test): host index_select,
            # no Spyre-dispatched op.
            return self._get_rotation_cache().index_select(0, idx.to(torch.int64)).to(self.dtype)
        if idx.dtype == torch.int64:
            logger.debug_once(
                "SpyreRoPE gather using torch-spyre's internal int64->int32 downcast "
                "(safe for positions < 2**31)."
            )
        return self._get_device_rotation_cache(target_device).index_select(
            0, convert(idx, device=target_device)
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Gather this pass's per-token rotation slice on the fly (on-device
        # index_select), then apply the rotation through the opaque spyre_rope_rotate
        # op so the 2x2 rotation runs eagerly on Spyre (its fused lowering corrupts
        # under whole-model compile).
        rot = self.gather_rotation(positions, query.device)
        out_query = torch.ops.vllm.spyre_rope_rotate(
            query,  # ty: ignore[invalid-argument-type]
            rot,
            self.head_size,
        )
        out_key = (
            torch.ops.vllm.spyre_rope_rotate(
                key,  # ty: ignore[invalid-argument-type]
                rot,
                self.head_size,
            )
            if key is not None
            else None
        )
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


def _rope_rotate_op_func(x: torch.Tensor, rot: torch.Tensor, head_size: int) -> torch.Tensor:
    """Opaque-op body: apply the 2x2 rotation eagerly on-device."""
    return _rotate_neox_2x2(x, rot, head_size)


def _rope_rotate_op_fake(x: torch.Tensor, rot: torch.Tensor, head_size: int) -> torch.Tensor:
    return torch.empty_like(x)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rope_rotate custom op. OOT class replacement happens at import
    time via ``@RotaryEmbeddingBase.register_oot()``."""
    direct_register_custom_op(
        op_name="spyre_rope_rotate",
        op_func=_rope_rotate_op_func,
        fake_impl=_rope_rotate_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_rope_rotate")
