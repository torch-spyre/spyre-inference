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
is no dependency on `fms`). The frequency cache is indexed on CPU as Spyre has
no eager index_select. ``_SpyreModelWrapper`` calls ``gather_rotation`` before
the model forward (while positions are still on the host), moves the gathered
slice to Spyre, and stashes it in the vLLM forward context. ``forward_oot``
fetches that shared slice through the opaque ``spyre_rope_rot`` op (keeping
the forward-context read out of torch.compile graphs) and applies the rotation.

Configs without an on-device path (gptj/interleaved, partial rotary, unaligned
head_size) have no Spyre RoPE kernel and raise ``NotImplementedError`` at
construction rather than silently running on CPU.
"""

import itertools
from functools import lru_cache

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.base import (
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
from vllm.model_executor.layers.rotary_embedding.llama3_rope import (
    Llama3RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert

logger = init_logger(__name__)

# Spyre stick size is 128 bytes = 64 float16 elements. The 2x2 layout's innermost
# dim is rotary_dim // 2; torch-spyre  can only stick it when it is a multiple of 64, i.e.
# rotary_dim % 128 == 0. Otherwise RoPE has no Spyre kernel and the config is rejected.
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
    # ToDo boh, ysc: maybe need .contiguous() here? Not present in fms either though
    # Split-half pairing: x_pairs[..., 0, :] = first half, [..., 1, :] = second half.
    x_pairs = x.view(num_tokens, -1, 2, head_size // 2)  # [T, H, 2, D/2]
    # out[..., a, :] = sum_b rot[..., a, b, :] * x_pairs[..., b, :]
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)  # [T, H, 2, D/2]
    return out.flatten(-2).view(x.shape)  # [T, H, D] -> original shape


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Runs the 2x2 rotation on the Spyre device for supported configs (neox, full
    rotary, stick-aligned inner dim); unsupported configs raise
    ``NotImplementedError`` at construction. The 2x2 rotation-matrix cache is
    derived from the base ``cos_sin_cache`` (so all rope-scaling variants are
    inherited) and kept on CPU. It is built lazily.
    """

    _key_counter = itertools.count()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # On-device rotation needs neox (split-half) pairing, FULL rotary
        # (rotary_dim == head_size; partial-rotary slicing has no Spyre kernel),
        # and a stick-aligned inner dim (rotary_dim % 128 == 0). Anything else
        # (gptj/interleaved, partial rotary, unaligned head_size like 64) has no
        # Spyre RoPE kernel and is rejected here rather than run on CPU.
        if not (
            self.is_neox_style
            and self.rotary_dim == self.head_size
            and (self.rotary_dim // 2) % _SPYRE_STICK == 0
        ):
            raise NotImplementedError(
                "SpyreRoPE supports only neox-style full rotary with a stick-aligned "
                f"inner dim (rotary_dim % {2 * _SPYRE_STICK} == 0); got "
                f"is_neox_style={self.is_neox_style}, rotary_dim={self.rotary_dim}, "
                f"head_size={self.head_size}."
            )
        self._rotation_cache: torch.Tensor | None = None
        # Stable per-instance (== per rope-config) key for the forward-context
        # rotation dict, passed through the opaque spyre_rope_rot op.
        self._rope_key = f"spyre_rope_{next(self._key_counter)}"
        # CPU-pinned reference to cos_sin_cache: the registered buffer gets DMA'd
        # to Spyre by torch_spyre.model_utils.load_model_to_spyre, but the 2x2
        # rotation cache is built (and index_select'd) on the host.
        self._cpu_cos_sin_cache = self.cos_sin_cache

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache from cos_sin_cache.

        cos_sin_cache is [max_pos, rotary_dim] = cat((cos, sin)); reshape into
        [max_pos, 2, 2, rotary_dim//2] rotation matrices [[cos, -sin], [sin, cos]].
        """
        # Built lazily on the first gather_rotation() call during warmup.
        if self._rotation_cache is None:
            cos, sin = self._cpu_cos_sin_cache.chunk(2, dim=-1)  # [max_pos, Dr/2]
            self._rotation_cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self._cpu_cos_sin_cache.shape[0], 2, 2, self.rotary_dim // 2
            )
        return self._rotation_cache

    def gather_rotation(
        self, positions: torch.Tensor, target_device: torch.device
    ) -> torch.Tensor | None:
        """Gather this pass's per-token 2x2 rotation slice and move it to Spyre.

        Called once per forward pass by ``_SpyreModelWrapper`` before positions
        are converted to Spyre, so the gather runs on the host with no D2H and the
        result is shared by every attention layer via the forward context. Returns
        ``None`` for multi-dim (mrope/xdrope) positions, which have no Spyre path.
        """
        cpu_positions = convert(positions, device="cpu")
        assert cpu_positions is not None
        if cpu_positions.dim() > 1:
            # mrope/xdrope positions are multi-dim and have no Spyre RoPE path.
            return None
        pos = cpu_positions.flatten().to(torch.int64)
        selected = self._get_rotation_cache().index_select(0, pos)
        return convert(selected, device=target_device, dtype=self.dtype)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on Spyre
        target_device = positions.device
        target_dtype = query.dtype

        # The per-token rotation slice is gathered ONCE before the model forward
        # (in _SpyreModelWrapper) and stashed in the forward context; here we only
        # fetch it and apply the rotation on Spyre. The fetch goes through the
        # opaque spyre_rope_rot op so the forward-context read stays out of the
        # torch.compile graph — the slice enters the graph as an op output rather
        # than a baked constant.
        rot = torch.ops.vllm.spyre_rope_rot(
            positions,  # ty: ignore[invalid-argument-type]
            self._rope_key,  # ty: ignore[invalid-argument-type]
            self.head_size,
        )
        q = convert(query, device=target_device, dtype=target_dtype)
        out_query = _rotate_neox_2x2(q, rot, self.head_size)
        out_key = None
        if key is not None:
            k = convert(key, device=target_device, dtype=target_dtype)
            out_key = _rotate_neox_2x2(k, rot, self.head_size)
        return out_query, out_key


@RotaryEmbeddingBase.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(_SpyreRotaryMixin, RotaryEmbedding):
    """OOT RotaryEmbedding that applies the rotation on Spyre.

    Supported configs run the 2x2 rotation on-device; unsupported configs raise
    NotImplementedError at construction.
    """

    pass


@RotaryEmbeddingBase.register_oot(name="Llama3RotaryEmbedding")
class SpyreLlama3RotaryEmbedding(_SpyreRotaryMixin, Llama3RotaryEmbedding):
    """OOT Llama3RotaryEmbedding that runs rotary computation on Spyre."""

    pass


def _rope_rot_op_func(positions: torch.Tensor, rope_key: str, head_size: int) -> torch.Tensor:
    """Opaque-op body: return this forward pass's 2x2 rotation slice on Spyre.

    ``_SpyreModelWrapper`` pre-gathers the slice and stashes it in the forward
    context (keyed by ``rope_key``) before every model forward, so this is a pure
    lookup. Hidden behind ``torch.ops.vllm.spyre_rope_rot`` so the forward-context
    read is not traced into outer torch.compile graphs. ``positions``/``head_size``
    are unused here but define the output shape for the fake impl below.
    """
    rope_rot = get_forward_context().additional_kwargs.get("spyre_rope_rot", {})
    if rope_key not in rope_rot:
        raise RuntimeError(f"SpyreRoPE: rotation slice for '{rope_key}' not primed")
    return rope_rot[rope_key]


def _rope_rot_op_fake(positions: torch.Tensor, rope_key: str, head_size: int) -> torch.Tensor:
    return torch.empty(
        (positions.shape[0], 2, 2, head_size // 2),
        dtype=torch.float16,
        device=positions.device,
    )


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rope_rot custom op. The OOT class replacement happens at
    import time via ``@RotaryEmbeddingBase.register_oot()``."""
    direct_register_custom_op(
        op_name="spyre_rope_rot",
        op_func=_rope_rot_op_func,
        fake_impl=_rope_rot_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_rope_rot")
