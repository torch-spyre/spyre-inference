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

Configs without an on-device path (gptj/interleaved, partial rotary) have no
Spyre RoPE kernel and raise ``NotImplementedError`` at construction rather than
silently running on CPU. Full neox rotary is supported for any head_size: when
the 2x2 layout's inner dim (``rotary_dim // 2``) is stick-aligned (e.g.
head_size=128 -> 64) the whole rotation runs on Spyre with a pure view; when it is
not (e.g. head_size=64 -> 32) the split-half reshape has a sub-stick stride that the
Spyre inductor rejects, so each half is zero-padded up to the next stick multiple
on-device with a constant ``{0, 1}`` matmul (no host round-trip) before the rotation.
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
# dim is rotary_dim // 2. When it is not a multiple of 64 (e.g. head_size=64 -> 32),
# the split-half reshape has a sub-stick pairing-axis stride that torch-spyre cannot
# read on device; each half is then zero-padded up to the next stick multiple on-device
# with a constant {0,1} matmul (see _get_expand_matrix / _rotate_neox_2x2).
_SPYRE_STICK = 64


def round_up(n: int, m: int = _SPYRE_STICK) -> int:
    """Round ``n`` up to the nearest multiple of ``m`` (the Spyre stick size)."""
    return ((n + m - 1) // m) * m


@lru_cache
def _get_expand_matrix(
    inner: int, padded: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Constant ``{0, 1}`` matrix ``E`` of shape ``[2*inner, 2*padded]`` that zero-pads each
    neox half up to the stick-aligned ``padded`` **on-device**, via ``x @ E``.

    ``x @ E`` maps ``x[:inner] -> out[:inner]`` and ``x[inner:] -> out[padded:padded+inner]``
    with zeros in the pad lanes, i.e. the flattened ``[T, H, 2, padded]`` pad layout. Doing
    the pad this way (instead of ``x.view(T, H, 2, inner)`` + host round-trip) never
    materializes the sub-stick ``[.,2,inner]`` view: ``E``'s output is a fresh contiguous
    ``[.,2*padded]`` whose ``view(.,2,padded)`` has a stick-aligned pairing stride. Cached
    per ``(inner, padded, device, dtype)``.
    """
    e = torch.zeros(2 * inner, 2 * padded, dtype=dtype)
    idx = torch.arange(inner)
    e[idx, idx] = 1  # first half:  x[:inner]      -> out[:inner]
    e[inner + idx, padded + idx] = 1  # second half: x[inner:] -> out[padded:padded+inner]
    return convert(e, device=device, dtype=dtype)


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full RoPE via 2x2 rotation matrices (neox / split-half pairing).

    Full rotary only (rotary_dim == head_size). Partial rotary needs a slicing,
    so it has no Spyre kernel and never reaches this function.

    The 2x2 layout's inner (stick) dim is ``head_size // 2``. When that is a stick
    multiple (e.g. head_size=128 -> 64), ``x`` is reshaped directly with a pure view
    and everything runs on Spyre. When it is not (e.g. head_size=64 -> 32), the
    split-half view ``x.view(T, H, 2, inner)`` has a sub-stick pairing-axis stride
    (``inner`` = head_size/2, not a multiple of 64), which torch-spyre's inductor
    rejects (``Unexpected stick expression`` on any read of that view). For the unaligned
    case the pad-to-stick is therefore done **on-device with a constant matmul**: ``x @ E``
    (``E`` a ``{0, 1}`` matrix, see ``_get_expand_matrix``) writes a fresh contiguous
    ``[T, H, 2*padded]`` holding each neox half zero-padded to ``padded``; ``view(.,2,padded)``
    then has a stick-aligned pairing stride. The sub-stick ``[.,2,inner]`` view never
    exists, so nothing (matmul, rotation, or the offset-0 ``[..., :inner]`` padding strip)
    hits the stick rejection. Padded lane values are irrelevant: the rotation cache is zero
    in the padded region.

    Args:
        x: query or key, shape ``[T, H*head_size]`` or ``[T, H, head_size]``.
        rot: per-token rotation matrices, shape ``[T, 2, 2, padded]`` where
            ``padded >= head_size//2``, ``[[cos, -sin], [sin, cos]]`` on the leading
            2x2 axes with zeros in the padded tail.
        head_size: per-head dim (== rotary_dim).

    Returns:
        Rotated tensor with the same shape as ``x``.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    padded = rot.shape[-1]
    if padded != inner:
        # Sub-stick pairing-axis stride: the [T, H, 2, inner] split view is unreadable on
        # Spyre. Pad each half up to the stick-aligned `padded` on-device via a constant
        # {0,1} matmul (x @ E), yielding a contiguous [T, H, 2*padded] whose view has a
        # stick-aligned pairing stride -- the [.,2,inner] layout is never materialized.
        e = _get_expand_matrix(inner, padded, x.device, x.dtype)
        x_pairs = (x.view(num_tokens, -1, head_size) @ e).view(num_tokens, -1, 2, padded)
    else:
        # Split-half pairing via a pure view: x_pairs[..., 0, :] = first half,
        # [..., 1, :] = second half. The view keeps the neox halves without slicing.
        x_pairs = x.view(num_tokens, -1, 2, inner)  # [T, H, 2, inner]
    # out[..., a, :] = sum_b rot[..., a, b, :] * x_pairs[..., b, :]
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)  # [T, H, 2, padded]
    if padded != inner:
        # Strip the padding (offset-0 slice) before reconstructing the head. The slice
        # is non-contiguous, so make the reshape explicit with .contiguous().
        out = out[..., :inner].contiguous()  # [T, H, 2, inner]
    return out.flatten(-2).view(x.shape)  # [T, H, D] -> original shape


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Runs the 2x2 rotation on the Spyre device for supported configs (neox, full
    rotary); unsupported configs raise ``NotImplementedError`` at construction. A
    non-stick-aligned inner dim is handled by zero-padding it up to the next stick
    multiple for the rotation. The 2x2 rotation-matrix cache is derived from the
    base ``cos_sin_cache`` (so all rope-scaling variants are inherited) and kept on
    CPU. It is built lazily.
    """

    _key_counter = itertools.count()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # On-device rotation needs neox (split-half) pairing and FULL rotary
        # (rotary_dim == head_size; partial-rotary slicing has no Spyre kernel).
        # gptj/interleaved and partial rotary have no Spyre RoPE kernel and are
        # rejected here rather than run on CPU. An unaligned inner dim (e.g.
        # head_size=64 -> inner 32) is supported via padding in _rotate_neox_2x2.
        if not (self.is_neox_style and self.rotary_dim == self.head_size):
            raise NotImplementedError(
                "SpyreRoPE supports only neox-style full rotary (rotary_dim == "
                f"head_size); got is_neox_style={self.is_neox_style}, "
                f"rotary_dim={self.rotary_dim}, head_size={self.head_size}."
            )
        # Inner (stick) dim of the 2x2 layout, padded up to the next stick multiple.
        self._padded_inner = round_up(self.rotary_dim // 2)
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
        The inner dim is zero-padded (at the end) up to the next stick multiple so
        the on-device rotation runs stick-aligned; when already aligned this is a
        no-op. All CPU work, so the padding op is unconstrained.
        """
        # Built lazily on the first gather_rotation() call during warmup.
        if self._rotation_cache is None:
            inner = self.rotary_dim // 2
            cos, sin = self._cpu_cos_sin_cache.chunk(2, dim=-1)  # [max_pos, inner]
            cache = torch.stack([cos, -sin, sin, cos], dim=1).view(
                self._cpu_cos_sin_cache.shape[0], 2, 2, inner
            )
            if self._padded_inner != inner:
                cache = torch.nn.functional.pad(cache, (0, self._padded_inner - inner))
            self._rotation_cache = cache
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
    # Inner dim is padded up to the next stick multiple (see _rotate_neox_2x2), so the
    # fake/meta shape must match the padded rotation slice torch.compile sees.
    return torch.empty(
        (positions.shape[0], 2, 2, round_up(head_size // 2)),
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
