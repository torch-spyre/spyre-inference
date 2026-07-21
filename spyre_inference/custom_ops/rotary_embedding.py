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
rotation-matrix formulation (ported from foundation-model-stack). The frequency
cache is index_select'd on CPU (Spyre has no eager index_select):
``_SpyreModelWrapper`` calls ``gather_rotation`` before the model forward, moves
the gathered slice to Spyre, and stashes it in the vLLM forward context;
``forward_oot`` fetches it through the opaque ``spyre_rope_rot`` op (keeping the
forward-context read out of torch.compile graphs) and applies the rotation.

Only neox-style full rotary is supported; other configs raise
``NotImplementedError`` at construction instead of silently falling back to CPU.
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

    _key_counter = itertools.count()

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
        self._rope_key = f"spyre_rope_{next(self._key_counter)}"
        # cos_sin_cache is DMA'd to Spyre by load_model_to_spyre; keep a CPU
        # reference since the rotation cache is index_select'd on the host.
        self._cpu_cos_sin_cache = self.cos_sin_cache

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim to the
        next stick multiple."""
        if self._rotation_cache is None:
            inner = self.rotary_dim // 2
            cos, sin = self._cpu_cos_sin_cache.chunk(2, dim=-1)
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
        """Gather this pass's per-token 2x2 rotation slice on the host and move it to
        Spyre. Returns ``None`` for multi-dim (mrope/xdrope) positions."""
        cpu_positions = convert(positions, device="cpu")
        assert cpu_positions is not None
        if cpu_positions.dim() > 1:
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
        # Fetch the pre-gathered slice via the opaque op so the forward-context read
        # stays out of the torch.compile graph.
        rot = torch.ops.vllm.spyre_rope_rot(
            positions,  # ty: ignore[invalid-argument-type]
            self._rope_key,  # ty: ignore[invalid-argument-type]
            self.head_size,
        )
        # query/key arrive on Spyre from the QKV projection; rot is primed on Spyre.
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


def _rope_rot_op_func(positions: torch.Tensor, rope_key: str, head_size: int) -> torch.Tensor:
    """Opaque-op body: return the pre-gathered 2x2 rotation slice (keyed by ``rope_key``
    in the forward context). positions/head_size only shape the fake impl below."""
    rope_rot = get_forward_context().additional_kwargs.get("spyre_rope_rot", {})
    if rope_key not in rope_rot:
        raise RuntimeError(f"SpyreRoPE: rotation slice for '{rope_key}' not primed")
    return rope_rot[rope_key]


def _rope_rot_op_fake(positions: torch.Tensor, rope_key: str, head_size: int) -> torch.Tensor:
    return torch.empty(
        (positions.shape[0], 2, 2, round_up(head_size // 2, _SPYRE_STICK)),
        dtype=torch.float16,
        device=positions.device,
    )


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rope_rot custom op. OOT class replacement happens at import
    time via ``@RotaryEmbeddingBase.register_oot()``."""
    direct_register_custom_op(
        op_name="spyre_rope_rot",
        op_func=_rope_rot_op_func,
        fake_impl=_rope_rot_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_rope_rot")
