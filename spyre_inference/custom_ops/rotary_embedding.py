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
head_size) and ``SPYRE_INFERENCE_ROPE_DEVICE=cpu`` fall back to a CPU
implementation wrapped in the opaque ``spyre_rotary_cpu`` op so torch.compile
does not trace into ``RotaryEmbedding.forward_static``.
"""

import itertools
import os
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

from .utils import convert, get_layer, register_layer

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
    # ToDo boh, ysc: maybe need .contiguous() here? Not present in fms either though
    # Split-half pairing: x_pairs[..., 0, :] = first half, [..., 1, :] = second half.
    x_pairs = x.view(num_tokens, -1, 2, head_size // 2)  # [T, H, 2, D/2]
    # out[..., a, :] = sum_b rot[..., a, b, :] * x_pairs[..., b, :]
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)  # [T, H, 2, D/2]
    return out.flatten(-2).view(x.shape)  # [T, H, D] -> original shape


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Supported configs (neox, full rotary, stick-aligned inner dim) run the 2x2
    rotation on the Spyre device; everything else falls back to the opaque
    ``spyre_rotary_cpu`` op. The 2x2 rotation-matrix cache is derived from the
    base ``cos_sin_cache`` (so all rope-scaling variants are inherited) and kept
    on CPU. It is built lazily.
    """

    _key_counter = itertools.count()

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
        # Stable per-instance (== per rope-config) key for the forward-context
        # rotation dict, passed through the opaque spyre_rope_rot op.
        self._rope_key = f"spyre_rope_{next(self._key_counter)}"
        # CPU-pinned reference to cos_sin_cache: the registered buffer gets DMA'd
        # to Spyre by torch_spyre.model_utils.load_model_to_spyre, but the 2x2
        # rotation cache and the CPU-fallback op both index it on the host.
        self._cpu_cos_sin_cache = self.cos_sin_cache
        # Registry key so the spyre_rotary_cpu fallback op can recover this instance.
        self._spyre_layer_name = register_layer(self, "spyre_rotary")

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
        ``None`` for configs that use the CPU fallback (nothing is stashed, and
        ``forward_oot`` takes its fallback branch instead).
        """
        if not (
            self._spyre_rope_supported
            and os.environ.get("SPYRE_INFERENCE_ROPE_DEVICE", "spyre") == "spyre"
        ):
            return None
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

        # CPU fallback (SPYRE_INFERENCE_ROPE_DEVICE="cpu" or an unsupported config):
        # route through the opaque spyre_rotary_cpu op so torch.compile does not trace
        # into RotaryEmbedding.forward_static.
        if not (
            self._spyre_rope_supported
            and os.environ.get("SPYRE_INFERENCE_ROPE_DEVICE", "spyre") == "spyre"
        ):
            # infer_schema rejects Optional[Tensor] returns, so use an empty
            # tensor sentinel across the op boundary.
            key_in = (
                key if key is not None else torch.empty(0, device=query.device, dtype=query.dtype)
            )
            # cos_sin_cache is fetched inside the op via get_layer(layer_name);
            # torch.ops dispatcher signature is opaque to the type checker.
            out_q, out_k = torch.ops.vllm.spyre_rotary_cpu(
                positions,  # ty: ignore[invalid-argument-type]
                query,  # ty: ignore[invalid-argument-type]
                key_in,  # ty: ignore[invalid-argument-type]
                self._spyre_layer_name,  # ty: ignore[invalid-argument-type]
            )
            if key is None:
                return out_q, None
            return out_q, out_k

        # On-Spyre path: the per-token rotation slice is gathered ONCE before the
        # model forward (in _SpyreModelWrapper) and stashed in the forward context;
        # here we only fetch it and apply the rotation on Spyre. The fetch goes
        # through the opaque spyre_rope_rot op so the forward-context read stays out
        # of the torch.compile graph — the slice enters the graph as an op output
        # rather than a baked constant.
        rot = torch.ops.vllm.spyre_rope_rot(  # ty: ignore[invalid-argument-type]
            positions, self._rope_key, self.head_size
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

    Supported configs run the 2x2 rotation on-device; unsupported configs and
    SPYRE_INFERENCE_ROPE_DEVICE=cpu fall back to the opaque spyre_rotary_cpu op.
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


def _rotary_cpu_op_func(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos_sin_cache is fetched from layer._cpu_cos_sin_cache (a CPU-pinned
    # reference saved in __init__), since the registered buffer gets DMA'd
    # to Spyre by torch_spyre.model_utils.load_model_to_spyre.
    layer = get_layer(layer_name)
    target_device = positions.device
    target_dtype = query.dtype

    cpu_positions = positions.to("cpu")
    cpu_query = query.to("cpu")
    cpu_key = key.to("cpu") if key.numel() > 0 else None

    out_q, out_k = RotaryEmbedding.forward_static(
        positions=cpu_positions,
        query=cpu_query,
        key=cpu_key,
        head_size=layer.head_size,
        rotary_dim=layer.rotary_dim,
        cos_sin_cache=layer._cpu_cos_sin_cache,
        is_neox_style=layer.is_neox_style,
    )

    out_q = out_q.to(device=target_device, dtype=target_dtype)
    if out_k is None:
        out_k = torch.empty(0, device=target_device, dtype=target_dtype)
    else:
        out_k = out_k.to(device=target_device, dtype=target_dtype)
    return out_q, out_k


def _rotary_cpu_op_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_q = torch.empty(query.shape, dtype=query.dtype, device=positions.device)
    out_k = torch.empty(key.shape, dtype=query.dtype, device=positions.device)
    return out_q, out_k


@lru_cache(maxsize=1)
def register():
    """Register both the on-device (spyre_rope_rot) and CPU-fallback (spyre_rotary_cpu)
    custom ops. The OOT class replacement happens at import time via
    ``@RotaryEmbeddingBase.register_oot()``."""
    direct_register_custom_op(
        op_name="spyre_rope_rot",
        op_func=_rope_rot_op_func,
        fake_impl=_rope_rot_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    direct_register_custom_op(
        op_name="spyre_rotary_cpu",
        op_func=_rotary_cpu_op_func,
        fake_impl=_rotary_cpu_op_fake,
        mutates_args=[],
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom ops: spyre_rope_rot, spyre_rotary_cpu")
