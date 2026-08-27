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
both directly in the full-model compile graph.

The cache must be materialized on-device *before* compile: building it inside the traced
forward (host chunk/stack/view then device transfer) segfaults libsenlib during warmup.
``_apply`` primes it when the module moves to Spyre, ahead of ``torch.compile``.

The cache is flattened to 2D and placed rows-outermost (see ``place_row_gathered``).

Only neox-style full rotary uses the on-device 2x2 kernel. Other configs (partial
rotary, where ``rotary_dim < head_size``, and gptj/interleaved) fall back to the native
``RotaryEmbedding`` forward, run on the host inside the opaque op
``torch.ops.vllm.spyre_partial_rotary``: the native half-split slices the head at a
sub-stick offset (``rotary_dim // 2``) that cannot lower on-device, and keeping it opaque
hides the host transfer from the compile graph (a traced ``spyre->cpu`` copy segfaults
libsenlib during warmup). Partial rotary is used by some vision-language models (GLM-OCR).
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
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import place_row_gathered

logger = init_logger(__name__)


def _partial_rotary_op_func(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    # Rotate on the host: the sub-stick head slice (rotary_dim < head_size) cannot lower
    # on Spyre, and hiding it in an opaque op keeps the transfer out of the compile graph.
    out, _ = RotaryEmbedding.forward_static(
        positions.cpu(),
        x.cpu(),
        None,
        head_size,
        rotary_dim,
        cos_sin_cache.cpu().to(x.dtype),
        is_neox_style,
    )
    return out.to(x.device)


def _partial_rotary_op_fake(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


@lru_cache(maxsize=1)
def _register_partial_rotary_op():
    # CompositeExplicitAutograd so the op dispatches regardless of input device
    # (query/key arrive on Spyre, cos_sin_cache stays CPU-pinned).
    direct_register_custom_op(
        op_name="spyre_partial_rotary",
        op_func=_partial_rotary_op_func,
        fake_impl=_partial_rotary_op_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_partial_rotary")


def _rotate_neox_2x2(
    x: torch.Tensor,
    rot: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Apply full neox RoPE via per-token 2x2 rotation matrices.

    ``x`` is [T, H*head_size] or [T, H, head_size]; ``rot`` is [T, 2, 2, head_size // 2].
    The inner dim head_size // 2 is stick-aligned (the platform pads head_dim to a
    128-multiple before RoPE is built), so the split-half pairing is a pure view.
    Returns the rotated tensor with ``x``'s shape.
    """
    num_tokens = x.shape[0]
    inner = head_size // 2
    x_pairs = x.view(num_tokens, -1, 2, inner)
    out = (rot.unsqueeze(1) * x_pairs.unsqueeze(-3)).sum(dim=-2)
    return out.flatten(-2).view(x.shape)


class _SpyreRotaryMixin:
    """Spyre RoPE wiring shared by the base and llama3 OOT classes.

    Runs the 2x2 rotation on Spyre for neox-style full rotary; any other config
    (partial rotary or gptj/interleaved) falls back to the native ``RotaryEmbedding``
    forward. The rotation cache is derived lazily from the base ``cos_sin_cache``
    (inheriting all rope-scaling variants) and kept on CPU.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only neox full rotary has a Spyre 2x2 kernel; gptj/interleaved and partial
        # rotary (rotary_dim < head_size) fall back to the native forward instead.
        self._spyre_native_fallback = not (self.is_neox_style and self.rotary_dim == self.head_size)
        if self._spyre_native_fallback:
            logger.warning_once(
                "SpyreRoPE: config unsupported by the on-device 2x2 kernel "
                "(is_neox_style=%s, rotary_dim=%s, head_size=%s); falling back to the "
                "native RotaryEmbedding forward.",
                self.is_neox_style,
                self.rotary_dim,
                self.head_size,
            )
            # Register here, not in forward_oot: registration must not be traced (Dynamo
            # graph-breaks trying to enter infer_schema inside a compiled forward).
            _register_partial_rotary_op()
            return
        self._padded_inner = self.rotary_dim // 2
        self._rotation_cache: torch.Tensor | None = None
        self._device_rotation_cache: torch.Tensor | None = None

    def _apply(self, fn, recurse=True):
        if self._spyre_native_fallback:
            # Keep cos_sin_cache CPU-pinned: spyre_partial_rotary reads it on the host,
            # so relocating it to Spyre would only add a round-trip per call.
            return self
        # Skip super()._apply: cos_sin_cache is intentionally CPU-pinned and this module
        # holds no other movable tensor, so there is nothing to relocate. We instead prime
        # the device rotation cache here (before torch.compile traces forward_oot).
        self._device_rotation_cache = place_row_gathered(
            self._get_device_rotation_cache(), fn, "RoPE rotation cache"
        )
        return self

    def _get_rotation_cache(self) -> torch.Tensor:
        """Lazily build the CPU 2x2 rotation cache [max_pos, 2, 2, _padded_inner] from
        cos_sin_cache ([[cos, -sin], [sin, cos]]), zero-padding the inner dim up to
        _padded_inner when a padded head injected a narrower original-frequency cache."""
        if self._rotation_cache is None:
            # Derive inner from the cache actually present, not rotary_dim: when a
            # head is padded (head_size=64 -> 128), fix_padded_rope injects the
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
        ``index_select`` (single-row gather has a kernel since torch-spyre#3418)."""
        if self._device_rotation_cache is None:
            self._device_rotation_cache = self._get_rotation_cache().flatten(1).contiguous()
        return self._device_rotation_cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._spyre_native_fallback:
            out_query = torch.ops.vllm.spyre_partial_rotary(
                query,
                positions,
                self.cos_sin_cache,
                self.head_size,
                self.rotary_dim,
                self.is_neox_style,
            )
            out_key = (
                torch.ops.vllm.spyre_partial_rotary(
                    key,
                    positions,
                    self.cos_sin_cache,
                    self.head_size,
                    self.rotary_dim,
                    self.is_neox_style,
                )
                if key is not None
                else None
            )
            return out_query, out_key
        # Cache was primed in _apply before compile, so only the index_select is traced.
        cache = self._get_device_rotation_cache()
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
