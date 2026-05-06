# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacements for RotaryEmbedding and ApplyRotaryEmb.

Mirrors the upstream `RotaryEmbedding.forward_static` and
`ApplyRotaryEmb.forward_static` bodies. Spyre does not support dynamic
tensor indexing today (index_select, chunk, last-dim slicing), so those
ops are quarantined on CPU; the rotation arithmetic (mul/add/sub/cat)
runs on the spyre device via the overridden ApplyRotaryEmb. When spyre
gains indexing support, the CPU hops can be dropped and the code
collapses onto the upstream forward_native.
"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .utils import convert


logger = init_logger(__name__)


@ApplyRotaryEmb.register_oot(name="ApplyRotaryEmb")
class SpyreApplyRotaryEmb(ApplyRotaryEmb):
    """Spyre OOT variant of ApplyRotaryEmb.

    Template: vllm/model_executor/layers/rotary_embedding/common.py
    Same math as the upstream `ApplyRotaryEmb.forward_static`; the one
    unsupported op (`torch.chunk(x, 2, -1)` for neox / strided slicing
    for gptj) is done on CPU. Elementwise mul/add/sub and cat stay on
    device.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        origin_dtype = x.dtype
        if self.enable_fp32_compute:
            x = x.float()

        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        # CPU: chunk / strided slice (spyre indexing not supported today).
        # Non-contiguous chunk/slice views must be materialized before
        # transfer -- spyre's .to() on non-contiguous views misbehaves.
        device = x.device
        x_cpu = convert(x, device="cpu")
        if self.is_neox_style:
            x1_cpu, x2_cpu = torch.chunk(x_cpu, 2, dim=-1)
        else:
            x1_cpu = x_cpu[..., ::2]
            x2_cpu = x_cpu[..., 1::2]
        x1 = convert(x1_cpu.contiguous(), device=device)
        x2 = convert(x2_cpu.contiguous(), device=device)

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if self.is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if self.enable_fp32_compute:
            output = output.to(origin_dtype)
        return output


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """Spyre OOT variant of RotaryEmbedding.

    The full cos/sin cache is pinned on CPU. Each forward indexes the
    cache on CPU by position (the one indexing op spyre doesn't support
    today), transfers the per-token cos/sin slices to spyre, and
    delegates the rotation to the OOT-registered `SpyreApplyRotaryEmb`
    which runs entirely on spyre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre RotaryEmbedding")

        # Pin a CPU copy of the cos/sin cache. Plain attribute (not a
        # buffer) so module.to(device) cannot relocate it -- forward_oot
        # requires it to be CPU-resident for the index_select.
        self._cos_sin_cache_cpu = self.cos_sin_cache.detach().clone()

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Positions arrive on spyre; we need them on CPU for the
        # cache index_select (the one op spyre does not support).
        position_device = positions.device
        positions_cpu = convert(positions, device="cpu").flatten()

        num_tokens = positions_cpu.shape[0]
        cos_sin_cpu = self._cos_sin_cache_cpu.index_select(0, positions_cpu)
        cos_cpu, sin_cpu = cos_sin_cpu.chunk(2, dim=-1)

        # Chunk views are non-contiguous; spyre transfers of non-contiguous
        # tensors misbehave, so materialize before moving to device.
        cos = convert(cos_cpu.contiguous(), device=position_device)
        sin = convert(sin_cpu.contiguous(), device=position_device)

        # q/k arrive on CPU in the current runner; move to the same
        # device as positions so the rotation math runs on spyre.
        query_dev = convert(query, device=position_device)
        key_dev = convert(key, device=position_device) if key is not None else None

        query_dev = self._rope(query_dev, cos, sin, num_tokens)
        if key_dev is not None:
            key_dev = self._rope(key_dev, cos, sin, num_tokens)

        return (
            convert(query_dev, device="cpu"),
            convert(key_dev, device="cpu") if key_dev is not None else None,
        )

    def _rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Apply RoPE. For partial RoPE (rotary_dim < head_size) fall
        back to a CPU slice for the rotary/pass split, since a view
        cannot express that split. Otherwise run entirely on-device.
        """
        x_shape = x.shape
        x_device = x.device
        x = x.view(num_tokens, -1, self.head_size)

        if self.rotary_dim == self.head_size:
            # Full RoPE: entire head is rotated, single view+apply.
            return self.apply_rotary_emb(x, cos, sin).reshape(x_shape)

        # Partial RoPE: split rotary/pass on CPU (non-contiguous slices
        # must be materialized before transfer to spyre).
        x_cpu = convert(x, device="cpu")
        x_rot_cpu = x_cpu[..., :self.rotary_dim].contiguous()
        x_pass_cpu = x_cpu[..., self.rotary_dim:].contiguous()
        x_rot = convert(x_rot_cpu, device=x_device)
        x_pass = convert(x_pass_cpu, device=x_device)

        x_rot = self.apply_rotary_emb(x_rot, cos, sin)
        out = torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)
        return out
