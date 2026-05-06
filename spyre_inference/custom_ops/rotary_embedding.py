# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacements for RotaryEmbedding and ApplyRotaryEmb.

Mirrors the upstream `forward_static` bodies. 

Limitations:
    *) All ops run on CPU today: spyre lacks dynamic tensor indexing,
    such as `index_select`, `chunk`, last-dim slicing, etc. 
    *) FP16 multiply diverges from CPU, which flips tokens under greedy decoding. 
    Thus, for the moment, the multiplications in `SpyreApplyRotaryEmb` need to run on CPU
    *) No promotion of the data types, as this is not yet supported in torch-spyre.
    
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
    Math is identical to `ApplyRotaryEmb.forward_static`; runs on CPU
    and restores the input device on the output.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        x_cpu = convert(x, device="cpu")
        cos_cpu = convert(cos, device="cpu")
        sin_cpu = convert(sin, device="cpu")

        origin_dtype = x_cpu.dtype
        if self.enable_fp32_compute:
            raise RuntimeError("Spyre currently doesn't support dtype upcasting!")
            x_cpu = x_cpu.float()

        cos_cpu = cos_cpu.unsqueeze(-2).to(x_cpu.dtype)
        sin_cpu = sin_cpu.unsqueeze(-2).to(x_cpu.dtype)

        if self.is_neox_style:
            x1, x2 = torch.chunk(x_cpu, 2, dim=-1)
        else:
            x1 = x_cpu[..., ::2]
            x2 = x_cpu[..., 1::2]

        o1 = x1 * cos_cpu - x2 * sin_cpu
        o2 = x2 * cos_cpu + x1 * sin_cpu

        if self.is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if self.enable_fp32_compute:
            output = output.to(origin_dtype)
        return convert(output, device=device)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """Spyre OOT variant of RotaryEmbedding.

    Mirrors `RotaryEmbedding.forward_static`. Runs on CPU today; the
    OOT structure is preserved for a later on-device migration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre RotaryEmbedding")

        # Plain attribute (not a buffer) so module.to(device) cannot
        # relocate it -- index_select requires a CPU-resident cache.
        self._cos_sin_cache_cpu = self.cos_sin_cache

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on spyre; query/key arrive on CPU (qkv split
        # in GraniteAttention runs on CPU). Restore their original
        # devices on the outputs so downstream layers see no shift.
        query_device = query.device
        key_device = key.device if key is not None else None

        positions_cpu = convert(positions, device="cpu").flatten()
        query_cpu = convert(query, device="cpu")
        key_cpu = convert(key, device="cpu") if key is not None else None

        num_tokens = positions_cpu.shape[0]
        cos_sin_cpu = self._cos_sin_cache_cpu.index_select(0, positions_cpu)
        cos_cpu, sin_cpu = cos_sin_cpu.chunk(2, dim=-1)

        query_cpu = self._rope(query_cpu, cos_cpu, sin_cpu, num_tokens)
        if key_cpu is not None:
            key_cpu = self._rope(key_cpu, cos_cpu, sin_cpu, num_tokens)

        return (
            convert(query_cpu, device=query_device),
            convert(key_cpu, device=key_device) if key_cpu is not None else None,
        )

    def _rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(num_tokens, -1, self.head_size)

        if self.rotary_dim == self.head_size:
            return self.apply_rotary_emb(x, cos, sin).reshape(x_shape)

        x_rot = x[..., :self.rotary_dim]
        x_pass = x[..., self.rotary_dim:]
        x_rot = self.apply_rotary_emb(x_rot, cos, sin)
        return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)
