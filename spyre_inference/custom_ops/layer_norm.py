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

"""Spyre-safe ``torch.nn.LayerNorm`` replacement.

``torch.nn.LayerNorm`` is plain PyTorch, not a vLLM ``CustomOp`` (unlike
``RMSNorm``), so it can't be intercepted via ``register_oot``. Its default
forward goes through ``aten.layer_norm.default``, which torch-spyre decomposes
into three native ops (``exx2`` -> ``layernormscale`` -> ``layernormnorm``,
see ``torch_spyre._inductor.decompositions.spyre_layer_norm``). That
decomposition is wrapped in an ``_OPWrapper`` that unconditionally
``torch.compile``s it in isolation the first time it's dispatched eagerly --
regardless of ``--enforce-eager``. For call sites outside any per-block
compiled region (e.g. CLIP's ``pre_layrnorm``/``post_layernorm``), the
resulting fused kernel fails at the native ``dxp_standalone`` compiler stage
("Not enough dimensions" matching ``layernormnorm.ddl``'s fixed 5-dim
signature).

Fix: ``SpyreLayerNorm`` reimplements the forward pass with plain
mean/var/rsqrt arithmetic, so ``aten.layer_norm.default`` (and its crashing
decomposition) is never invoked at all. The mean/var reduction accumulates in
fp32 -- unlike fp16, which overflows to ``inf`` once ``|x - mean| > 256`` and
silently zeroes the row via ``rsqrt(inf)`` -- matching what ``F.layer_norm``
itself does internally, and needed here because CLIP's boundary norms are
exactly where activation outliers of that magnitude show up.

This is a drop-in subclass, not a global monkeypatch of
``torch.nn.LayerNorm`` -- only the specific boundary LayerNorms that actually
hit the crash (currently: CLIP's, patched in ``spyre_inference.multimodal.clip``)
should be swapped to it. Most ``LayerNorm`` call sites live inside a per-block
``torch.compile`` region and never take the crashing eager path in the first
place, so patching them too would be unnecessary blast radius onto unrelated
models.
"""

from __future__ import annotations

import torch

from .lazy_compile import CompileOutermost, maybe_compile


def _layer_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    # Casting back to fp16 before the affine step -- rather than after -- hits a
    # torch-spyre layout limitation ("Multi-arg pointwise with mixed EA") when
    # multiplying by `weight`: the fp32 reduction's device layout doesn't
    # broadcast against a plain fp16 parameter. Keeping weight/bias in fp32 too
    # and casting only the final result avoids it.
    input_dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight.float()
    if bias is not None:
        x_norm = x_norm + bias.float()
    return x_norm.to(input_dtype)


class SpyreLayerNorm(CompileOutermost, torch.nn.LayerNorm):
    """``torch.nn.LayerNorm`` that never invokes ``aten.layer_norm.default`` on Spyre."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if input.device.type != "spyre":
            return super().forward(input)
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.elementwise_affine else None
        return self._spyre_forward(input, weight, bias)

    @maybe_compile
    def _spyre_forward(
        self, x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return _layer_norm_kernel(x, weight, bias, self.eps)
