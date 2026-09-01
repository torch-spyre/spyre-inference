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

"""Spyre-safe ``torch.nn.LayerNorm``.

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

Fix: reimplement the forward pass with plain mean/var/rsqrt arithmetic, same
as ``SpyreRMSNorm`` does for RMSNorm, so ``aten.layer_norm.default`` (and its
crashing decomposition) is never invoked at all.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_orig_init = torch.nn.LayerNorm.__init__
_orig_forward = torch.nn.LayerNorm.forward


def _spyre_layer_norm_init(self: torch.nn.LayerNorm, *args, **kwargs) -> None:
    _orig_init(self, *args, **kwargs)
    # Sampled here, same as `CompileOutermost.__init__`: construction is the only
    # point where the vLLM config context is guaranteed live; a call during
    # warm-up (before/outside any per-block compile) is not.
    from vllm.config import CompilationMode, get_cached_compilation_config

    mode = get_cached_compilation_config().mode
    self._spyre_compile_enabled = mode is not CompilationMode.NONE
    self._spyre_kernel = None


def _layer_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def _spyre_layer_norm_forward(self: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "spyre":
        return _orig_forward(self, x)

    weight = self.weight if self.elementwise_affine else None
    bias = self.bias if self.elementwise_affine else None

    # Already inside a per-block compiled graph (STOCK_TORCH_COMPILE compiles one
    # transformer block at a time): inline directly, don't re-enter torch.compile.
    if torch.compiler.is_compiling():
        return _layer_norm_kernel(x, weight, bias, self.eps)

    kernel = self._spyre_kernel
    if kernel is None:
        if not self._spyre_compile_enabled:
            kernel = _layer_norm_kernel
        else:
            from vllm.platforms import current_platform

            logger.info_once(
                "Compiling torch.nn.LayerNorm as its own graph: no enclosing graph "
                "covers it."
            )
            # dynamic=False is mandatory: the Spyre backend rejects SymInt shapes.
            kernel = torch.compile(
                _layer_norm_kernel,
                backend=current_platform.simple_compile_backend,
                fullgraph=True,
                dynamic=False,
            )
        self._spyre_kernel = kernel
    return kernel(x, weight, bias, self.eps)


def register() -> None:
    """Monkeypatch ``torch.nn.LayerNorm`` to bypass torch-spyre's
    ``exx2``/``layernormscale``/``layernormnorm`` decomposition."""
    torch.nn.LayerNorm.__init__ = _spyre_layer_norm_init
    torch.nn.LayerNorm.forward = _spyre_layer_norm_forward
    logger.debug_once("Patched torch.nn.LayerNorm for Spyre")
