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

"""Native-path SwiGLU MLP ``intermediate_size`` padding to a stick-aligned width.

An ``intermediate_size`` that is not a multiple of the 64-element fp16 stick makes
``SiluAndMul`` slice the fused gate+up tensor's second half at an unaligned offset,
which Spyre inductor cannot lower. ``TorchSpyrePlatform._maybe_pad_intermediate_size``
rounds it up to a 64-multiple before the model is built; the pass here zero-fills the
added gate/up output rows and down_proj input columns as the checkpoint streams in.

Zero-padding is arithmetically inert for SwiGLU (``silu(0) = 0``): unlike QK-norm
padding, nothing normalizes over ``intermediate_size`` so no rescale is needed, and
there is no RoPE half-split so plain end-padding (not interleaving) suffices.

Scope: dense SwiGLU (``gate_proj``/``up_proj``/``down_proj``, fused or separate); MoE
experts (``moe_intermediate_size``) are out of scope — a fused expert tensor differs.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

BLOCK_SIZE = 64
_ORIG_ATTR = "_spyre_orig_intermediate_size"


def original_intermediate_size(hf_config) -> int | None:
    """The pre-pad intermediate_size if the platform padded this model, else None."""
    return getattr(hf_config, _ORIG_ATTR, None)


def intermediate_padding_active(hf_config) -> bool:
    """True when the platform padded this model's intermediate_size for alignment."""
    return original_intermediate_size(hf_config) is not None


def _pad_rows_end(w: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    """Zero-pad the output dim (dim 0) ``[orig, ...] -> [padded, ...]``. gate/up."""
    return F.pad(w, (0,) * (2 * (w.ndim - 1)) + (0, padded - orig))


def _pad_cols_end(w: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    """Zero-pad the input dim (dim 1) ``[hidden, orig] -> [hidden, padded]``. down."""
    return F.pad(w, (0, padded - orig))


def _pad_weight(name: str, w: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    """Dispatch a single checkpoint tensor to the right end-padding by its name."""
    # Must precede the up_proj test: "gate_up_proj.*" also ends with "up_proj.*".
    if name.endswith(("gate_up_proj.weight", "gate_up_proj.bias")) and w.shape[0] == 2 * orig:
        gate, up = w.chunk(2, dim=0)
        return torch.cat([_pad_rows_end(gate, orig, padded), _pad_rows_end(up, orig, padded)])
    if name.endswith(("gate_proj.weight", "gate_proj.bias", "up_proj.weight", "up_proj.bias")):
        return _pad_rows_end(w, orig, padded) if w.shape[0] == orig else w
    # down_proj input columns line up with the padded activation lanes.
    if name.endswith("down_proj.weight") and w.ndim == 2 and w.shape[1] == orig:
        return _pad_cols_end(w, orig, padded)
    return w


def install_mlp_pad_weight_loader(model_loader, hf_config) -> None:
    """Wrap ``model_loader.get_all_weights`` to zero-pad the MLP tensors to ``padded``.

    Runs on the raw ``(name, tensor)`` stream before vLLM's ``WeightsMapper`` and
    ``weight_loader`` (which narrow/assert against the now-padded params). Full
    unsharded tensors are end-padded, so TP narrowing downstream still selects
    clean partitions (primary target is TP=1). Composes with the head-pad loader:
    the two transforms touch disjoint tensor names.
    """
    if not intermediate_padding_active(hf_config):
        return
    if not hasattr(model_loader, "get_all_weights"):
        logger.warning(
            "MLP padding active but %s has no get_all_weights; weights not padded.",
            type(model_loader).__name__,
        )
        return

    orig = getattr(hf_config, _ORIG_ATTR)
    padded = hf_config.intermediate_size

    original_get_all_weights = model_loader.get_all_weights

    def padded_get_all_weights(model_config, model) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in original_get_all_weights(model_config, model):
            yield name, _pad_weight(name, weight, orig, padded)

    model_loader.get_all_weights = padded_get_all_weights


def verify_padded_intermediate_size(model, hf_config) -> None:
    """Fail loudly if any SwiGLU MLP was still built at the unpadded width.

    Guards the silent-corruption path: the linear weight loader narrows an
    over-wide tensor to the param width without raising, so a module the config
    override failed to reach loads truncated weights and produces plausible
    garbage instead of an error.
    """
    if not intermediate_padding_active(hf_config):
        return
    padded = hf_config.intermediate_size
    bad = sorted(
        {
            f"{name}(input_size={module.input_size})"
            for name, module in model.named_modules()
            if name.endswith("down_proj") and getattr(module, "input_size", padded) != padded
        }
    )
    if bad:
        raise RuntimeError(
            f"Spyre padded MLP intermediate_size to {padded}, but these down_proj "
            f"layers were built at a different width, so their weights would load "
            f"truncated: {', '.join(bad)}"
        )
