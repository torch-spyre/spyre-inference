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

"""CLIP boundary-LayerNorm workaround for Spyre.

Only ``vision_model.pre_layrnorm``/``post_layernorm`` and
``text_model.final_layer_norm`` are swapped to ``SpyreLayerNorm``. Those three
sit at the model boundary, outside any per-block compiled graph, which is
what triggers the crash ``SpyreLayerNorm`` works around (see
``spyre_inference.custom_ops.layer_norm``). ``CLIPEncoderLayer.layer_norm1``/
``layer_norm2`` are traced inside the per-block ``torch.compile`` region
already and never hit that crashing path, so they're left as plain
``nn.LayerNorm`` -- swapping them too would be unnecessary.

Applied to the already-loaded model instance (weights included), so the
replacement ``SpyreLayerNorm`` here copies the original's already-loaded
weight/bias explicitly, rather than relying on a later ``load_weights()``
pass to populate them.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.layer_norm import SpyreLayerNorm

logger = init_logger(__name__)


def _to_spyre_layer_norm(ln: torch.nn.LayerNorm, device: torch.device) -> torch.nn.LayerNorm:
    new_ln = SpyreLayerNorm(
        list(ln.normalized_shape),
        eps=ln.eps,
        elementwise_affine=ln.elementwise_affine,
        bias=ln.bias is not None,
    ).to(device=device, dtype=ln.weight.dtype if ln.elementwise_affine else torch.float16)
    if ln.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.copy_(ln.weight)
            if ln.bias is not None:
                new_ln.bias.copy_(ln.bias)
    return new_ln


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Swap CLIP's three boundary LayerNorms for ``SpyreLayerNorm``, in place.

    The ``isinstance`` checks are a second line of defense on top of the
    ``model_type == "clip"`` dispatch gate in ``multimodal/__init__.py``: they
    keep this a no-op (rather than an ``AttributeError`` on ``normalized_shape``)
    for any boundary norm that isn't a plain ``nn.LayerNorm``.
    """
    text_model = getattr(model, "text_model", None)
    if text_model is not None:
        ln = getattr(text_model, "final_layer_norm", None)
        if isinstance(ln, torch.nn.LayerNorm):
            text_model.final_layer_norm = _to_spyre_layer_norm(ln, device)

    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        pre_ln = getattr(vision_model, "pre_layrnorm", None)
        if isinstance(pre_ln, torch.nn.LayerNorm):
            vision_model.pre_layrnorm = _to_spyre_layer_norm(pre_ln, device)
        post_ln = getattr(vision_model, "post_layernorm", None)
        if isinstance(post_ln, torch.nn.LayerNorm):
            vision_model.post_layernorm = _to_spyre_layer_norm(post_ln, device)

    logger.info_once(
        "Spyre: CLIP's boundary LayerNorms (pre_layrnorm/post_layernorm/"
        "final_layer_norm) use SpyreLayerNorm; layer_norm1/layer_norm2 inside "
        "encoder blocks are unaffected."
    )
