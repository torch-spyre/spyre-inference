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

"""Spyre adaptations for vLLM's Gemma-4 model."""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def install_spyre_patches() -> None:
    """Scale Gemma-4 embeddings by a Python float instead of the 0-d ``normalizer``.

    ``model.to("spyre")`` leaves ``Gemma4SelfDecoderLayers``' aliased ``normalizer`` on a
    0-d CPU tensor that torch-spyre cannot tile ("does not have FixedTiledLayout"). A
    scalar multiply lowers to ``aten.mul.Scalar`` with no 0-d operand and is numerically
    identical. The float is precomputed in ``__init__`` because ``forward`` is
    ``@support_torch_compile`` — computing it there would lift the 0-d tensor back into
    the traced graph.
    """
    from vllm.model_executor.models.gemma4 import Gemma4SelfDecoderLayers

    if getattr(Gemma4SelfDecoderLayers, "_spyre_patched", False):
        return

    orig_init = Gemma4SelfDecoderLayers.__init__

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        self._spyre_normalizer_scale = float(self.normalizer)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self._spyre_normalizer_scale

    Gemma4SelfDecoderLayers.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers.embed_input_ids = embed_input_ids  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers._spyre_patched = True
    logger.info(
        "Spyre: Gemma-4 embeddings scaled by a precomputed Python float "
        "(avoids a 0-d CPU normalizer tensor, eager and torch.compile)."
    )
