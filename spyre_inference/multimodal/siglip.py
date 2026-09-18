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

"""SigLIP vision-encoder workarounds for Spyre."""

from __future__ import annotations

import torch


def patch_siglip_vision_embeddings(model: torch.nn.Module, device: torch.device) -> None:
    """Move SiglipVisionEmbeddings position_embedding and position_ids to CPU.

    aten.embedding(position_embedding.weight, position_ids) called eagerly on
    Spyre tensors hits torch-spyre's compile_once eager kernel, causing Dynamo
    re-entrancy / RecursionError when tracing. Keeping the position embedding lookup
    on CPU avoids this.
    """
    try:
        from vllm.model_executor.models.siglip import SiglipVisionEmbeddings
    except ImportError:
        return

    def _siglip_embeddings_forward(
        self: SiglipVisionEmbeddings,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        if interpolate_pos_encoding:
            embeddings += self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # Both the embedding lookup and the add run on CPU to avoid
            # torch-spyre's compile_once re-entrancy (aten.embedding and
            # aten.add both hit compile_once when called eagerly on Spyre).
            pos_emb = self.position_embedding(self.position_ids)
            embeddings = (embeddings.to("cpu") + pos_emb).to(device)
        return embeddings

    for module in model.modules():
        if isinstance(module, SiglipVisionEmbeddings):
            module.position_embedding.to("cpu")
            module.register_buffer(
                "position_ids",
                module.position_ids.to("cpu"),
                persistent=False,
            )
            module.forward = _siglip_embeddings_forward.__get__(module)  # ty: ignore[method-assign]


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply SigLIP vision workarounds."""
    patch_siglip_vision_embeddings(model, device)
