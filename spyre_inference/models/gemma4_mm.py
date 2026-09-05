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

"""Spyre adaptation for vLLM's multimodal Gemma-4 wrapper.

Its own module, not part of ``models.gemma4``: ``apply_prelaunch_overrides``
imports that one for every launch, and pulling the vision / audio towers in with
it would cost every other model. Registration here stays lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

if TYPE_CHECKING:
    from spyre_inference.models.gemma4 import SpyreGemma4ForCausalLM


class SpyreGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Forward vLLM's post-load hook down to the text backbone.

    vLLM calls ``process_weights_after_loading`` on the top-level model only. The
    wrapper builds its language model through the registry, so that model is already
    ``SpyreGemma4ForCausalLM`` — but its hook, which lays out the Gemma-4 MoE expert
    stacks, would never fire. Reachable only when the user sets ``hf_overrides`` and
    so opts out of ``gemma4.force_text_backbone``.
    """

    def process_weights_after_loading(self) -> None:
        cast("SpyreGemma4ForCausalLM", self.language_model).process_weights_after_loading()
