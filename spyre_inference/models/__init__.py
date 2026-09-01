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

"""Model-specific Spyre adaptations, per architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs


def apply_prelaunch_overrides(engine_args: EngineArgs) -> None:
    """Apply per-model EngineArgs overrides that must run before create_model_config
    builds the ModelConfig (e.g. text-only backbone selection)."""
    from spyre_inference.models import gemma4

    gemma4.force_text_backbone(engine_args)


def install_pooling_model_patches() -> None:
    """Install encoder/pooling model adapters (BERT / RoBERTa token_type, …)."""
    from spyre_inference.models import bert, roberta

    bert.install_spyre_patches()
    roberta.install_spyre_patches()


def install_decoder_model_patches() -> None:
    """Install decoder/generative model adapters (Gemma-4 embed scale, …)."""
    from spyre_inference.models import gemma4

    gemma4.install_spyre_patches()
