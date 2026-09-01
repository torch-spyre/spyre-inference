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

from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def force_text_backbone(engine_args: EngineArgs) -> None:
    """Default gemma-4 to its text-only backbone (it ships multimodal).

    Sets ``hf_overrides["architectures"]`` so ``create_model_config`` resolves
    ``Gemma4ForCausalLM`` instead of the multimodal default. Skipped when the user
    already pinned an architecture (dict or callable ``hf_overrides``).
    """
    ov = engine_args.hf_overrides
    user_arch = callable(ov) or (isinstance(ov, dict) and "architectures" in ov)
    if "gemma-4" in (engine_args.model or "").lower() and not user_arch and isinstance(ov, dict):
        overrides = cast("dict[str, Any]", ov)
        overrides["architectures"] = ["Gemma4ForCausalLM"]
        logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def install_spyre_patches() -> None:
    """Register Gemma-4's aliased ``normalizer`` as a buffer so it follows the model.

    ``Gemma4SelfDecoderLayers`` stores ``normalizer`` as a plain tensor attribute
    aliased from ``Gemma4Model``'s buffer. ``model.to("spyre")`` rebinds the parent's
    buffer to a device tensor but leaves this alias on CPU, so the compiled
    ``embed_input_ids`` feeds a 0-d CPU tensor into Inductor, which has no notion of a
    live CPU graph input. Re-registering it as a buffer restores the parent's documented
    intent (move with the model, interact with torch.compile) and needs no change to the
    embedding math. A device-side 0-d scalar lowers fine.
    """
    from vllm.model_executor.models.gemma4 import Gemma4SelfDecoderLayers

    if getattr(Gemma4SelfDecoderLayers, "_spyre_patched", False):
        return

    orig_init = Gemma4SelfDecoderLayers.__init__

    def __init__(self, *args, **kwargs) -> None:
        orig_init(self, *args, **kwargs)
        normalizer = self.normalizer
        del self.normalizer
        self.register_buffer("normalizer", normalizer, persistent=False)

    Gemma4SelfDecoderLayers.__init__ = __init__  # ty: ignore[invalid-assignment]
    Gemma4SelfDecoderLayers._spyre_patched = True
    logger.info(
        "Spyre: Gemma-4 normalizer registered as a buffer so it follows the model to "
        "device (upstream aliases it as a plain CPU attribute)."
    )
