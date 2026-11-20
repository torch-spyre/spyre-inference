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

"""Helpers shared by the EngineArgs hooks in ``models.apply_prelaunch_overrides``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config.multimodal import BaseDummyOptions

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers import PretrainedConfig
    from vllm.engine.arg_utils import EngineArgs


def load_hf_config(engine_args: EngineArgs) -> PretrainedConfig | None:
    """The checkpoint's config, or None on any load failure so ModelConfig raises the real one."""
    from vllm.transformers_utils.config import get_config

    try:
        return get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return None


def disable_modalities(engine_args: EngineArgs, modalities: Iterable[str], model: str) -> None:
    """Cap each of ``modalities`` at 0 per prompt, raising if the user asked for one."""
    # A limit of 0 makes the frontend refuse the input per request; past it, the worker
    # raises mid-step and takes the engine down with every in-flight request.
    limits = dict(engine_args.limit_mm_per_prompt or {})
    for modality in modalities:
        value = limits.get(modality)
        # A dict without "count" takes vLLM's default, which enables the modality.
        if isinstance(value, dict):
            value = value.get("count", BaseDummyOptions().count)
        if value:
            raise NotImplementedError(
                f"{model} {modality} input is not supported on Spyre; remove it from "
                "limit_mm_per_prompt."
            )
        limits[modality] = 0
    engine_args.limit_mm_per_prompt = limits
