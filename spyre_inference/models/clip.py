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

"""CLIP EngineArgs overrides that must land before ModelConfig is built.

The LayerNorm boundary-norm swap lives in ``multimodal/clip.py`` instead
(an instance-level, post-load patch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def force_disable_chunked_prefill(engine_args: EngineArgs) -> None:
    """SpyreAllPool (CLIP's token-level pooler) rejects chunked prefill outright,
    but ModelConfig.is_chunked_prefill_supported doesn't exclude ALL-type token
    pooling, so it isn't disabled automatically. Mirrors
    models.gemma4.force_text_backbone's early get_config() probe.
    """
    if engine_args.enable_chunked_prefill is not None:
        return
    from vllm.transformers_utils.config import get_config

    try:
        hf_config = get_config(
            engine_args.hf_config_path or engine_args.model,
            engine_args.trust_remote_code,
            engine_args.revision,
            engine_args.code_revision,
            engine_args.config_format,
            token=engine_args.hf_token,
        )
    except Exception:
        return
    if getattr(hf_config, "model_type", None) != "clip":
        return
    engine_args.enable_chunked_prefill = False
    logger.info("CLIP: disabling chunked prefill (unsupported with token-level pooling).")
