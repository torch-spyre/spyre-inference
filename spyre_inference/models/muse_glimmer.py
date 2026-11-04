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

"""Muse Glimmer (text backbone) on Spyre.

Upstream's model (``vllm.model_executor.models.muse_glimmer``) runs as written except for
its RMSNorm, a plain ``nn.Module`` computing in fp32 that the platform cannot override.
Outside a block graph (the embedding norm, the final norm) it runs op by op in eager mode,
where it returns wrong values, and infs at the final norm. Every instance becomes the vLLM
norm with the same math, which Spyre compiles wherever it runs: ``GemmaRMSNorm`` for the
``1 + w`` sandwich norms, ``RMSNorm`` for the rest.

The vision tower is not adapted, so a checkpoint carrying one serves text only; see
``force_language_model_only``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models import muse_glimmer as mg

if TYPE_CHECKING:
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)


def force_language_model_only(engine_args: EngineArgs) -> None:
    """Default a Muse Glimmer checkpoint to ``language_model_only``.

    Upstream then builds the vision tower as a placeholder, so its weights are neither
    loaded nor moved to the card, and the frontend refuses image and video input per
    request. Images are not supported on Spyre yet: the tower would run op by op, and its
    host-built position tables have no device layout.
    """
    if engine_args.language_model_only or engine_args.limit_mm_per_prompt:
        return
    from vllm.transformers_utils.config import get_config

    # On any load failure, defer to ModelConfig, which raises the real error.
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
    if getattr(hf_config, "model_type", None) != "muse_glimmer":
        return
    engine_args.language_model_only = True
    logger.info("Muse Glimmer: serving text only (language_model_only=True).")


def _spyre_rms_norm(
    dim: int | None = None, eps: float = 1e-6, with_scale: bool = True, weight_offset: int = 0
) -> nn.Module:
    """``MuseGlimmerRMSNorm``'s constructor, building the vLLM norm of the same math."""
    if not with_scale:
        # The width only sizes RMSNorm's unused ones-vector; the QK norms pass none.
        return RMSNorm(dim or 1, eps=eps, has_weight=False)
    if dim is None:
        raise ValueError("a scaled MuseGlimmerRMSNorm needs its width")
    if weight_offset == 1:
        return GemmaRMSNorm(dim, eps=eps)
    if weight_offset != 0:
        raise NotImplementedError(f"MuseGlimmerRMSNorm weight_offset={weight_offset}")
    return RMSNorm(dim, eps=eps)


@contextlib.contextmanager
def _spyre_norms():
    """Build upstream's model with ``_spyre_rms_norm``, which its classes look up as a
    module global at construction."""
    saved = mg.MuseGlimmerRMSNorm
    mg.MuseGlimmerRMSNorm = _spyre_rms_norm  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        mg.MuseGlimmerRMSNorm = saved


class SpyreMuseGlimmerForCausalLM(mg.MuseGlimmerForCausalLM):
    """Muse Glimmer on Spyre with norms the platform compiles."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        with _spyre_norms():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
