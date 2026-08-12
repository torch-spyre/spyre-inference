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

"""Spyre adaptations for vLLM RoBERTa / XLM-R pooling models.

RoBERTa re-exports BERT's ``_encode_token_type_ids`` /
``_decode_token_type_ids`` into its own module globals, so the side-buffer
adapter must be installed here as well as on ``bert``.
"""

from __future__ import annotations

from vllm.logger import init_logger

from spyre_inference.models.token_type_adapter import install_on

logger = init_logger(__name__)


def install_spyre_patches() -> None:
    """Install token_type side-buffer adapter on the RoBERTa module namespace."""
    from vllm.model_executor.models import roberta

    if not hasattr(roberta, "_encode_token_type_ids") or not hasattr(
        roberta, "_decode_token_type_ids"
    ):
        logger.debug("Spyre: RoBERTa module has no token_type helpers; skipping adapter")
        return

    install_on(roberta)
    logger.info(
        "Spyre: RoBERTa token_type_ids use side-buffer adapter "
        "(skip vLLM bit-pack; torch-spyre#3509)"
    )
