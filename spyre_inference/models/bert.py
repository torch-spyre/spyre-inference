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

"""Spyre adaptations for vLLM BERT-family pooling models."""

from __future__ import annotations

from vllm.logger import init_logger

from spyre_inference.models.token_type_adapter import install_on

logger = init_logger(__name__)


def install_spyre_patches() -> None:
    """Install BERT token_type side-buffer adapter (see ``token_type_adapter``)."""
    from vllm.model_executor.models import bert

    install_on(bert)
    logger.info(
        "Spyre: BERT token_type_ids use side-buffer adapter (skip vLLM bit-pack; torch-spyre#3509)"
    )
