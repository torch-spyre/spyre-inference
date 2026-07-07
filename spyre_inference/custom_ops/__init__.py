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

"""This module contains all custom ops for spyre"""

from functools import lru_cache

from . import logits_processor  # noqa: F401
from . import parallel_lm_head
from . import rms_norm
from . import rotary_embedding
from . import linear
from . import silu_and_mul
from . import utils
from . import vocab_parallel_embedding  # noqa: F401
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def register_all():
    logger.info("Registering custom ops for spyre_inference")
    rotary_embedding.register()
    utils.register()
