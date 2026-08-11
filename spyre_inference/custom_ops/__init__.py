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

from . import gelu_and_mul  # noqa: F401
from . import gemma_rms_norm  # noqa: F401
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
    _patch_element_arrangement()
    rotary_embedding.register()
    utils.register()
    vocab_parallel_embedding.register()


def _patch_element_arrangement():
    """Add ``QFP8WT`` to ``torch_spyre._C.ElementArrangement`` if missing.

    torch-spyre's inductor pass (work_division.py) references
    ``ElementArrangement.QFP8WT`` unconditionally on the right-hand side of a
    comparison even though the left side is guarded by ``hasattr``.  Older
    builds of ``torch_spyre._C`` don't include ``QFP8WT``, causing an
    ``AttributeError`` the first time any Spyre op is JIT-compiled.

    We inject a sentinel object that compares unequal to everything so the
    ``td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT``
    branch is simply never taken on these builds.
    """
    try:
        import torch_spyre._C as _C

        if not hasattr(_C.ElementArrangement, "QFP8WT"):

            class _NeverEqual:
                """Sentinel: always compares unequal, even to itself."""

                def __eq__(self, other):
                    return False

                def __hash__(self):
                    return 0

            _C.ElementArrangement.QFP8WT = _NeverEqual()
            logger.debug(
                "Patched torch_spyre._C.ElementArrangement.QFP8WT "
                "(missing from this build; injected NeverEqual sentinel)"
            )
    except Exception:
        # torch_spyre not available (e.g. CPU-only dev environment) — skip.
        pass
