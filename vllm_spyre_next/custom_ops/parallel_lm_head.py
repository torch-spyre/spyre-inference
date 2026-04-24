# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT replacement for ParallelLMHead.

Remove this file once all model execution happens on Spyre.
"""

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)

from functools import lru_cache

logger = init_logger(__name__)


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that keeps weights on CPU."""

    def _apply(self, fn, recurse=True):
        """No-op: keep lm_head weights on CPU.

        In the current Spyre eager-mode flow, hidden_states stay on CPU
        throughout the model. The lm_head weights must also be on CPU
        for F.linear to work without device mismatches.
        """
        return self

@lru_cache(maxsize=1)
def register():
    # No-op: ParallelLMHead doesn't require custom op registration.
    
    # Unlike other Spyre layers (RMSNorm, SiluAndMul, etc.), ParallelLMHead
    # only needs a class replacement that overrides _apply() to keep weights on CPU.
    # This replacement happens at import time via @ParallelLMHead.register_oot().
    pass
