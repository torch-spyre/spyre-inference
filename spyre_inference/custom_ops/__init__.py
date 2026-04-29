"""This module contains all custom ops for spyre"""

from functools import lru_cache

from . import parallel_lm_head
from . import rotary_embedding
from . import silu_and_mul
from . import vocab_parallel_embedding
from . import linear
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def register_all():
    logger.info("Registering custom ops for spyre_inference")
    vocab_parallel_embedding.register()
    parallel_lm_head.register()
    rotary_embedding.register()
    silu_and_mul.register()
    linear.register()

    # IR provider registration (triggered by import)
    logger.info("Registering vLLM IR providers for spyre_inference")
    from . import kernels as _kernels  # noqa: F401
