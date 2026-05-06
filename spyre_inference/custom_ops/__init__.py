"""This module contains all custom ops for spyre"""

from functools import lru_cache

from . import parallel_lm_head
from . import rms_norm
from . import rotary_embedding
from . import linear
from . import silu_and_mul
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def register_all():
    logger.info("Registering custom ops for spyre_inference")
    parallel_lm_head.register()
    rotary_embedding.register()
    rms_norm.register()
