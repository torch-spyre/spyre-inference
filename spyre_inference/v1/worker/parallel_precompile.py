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

"""The warm-up's first pass in forked compile workers, shared by the V1 and V2 runners.

``SPYRE_PARALLEL_WARMUP_PRECOMPILE=N`` traces the warm-up once with every graph's Spyre
inner compile forked into one of ``N`` processes (torch-spyre's ``fork_inner_compiles``
with stand-in graphs), then forgets the stand-ins so the runner's normal warm-up pass
retraces the same graphs into cache hits. Each runner supplies its own trace of the
buckets; this module owns the pool, its guards and the accounting.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)


def precompile_in_workers(
    trace: Callable[[Any], bool], warm_regions: Callable[[], None]
) -> Any | None:
    """Run ``trace(pool)`` with the inner compiles forked, then reset Dynamo.

    ``trace`` runs the warm-up's dummy runs inside the pool and returns ``True`` when the
    first bucket found every graph cached (a restart with warm caches), in which case
    its real graphs stay and nothing else runs. On a cold start ``warm_regions`` then runs
    the compiled regions a stand-in block never reaches (the experts behind the opaque
    MoE op), and the stand-in graphs are forgotten. Returns the check to ``finish()``
    after the normal pass, or ``None`` when the workers did not run.
    """
    try:
        from torch_spyre._inductor.parallel_precompile import (  # ty: ignore[unresolved-import]
            SecondPassCheck,
            fork_inner_compiles,
        )
    except ImportError:
        logger.warning(
            "SPYRE_PARALLEL_WARMUP_PRECOMPILE needs a torch-spyre with fork_inner_compiles "
            "(torch-spyre/torch-spyre#6262); compiling the buckets one after another."
        )
        return None
    if "placeholders" not in inspect.signature(fork_inner_compiles).parameters:
        logger.warning(
            "SPYRE_PARALLEL_WARMUP_PRECOMPILE needs fork_inner_compiles(placeholders=True); "
            "compiling the buckets one after another."
        )
        return None
    if torch._dynamo.config.caching_precompile:
        logger.info(
            "SPYRE_PARALLEL_WARMUP_PRECOMPILE stays off under TORCH_CACHING_PRECOMPILE: "
            "Dynamo packages need every compile's output code, which the workers' "
            "stand-ins do not provide; compiling the buckets one after another."
        )
        return None
    t0 = time.time()
    # At TP > 1 every rank precompiles, so stay within this rank's CPU share.
    workers = max(1, min(envs.SPYRE_PARALLEL_WARMUP_PRECOMPILE, torch.get_num_threads()))
    # The keyword lands with torch-spyre#6533 (checked by the signature test above); a dict
    # keeps the call type-checkable against a torch-spyre that predates it.
    options = {"placeholders": True}
    with fork_inner_compiles(workers, timeout_s=1800.0, **options) as pool:
        warm = trace(pool)
        if not warm:
            warm_regions()
    if not warm:
        # Forget the stand-in graphs; the normal pass retraces the buckets into cache hits.
        # A warm start traced real graphs, which the normal pass reuses.
        torch._dynamo.reset()
    logger.info(
        "Parallel warm-up precompile: %s in %.1fs.",
        "skipped, the caches are warm"
        if warm
        else f"{pool.submitted} graphs compiled in {workers} workers ({pool.failed} failed)",
        time.time() - t0,
    )
    if warm or not pool.submitted:
        return None
    return SecondPassCheck(expected=pool.ok)
