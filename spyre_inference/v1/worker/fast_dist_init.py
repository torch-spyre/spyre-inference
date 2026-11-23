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

"""Opt-in: start the card of a one-card worker while its weights are read.

At world size 1 nothing uses the spyreccl process group, but building it is what
starts the card (torch-spyre's runtime), and the seed and the memory snapshot on the
next lines of ``init_device`` need the card too. So the card start sits alone on the
critical path. With ``SPYRE_FAST_DIST_INIT=1`` the worker instead:

1. starts the runtime on a side thread at the top of ``init_device``. The call goes
   through ``ctypes``, which lets go of the GIL (the pybind ``start_runtime`` binding
   keeps it, so a side thread calling it would stop the main thread);
2. builds the process groups with gloo only;
3. seeds the host generators at once and the card's generators later;
4. waits for the card, seeds it and takes the memory snapshot after the weights are
   read and before the first upload (``join_after_weight_read``).

Nothing touches the card's memory before step 4, so the snapshot sees the same card as
it does today. ``SPYRE_FAST_DIST_INIT=gloo`` does step 2 only: a diagnostic that
shows what building the spyreccl groups costs apart from the card start.

Any other world size, the mp executor, DP > 1 or the flag unset: no change.
"""

import ctypes
import importlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.utils import request_memory

from spyre_inference import envs

logger = init_logger(__name__)

_ENV = "SPYRE_FAST_DIST_INIT"
# spyre::startRuntime() in torch_spyre/_C, the std::call_once that start_runtime and the
# stream pool also go through.
_START_RUNTIME_SYMBOL = "_ZN5spyre12startRuntimeEv"


def _mode(worker: Any) -> str | None:
    value = envs.SPYRE_FAST_DIST_INIT.strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return None
    if value not in ("1", "gloo"):
        raise ValueError(f"{_ENV}={value!r}: use 0, 1 or gloo.")
    parallel = worker.vllm_config.parallel_config
    if (
        parallel.world_size != 1
        or parallel.data_parallel_size != 1
        or worker.local_rank != 0
        or parallel.distributed_executor_backend != "uni"
    ):
        logger.warning(
            "Ignoring %s=%s: it only applies to one worker inside EngineCore (world size "
            "1, uni executor); here world_size=%d, data_parallel_size=%d, executor=%r.",
            _ENV,
            value,
            parallel.world_size,
            parallel.data_parallel_size,
            parallel.distributed_executor_backend,
        )
        return None
    return value


class CardStart:
    """The card's runtime, started early. ``join`` finishes the steps it put off."""

    def __init__(self, worker: Any, mode: str) -> None:
        self.worker = worker
        self.mode = mode
        self.seed: int | None = None
        self.joined = False
        self._ready = threading.Event()
        self._started_at = time.perf_counter()
        self._side_thread_s: float | None = None

    @property
    def defers_card_work(self) -> bool:
        return self.mode == "1"

    def start(self) -> None:
        if not self.defers_card_work:
            self._ready.set()
            return
        _C = importlib.import_module("torch_spyre._C")
        # What torch-spyre's _lazy_init does before start_runtime.
        from torch_spyre import logging_config

        logging_config._sync_cpp_config()
        try:
            if _C.device_count() < 1:
                raise RuntimeError("no visible card")
            start_runtime = getattr(ctypes.CDLL(_C.__file__), _START_RUNTIME_SYMBOL)
        except (OSError, AttributeError, RuntimeError) as e:
            # Still correct: the card then starts on the main thread at the join.
            logger.warning(
                "%s=1: cannot start the card on a side thread (%s); it starts after the "
                "weight read instead.",
                _ENV,
                e,
            )
            self._ready.set()
            return
        start_runtime.argtypes = []
        start_runtime.restype = None

        def run() -> None:
            t0 = time.perf_counter()
            start_runtime()
            self._side_thread_s = time.perf_counter() - t0
            self._ready.set()
            # Stay alive, idle: the runtime may keep state that belongs to the thread
            # that started it.
            threading.Event().wait()

        threading.Thread(target=run, name="spyre-card-start", daemon=True).start()
        logger.info("%s=1: starting the card on a side thread (world size 1).", _ENV)

    def seed_host_only(self, seed: int | None) -> None:
        """``set_random_seed`` without the card's generators; ``join`` seeds those."""
        self.seed = seed
        spyre = torch.spyre
        is_in_bad_fork = spyre._is_in_bad_fork
        # torch.manual_seed skips a custom device that reports a bad fork.
        spyre._is_in_bad_fork = lambda: True
        try:
            set_random_seed(seed)
        finally:
            spyre._is_in_bad_fork = is_in_bad_fork

    def join(self) -> None:
        if self.joined:
            return
        self.joined = True
        t0 = time.perf_counter()
        self._ready.wait()
        waited = time.perf_counter() - t0
        # The Python side of torch-spyre's start (pending device, initialized flag);
        # the runtime itself is already up, so start_runtime returns at once.
        torch.spyre._impl._lazy_init()
        if self.seed is not None:
            torch.spyre.manual_seed_all(self.seed)

        worker = self.worker
        snapshot = MemorySnapshot(device=worker._spyre_memory_device())
        if snapshot.torch_allocated:
            raise RuntimeError(
                f"{_ENV}=1: {snapshot.torch_allocated} bytes were on the card before the "
                f"memory snapshot, so it would undercount the free memory. Run without "
                f"{_ENV}."
            )
        worker.init_snapshot = snapshot
        worker.requested_memory = request_memory(snapshot, worker.cache_config)
        logger.debug("worker init memory snapshot: %r", snapshot)
        side = "n/a" if self._side_thread_s is None else f"{self._side_thread_s:.2f}s"
        logger.info(
            "%s=1: card ready; side thread start %s, joined %.2fs after init_device "
            "began it, waited %.2fs for it.",
            _ENV,
            side,
            time.perf_counter() - self._started_at,
            waited,
        )


def start(worker: Any) -> CardStart | None:
    """Start the card early when ``SPYRE_FAST_DIST_INIT`` applies; None otherwise."""
    mode = _mode(worker)
    if mode is None:
        return None
    card = CardStart(worker, mode)
    card.start()
    return card


def dist_backend(card: CardStart | None, default: str) -> str:
    """gloo only while the flag applies (no group at world size 1 needs spyreccl)."""
    return default if card is None else "gloo"


def defers_card_work(card: CardStart | None) -> bool:
    return card is not None and card.defers_card_work


@contextmanager
def join_after_weight_read(worker: Any) -> Iterator[None]:
    """Around ``load_model``: join the card start once the weights are read.

    The join runs in front of vLLM's ``process_weights_after_loading``, which follows
    the weight read in ``BaseModelLoader.load_model`` and comes before any upload. A
    loader that skips it gets the join after the load, where the snapshot check fails
    loudly instead of undercounting.
    """
    card = getattr(worker, "_fast_card_start", None)
    if card is None or card.joined or not card.defers_card_work:
        yield
        return
    from vllm.model_executor.model_loader import base_loader

    original = getattr(base_loader, "process_weights_after_loading", None)
    if original is None:
        logger.warning("%s=1: no hook after the weight read; joining before it.", _ENV)
        card.join()
        yield
        return

    def join_first(*args: Any, **kwargs: Any) -> Any:
        card.join()
        return original(*args, **kwargs)

    base_loader.process_weights_after_loading = join_first
    try:
        yield
    finally:
        base_loader.process_weights_after_loading = original
    card.join()
