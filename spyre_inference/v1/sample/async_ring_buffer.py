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

import contextlib
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Generator

import numpy as np
import torch


class AsyncRingBuffer(ABC):
    """Pre-generates data rows on a background thread via a ring buffer.

    Maintains a contiguous ``(S, V)`` buffer (``S = scale * max_batch_size``)
    and two shared counters:

    * ``_read_pos`` — next row index the consumer will read from.
    * ``_tail`` — upper bound (in unwrapped space) up to which the consumer
      may read without stalling.

    On init the buffer is fully filled and ``_tail = S``.  The consumer
    advances ``_read_pos`` after each call; when it approaches the end of the
    buffer it wraps back to 0.  Each consumed segment is enqueued for the
    background thread to refill, which increments ``_tail`` once done.

    Storage is a NumPy array shared with a Torch CPU view.  The producer
    thread must not call Torch ops: under the Spyre plugin, background-thread
    Torch tensor mutations can abort the producer, which then deadlocks
    ``borrow_rows`` after the first wrap.

    Args:
        vocab_size: Number of columns ``V``.
        max_batch_size: Maximum rows per :meth:`get_rows` call ``B``.
        scale: Buffer depth multiplier; ``S = scale * B``.  Must be >= 2 so
            there is always at least one full batch of pre-filled rows ahead
            of the consumer.
    """

    def __init__(
        self,
        vocab_size: int,
        max_batch_size: int,
        scale: int = 4,
    ) -> None:
        assert scale >= 2, "scale must be >= 2"
        self._V = vocab_size
        self._B = max_batch_size
        self._S = scale * max_batch_size

        # NumPy backing store; Torch view shares the same memory for zero-copy
        # borrows. Producer refills via NumPy only (see class docstring).
        self._np = np.empty((self._S, self._V), dtype=np.float32)
        self._buf = torch.from_numpy(self._np)

        self._error: BaseException | None = None

        # first-time buffer initialization
        self._refill_slice(0, self._S)
        self._tail: int = self._S
        self._read_pos: int = 0

        # _tail and _read_pos are guarded by _cond.
        self._cond = threading.Condition(threading.Lock())

        # Refill requests: (start, end, wrap)
        self._refill_q: queue.Queue[tuple[int, int, bool] | None] = queue.Queue()

        self._thread = threading.Thread(
            target=self._produce,
            name="async-ring-buffer",
            daemon=True,
        )
        self._thread.start()

    @abstractmethod
    def _refill_slice(self, start: int, end: int) -> None:
        """Fill ``self._np[start:end]`` with fresh values in-place (NumPy only)."""
        ...

    @property
    def vocab_size(self) -> int:
        return self._V

    def _raise_if_producer_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("async ring buffer producer failed") from self._error
        if not self._thread.is_alive() and self._error is None:
            # Thread exited without recording an error (e.g. unexpected break).
            raise RuntimeError("async ring buffer producer thread is not alive")

    @contextlib.contextmanager
    def borrow_rows(self, n: int) -> Generator[torch.Tensor, None, None]:
        """Context manager that yields a zero-copy ``(n, V)`` view.

        The backing rows are released for refill automatically when the
        ``with`` block exits, even if an exception is raised.  The view
        must not be used after the block.

        Args:
            n: Number of rows to borrow.  Must satisfy ``1 <= n <= B``.

        Raises:
            ValueError: If ``n`` is outside the valid range.

        Example::

            with buf.borrow_rows(batch_size) as noise:
                tokens = probs.div(noise).argmax(dim=-1)
        """
        if n > self._B or n < 1:
            raise ValueError(f"n (got {n}) must satisfy 1 <= n <= {self._B} (max_batch_size)")

        start = self._read_pos
        end = start + n

        # wait for the producer to fill up at least n many values ahead
        with self._cond:
            while self._tail < end:
                self._raise_if_producer_failed()
                self._cond.wait(timeout=1.0)
            self._raise_if_producer_failed()

        # get view (zero-copy into the shared Torch/NumPy buffer)
        view = self._buf[start:end]

        wrap: bool = end > self._S - self._B
        if wrap:
            with self._cond:
                self._tail -= self._S

            self._read_pos = 0
        else:
            self._read_pos = end

        try:
            # yield view to outside consumer
            yield view
        finally:
            # issue refill request once view has been consumed and returned
            self._refill_q.put((start, end, wrap))

    def _produce(self) -> None:
        try:
            while True:
                req = self._refill_q.get()

                # handle termination signal
                if req is None:
                    break

                # refill buffer (NumPy only — see class docstring)
                start, end, wrap = req
                self._refill_slice(start, end)

                increment = (self._S - start) if wrap else (end - start)
                with self._cond:
                    self._tail += increment
                    self._cond.notify_all()
        except BaseException as exc:
            self._error = exc
            with self._cond:
                self._cond.notify_all()
            raise

    def shutdown(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._refill_q.put(None)
        self._thread.join()


class AsyncExponential_RingBuffer(AsyncRingBuffer):
    """Ring buffer that pre-generates exponential log noise via Exp(1) then log."""

    def _refill_slice(self, start: int, end: int) -> None:
        # Match torch.Tensor.exponential_() default (rate=1) then log_().
        n = end - start
        out = self._np[start:end]
        out[:] = np.random.exponential(scale=1.0, size=(n, self._V))
        np.log(out, out=out)


class _AsyncCounterRingBuffer(AsyncRingBuffer):
    """Ring buffer that fills each row with the cumulative row index.

    Used in tests to verify that consumers receive the correct rows in order
    without repeating any.
    """

    def __init__(self, vocab_size: int, max_batch_size: int, scale: int = 4) -> None:
        self._total_generated: int = 0
        super().__init__(vocab_size, max_batch_size, scale)

    def _refill_slice(self, start: int, end: int) -> None:
        n = end - start
        for i in range(n):
            self._np[start + i, :] = self._total_generated
            self._total_generated += 1
