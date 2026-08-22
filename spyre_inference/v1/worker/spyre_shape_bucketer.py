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

"""Spyre shape bucketer for compilation warmup and runtime dispatch.

Manages a sorted set of pre-compiled bucket sizes and dispatches incoming
batch sizes to the nearest bucket >= actual num_tokens.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a Spyre compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


class SpyreShapeBucketer:
    """Dispatches runtime batch sizes to pre-compiled bucket sizes.

    Responsibilities:
    - Track compiled bucket sizes (from compile_sizes)
    - Find nearest bucket >= actual batch size
    - Provide padding coordination for execute_model
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        compilation_config = vllm_config.compilation_config
        sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
        self._bucket_sizes: list[int] = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False

        logger.info(
            "SpyreShapeBucketer initialized with %d bucket sizes: min=%d, max=%d",
            len(self._bucket_sizes),
            self._bucket_sizes[0] if self._bucket_sizes else 0,
            self._max_bucket_size,
        )

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def max_bucket_size(self) -> int:
        return self._max_bucket_size

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest bucket size >= num_tokens.

        Returns None if num_tokens exceeds the largest compiled bucket.
        The caller (execute_model) handles the None case by running the
        forward pass without bucket padding, which may trigger Dynamo
        recompilation for the unseen shape.
        """
        idx = bisect.bisect_left(self._bucket_sizes, num_tokens)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

    def dispatch(self, num_tokens: int) -> SpyreBucketDescriptor | None:
        """Compute padded batch descriptor for the given token count.

        Returns None if no suitable bucket exists.
        """
        padded = self.find_bucket(num_tokens)
        if padded is None:
            return None
        return SpyreBucketDescriptor(
            actual_num_tokens=num_tokens,
            padded_num_tokens=padded,
        )
