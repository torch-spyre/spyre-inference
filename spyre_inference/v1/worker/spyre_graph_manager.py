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

"""Spyre Graph Manager runtime bucket dispatch for compilation.

This class manages runtime bucket dispatch: padding inputs to the nearest
compiled bucket size and trimming outputs back to actual size.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass

from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a Spyre compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


class SpyreGraphManager:
    """Manages bucket dispatch for Spyre compilation.

    Responsibilities:
    - Track compiled bucket sizes (from compile_sizes)
    - Find nearest bucket >= actual batch size
    - Provide padding/trimming coordination for execute_model
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        compilation_config = vllm_config.compilation_config
        sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
        self._bucket_sizes: list[int] = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_captured = False

        logger.info(
            "SpyreGraphManager initialized with %d bucket sizes: min=%d, max=%d",
            len(self._bucket_sizes),
            self._bucket_sizes[0] if self._bucket_sizes else 0,
            self._max_bucket_size,
        )

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def is_captured(self) -> bool:
        return self._is_captured

    def mark_captured(self) -> None:
        self._is_captured = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest bucket size >= num_tokens.

        Returns None if num_tokens exceeds max bucket (fallback to eager).
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
