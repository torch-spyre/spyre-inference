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

"""Unit tests for SpyreLogitsProcessor (custom_ops/logits_processor.py).

SpyreLogitsProcessor overrides _get_logits to add .contiguous() — a workaround
for a torch-spyre compile issue with in-place scale. These tests verify the
OOT registration and the contiguous behavior on CPU.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class TestSpyreLogitsProcessorRegistration:
    """Test OOT registration of SpyreLogitsProcessor."""

    def test_is_subclass_of_logits_processor(self):
        """SpyreLogitsProcessor inherits from LogitsProcessor."""
        assert issubclass(SpyreLogitsProcessor, LogitsProcessor)

    def test_has_get_logits_method(self):
        """SpyreLogitsProcessor overrides _get_logits."""
        assert hasattr(SpyreLogitsProcessor, "_get_logits")
        # The method should be defined on SpyreLogitsProcessor itself
        assert "_get_logits" in SpyreLogitsProcessor.__dict__


class TestSpyreLogitsProcessorBehavior:
    """Test the .contiguous() workaround behavior."""

    def test_output_is_contiguous(self):
        """_get_logits returns a contiguous tensor regardless of input layout."""
        # Create a non-contiguous tensor to simulate strided views from Spyre
        base = torch.randn(4, 8, dtype=torch.float32)
        # Transpose makes it non-contiguous
        hidden_states = base.t()  # shape [8, 4], non-contiguous
        assert not hidden_states.is_contiguous()

        # Create a mock lm_head that returns a non-contiguous result
        # (simulating the parent's behavior before .contiguous())
        mock_lm_head = MagicMock(spec=VocabParallelEmbedding)

        # We need to test _get_logits directly, so we mock super()._get_logits
        # to return a non-contiguous tensor
        non_contiguous_logits = torch.randn(4, 10).t()[:, :4]  # non-contiguous slice
        assert not non_contiguous_logits.is_contiguous()

        with patch.object(
            LogitsProcessor, "_get_logits", return_value=non_contiguous_logits
        ):
            processor = SpyreLogitsProcessor(
                vocab_size=10,
                org_vocab_size=10,
                scale=1.0,
            )
            result = processor._get_logits(
                hidden_states=torch.randn(4, 8),
                lm_head=mock_lm_head,
                embedding_bias=None,
            )

        assert result is not None
        assert result.is_contiguous()

    def test_none_logits_passthrough(self):
        """_get_logits passes through None when parent returns None."""
        with patch.object(LogitsProcessor, "_get_logits", return_value=None):
            processor = SpyreLogitsProcessor(
                vocab_size=10,
                org_vocab_size=10,
                scale=1.0,
            )
            result = processor._get_logits(
                hidden_states=torch.randn(4, 8),
                lm_head=MagicMock(),
                embedding_bias=None,
            )

        assert result is None

    def test_already_contiguous_tensor(self):
        """_get_logits still works with already-contiguous tensors."""
        contiguous_logits = torch.randn(4, 10)
        assert contiguous_logits.is_contiguous()

        with patch.object(
            LogitsProcessor, "_get_logits", return_value=contiguous_logits
        ):
            processor = SpyreLogitsProcessor(
                vocab_size=10,
                org_vocab_size=10,
                scale=1.0,
            )
            result = processor._get_logits(
                hidden_states=torch.randn(4, 8),
                lm_head=MagicMock(),
                embedding_bias=None,
            )

        assert result is not None
        assert result.is_contiguous()

    def test_values_preserved_after_contiguous(self):
        """Tensor values are preserved after .contiguous() call."""
        original_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Create a non-contiguous view
        non_contig = original_data.t()

        with patch.object(
            LogitsProcessor, "_get_logits", return_value=non_contig
        ):
            processor = SpyreLogitsProcessor(
                vocab_size=3,
                org_vocab_size=3,
                scale=1.0,
            )
            result = processor._get_logits(
                hidden_states=torch.randn(2, 4),
                lm_head=MagicMock(),
                embedding_bias=None,
            )

        torch.testing.assert_close(result, non_contig.contiguous())
