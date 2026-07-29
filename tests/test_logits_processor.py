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

"""Unit tests for SpyreLogitsProcessor custom op.

Tests correctness and contiguous output guarantee of the Spyre OOT
LogitsProcessor replacement.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from spyre_inference.custom_ops import register_all


@pytest.fixture()
def _vllm_config(monkeypatch):
    """Set up a minimal vLLM config context for OOT instantiation tests.

    This mirrors the `default_vllm_config` fixture from the spyre testing
    plugin but is self-contained for environments where the plugin is not
    installed.
    """
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.forward_context import set_forward_context
    from vllm.platforms import PlatformEnum, current_platform

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)
    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield


class TestSpyreLogitsProcessorOOTRegistration:
    """Verify SpyreLogitsProcessor OOT registration and class swap."""

    @pytest.mark.usefixtures("_vllm_config")
    def test_oot_class_swap(self):
        """LogitsProcessor.__new__ should produce SpyreLogitsProcessor."""
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        # LogitsProcessor requires some args — use kwargs matching its __init__
        layer = LogitsProcessor(vocab_size=32000)
        assert isinstance(layer, SpyreLogitsProcessor)

    @pytest.mark.usefixtures("_vllm_config")
    def test_forward_method_selection(self):
        """dispatch_forward should have selected the OOT _get_logits path."""
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        layer = LogitsProcessor(vocab_size=32000)
        # The SpyreLogitsProcessor overrides _get_logits, not forward_oot,
        # so check that the class is correct (dispatch happens via inheritance).
        assert type(layer).__name__ == "SpyreLogitsProcessor"


class TestSpyreLogitsProcessorContiguousOutput:
    """Test that _get_logits returns a contiguous tensor."""

    def test_contiguous_output_from_noncontiguous_input(self):
        """The override ensures logits are contiguous even when super() returns
        a non-contiguous tensor (e.g. from a transpose or slice)."""
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        # Create a SpyreLogitsProcessor with a mock lm_head
        processor = SpyreLogitsProcessor.__new__(SpyreLogitsProcessor)
        processor.scale = 1.0
        processor.soft_cap = None
        processor.use_gather = True

        # Create a non-contiguous tensor that simulates what super()._get_logits
        # would return (e.g. from a matmul with transposed weight)
        batch_size, vocab_size = 4, 32000
        raw_logits = torch.randn(vocab_size, batch_size).t()  # non-contiguous
        assert not raw_logits.is_contiguous()

        # Patch the parent _get_logits to return our non-contiguous tensor
        with patch(
            "vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
            return_value=raw_logits,
        ):
            lm_head = MagicMock()
            result = processor._get_logits(
                hidden_states=torch.randn(batch_size, 768),
                lm_head=lm_head,
                embedding_bias=None,
            )

        assert result is not None
        assert result.is_contiguous()
        torch.testing.assert_close(result, raw_logits.contiguous())

    def test_none_logits_passthrough(self):
        """When super()._get_logits returns None, the override returns None."""
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        processor = SpyreLogitsProcessor.__new__(SpyreLogitsProcessor)
        processor.scale = 1.0
        processor.soft_cap = None
        processor.use_gather = True

        with patch(
            "vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
            return_value=None,
        ):
            lm_head = MagicMock()
            result = processor._get_logits(
                hidden_states=torch.randn(2, 768),
                lm_head=lm_head,
                embedding_bias=None,
            )

        assert result is None

    def test_already_contiguous_is_noop(self):
        """When logits are already contiguous, .contiguous() is a no-op."""
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        processor = SpyreLogitsProcessor.__new__(SpyreLogitsProcessor)
        processor.scale = 1.0
        processor.soft_cap = None
        processor.use_gather = True

        contiguous_logits = torch.randn(4, 32000)
        assert contiguous_logits.is_contiguous()

        with patch(
            "vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits",
            return_value=contiguous_logits,
        ):
            lm_head = MagicMock()
            result = processor._get_logits(
                hidden_states=torch.randn(4, 768),
                lm_head=lm_head,
                embedding_bias=None,
            )

        assert result is not None
        assert result.is_contiguous()
        # Should be the same data (contiguous() on contiguous tensor is self)
        assert result.data_ptr() == contiguous_logits.data_ptr()
