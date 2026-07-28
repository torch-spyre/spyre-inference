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

"""Unit tests for _SpyreModelWrapper (spyre_model_runner.py).

_SpyreModelWrapper handles:
- Input conversion (CPU → Spyre int64) for integer tensors
- Output conversion (Spyre → CPU) for all outputs
- RoPE rotation priming via forward context
- compute_logits delegation with device conversion
- Transparent attribute access delegation to the wrapped model

All tests run on CPU — no Spyre device required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper


class TestSpyreModelWrapperInputConversion:
    """Test input integer conversion in __call__."""

    def test_int32_input_ids_converted_to_int64(self):
        """input_ids (int32) should be converted to int64 on the target device."""
        model = Mock()
        model.return_value = torch.randn(4, 64)  # output on "cpu"

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        wrapper(input_ids=input_ids, positions=positions)

        # Check that model was called with converted tensors
        _, kwargs = model.call_args
        assert kwargs["input_ids"].dtype == torch.int64
        assert kwargs["positions"].dtype == torch.int64

    def test_int64_input_stays_int64(self):
        """int64 inputs should remain int64 (no unnecessary conversion)."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        wrapper(input_ids=input_ids)

        _, kwargs = model.call_args
        assert kwargs["input_ids"].dtype == torch.int64

    def test_float_inputs_not_converted(self):
        """Float tensors should not be converted by the int-conversion logic."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        float_input = torch.randn(4, 64, dtype=torch.float16)
        wrapper(hidden_states=float_input)

        _, kwargs = model.call_args
        assert kwargs["hidden_states"].dtype == torch.float16

    def test_none_inputs_passed_through(self):
        """None values should pass through unchanged."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        wrapper(input_ids=None, positions=None)

        _, kwargs = model.call_args
        assert kwargs["input_ids"] is None
        assert kwargs["positions"] is None


class TestSpyreModelWrapperOutputConversion:
    """Test output conversion back to CPU."""

    def test_tensor_output_moved_to_cpu(self):
        """Model output tensors should be moved to CPU."""
        model = Mock()
        # Simulate model returning a CPU tensor (since we test on CPU)
        output_tensor = torch.randn(4, 64)
        model.return_value = output_tensor

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        result = wrapper(input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64))

        assert result.device == torch.device("cpu")

    def test_tuple_output_all_moved_to_cpu(self):
        """Tuple outputs should have all tensors moved to CPU."""
        model = Mock()
        model.return_value = (torch.randn(4, 64), torch.randn(4, 32))

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        result = wrapper(input_ids=torch.tensor([1, 2], dtype=torch.int64))

        assert isinstance(result, tuple)
        assert all(t.device == torch.device("cpu") for t in result)


class TestSpyreModelWrapperComputeLogits:
    """Test compute_logits delegation."""

    def test_compute_logits_delegates_to_model(self):
        """compute_logits should call the inner model's compute_logits."""
        model = Mock()
        model.compute_logits = Mock(return_value=torch.randn(4, 1000))

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        hidden_states = torch.randn(4, 64)
        result = wrapper.compute_logits(hidden_states)

        model.compute_logits.assert_called_once()

    def test_compute_logits_result_type(self):
        """compute_logits should return logits tensor."""
        model = Mock()
        expected_logits = torch.randn(4, 1000)
        model.compute_logits = Mock(return_value=expected_logits)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        hidden_states = torch.randn(4, 64)
        result = wrapper.compute_logits(hidden_states)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 1000)


class TestSpyreModelWrapperAttributeDelegation:
    """Test __getattr__ and __setattr__ delegation."""

    def test_getattr_delegates_to_inner_model(self):
        """Attribute access should delegate to the inner model."""
        model = nn.Linear(10, 5)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        # Access model's attributes through wrapper
        assert wrapper.in_features == 10
        assert wrapper.out_features == 5

    def test_setattr_delegates_to_inner_model(self):
        """Setting attributes should delegate to the inner model."""
        model = nn.Linear(10, 5)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        wrapper.custom_attr = "test_value"
        assert model.custom_attr == "test_value"

    def test_internal_attrs_not_delegated(self):
        """Internal _model and _spyre_device should not be delegated."""
        model = nn.Linear(10, 5)
        device = torch.device("cpu")
        wrapper = _SpyreModelWrapper(model, device)

        # Access internal attributes via object.__getattribute__
        assert object.__getattribute__(wrapper, "_model") is model
        assert object.__getattribute__(wrapper, "_spyre_device") == device


class TestSpyreModelWrapperRopePriming:
    """Test RoPE rotation priming."""

    def test_prime_rope_skips_when_positions_none(self):
        """_prime_rope_rotation is a no-op when positions is None."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        rope_module = Mock()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[rope_module])

        # Should not raise even with rope_modules configured
        wrapper(input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64), positions=None)

        # gather_rotation should NOT be called when positions is None
        rope_module.gather_rotation.assert_not_called()

    def test_prime_rope_skips_when_no_rope_modules(self):
        """_prime_rope_rotation is a no-op when rope_modules is empty."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[])

        # Should not raise
        wrapper(input_ids=torch.tensor([1, 2], dtype=torch.int64), positions=torch.tensor([0, 1]))

    def test_prime_rope_calls_gather_rotation(self):
        """When positions and rope_modules exist, gather_rotation is called."""
        model = Mock()
        model.return_value = torch.randn(4, 64)

        rope_module = Mock()
        rope_module._rope_key = "layer_0"
        rope_module.gather_rotation = Mock(return_value=None)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[rope_module])

        # Need to mock the forward context to make priming work
        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.is_forward_context_available",
            return_value=True,
        ):
            wrapper(
                input_ids=torch.tensor([1, 2], dtype=torch.int64),
                positions=torch.tensor([0, 1]),
            )

        rope_module.gather_rotation.assert_called_once()


class TestSpyreModelWrapperEdgeCases:
    """Edge cases and integration patterns."""

    def test_positional_args_converted(self):
        """Positional tensor args with int dtype should also be converted."""
        model = Mock()
        model.return_value = torch.randn(2, 32)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        int_arg = torch.tensor([5, 6], dtype=torch.int32)
        wrapper(int_arg, some_kwarg=torch.tensor([7, 8], dtype=torch.int32))

        args, kwargs = model.call_args
        assert args[0].dtype == torch.int64
        assert kwargs["some_kwarg"].dtype == torch.int64

    def test_non_tensor_inputs_passed_through(self):
        """Non-tensor inputs (int, str, None) should pass through unchanged."""
        model = Mock()
        model.return_value = torch.randn(1, 32)

        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        wrapper(batch_size=4, name="test", flag=None)

        _, kwargs = model.call_args
        assert kwargs["batch_size"] == 4
        assert kwargs["name"] == "test"
        assert kwargs["flag"] is None
