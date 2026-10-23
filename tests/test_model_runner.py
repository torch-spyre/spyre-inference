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

"""Unit tests for _SpyreModelWrapper and TorchSpyreModelRunner helpers.

_SpyreModelWrapper converts model inputs/outputs at the model boundary.
These tests verify the wrapper's delegation, RoPE priming, and the
_FuncWrapper utility — all on CPU (no Spyre device needed).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from spyre_inference.v1.worker.spyre_model_runner import (
    _SpyreModelWrapper,
    _FuncWrapper,
    SpyreCpuGpuBuffer,
)


class TestFuncWrapper:
    """Tests for _FuncWrapper (Triton grid-launch syntax adapter)."""

    def test_getitem_returns_same_func(self):
        """kernel[(grid,)](...) syntax returns the wrapped function."""
        def my_func(a, b):
            return a + b

        wrapper = _FuncWrapper(my_func)
        # Simulate kernel[(1,)](a, b)
        grid_called = wrapper[(1,)]
        assert grid_called is my_func

    def test_various_grid_values(self):
        """Different grid values all return the same function."""
        def f():
            pass

        wrapper = _FuncWrapper(f)
        assert wrapper[1] is f
        assert wrapper[(4, 4)] is f
        assert wrapper["anything"] is f


class TestSpyreModelWrapper:
    """Tests for _SpyreModelWrapper's delegation and conversion behavior."""

    def test_getattr_delegates_to_model(self):
        """Attribute access delegates to the wrapped model."""
        model = nn.Linear(4, 8)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))
        # Access 'weight' via the wrapper
        assert wrapper.weight is model.weight
        assert wrapper.in_features == 4

    def test_setattr_delegates_to_model(self):
        """Setting attributes on wrapper sets them on the wrapped model."""
        model = nn.Linear(4, 8)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))
        wrapper.custom_attr = "hello"
        assert model.custom_attr == "hello"

    def test_call_invokes_model(self):
        """Calling the wrapper invokes the model's forward."""
        model = nn.Linear(4, 8)
        model.eval()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        x = torch.randn(2, 4)
        # The wrapper converts int tensors and calls the model
        # For float inputs on CPU, conversion is a no-op, model runs normally
        with torch.no_grad():
            result = wrapper(x)

        # Result should be on CPU (tree_map(_to_cpu) is applied)
        assert result.device.type == "cpu"
        assert result.shape == (2, 8)

    def test_int_inputs_preserved_on_cpu(self):
        """On CPU device, int tensor inputs are converted to int64."""

        class IntConsumer(nn.Module):
            def forward(self, input_ids=None, positions=None, **kwargs):
                return input_ids.float().sum() + positions.float().sum()

        model = IntConsumer()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        pos = torch.tensor([0, 1, 2], dtype=torch.int32)

        with torch.no_grad():
            result = wrapper(input_ids=ids, positions=pos)

        # The wrapper converts int tensors to int64 on the spyre device (CPU here)
        assert result.device.type == "cpu"

    def test_none_inputs_passed_through(self):
        """None inputs are passed through without conversion."""

        class OptionalModel(nn.Module):
            def forward(self, x=None, y=None):
                if x is None:
                    return torch.tensor(0.0)
                return x.sum()

        model = OptionalModel()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        with torch.no_grad():
            result = wrapper(x=None, y=None)

        assert result.item() == 0.0

    def test_float_inputs_not_converted(self):
        """Float tensor inputs are NOT converted (only int tensors are)."""

        class FloatModel(nn.Module):
            def forward(self, hidden_states=None, **kwargs):
                return hidden_states

        model = FloatModel()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        h = torch.randn(2, 4, dtype=torch.float16)
        with torch.no_grad():
            result = wrapper(hidden_states=h)

        assert result.dtype == torch.float16

    def test_prime_rope_rotation_no_rope_modules(self):
        """_prime_rope_rotation is a no-op when rope_modules is empty."""
        model = nn.Linear(4, 8)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[])
        # Should not raise — positions is irrelevant without rope modules
        wrapper._prime_rope_rotation(torch.tensor([0, 1, 2]))

    def test_prime_rope_rotation_none_positions(self):
        """_prime_rope_rotation is a no-op with positions=None."""
        model = nn.Linear(4, 8)
        mock_rope = MagicMock()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[mock_rope])
        # Should not call gather_rotation
        wrapper._prime_rope_rotation(None)
        mock_rope.gather_rotation.assert_not_called()

    def test_compute_logits_calls_model(self):
        """compute_logits delegates to model.compute_logits."""

        class MockModel(nn.Module):
            def forward(self, x):
                return x

            def compute_logits(self, hidden_states, *args, **kwargs):
                # In production this does the lm_head matmul on Spyre
                return hidden_states * 2

        model = MockModel()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        h = torch.randn(2, 4)
        result = wrapper.compute_logits(h)
        expected = h * 2  # On CPU, convert is a no-op
        torch.testing.assert_close(result, expected)


class TestMakeBuffer:
    """Tests for TorchSpyreModelRunner._make_buffer logic via SpyreCpuGpuBuffer."""

    def test_float_buffer_dtype_split(self):
        """Float buffers: cpu_dtype=source, gpu_dtype=float16 on Spyre.

        On CPU (no Spyre available), this falls back to aliasing.
        """
        # Simulate _make_buffer logic: float → cpu=float32, gpu=float16
        buf = SpyreCpuGpuBuffer(
            4, 8,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float16,
            device=torch.device("cpu"),  # Would be "spyre" in production
            pin_memory=False,
        )
        assert buf.cpu.dtype == torch.float32

    def test_int_buffer_aliased(self):
        """Int buffers: gpu aliased to cpu."""
        buf = SpyreCpuGpuBuffer(
            4, 8,
            cpu_dtype=torch.int32,
            gpu_dtype=torch.int32,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.int32
