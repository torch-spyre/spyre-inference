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

"""Unit tests for custom_ops/utils.py (convert utility and op registration).

The `convert` function and `_convert_op_func` handle device/dtype
conversion with the Spyre dtype-detour pattern. These tests run on CPU —
no Spyre device needed — and verify the core logic paths.
"""

import pytest
import torch

from spyre_inference.custom_ops.utils import (
    _convert_op_func,
    _convert_op_fake,
    convert,
    register,
)


class TestConvertOpFunc:
    """Direct tests of _convert_op_func (the real op implementation)."""

    def test_noop_same_device_same_dtype(self):
        """No-op when device and dtype are already correct."""
        t = torch.randn(3, 4, dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.float32)
        # Should return the same tensor (identity)
        assert result is t

    def test_dtype_change_on_cpu(self):
        """dtype change on CPU: float32 → float16."""
        t = torch.randn(2, 3, dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        assert result.shape == (2, 3)

    def test_dtype_change_preserves_values(self):
        """dtype change preserves tensor values within precision limits."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.float16)
        torch.testing.assert_close(
            result.float(), t, atol=1e-3, rtol=1e-3
        )

    def test_none_device_keeps_current(self):
        """device=None keeps the tensor on its current device."""
        t = torch.randn(4, dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=None, dtype=torch.float16)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float16

    def test_none_dtype_keeps_current(self):
        """dtype=None keeps the tensor's current dtype."""
        t = torch.randn(4, dtype=torch.float16, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=None)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_both_none_returns_original(self):
        """Both device=None and dtype=None returns the original tensor."""
        t = torch.randn(4, dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=None, dtype=None)
        assert result is t

    def test_int_dtype_conversion(self):
        """Integer dtype conversion: int32 → int64."""
        t = torch.tensor([1, 2, 3], dtype=torch.int32, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.int64)
        assert result.dtype == torch.int64
        torch.testing.assert_close(result, torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_float_to_int_conversion(self):
        """Float → int conversion (truncation)."""
        t = torch.tensor([1.7, 2.3, 3.9], dtype=torch.float32, device="cpu")
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.int64)
        assert result.dtype == torch.int64
        # torch.to truncates toward zero
        expected = torch.tensor([1, 2, 3], dtype=torch.int64)
        torch.testing.assert_close(result, expected)


class TestConvertOpFake:
    """Tests for the fake implementation (used by torch.compile)."""

    def test_fake_returns_correct_shape_dtype_device(self):
        """Fake impl returns tensor with correct metadata."""
        t = torch.randn(3, 4, dtype=torch.float32, device="cpu")
        result = _convert_op_fake(t, device=torch.device("cpu"), dtype=torch.float16)
        assert result.shape == (3, 4)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_fake_none_device_keeps_current(self):
        """Fake impl with device=None uses tensor's device."""
        t = torch.randn(2, dtype=torch.float32, device="cpu")
        result = _convert_op_fake(t, device=None, dtype=torch.float16)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float16

    def test_fake_none_dtype_keeps_current(self):
        """Fake impl with dtype=None uses tensor's dtype."""
        t = torch.randn(2, dtype=torch.float16, device="cpu")
        result = _convert_op_fake(t, device=torch.device("cpu"), dtype=None)
        assert result.dtype == torch.float16


class TestConvertPublicAPI:
    """Tests for the public convert() wrapper."""

    def test_none_input_returns_none(self):
        """None input short-circuits to None (no op call)."""
        result = convert(None, device="cpu", dtype=torch.float16)
        assert result is None

    def test_string_device_converted(self):
        """String device arg is properly converted to torch.device."""
        # Ensure register() is called so the op is available
        register()
        t = torch.randn(3, dtype=torch.float32, device="cpu")
        result = convert(t, device="cpu", dtype=torch.float16)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float16

    def test_torch_device_arg(self):
        """torch.device arg works directly."""
        register()
        t = torch.randn(3, dtype=torch.float32, device="cpu")
        result = convert(t, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16

    def test_no_args_noop(self):
        """convert with no device/dtype is a no-op."""
        register()
        t = torch.randn(3, dtype=torch.float32, device="cpu")
        result = convert(t)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"


class TestRegister:
    """Tests for the register() function."""

    def test_register_is_idempotent(self):
        """Calling register() multiple times doesn't raise."""
        register()
        register()  # second call should be a no-op due to lru_cache

    def test_op_available_after_register(self):
        """torch.ops.vllm.spyre_convert is available after registration."""
        register()
        assert hasattr(torch.ops.vllm, "spyre_convert")

    def test_registered_op_matches_func(self):
        """The registered op produces the same result as direct _convert_op_func."""
        register()
        t = torch.randn(4, dtype=torch.float32, device="cpu")
        direct = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.float16)
        via_op = torch.ops.vllm.spyre_convert(t, torch.device("cpu"), torch.float16)
        torch.testing.assert_close(direct, via_op)
