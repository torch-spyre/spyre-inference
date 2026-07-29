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

"""Unit tests for spyre_inference/custom_ops/utils.py.

The `convert` function is the foundation of all device/dtype transfers in
SpyreModelRunner and custom ops. It routes through a registered custom op
(spyre_convert) to hide transfers from torch.compile.

These tests exercise the _convert_op_func logic on CPU (the underlying
implementation handles the Spyre dtype detour).
"""

import pytest
import torch

from spyre_inference.custom_ops.utils import _convert_op_func, convert, register


# ---------------------------------------------------------------------------
# _convert_op_func (raw implementation, no custom op dispatch)
# ---------------------------------------------------------------------------


class TestConvertOpFunc:
    """Test _convert_op_func logic paths on CPU."""

    def test_noop_when_device_and_dtype_match(self):
        """No-op when target device and dtype are already correct."""
        t = torch.randn(4, dtype=torch.float32)
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=torch.float32)
        # Should be the exact same object (no copy)
        assert result is t

    def test_noop_when_both_none(self):
        """device=None and dtype=None means keep everything → same object."""
        t = torch.randn(4, dtype=torch.float32)
        result = _convert_op_func(t, device=None, dtype=None)
        assert result is t

    def test_dtype_change_on_cpu(self):
        """dtype change on CPU tensor does not detour through device change."""
        t = torch.randn(4, dtype=torch.float32)
        result = _convert_op_func(t, device=None, dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        torch.testing.assert_close(result.float(), t, atol=1e-3, rtol=1e-3)

    def test_device_change_same_dtype(self):
        """Device change without dtype change (CPU→CPU is a no-op detectable)."""
        t = torch.randn(4, dtype=torch.float32)
        # Targeting same device type → should be identity
        result = _convert_op_func(t, device=torch.device("cpu"), dtype=None)
        assert result is t

    def test_dtype_and_device_change_cpu_to_cpu(self):
        """Simultaneous dtype + device change (both CPU, just dtype differs)."""
        t = torch.randn(4, dtype=torch.float32)
        result = _convert_op_func(
            t, device=torch.device("cpu"), dtype=torch.float16
        )
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_preserves_shape(self):
        """Output shape matches input shape for all conversions."""
        t = torch.randn(3, 5, 7, dtype=torch.float32)
        result = _convert_op_func(t, device=None, dtype=torch.float16)
        assert result.shape == (3, 5, 7)

    def test_int_to_int64(self):
        """int32 → int64 conversion (the most common use in the wrapper)."""
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        result = _convert_op_func(t, device=None, dtype=torch.int64)
        assert result.dtype == torch.int64
        assert (result == t.to(torch.int64)).all()

    def test_bool_tensor(self):
        """Bool tensor conversion to int."""
        t = torch.tensor([True, False, True], dtype=torch.bool)
        result = _convert_op_func(t, device=None, dtype=torch.int64)
        assert result.dtype == torch.int64
        assert result.tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------
# convert() wrapper function
# ---------------------------------------------------------------------------


class TestConvertWrapper:
    """Test the convert() Python wrapper (routes through custom op)."""

    def test_none_passthrough(self):
        """None input returns None without calling the custom op."""
        result = convert(None, device="cpu", dtype=torch.float16)
        assert result is None

    def test_string_device_accepted(self):
        """Device can be passed as a string (converted internally)."""
        # This exercises the isinstance(device, str) → torch.device path
        t = torch.randn(4, dtype=torch.float32)
        result = convert(t, device="cpu", dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_torch_device_accepted(self):
        """Device can be passed as torch.device."""
        t = torch.randn(4, dtype=torch.float32)
        result = convert(t, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16

    def test_no_args_is_noop(self):
        """convert(t) with no device/dtype is identity."""
        t = torch.randn(4, dtype=torch.float32)
        result = convert(t)
        torch.testing.assert_close(result, t)

    def test_dtype_only(self):
        """convert(t, dtype=...) changes dtype, keeps device."""
        t = torch.randn(4, dtype=torch.float32)
        result = convert(t, dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device == t.device


# ---------------------------------------------------------------------------
# register() idempotency
# ---------------------------------------------------------------------------


class TestRegister:
    """Test that register() can be called multiple times (lru_cache)."""

    def test_register_idempotent(self):
        """Calling register() twice does not raise."""
        register()
        register()
        # The op should be callable
        t = torch.randn(4)
        result = torch.ops.vllm.spyre_convert(t, None, None)
        torch.testing.assert_close(result, t)
