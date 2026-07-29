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

"""Unit tests for spyre_inference/custom_ops/utils.py (convert utilities).

Tests target `_convert_op_func` (the opaque-op body) directly on CPU since
the Spyre dtype-detour logic is testable without hardware. The public
`convert()` wrapper is tested for None passthrough and custom-op dispatch.
"""

import pytest
import torch


class TestConvertOpFunc:
    """Tests for _convert_op_func — the kernel behind spyre_convert."""

    def test_noop_same_device_same_dtype(self):
        """No-op when device and dtype already match target."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(4, 8, dtype=torch.float32)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=torch.float32)
        # Should return the same tensor (identity, no copy)
        assert result is x

    def test_noop_none_device_none_dtype(self):
        """No-op when both device=None and dtype=None (defaults to current)."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(4, 8, dtype=torch.float16)
        result = _convert_op_func(x, device=None, dtype=None)
        assert result is x

    def test_dtype_change_cpu_to_cpu(self):
        """Dtype change on CPU (the normal non-Spyre case)."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(4, 8, dtype=torch.float32)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        torch.testing.assert_close(result.float(), x, atol=1e-3, rtol=1e-3)

    def test_dtype_change_preserves_shape(self):
        """Dtype conversion preserves tensor shape."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(3, 5, 7, dtype=torch.float32)
        result = _convert_op_func(x, device=None, dtype=torch.float16)
        assert result.shape == (3, 5, 7)

    def test_int_to_int64_conversion(self):
        """Int32 to int64 conversion (the _SpyreModelWrapper input path)."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=torch.int64)
        assert result.dtype == torch.int64
        assert torch.equal(result, x.to(torch.int64))

    def test_device_only_change(self):
        """Device change without dtype change (CPU to CPU is identity)."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(4, dtype=torch.float16)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=None)
        # Same device type → should be identity
        assert result is x

    def test_device_and_dtype_change_together(self):
        """Both device and dtype change in one call."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.randn(4, dtype=torch.float32)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_empty_tensor(self):
        """Empty tensor conversion does not crash."""
        from spyre_inference.custom_ops.utils import _convert_op_func

        x = torch.empty(0, 8, dtype=torch.float32)
        result = _convert_op_func(x, device=torch.device("cpu"), dtype=torch.float16)
        assert result.shape == (0, 8)
        assert result.dtype == torch.float16


class TestConvertPublicAPI:
    """Tests for the public convert() wrapper."""

    def test_none_passthrough(self):
        """convert(None) returns None without touching custom op."""
        from spyre_inference.custom_ops.utils import convert

        result = convert(None, device="cpu", dtype=torch.float16)
        assert result is None

    def test_string_device_converted(self):
        """String device argument is converted to torch.device."""
        from spyre_inference.custom_ops.utils import convert, register

        # Ensure the custom op is registered
        register()

        x = torch.randn(4, dtype=torch.float32)
        result = convert(x, device="cpu", dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_torch_device_argument(self):
        """torch.device argument works directly."""
        from spyre_inference.custom_ops.utils import convert, register

        register()

        x = torch.randn(4, dtype=torch.float32)
        result = convert(x, device=torch.device("cpu"), dtype=torch.float16)
        assert result.dtype == torch.float16

    def test_none_device_none_dtype_passthrough(self):
        """No device/dtype specified → no conversion."""
        from spyre_inference.custom_ops.utils import convert, register

        register()

        x = torch.randn(4, dtype=torch.float32)
        result = convert(x, device=None, dtype=None)
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"


class TestConvertOpFake:
    """Tests for _convert_op_fake (the fake/meta tensor implementation)."""

    def test_fake_produces_correct_shape_and_dtype(self):
        """Fake impl returns empty tensor with target shape/device/dtype."""
        from spyre_inference.custom_ops.utils import _convert_op_fake

        x = torch.randn(3, 5, dtype=torch.float32)
        result = _convert_op_fake(x, device=torch.device("cpu"), dtype=torch.float16)
        assert result.shape == (3, 5)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"

    def test_fake_defaults_to_input_when_none(self):
        """Fake impl uses input device/dtype when target is None."""
        from spyre_inference.custom_ops.utils import _convert_op_fake

        x = torch.randn(4, dtype=torch.float16)
        result = _convert_op_fake(x, device=None, dtype=None)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"


class TestRegister:
    """Tests for the register() function."""

    def test_register_idempotent(self):
        """register() can be called multiple times without error (lru_cache)."""
        from spyre_inference.custom_ops.utils import register

        register()
        register()  # second call is cached no-op

    def test_custom_op_available_after_register(self):
        """After register(), torch.ops.vllm.spyre_convert is callable."""
        from spyre_inference.custom_ops.utils import register

        register()
        assert hasattr(torch.ops.vllm, "spyre_convert")

        x = torch.randn(4, dtype=torch.float32)
        result = torch.ops.vllm.spyre_convert(x, torch.device("cpu"), torch.float16)
        assert result.dtype == torch.float16
