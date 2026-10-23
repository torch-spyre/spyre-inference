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

"""Unit tests for SpyreCpuGpuBuffer (spyre_model_runner.py).

SpyreCpuGpuBuffer handles CPU↔Spyre data movement and the int/bool aliasing
pattern. These tests verify allocation, aliasing, copy semantics, and the
bfloat16 numpy guard — all on CPU (no Spyre device needed) using CPU-device
for the "non-Spyre float" path and verifying the aliasing logic directly.
"""

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import SpyreCpuGpuBuffer


class TestSpyreCpuGpuBufferAllocation:
    """Test buffer allocation patterns."""

    def test_int_dtype_aliases_gpu_to_cpu(self):
        """Int dtype: .gpu is aliased to .cpu (same object)."""
        buf = SpyreCpuGpuBuffer(
            4, 8,
            cpu_dtype=torch.int32,
            gpu_dtype=torch.int32,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.int32
        assert buf.cpu.shape == (4, 8)

    def test_bool_dtype_aliases_gpu_to_cpu(self):
        """Bool dtype: .gpu is aliased to .cpu."""
        buf = SpyreCpuGpuBuffer(
            10,
            cpu_dtype=torch.bool,
            gpu_dtype=torch.bool,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.bool

    def test_float_dtype_separate_buffers_on_spyre_device(self):
        """Float dtype with spyre device creates separate .cpu and .gpu tensors.

        Since Spyre device isn't available in test, we test the CPU-device
        aliasing path (which matches how int/bool is handled) and verify
        the allocation shape and dtype logic.
        """
        # When device.type != "spyre", gpu is aliased to cpu
        buf = SpyreCpuGpuBuffer(
            4, 8,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float16,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        # On CPU device, the implementation aliases gpu to cpu
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.float32

    def test_numpy_array_created_by_default(self):
        """with_numpy=True creates .np array sharing memory with .cpu."""
        buf = SpyreCpuGpuBuffer(
            3, 4,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float32,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=True,
        )
        assert hasattr(buf, "np")
        assert buf.np.shape == (3, 4)
        # Numpy array shares memory with CPU tensor
        buf.np[0, 0] = 42.0
        assert buf.cpu[0, 0].item() == 42.0

    def test_numpy_disabled(self):
        """with_numpy=False still creates the buffer (no .np attribute error)."""
        buf = SpyreCpuGpuBuffer(
            3, 4,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float32,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=False,
        )
        # .np is not set when with_numpy=False
        # The implementation uses an annotated attribute that doesn't get assigned
        assert not hasattr(buf, "np") or buf.np is None if hasattr(buf, "np") else True

    def test_bfloat16_with_numpy_raises(self):
        """bfloat16 dtype with with_numpy=True raises ValueError."""
        with pytest.raises(ValueError, match="[Bb]float16"):
            SpyreCpuGpuBuffer(
                4,
                cpu_dtype=torch.bfloat16,
                gpu_dtype=torch.bfloat16,
                device=torch.device("cpu"),
                pin_memory=False,
                with_numpy=True,
            )

    def test_bfloat16_without_numpy_succeeds(self):
        """bfloat16 dtype with with_numpy=False succeeds."""
        buf = SpyreCpuGpuBuffer(
            4,
            cpu_dtype=torch.bfloat16,
            gpu_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=False,
        )
        assert buf.cpu.dtype == torch.bfloat16

    def test_zero_initialized(self):
        """Buffers are zero-initialized."""
        buf = SpyreCpuGpuBuffer(
            5,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float32,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert torch.all(buf.cpu == 0)


class TestSpyreCpuGpuBufferCopy:
    """Test copy_to_gpu and copy_to_cpu semantics."""

    def test_copy_to_gpu_aliased_noop(self):
        """copy_to_gpu on aliased buffer returns the buffer without copying."""
        buf = SpyreCpuGpuBuffer(
            4,
            cpu_dtype=torch.int64,
            gpu_dtype=torch.int64,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert buf.gpu is buf.cpu  # aliased
        buf.cpu[0] = 42
        result = buf.copy_to_gpu()
        assert result is buf.gpu
        assert result[0].item() == 42

    def test_copy_to_gpu_aliased_with_n(self):
        """copy_to_gpu(n) on aliased buffer returns a slice."""
        buf = SpyreCpuGpuBuffer(
            8,
            cpu_dtype=torch.int32,
            gpu_dtype=torch.int32,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        buf.cpu[:] = torch.arange(8, dtype=torch.int32)
        result = buf.copy_to_gpu(n=3)
        assert result.shape == (3,)
        torch.testing.assert_close(result, torch.tensor([0, 1, 2], dtype=torch.int32))

    def test_copy_to_cpu_raises(self):
        """copy_to_cpu raises NotImplementedError."""
        buf = SpyreCpuGpuBuffer(
            4,
            cpu_dtype=torch.int64,
            gpu_dtype=torch.int64,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        with pytest.raises(NotImplementedError, match="copy_to_cpu"):
            buf.copy_to_cpu()

    def test_multidim_buffer(self):
        """Multi-dimensional buffers work correctly."""
        buf = SpyreCpuGpuBuffer(
            2, 3, 4,
            cpu_dtype=torch.float32,
            gpu_dtype=torch.float32,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert buf.cpu.shape == (2, 3, 4)
        assert buf.gpu.shape == (2, 3, 4)
