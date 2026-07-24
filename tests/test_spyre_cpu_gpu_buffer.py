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

"""Unit tests for SpyreCpuGpuBuffer async H2D staging copies."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import SpyreCpuGpuBuffer


def _float_buffer_without_device_alloc() -> SpyreCpuGpuBuffer:
    """Build a float dual-buffer without touching the Spyre runtime.

    ``SpyreCpuGpuBuffer`` only takes the async H2D path when ``.gpu is not
    .cpu``. For unit-testing the ``non_blocking=True`` contract we can
    assemble that state on CPU.
    """
    buf = SpyreCpuGpuBuffer.__new__(SpyreCpuGpuBuffer)
    buf.cpu = torch.zeros(8, dtype=torch.float32)
    buf.gpu = torch.zeros(8, dtype=torch.float16)  # distinct storage
    return buf


def _assert_non_blocking_true(mock_copy) -> None:
    mock_copy.assert_called_once()
    args, kwargs = mock_copy.call_args
    if "non_blocking" in kwargs:
        assert kwargs["non_blocking"] is True
    else:
        # copy_(self, src, non_blocking=True) or copy_(src, True)
        assert args[-1] is True


def test_float_copy_to_gpu_passes_non_blocking_true():
    """Float staging must request async H2D via non_blocking=True."""
    buf = _float_buffer_without_device_alloc()
    buf.cpu.copy_(torch.arange(8, dtype=torch.float32))

    with patch.object(torch.Tensor, "copy_", autospec=True) as mock_copy:
        # autospec'd Tensor.copy_ is invoked as copy_(src, *, non_blocking=...)
        # (receiver is bound; not passed as a positional arg).
        mock_copy.side_effect = lambda *args, **kwargs: buf.gpu
        out = buf.copy_to_gpu()

    assert out is buf.gpu
    _assert_non_blocking_true(mock_copy)


def test_float_copy_to_gpu_partial_slice_passes_non_blocking_true():
    """Partial float H2D also uses non_blocking=True."""
    buf = _float_buffer_without_device_alloc()

    with patch.object(torch.Tensor, "copy_", autospec=True) as mock_copy:
        mock_copy.side_effect = lambda *args, **kwargs: buf.gpu[:3]
        buf.copy_to_gpu(n=3)

    _assert_non_blocking_true(mock_copy)


def test_int_buffer_copy_to_gpu_is_alias_noop():
    """Int/bool buffers stay CPU-aliased; copy_to_gpu must not DMA."""
    buf = SpyreCpuGpuBuffer(
        8,
        cpu_dtype=torch.int32,
        gpu_dtype=torch.int32,
        device=torch.device("cpu"),
        pin_memory=False,
        with_numpy=False,
    )
    assert buf.gpu is buf.cpu

    with patch.object(torch.Tensor, "copy_", autospec=True) as mock_copy:
        out = buf.copy_to_gpu()

    assert out is buf.gpu
    mock_copy.assert_not_called()


def test_float_async_h2d_matches_after_synchronize():
    """End-to-end on Spyre: async float H2D + synchronize is correct."""
    if not hasattr(torch, "spyre"):
        pytest.skip("torch.spyre not available")
    try:
        if torch.spyre.device_count() < 1:
            pytest.skip("no Spyre devices visible")
    except Exception:
        pytest.skip("Spyre runtime unavailable")

    buf = SpyreCpuGpuBuffer(
        16,
        cpu_dtype=torch.float32,
        gpu_dtype=torch.float16,
        device=torch.device("spyre"),
        pin_memory=False,
        with_numpy=False,
    )
    expected = torch.arange(16, dtype=torch.float32)
    buf.cpu.copy_(expected)

    out = buf.copy_to_gpu()
    torch.spyre.synchronize(out.device)

    torch.testing.assert_close(out.cpu().float(), expected, atol=0, rtol=0)
