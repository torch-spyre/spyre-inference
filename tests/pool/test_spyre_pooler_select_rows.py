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

"""Test select_rows row-gather correctness, including the storage_offset
check that avoids an unnecessary clone for full, unsliced buffers."""

import sys

import pytest
import torch

from spyre_inference.v1.pool.spyre_pooler import select_rows


def test_select_rows_cpu_matches_index_select():
    """Off-device (CPU) path is a plain torch.index_select; no _base handling."""
    torch.manual_seed(0)
    hidden = torch.randn(10, 16, dtype=torch.float16)
    row_indices = torch.tensor([0, 3, 7])

    actual = select_rows(hidden, row_indices)
    expected = torch.index_select(hidden, 0, row_indices)

    torch.testing.assert_close(actual, expected)


def test_select_rows_cpu_handles_2d_pack_indices():
    """[B, L] index tensors (SpyreAllPool 'pack' case) are flattened before the
    gather."""
    torch.manual_seed(1)
    hidden = torch.randn(10, 16, dtype=torch.float16)
    row_indices = torch.tensor([[0, 1], [2, 3]])

    actual = select_rows(hidden, row_indices)
    expected = torch.index_select(hidden, 0, row_indices.reshape(-1))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_scheduled_tokens", [5, 10, 20])
def test_select_rows_on_spyre_from_full_prefix_slice(num_scheduled_tokens):
    """hidden_states arrives as a slice of a larger persistent buffer, mirroring
    vLLM's `hidden_states[:num_scheduled_tokens]` trim before pooling. A nonzero
    storage_offset (mid-buffer slice) triggers a clone; a zero-offset slice
    (num_scheduled_tokens == buffer_size) does not."""
    torch.manual_seed(2)
    buffer_size, hidden_size = 20, 64
    buffer = torch.randn(buffer_size, hidden_size, dtype=torch.float16)
    sliced = buffer[:num_scheduled_tokens]
    # Sanity: really is a view sharing storage, not a fresh tensor. Can't assert
    # `sliced._base is not None` here: real vLLM forward passes run under
    # `@torch.inference_mode()` (spyre_model_runner.py, gpu_model_runner.py), and
    # this suite's session fixture does the same -- inference mode skips the
    # autograd view-tracking that populates `_base`, even though the storage is
    # genuinely shared.
    assert sliced.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()

    row_indices = torch.tensor([0, num_scheduled_tokens - 1])
    expected = torch.index_select(sliced.clone(), 0, row_indices)

    actual = select_rows(sliced.to("spyre"), row_indices)

    assert actual.device.type == "spyre"
    torch.testing.assert_close(actual.cpu(), expected)


def test_select_rows_on_spyre_offset_matches_sliced_rows():
    """A mid-buffer slice has a nonzero storage_offset; select_rows clones it
    before index_select so the native scheduler can map the source correctly."""
    torch.manual_seed(3)
    buffer = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4).to(torch.float16)
    sliced = buffer[5:15]  # rows 5..14 of the base buffer
    row_indices = torch.tensor([0, 9])  # first/last row of the *slice*

    actual = select_rows(sliced.to("spyre"), row_indices)

    expected = torch.stack([buffer[5], buffer[14]])
    torch.testing.assert_close(actual.cpu(), expected)


def test_select_rows_on_spyre_non_view_hidden_states():
    """A fresh (non-view) tensor has storage_offset 0; select_rows skips the
    clone and indexes it directly."""
    torch.manual_seed(4)
    hidden = torch.randn(10, 16, dtype=torch.float16)
    assert hidden.storage_offset() == 0
    row_indices = torch.tensor([2, 5])

    actual = select_rows(hidden.to("spyre"), row_indices)

    expected = torch.index_select(hidden, 0, row_indices)
    torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
