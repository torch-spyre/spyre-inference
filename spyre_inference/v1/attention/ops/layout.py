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

"""Device layout for the paged KV cache, and the stick geometry its index rows follow."""

import torch

# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables.
INT32_ELEMS_PER_STICK = 32


def slot_major_kv_layout(num_slots: int, num_kv_heads: int, head_size: int, dtype: torch.dtype):
    """Slot-axis-outermost layout. The default tiled layout spreads the slot index
    across two device dims, making the indirect store write to the wrong rows
    (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def head_major_kv_layout(num_pages: int, block_size: int, head_size: int, dtype: torch.dtype):
    """Head-major cache with the (page, kv_head) row the kernel gathers at device dim 0.

    ``num_pages`` counts folded rows, num_blocks * num_kv_heads. The indexed axis has to
    stay whole at device position 0, or the gather costs the whole tensor rather than one
    page (torch-spyre#3705).
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_pages, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )
