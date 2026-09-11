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

"""Device layouts for the paged KV cache, and the stick geometry the index rows follow.

Two cache forms exist, selected by ``SPYRE_ATTN_HEAD_MAJOR_KV``:

* slot-major (default), allocated ``[num_blocks, block_size, num_kv_heads, head_size]``
* head-major, allocated ``[num_blocks, num_kv_heads, block_size, head_size]``

Both hold the same elements; they differ in which axis is contiguous within a block,
and hence in what a page gather costs. See ``ops.page_attn``.
"""

import torch

# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables.
INT32_ELEMS_PER_STICK = 32


def stick_aligned_len(n: int) -> int:
    """Round n up to a whole number of int32 sticks (see INT32_ELEMS_PER_STICK)."""
    return (n + INT32_ELEMS_PER_STICK - 1) // INT32_ELEMS_PER_STICK * INT32_ELEMS_PER_STICK


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


def head_major_kv_layout(num_rows: int, block_size: int, head_size: int, dtype: torch.dtype):
    """Tile-outermost layout for the head-major cache; `num_rows` counts (page, kv head).

    The indexed axis must sit at device position 0 or index_select costs the whole
    tensor, so the cache is materialised with this layout rather than viewed out of
    slot-major. The allocation is 4-D while the device layout keeps (page, kv head) as
    one extent: merging device dims is the direction torch-spyre lowers, splitting
    device dim 0 is not.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype, get_elem_in_stick

    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_rows, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )
