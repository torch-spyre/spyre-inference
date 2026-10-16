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

"""KV store into the head-major paged cache. See ``SpyreHeadMajorAttentionImpl``."""


def reshape_and_cache_head_major_kernel(key, value, k_rows, v_rows, row_index):
    """Store one head at a time: head-major puts a token's heads block_size rows apart.

    k/v_rows are [num_blocks * num_kv_heads * block_size, head_size] views; row_index is
    one [T] int64 tensor per KV head. A single index over a flattened (T, KV) source does
    not compile (UnalignedStickSplit on the merged row axis).
    """
    # `key` is a strided view of the fused QKV projection, and the per-head slice of one
    # stores wrong values on device, so the source is materialized here. Not contiguous():
    # at one token the view already reports contiguous. clone() does not fix it either.
    key = key * 1.0
    value = value * 1.0
    for h, idx in enumerate(row_index):
        k_rows.index_copy_(0, idx, key[:, h])
        v_rows.index_copy_(0, idx, value[:, h])
