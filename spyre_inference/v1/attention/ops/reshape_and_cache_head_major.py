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

import torch


def quantize_kv(x, scale, dtype):
    """``x / scale`` saturated to float8; ``spyre.qfp8ch`` on Spyre, where ``.to`` falls
    back to CPU and writes bytes in the wrong in-stick order."""
    fmax = torch.finfo(dtype).max
    if scale != 1.0:
        x = x / scale
    x = torch.clamp(x, -fmax, fmax)
    if x.device.type == "spyre":
        return torch.ops.spyre.qfp8ch(x)  # ty: ignore[invalid-argument-type]
    return x.to(dtype)


def reshape_and_cache_head_major_kernel(
    key, value, k_rows, v_rows, row_index, k_scale=None, v_scale=None
):
    """Store one head at a time: head-major puts a token's heads block_size rows apart.

    k/v_rows are [num_blocks * num_kv_heads * block_size, head_size] views; row_index is
    one [T] int64 tensor per KV head. A single index over a flattened (T, KV) source does
    not compile (UnalignedStickSplit on the merged row axis). With k/v_scale the rows are
    float8.
    """
    # `key` is a strided view of the fused QKV projection, and the per-head slice of one
    # stores wrong values on device, so the source is materialized here. Not contiguous():
    # at one token the view already reports contiguous. clone() does not fix it either.
    # Quantizing is an elementwise pass too, so it materializes the source the same way.
    if k_scale is None:
        key = key * 1.0
        value = value * 1.0
    else:
        key = quantize_kv(key, k_scale, k_rows.dtype)
        value = quantize_kv(value, v_scale, v_rows.dtype)
    for h, idx in enumerate(row_index):
        k_rows.index_copy_(0, idx, key[:, h])
        v_rows.index_copy_(0, idx, value[:, h])
