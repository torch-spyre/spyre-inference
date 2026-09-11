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

"""KV store, one kernel per cache layout. See ``SpyreAttentionImpl.kv_slot_views``."""


def reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    """Scatter K/V into the slot-major cache, whose rows are (page, token) pairs."""
    k_slots.index_copy_(0, slot_mapping, key)
    v_slots.index_copy_(0, slot_mapping, value)


def head_major_reshape_and_cache_kernel(key, value, k_slots, v_slots, slot_mapping):
    """Scatter K/V into the head-major cache, one index_copy_ per KV head.

    Unrolled rather than one copy over a flattened source: head-major puts a token's
    heads block_size rows apart, so no single copy addresses them, and every spelling
    of the reshape that would fails to lower at the decode width. The copies fuse into
    one kernel.

    ``slot_mapping`` is one int64 tensor per KV head, each its own allocation: an index
    tensor reaches the hardware as a tensor argument, so a row slice of a 2-D table
    would have its offset dropped (torch-spyre#3770). ``key``/``value`` must be their
    own allocations too -- see the clone in ``do_kv_cache_update``.
    """
    for h in range(len(slot_mapping)):
        rows = slot_mapping[h]
        k_slots.index_copy_(0, rows, key.select(1, h))
        v_slots.index_copy_(0, rows, value.select(1, h))
