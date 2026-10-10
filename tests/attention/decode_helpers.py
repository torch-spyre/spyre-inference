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

import torch


def _decode_reference_fp32(
    query: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_ids: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    qpk: int,
    head_size: int,
) -> torch.Tensor:
    """Per-sequence softmax over each sequence's own blocks, no chunking.

    page_ids: [num_seqs, num_blocks]. mask: [num_seqs, num_blocks, block_size].
    """
    num_seqs, num_blocks = page_ids.shape
    out = torch.zeros(num_seqs, num_kv_heads * qpk, head_size, dtype=torch.float32)
    for s in range(num_seqs):
        q = query[s].reshape(num_kv_heads, qpk, head_size)
        k = torch.cat([k_pages[page_ids[s, b]] for b in range(num_blocks)], dim=0)
        v = torch.cat([v_pages[page_ids[s, b]] for b in range(num_blocks)], dim=0)
        # [KV, qpk, kv_len]
        scores = torch.einsum("hqd,thd->hqt", q, k) * scale + mask[s].reshape(-1)
        probs = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("hqt,thd->hqd", probs, v).reshape(num_kv_heads * qpk, head_size)
    return out.reshape(num_seqs, num_kv_heads * qpk, head_size)
