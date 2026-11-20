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

"""The sort-free top-k/top-p V2 falls back to on an extension without the csrc/ filters."""

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from spyre_inference.v1.worker.spyre.kernels import apply_top_k_top_p_sort_free


# Mixed per-row k, plus a no-top-k row (k == vocab) that takes the full sort.
# fp16-rounded logits tie; "wide" ties 300 tokens at the top, past the 2*k window.
# When a tie straddles the top-p cutoff, which of the tied tokens survives is
# arbitrary in upstream's unstable sort too, so there only the kept values (not
# their positions) must match.
@pytest.mark.parametrize("ties", ["none", "fp16", "wide"])
@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("vocab", [4096, 262144])
def test_topk_topp_sort_free_matches_full_sort(rows: int, vocab: int, ties: str) -> None:
    torch.manual_seed(rows)
    x = torch.randn(rows, vocab) * 3
    if ties == "fp16":
        x = x.half().float()
    elif ties == "wide":
        x[:, :300] = 20.0
    p = torch.rand(rows) * 0.5 + 0.5
    for k in (torch.randint(1, 65, (rows,)), torch.tensor([vocab] + [64] * (rows - 1))):
        expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
        actual = apply_top_k_top_p_sort_free(x.clone(), k, p)
        if ties == "none":
            assert torch.equal(actual, expected)
        assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
