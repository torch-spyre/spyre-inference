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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Compare Spyre refinement helpers with the independent merged-vLLM reference."""

import pytest
import torch
from xpress_reference import merged_refine

from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead, select_feedback


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("topc", [128, 512])
@pytest.mark.parametrize("passes", [0, 1, 6])
def test_refinement_matches_merged_vllm(batch, topc, passes):
    torch.manual_seed(20261027)
    head = XPressRefinerHead(1031, 128, 16, rank=64, mlp_hidden=128, topc=topc).double()
    with torch.no_grad():
        head.mix_L.copy_(torch.randn_like(head.mix_L).tril() / 8 + torch.eye(16))
        hidden = torch.randn(batch, 16, 128, dtype=torch.float64)
        base = torch.randn(batch, 16, 1031, dtype=torch.float64)
        anchors = torch.arange(batch) + 1025
        expected = merged_refine(head, base, hidden, anchors, passes)
        scores = base[:, 1:]
        actual = scores.argmax(-1)
        cache = head.project_hidden_cache(hidden)
        scores, candidates = scores.topk(topc, dim=-1)
        previous = torch.cat((anchors[:, None], anchors[:, None], actual[:, :-1]), dim=1)
        embeddings, feedback, readout = head.prepare_feedback(candidates.flatten(), previous)
        for _ in range(passes):
            logits = head.refine_feedback(scores, feedback, cache, readout)
            indices, feedback = select_feedback(
                logits, embeddings, torch.arange(topc).half(), feedback
            )
            actual = candidates.gather(-1, indices.long().unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
