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

"""CPU oracle for vLLM XPress at ce9467b757, independent of Spyre's split projections."""

import torch
from torch.nn import functional as F


def merged_refine(head, base, hidden, anchors, passes):
    summary = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    cache = torch.cat((head.down_h(hidden), head.down_g(summary)), dim=-1)
    draft = base[:, 1:].argmax(-1)
    candidates = None
    if head.topc:
        _, candidates = base[:, 1:].topk(min(head.topc, base.shape[-1]), dim=-1)
    for _ in range(passes):
        # XPressSpeculator uses the verified anchor as its own predecessor.
        previous = torch.cat((anchors[:, None], anchors[:, None], draft[:, :-1]), dim=1)
        x = head.in_proj(torch.cat((cache, head.w1(previous)), dim=-1))
        x = torch.bmm(head.mix_L.to(x.dtype), x.permute(2, 1, 0)).permute(2, 1, 0)
        latent = x + head.mlp_down(F.silu(head.mlp_gate(x)) * head.mlp_up(x))
        scores = base[:, 1:] + head.w2(latent)[:, 1:]
        if candidates is None:
            draft = scores.argmax(-1)
        else:
            winners = scores.gather(-1, candidates).argmax(-1, keepdim=True)
            draft = candidates.gather(-1, winners).squeeze(-1)
    return draft
