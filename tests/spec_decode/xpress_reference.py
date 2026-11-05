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

"""CPU oracle: upstream XPress's torch path, independent of Spyre's split projections."""

import torch
from vllm.model_executor.models.qwen3_xpress import XPressRefinerHead


def merged_refine(head, base, hidden, anchors, passes):
    cache = XPressRefinerHead.hidden_cache(head, hidden)
    draft = base[:, 1:].argmax(-1)
    candidates = None
    if head.topc:
        _, candidates = base[:, 1:].topk(min(head.topc, base.shape[-1]), dim=-1)
    for _ in range(passes):
        # XPressSpeculator uses the verified anchor as its own predecessor.
        previous = torch.cat((anchors[:, None], anchors[:, None], draft[:, :-1]), dim=1)
        scores = base[:, 1:] + XPressRefinerHead.refine_bias(head, previous, cache)[:, 1:]
        if candidates is None:
            draft = scores.argmax(-1)
        else:
            winners = scores.gather(-1, candidates).argmax(-1, keepdim=True)
            draft = candidates.gather(-1, winners).squeeze(-1)
    return draft
