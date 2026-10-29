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

from types import MethodType, SimpleNamespace

import pytest
import torch
from xpress_reference import merged_refine

from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.mark.parametrize(
    "topc,vocab", [(0, 73), (1, 73), (7, 73), (73, 73), (128, 1031), (512, 1031)]
)
@pytest.mark.parametrize("passes", [0, 2, 6])
@pytest.mark.parametrize("batch", [1])
def test_serving_selection_loop_matches_reference_and_transfers_only_candidate_scores(
    topc, vocab, passes, batch
):
    torch.manual_seed(81)
    head = XPressRefinerHead(vocab, 128, 16, rank=64, mlp_hidden=128, topc=topc).eval()
    hidden, base = torch.randn(batch * 16, 128), torch.randn(batch * 16, vocab)
    anchor = torch.arange(batch) + 3
    model = SimpleNamespace(
        xpress_head=head,
        xpress_num_passes=passes,
        xpress_topc=topc,
        compute_device_logits=lambda _: base,
        _hidden_cache=head.project_hidden_cache,
        _refine=head.refine_full,
        _gather_readout=head.gather_readout,
        _refine_candidates=head.refine_candidates,
    )
    transferred_bytes = []

    def copy_logits(logits):
        result = SpyreDFlashQwen3ForCausalLM._copy_logits_to_cpu(model, logits)
        transferred_bytes.append(result.numel() * result.element_size())
        return result

    model._copy_logits_to_cpu = copy_logits
    model._argmax_on_cpu = MethodType(SpyreDFlashQwen3ForCausalLM._argmax_on_cpu, model)
    with torch.inference_mode():
        expected = [
            merged_refine(
                head,
                base.view(batch, 16, vocab)[i : i + 1],
                hidden.view(batch, 16, 128)[i : i + 1],
                anchor[i : i + 1],
                passes,
            )[0].tolist()
            for i in range(batch)
        ]
        actual = SpyreDFlashQwen3ForCausalLM.propose_block(model, hidden, anchor)
    assert actual == expected
    assert sum(transferred_bytes) == batch * (16 * vocab + passes * 15 * (topc or vocab)) * 4
