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
    for name in (
        "device_to_host_seconds",
        "host_argmax_seconds",
        "host_topk_seconds",
        "host_to_device_seconds",
        "candidate_gather_seconds",
        "candidate_transfer_bytes",
        "refiner_seconds",
        "logits_seconds",
        "selection_calls",
        "logits_transfer_bytes",
    ):
        setattr(model, name, 0)
    for name in ("_copy_logits_to_cpu", "_argmax_on_cpu"):
        setattr(model, name, MethodType(getattr(SpyreDFlashQwen3ForCausalLM, name), model))
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
    assert model.logits_transfer_bytes == batch * (16 * vocab + passes * 15 * (topc or vocab)) * 4
