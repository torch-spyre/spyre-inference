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
from unittest.mock import Mock

import pytest
import torch
from xpress_reference import merged_refine

from spyre_inference import envs
from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead, select_feedback
from spyre_inference.v1.worker import compile_guard


@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize(
    "block,topc,vocab",
    [(8, 1, 1031), (8, 7, 1031), (16, 65, 1031), (16, 256, 1031), (32, 1024, 2049), (32, 2048, 73)],
)
def test_feedback_compilation_requires_opt_in(monkeypatch, feedback, block, topc, vocab):
    head = XPressRefinerHead(vocab, 128, block, rank=64, mlp_hidden=128, topc=topc)
    model = SimpleNamespace(
        model=SimpleNamespace(prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]),
        xpress_head=head,
        xpress_topc=topc,
        config=SimpleNamespace(vocab_size=vocab),
    )
    compiled, watched = [], []

    def compile(function, **kwargs):
        compiled.append(function)
        return function

    monkeypatch.setattr(head, "prepare_for_spyre", lambda: None)
    monkeypatch.setattr(torch, "compile", compile)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_FEEDBACK", feedback)
    monkeypatch.setattr(compile_guard, "watch", lambda function, label: watched.append(function))
    SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
    assert model._device_feedback_enabled is feedback
    if feedback:
        count = min(topc, vocab)
        assert torch.equal(model._feedback_order, torch.arange(count + -count % 64).half())
    for function in (head.prepare_feedback, head.refine_feedback, select_feedback):
        assert (function in compiled) is feedback
        assert (function in watched) is feedback


@pytest.mark.parametrize("topc", [0, 1025, 2048])
def test_feedback_rejects_unsupported_configuration(monkeypatch, topc):
    head = XPressRefinerHead(2049, 128, 16, rank=64, mlp_hidden=128, topc=topc)
    model = SimpleNamespace(
        model=SimpleNamespace(prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]),
        xpress_head=head,
        xpress_topc=topc,
        config=SimpleNamespace(vocab_size=2049),
    )
    monkeypatch.setattr(head, "prepare_for_spyre", lambda: None)
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_FEEDBACK", True)
    with pytest.raises(ValueError, match="requires a shortlist of 1 to 1024 candidates"):
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)


@pytest.mark.parametrize("kind", ["random", "all_equal", "negative", "edge_ties"])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize(
    "block,topc", [(8, 1), (8, 7), (8, 64), (16, 65), (16, 128), (16, 256), (16, 512), (32, 1024)]
)
def test_feedback_keeps_first_ties_and_request_anchors(kind, batch, block, topc):
    torch.manual_seed(39)
    slots = block - 1
    scores = torch.randn(batch, slots, topc).half()
    if kind == "all_equal":
        scores.zero_()
    elif kind == "negative":
        scores = -(torch.arange(topc) + 1).view(1, 1, -1).expand(batch, slots, -1).half()
    elif kind == "edge_ties":
        scores.fill_(-4)
        for row in range(slots):
            index = (31, 32, 63, 64, topc // 2 - 1, topc // 2, topc - 2, topc - 1)[row % 8]
            index = min(topc - 1, max(0, index))
            scores[:, row, index] = scores[:, row, topc - 1] = 9
    embeddings = torch.randn(batch * slots * topc, 64).half()
    previous = torch.randn(batch, block, 64).half()
    indices, feedback = select_feedback(scores, embeddings, torch.arange(topc).half(), previous)
    expected = scores.argmax(-1)
    assert torch.equal(indices.long(), expected)
    winners = (
        embeddings.view(batch, slots, topc, 64)
        .gather(2, expected[..., None, None].expand(-1, -1, 1, 64))
        .squeeze(2)
    )
    assert torch.equal(feedback, torch.cat((previous[:, :2], winners[:, :-1]), dim=1))


@pytest.mark.parametrize("passes", [0, 1, 6])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("block,topc", [(8, 7), (8, 65), (16, 128), (16, 512), (32, 256)])
def test_feedback_serving_loop_matches_upstream_without_per_pass_logit_copies(
    passes, batch, block, topc
):
    torch.manual_seed(81)
    head = XPressRefinerHead(1031, 128, block, rank=64, mlp_hidden=128, topc=topc).eval()
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.01)
    hidden, base = torch.randn(batch * block, 128), torch.randn(batch * block, 1031)
    anchor = torch.arange(batch) + 1025
    model = SimpleNamespace(
        xpress_head=head,
        xpress_num_passes=passes,
        xpress_topc=topc,
        _device_feedback_enabled=True,
        _feedback_order=torch.arange(topc + -topc % 64).half(),
        compute_device_logits=lambda _: base,
        _hidden_cache=head.project_hidden_cache,
        _prepare_feedback=Mock(wraps=head.prepare_feedback),
        _refine_feedback=Mock(wraps=head.refine_feedback),
        _select_feedback=Mock(wraps=select_feedback),
    )
    for name in ("_copy_logits_to_cpu", "_argmax_on_cpu"):
        method = MethodType(getattr(SpyreDFlashQwen3ForCausalLM, name), model)
        setattr(model, name, Mock(wraps=method))
    with torch.inference_mode():
        expected = merged_refine(
            head, base.view(batch, block, 1031), hidden.view(batch, block, 128), anchor, passes
        ).tolist()
        actual = SpyreDFlashQwen3ForCausalLM.propose_block(model, hidden, anchor)
    assert actual == expected
    assert model._copy_logits_to_cpu.call_count == 1
    assert model._copy_logits_to_cpu.call_args.args[0].numel() == base.numel()
    assert model._argmax_on_cpu.call_count == 1
    assert model._prepare_feedback.call_count == bool(passes)
    assert model._refine_feedback.call_count == model._select_feedback.call_count == passes
