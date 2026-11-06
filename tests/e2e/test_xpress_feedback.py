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

import warnings
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from torch.nn import functional as F


@pytest.mark.parametrize("block,topc", [(8, 1), (16, 65), (32, 1024)])
def test_batched_feedback_preserves_selection_and_request_anchors(block, topc):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    import torch_spyre

    torch_spyre._autoload()
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.v1.spec_decode.xpress_head import select_feedback

    select = torch.compile(select_feedback, fullgraph=True, dynamic=False)
    torch.manual_seed(39)
    slots = block - 1
    count = topc + -topc % 64
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        order = F.pad(torch.arange(topc).half(), (0, count - topc), value=-1).to("spyre")
        for batch in (1, 2, 4):
            for kind in ("random", "all_equal", "edge_ties", "negative", "nonfinite"):
                scores = torch.randn(batch, slots, topc).half()
                if kind == "all_equal":
                    scores.zero_()
                elif kind == "negative":
                    scores = (
                        -(torch.arange(topc) + 1).view(1, 1, -1).expand(batch, slots, -1).half()
                    )
                elif kind == "edge_ties":
                    scores.fill_(-4)
                    for row in range(slots):
                        index = (31, 32, 63, 64, topc // 2 - 1, topc // 2, topc - 2, topc - 1)[
                            row % 8
                        ]
                        index = min(topc - 1, max(0, index))
                        scores[:, row, index] = scores[:, row, topc - 1] = 9
                elif kind == "nonfinite":
                    scores[:, 0] = torch.nan
                    scores[:, 1] = -torch.inf
                    scores[:, 2] = torch.inf
                    scores[:, 3, 0] = torch.inf
                    scores[:, 3, topc // 2 :] = torch.nan
                embeddings = torch.randn(batch * slots * count, 64).half().to("spyre")
                previous = torch.randn(batch, block, 64).half().to("spyre")
                device_scores = F.pad(scores, (0, count - topc), value=torch.nan).to("spyre")
                # DLF16 upload can round close scores into ties; use resident values.
                expected = device_scores.cpu()[..., :topc].argmax(-1)
                winners = (
                    embeddings.cpu()
                    .view(batch, slots, count, 64)
                    .gather(2, expected[..., None, None].expand(-1, -1, 1, 64))
                    .squeeze(2)
                )
                expected_feedback = torch.cat((previous.cpu()[:, :2], winners[:, :-1]), dim=1)
                indices, feedback = select(device_scores, embeddings, order, previous)
                assert torch.equal(indices.cpu().long(), expected), (batch, kind)
                assert torch.equal(feedback.cpu(), expected_feedback), (batch, kind)


@pytest.mark.parametrize(
    "batch,block,topc",
    [
        (4, 8, 1),
        (4, 8, 65),
        (4, 32, 256),
        pytest.param(
            1,
            16,
            512,
            marks=pytest.mark.xfail(
                strict=True,
                reason="deeptools sbf-ddc 'Incorrect chunk size' crash, likely the same "
                "#4920 regression as deeptools#4965; remove on the next RPM bump once "
                "the fix is in",
            ),
        ),
    ],
)
def test_feedback_drives_refinement_without_per_pass_host_selection(batch, block, topc):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    import torch_spyre

    torch_spyre._autoload()
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
    from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead
    from spyre_inference.v1.worker import compile_guard

    torch.manual_seed(918)
    vocab, passes = 151936, 6
    head = XPressRefinerHead(vocab, 128, block, rank=256, mlp_hidden=512, topc=topc).eval()
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.01)
    head = head.half().to("spyre")
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        hidden = torch.randn(batch * block, 128).half().to("spyre")
        anchors = torch.arange(batch).int() * 20003 + 65537
        base = torch.randn(batch * block, vocab).half().to("spyre")
        resident = base.cpu().float().view(batch, block, vocab)
        candidates = resident[:, 1:].topk(topc, dim=-1).indices
        padded_candidates = F.pad(candidates, (0, -topc % 64))
        draft = resident[:, 1:].argmax(-1)
        model = SimpleNamespace(
            model=SimpleNamespace(
                prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]
            ),
            config=SimpleNamespace(vocab_size=vocab),
            xpress_head=head,
            xpress_topc=topc,
            xpress_num_passes=passes,
            compute_device_logits=lambda _: base,
        )
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
        for name in ("_copy_logits_to_cpu", "_argmax_on_cpu"):
            method = MethodType(getattr(SpyreDFlashQwen3ForCausalLM, name), model)
            setattr(model, name, Mock(wraps=method))
        host_input, host_readout = head.w1.weight.cpu(), head.w2.weight.cpu()
        prepare, refine, select = (
            model._prepare_feedback,
            model._refine_feedback,
            model._select_feedback,
        )
        state = {}

        def checked_prepare(ids, previous):
            expected_ids = torch.cat((anchors[:, None].expand(-1, 2), draft[:, :-1]), dim=1)
            assert torch.equal(ids.cpu().long(), padded_candidates.flatten())
            assert torch.equal(previous.cpu().long(), expected_ids)
            embeddings, feedback, readout = prepare(ids, previous)
            assert torch.equal(embeddings.cpu(), host_input[padded_candidates.flatten()])
            assert torch.equal(readout.cpu(), host_readout[padded_candidates.flatten()])
            state["feedback"] = host_input[expected_ids]
            assert torch.equal(feedback.cpu(), state["feedback"])
            return embeddings, feedback, readout

        def checked_refine(base, previous, cache, readout):
            assert torch.equal(previous.cpu(), state["feedback"])
            return refine(base, previous, cache, readout)

        def checked_select(scores, embeddings, order, previous):
            expected = scores.cpu()[..., :topc].argmax(-1)
            indices, feedback = select(scores, embeddings, order, previous)
            assert torch.equal(indices.cpu().long(), expected)
            state["draft"] = candidates.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
            expected_ids = torch.cat(
                (anchors[:, None].expand(-1, 2), state["draft"][:, :-1]), dim=1
            )
            state["feedback"] = host_input[expected_ids]
            assert torch.equal(feedback.cpu(), state["feedback"])
            return indices, feedback

        model._prepare_feedback = Mock(wraps=checked_prepare)
        model._refine_feedback = Mock(wraps=checked_refine)
        model._select_feedback = Mock(wraps=checked_select)
        propose = MethodType(SpyreDFlashQwen3ForCausalLM.propose_block, model)
        assert propose(hidden, anchors) == state["draft"].tolist()
        compile_guard.arm(compile_guard.CompileGuardLevel.ERROR)
        assert propose(hidden, anchors) == state["draft"].tolist()
        assert model._copy_logits_to_cpu.call_count == model._argmax_on_cpu.call_count == 2
        assert model._prepare_feedback.call_count == 2
        assert model._refine_feedback.call_count == model._select_feedback.call_count == 2 * passes
