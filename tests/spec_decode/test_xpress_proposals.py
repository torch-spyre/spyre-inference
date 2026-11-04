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
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
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


@pytest.fixture
def upstream_xpress_model(monkeypatch):
    from vllm.model_executor.models import qwen3_xpress
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    from spyre_inference.models.qwen3_dflash import SpyreQwen3XPressForCausalLM

    config = SimpleNamespace(
        vocab_size=73,
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        num_hidden_layers=0,
        dflash_config={"target_layer_ids": [0, 2]},
        xpress_block_size=16,
        xpress_rank=64,
        xpress_mlp_hidden=128,
        xpress_topc=7,
        xpress_num_passes=2,
        draft_vocab_size=73,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config)),
        model_config=SimpleNamespace(get_num_layers=lambda _: 4, get_vocab_size=lambda: 73),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )

    def backbone_init(self, *, vllm_config, start_layer_id=0, prefix=""):
        torch.nn.Module.__init__(self)
        self.config = config
        self.fc = torch.nn.Linear(256, 128, bias=False)
        self.hidden_norm = torch.nn.RMSNorm(128)
        self.norm = torch.nn.RMSNorm(128)
        self.layers = torch.nn.ModuleList()

    monkeypatch.setattr(DFlashQwen3Model, "__init__", backbone_init)
    monkeypatch.setattr(qwen3_xpress, "ParallelLMHead", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(qwen3_xpress, "LogitsProcessor", lambda *a, **k: torch.nn.Identity())
    return SpyreQwen3XPressForCausalLM(vllm_config=vllm_config)


def test_upstream_xpress_construction_builds_the_spyre_classes(upstream_xpress_model):
    from vllm.model_executor.models import qwen3_xpress

    from spyre_inference.models.qwen3_dflash import SpyreQwen3XPressModel

    model = upstream_xpress_model

    assert type(model.model) is SpyreQwen3XPressModel
    head = model.xpress_head
    assert head is model.model.xpress_head and type(head) is XPressRefinerHead
    assert (head.rank, head.topc, model.xpress_num_passes) == (64, 7, 2)
    assert "refine_bias" not in head.__dict__, "upstream's compiled refine_bias was left bound"
    assert qwen3_xpress.XPressRefinerHead is not XPressRefinerHead, "class swap leaked"


@pytest.fixture
def upstream_xpress_weights(upstream_xpress_model):
    model = upstream_xpress_model
    weights = {
        name: torch.full_like(value, 0.125)
        for name, value in model.model.state_dict().items()
        if not name.startswith("xpress_head.")
    }
    head = model.xpress_head
    state = head.state_dict()
    for index, (name, dest) in enumerate(head.HYBRID_KEY_MAP.items(), start=1):
        reference = head.mix_L if name == "mix.L" else state[dest]
        weights["xpress_head." + name] = torch.full_like(reference, index / 32)
    return weights


def test_upstream_xpress_checkpoint_roundtrip(upstream_xpress_model, upstream_xpress_weights):
    model, weights = upstream_xpress_model, upstream_xpress_weights
    model.load_weights(weights.items())
    head = model.xpress_head
    state = head.state_dict()
    for name, dest in head.HYBRID_KEY_MAP.items():
        expected = weights["xpress_head." + name]
        if name == "mix.L":
            expected = expected.tril() + torch.eye(head.block_size, dtype=expected.dtype)
            dest = "mix_L"
        torch.testing.assert_close(state[dest], expected, atol=0, rtol=0)
    assert head._fused_buf is None
    restored = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128, topc=7)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.state_dict(), state, atol=0, rtol=0)


@pytest.mark.parametrize("name", tuple(XPressRefinerHead.HYBRID_KEY_MAP))
def test_missing_xpress_head_weight_fails(upstream_xpress_model, upstream_xpress_weights, name):
    del upstream_xpress_weights["xpress_head." + name]
    with pytest.raises(ValueError, match=name):
        upstream_xpress_model.load_weights(upstream_xpress_weights.items())


def test_raw_xpress_mixer_cannot_broadcast(upstream_xpress_model, upstream_xpress_weights):
    upstream_xpress_weights["xpress_head.mix.L"] = upstream_xpress_weights["xpress_head.mix.L"][:1]
    with pytest.raises(ValueError, match="mix.L shape"):
        upstream_xpress_model.load_weights(upstream_xpress_weights.items())


def test_folded_xpress_mixer_is_not_a_published_weight(
    upstream_xpress_model, upstream_xpress_weights
):
    upstream_xpress_weights["xpress_head.mix_L"] = upstream_xpress_model.xpress_head.mix_L.clone()
    with pytest.raises(ValueError, match="unexpected=.*mix_L"):
        upstream_xpress_model.load_weights(upstream_xpress_weights.items())
