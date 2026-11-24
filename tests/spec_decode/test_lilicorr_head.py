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

"""Spyre LiLiCorr head matches upstream, including after the OOT weight transpose."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.lilicorr import (
    LiLiCorrForCausalLM,
    LiLiCorrHead,
    LiLiCorrLatticeAttention,
)

from spyre_inference.models import lilicorr as spyre_lilicorr
from spyre_inference.models.lilicorr import (
    SpyreLiLiCorrForCausalLM,
    SpyreLiLiCorrHead,
    SpyreLiLiCorrLatticeAttention,
)


@pytest.fixture(autouse=True)
def head_tp_metadata(monkeypatch):
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))


def _config(**overrides):
    config = dict(
        candidate_topk=4,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        factor_dim=4,
        vector_eps=1e-6,
        logit_scale=3.0,
    )
    config.update(overrides)
    return {f"lilicorr_{key}": value for key, value in config.items()}


def _make_heads(head_width: int, block_size: int = 4):
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        )
    ):
        spyre = SpyreLiLiCorrHead(
            model_hidden_size=16,
            block_size=block_size,
            rms_norm_eps=1e-6,
            config=_config(hidden_size=head_width),
        )
        upstream = LiLiCorrHead(
            model_hidden_size=16,
            block_size=block_size,
            rms_norm_eps=1e-6,
            config=_config(hidden_size=head_width),
        )
    return spyre, upstream


def _initialize(head: LiLiCorrHead, dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, LinearBase):
                for parameter in module.parameters():
                    parameter.weight_loader(parameter, torch.randn_like(parameter) * 0.1)
        for layer in head.layers:
            layer.attn.in_proj_weight.normal_(std=0.1)
            layer.attn.in_proj_bias.normal_(std=0.1)
        head.relative_slot_bias.normal_(std=0.1)
        head.same_slot_bias.normal_(std=0.1)
        head.slot_embedding.normal_(std=0.1)
        head.rank_embedding.normal_(std=0.1)
    head.to(dtype)


def _transpose_like_process_weights(head: SpyreLiLiCorrHead) -> None:
    """Transpose the modules whose weights the host feature path stashed at load.

    The factor/edge linears are now used through their OOT modules, so their
    orientation is the module's concern; only the feature MLP is read on the host.
    """
    for module in (head.feature_mlp.up_proj, head.feature_mlp.down_proj):
        module.weight.data = module.weight.data.t().contiguous()


@pytest.mark.parametrize(
    "head_width,slots",
    [
        pytest.param(8, 3, id="projected-multi-slot"),
        pytest.param(16, 1, id="identity-shortened-single-slot"),
        pytest.param(8, 2, id="projected-shortened-multi-slot"),
    ],
)
def test_spyre_head_matches_upstream(head_width, slots):
    # Host fp16 inference is not portable across CPU builds; the device path is
    # fp16 and qualified separately.
    dtype = torch.float32
    spyre, upstream = _make_heads(head_width)
    _initialize(spyre, dtype)
    upstream.load_state_dict(spyre.state_dict())
    upstream.to(dtype)

    torch.manual_seed(11)
    token_embeddings = torch.randn(2, slots, 4, 16, dtype=dtype)
    log_probs = torch.randn(2, slots, 23).log_softmax(-1).topk(4).values
    pass_hidden = torch.randn(2, slots, 16, dtype=dtype)
    anchor_hidden = torch.randn(2, 16, dtype=dtype)
    anchor_valid = torch.tensor([True, False])

    spyre.materialize_inference_buffers(torch.device("cpu"), dtype)
    upstream.materialize_inference_buffers(torch.device("cpu"), dtype)
    # The loader transposes the linear weights after materialization; the fused
    # buffers must have been stashed on the host before that.
    _transpose_like_process_weights(spyre)
    spyre.prepare_for_spyre(torch.device("cpu"), dtype)
    spyre._score_lattice_compiled = spyre._score_lattice

    actual = spyre.score(
        token_embeddings=token_embeddings,
        candidate_log_probs=log_probs,
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid.to(dtype),
    )
    expected = upstream(token_embeddings, log_probs, pass_hidden, anchor_hidden, anchor_valid)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_attention_matches_upstream_and_is_contiguous():
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        )
    ):
        spyre = SpyreLiLiCorrLatticeAttention(64, 1)
        upstream = LiLiCorrLatticeAttention(64, 1)
    torch.manual_seed(5)
    with torch.no_grad():
        for head in (spyre, upstream):
            head.in_proj_weight.normal_(std=0.1)
            head.in_proj_bias.normal_(std=0.1)
            head.out_proj.weight_loader(
                head.out_proj.weight, torch.randn_like(head.out_proj.weight) * 0.1
            )
    upstream.load_state_dict(spyre.state_dict())

    hidden = torch.randn(2, 4, 64)
    bias = torch.randn(1, 4, 4)
    torch.testing.assert_close(spyre(hidden, bias), upstream(hidden, bias), atol=1e-5, rtol=1e-5)


def test_prepare_before_materialize_is_rejected():
    spyre, _ = _make_heads(8)
    with pytest.raises(RuntimeError, match="materialized"):
        spyre.prepare_for_spyre(torch.device("cpu"), torch.float32)


def test_materialize_after_prepare_is_rejected():
    """A draft reload would rebuild the head buffers from transposed weights."""
    spyre, _ = _make_heads(8)
    _initialize(spyre, torch.float32)
    spyre.materialize_inference_buffers(torch.device("cpu"), torch.float32)
    spyre.prepare_for_spyre(torch.device("cpu"), torch.float32)
    with pytest.raises(NotImplementedError, match="reload"):
        spyre.materialize_inference_buffers(torch.device("cpu"), torch.float32)


def test_load_weights_validates_the_backbone_and_keeps_lilicorr(monkeypatch):
    validated: set[str] = set()
    passed: set[str] = set()

    def fake_validate(config, weights):
        validated.update(weights)

    def fake_upstream_load(self, weights):
        passed.update(name for name, _ in weights)

    monkeypatch.setattr(spyre_lilicorr, "validate_dflash_weights", fake_validate)
    monkeypatch.setattr(LiLiCorrForCausalLM, "load_weights", fake_upstream_load)

    model = object.__new__(SpyreLiLiCorrForCausalLM)
    model.config = SimpleNamespace()
    weights = {
        "fc.weight": torch.zeros(1),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "lilicorr.token_proj.weight": torch.zeros(1),
        "attention_conv.kernel_projection.weight": torch.zeros(1),
    }
    SpyreLiLiCorrForCausalLM.load_weights(model, weights.items())

    assert validated == {
        "fc.weight",
        "layers.0.self_attn.q_proj.weight",
    }
    assert passed == set(weights)


def test_topk_log_probs_masks_padding_and_matches_log_softmax():
    torch.manual_seed(3)
    logits = torch.randn(2, 10)
    logits[:, 7:] = -float("inf")
    ids, probs = spyre_lilicorr._topk_log_probs(
        logits, vocab_size=7, top_k=3, scale=1.0, soft_cap=None
    )
    expected = logits[:, :7].log_softmax(-1).topk(3, dim=-1)
    torch.testing.assert_close(ids, expected.indices)
    torch.testing.assert_close(probs, expected.values)


def test_topk_log_probs_applies_soft_cap_then_scale():
    torch.manual_seed(4)
    logits = torch.randn(2, 7)
    ids, probs = spyre_lilicorr._topk_log_probs(
        logits, vocab_size=7, top_k=3, scale=2.0, soft_cap=5.0
    )
    scaled = torch.tanh(logits / 5.0) * 5.0 * 2.0
    expected = scaled.log_softmax(-1).topk(3, dim=-1)
    torch.testing.assert_close(ids, expected.indices)
    torch.testing.assert_close(probs, expected.values)


def test_gather_full_weight_is_identity_at_tp1(monkeypatch):
    monkeypatch.setattr(spyre_lilicorr, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    shard = torch.randn(3, 5, dtype=torch.float16)
    gathered = spyre_lilicorr._gather_full_weight(shard, dim=0)
    assert gathered.dtype == torch.float32
    torch.testing.assert_close(gathered, shard.float())


def test_compute_candidates_outside_tp1_gathers_and_takes_topk():
    """The TP>1 branch returns the same top-k/log-probs as a full-vocab softmax."""
    model = object.__new__(SpyreLiLiCorrForCausalLM)
    model.lm_head = SimpleNamespace(tp_size=2, org_vocab_size=4)
    model.candidate_logits_processor = SimpleNamespace(
        _apply_head=lambda lm_head, hidden: hidden,
        _gather_logits=lambda logits: logits,
        scale=1.0,
        soft_cap=None,
    )
    model.model = SimpleNamespace(lilicorr=SimpleNamespace(candidate_topk=2))
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, float("-inf"), float("-inf")]])

    ids, probs = model.compute_candidates(logits)

    expected = logits.log_softmax(-1).topk(2, dim=-1)
    torch.testing.assert_close(ids, expected.indices)
    torch.testing.assert_close(probs, expected.values)
