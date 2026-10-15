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

"""Tests for `spyre_inference/models/clip.py`. `get_config` is monkeypatched
to a stub -- no real HF config fetch -- and only the layout test needs a card."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import vllm.model_executor.models.clip as vllm_clip
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.clip import CLIPEmbeddingModel
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

from spyre_inference.models.clip import (
    CausalEncoderOnlyAttention,
    SpyreCLIPEmbeddingModel,
    force_disable_chunked_prefill,
    has_text_tokens,
    merge_text_and_vision,
    select_class_rows,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import build_encoder_plan


def _fake_engine_args(model_type: str | None, enable_chunked_prefill=None):
    return types.SimpleNamespace(
        enable_chunked_prefill=enable_chunked_prefill,
        hf_config_path=None,
        model="some/model",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
        _model_type=model_type,
    )


def _patch_get_config(monkeypatch, model_type: str | None):
    import spyre_inference.models.clip as clip_mod

    def _fake_get_config(*args, **kwargs):
        return types.SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config", _fake_get_config, raising=False
    )
    return clip_mod


def test_disables_chunked_prefill_for_clip(monkeypatch):
    _patch_get_config(monkeypatch, "clip")
    args = _fake_engine_args("clip")

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is False


def test_leaves_non_clip_models_untouched(monkeypatch):
    _patch_get_config(monkeypatch, "bert")
    args = _fake_engine_args("bert")

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is None


def test_respects_an_explicit_user_choice(monkeypatch):
    _patch_get_config(monkeypatch, "clip")
    args = _fake_engine_args("clip", enable_chunked_prefill=True)

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is True


def test_tolerates_config_load_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _raise, raising=False)
    args = _fake_engine_args(None)

    force_disable_chunked_prefill(args)  # must not raise

    assert args.enable_chunked_prefill is None


def test_text_tower_is_built_with_causal_encoder_attention(monkeypatch):
    """The text tower takes its layers from the clip module's ``Attention``, and the
    swap is undone once the model is built."""
    upstream = vllm_clip.Attention
    seen = {}

    def init(self, *, vllm_config, prefix=""):
        torch.nn.Module.__init__(self)
        seen["attention"] = vllm_clip.Attention

    monkeypatch.setattr(CLIPEmbeddingModel, "__init__", init)
    SpyreCLIPEmbeddingModel(vllm_config=None)

    assert seen["attention"] is CausalEncoderOnlyAttention
    assert vllm_clip.Attention is upstream


@pytest.mark.parametrize("runner_causal", [True, False])
def test_text_attention_mask_is_causal_whatever_the_runner_passes(
    default_vllm_config, monkeypatch, runner_causal
):
    """CLIP text gets a causal ``[N, 1, L, L]`` mask from its own builder, so a change to
    vLLM's ``causal=True`` default cannot make it bidirectional."""
    layer = CausalEncoderOnlyAttention(4, 64, 64**-0.5, prefix="text_model.layers.0.attn")
    cfg = get_current_vllm_config()
    monkeypatch.setattr(cfg.model_config, "get_num_attention_heads", lambda *a: 4)
    monkeypatch.setattr(cfg.model_config, "get_num_kv_heads", lambda *a: 4)
    monkeypatch.setattr(cfg.cache_config, "block_size", 64)
    spec = EncoderOnlyAttentionSpec(
        block_size=64, num_kv_heads=4, head_size=64, dtype=torch.float16
    )
    builder_cls = layer.get_attn_backend().get_builder_cls()
    builder = builder_cls(spec, [layer.layer_name], cfg, torch.device("cpu"))
    cu = torch.tensor([0, 9, 39], dtype=torch.int32)
    metadata = builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=cu,
            query_start_loc_cpu=cu,
            seq_lens=torch.tensor([9, 30], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=39,
            max_query_len=30,
            max_seq_len=30,
            block_table_tensor=torch.zeros(2, 1, dtype=torch.int32),
            slot_mapping=torch.arange(39, dtype=torch.int64),
            causal=runner_causal,
        ),
    )

    plans = build_encoder_plan(
        metadata,
        rectangles=(),
        width_cap_for={},
        device=torch.device("cpu"),
        dtype=torch.float16,
        batched=False,
    )
    assert [tuple(p.mask.shape) for p in plans] == [(1, 1, 64, 64)] * 2
    for plan in plans:
        assert plan.mask[0, 0, 0, 1] < 0, "query 0 sees key 1"
        assert plan.mask[0, 0, 1, 0] == 0, "query 1 does not see key 0"


def test_text_only_step_runs_text_tower():
    assert has_text_tokens(None, None)
    assert has_text_tokens([], torch.zeros(4, dtype=torch.bool))


def test_image_only_step_skips_text_tower():
    assert not has_text_tokens([torch.ones(2, 8)], torch.ones(2, dtype=torch.bool))


def test_mixed_step_runs_text_tower():
    # The case vLLM 0.28 got wrong: any image made the whole step image-only.
    mask = torch.tensor([True, False, False, False])
    assert has_text_tokens([torch.ones(1, 8)], mask)


def test_merge_keeps_vision_on_image_rows_and_text_elsewhere():
    text = torch.zeros(4, 2)
    vision = torch.ones(4, 2)
    out = merge_text_and_vision(text, vision, torch.tensor([False, True, False, False]))
    assert torch.equal(out, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_merge_pads_a_short_mask_with_text_rows():
    out = merge_text_and_vision(torch.zeros(4, 2), torch.ones(4, 2), torch.tensor([True]))
    assert torch.equal(out[0], torch.ones(2))
    assert torch.equal(out[1:], torch.zeros(3, 2))


def test_merge_without_image_rows_returns_text():
    text = torch.randn(3, 2)
    assert merge_text_and_vision(text, torch.ones(3, 2), torch.zeros(3, dtype=torch.bool)) is text


def test_merge_output_keeps_the_standard_device_layout():
    """A device-side [N, 1] mask view gave torch.where's output a layout that index_select
    faults the card on past row 63 (torch-spyre#5230).

    Checks the layout only, never gathers, so it cannot fault the card.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")
    text = torch.randn(512, 512, dtype=torch.float16).to("spyre")
    vision = torch.randn(512, 512, dtype=torch.float16).to("spyre")
    mask = torch.zeros(512, dtype=torch.bool)
    mask[[0, 100]] = True
    out = merge_text_and_vision(text, vision, mask)
    assert out.device_tensor_layout().stride_map == text.device_tensor_layout().stride_map


@pytest.mark.parametrize("batch", [1, 3])
def test_select_class_rows_matches_the_class_token_slice(batch):
    feats = torch.randn(batch, 50, 8)
    assert torch.equal(select_class_rows(feats), feats[:, :1, :])


def _align(mask: torch.Tensor, num_tokens: int, grid=None) -> types.SimpleNamespace:
    model = types.SimpleNamespace(_spyre_mm_token_mask=mask, _spyre_has_text_tokens=True)
    SpyreCLIPEmbeddingModel.spyre_align_token_mask(model, num_tokens, grid)
    return model


def test_padding_rows_do_not_count_as_text():
    # Two image rows, then bucket padding: the step is image-only.
    model = _align(torch.tensor([True, True, False, False]), num_tokens=2)
    assert model._spyre_has_text_tokens is False
    assert torch.equal(model._spyre_mm_token_mask, torch.tensor([True, True]))


def test_a_real_text_row_keeps_the_text_tower():
    model = _align(torch.tensor([True, False, False, False]), num_tokens=3)
    assert model._spyre_has_text_tokens is True


def test_mask_is_laid_out_like_the_grid():
    # Packed: text (2 rows), then image (1 row). Grid extent 4: request i starts at 4*i,
    # so the image row moves from packed row 2 to grid row 4.
    model = _align(torch.tensor([False, False, True, False]), 3, grid=(4, 2, [2, 1]))
    expected = torch.tensor([False, False, False, False, True, False, False, False])
    assert torch.equal(model._spyre_mm_token_mask, expected)


def test_forward_routes_each_row_itself():
    """Text-tower output on text rows and the vision embedding on image rows, without
    upstream's ``forward``, whose image-row handling differs across vLLM versions."""
    embeds = torch.arange(8.0).view(4, 2)
    model = types.SimpleNamespace(
        _spyre_has_text_tokens=True,
        _spyre_mm_token_mask=torch.tensor([False, True, False, False]),
        text_embed_dim=2,
        get_text_features=lambda ids, pos, emb: -emb,
    )
    out = SpyreCLIPEmbeddingModel.forward(model, None, None, inputs_embeds=embeds)
    assert torch.equal(out, torch.where(model._spyre_mm_token_mask[:, None], embeds, -embeds))

    model._spyre_has_text_tokens = False
    assert SpyreCLIPEmbeddingModel.forward(model, None, None, inputs_embeds=embeds) is embeds


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
