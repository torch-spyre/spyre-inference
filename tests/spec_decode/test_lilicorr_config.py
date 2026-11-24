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

from types import SimpleNamespace as NS

import pytest
import torch

from spyre_inference.v1.spec_decode.config import (
    is_lilicorr,
    normalize_speculative_config,
    validate_dflash_config,
)


def _lilicorr_config(**draft_overrides):
    draft_config = {
        "lilicorr_enabled": True,
        "lilicorr_candidate_topk": 4,
        "lilicorr_hidden_size": 128,
        "lilicorr_num_heads": 2,
        "block_size": 16,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "conv_kernel_size": 0,
        "conv_group_size": 0,
        "use_swa": False,
    }
    draft_config.update(draft_overrides)
    draft = NS(
        architectures=["LiLiCorrDraftModel"],
        hidden_size=4096,
        vocab_size=151936,
        num_hidden_layers=5,
        layer_types=["full_attention"] * 5,
        dflash_config=draft_config,
        logit_scale=1.0,
        attention_bias=False,
    )
    spec = NS(
        use_dflash=lambda: True,
        draft_shares_target_positions=lambda: True,
        max_model_len=None,
        draft_model_config=NS(
            hf_config=draft,
            architecture="LiLiCorrDraftModel",
            architectures=["LiLiCorrDraftModel"],
            quantization=None,
            dtype=torch.float16,
            max_model_len=2048,
        ),
        num_speculative_tokens=15,
        max_num_new_slots_for_drafting=15,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        enable_adaptive_verification=False,
        draft_lm_head_quantization=None,
    )
    target = NS(model_type="qwen3", hidden_size=4096, vocab_size=151936, num_hidden_layers=36)
    return NS(
        speculative_config=spec,
        model_config=NS(
            hf_text_config=target,
            dtype=torch.float16,
            enforce_eager=False,
            quantization=None,
            is_multimodal_model=False,
            max_model_len=2048,
            architectures=["Qwen3ForCausalLM"],
        ),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(max_num_seqs=4, max_num_batched_tokens=128),
        cache_config=NS(enable_prefix_caching=False),
    )


def test_lilicorr_config_is_accepted():
    config = _lilicorr_config()
    validate_dflash_config(config)
    assert is_lilicorr(config.speculative_config)
    assert not is_lilicorr(NS(draft_model_config=NS(architectures=["DFlashDraftModel"])))


def test_lilicorr_accepts_probabilistic_proposals():
    config = _lilicorr_config()
    config.speculative_config.draft_sample_method = "probabilistic"
    validate_dflash_config(config)


def test_lilicorr_accepts_tensor_parallelism():
    config = _lilicorr_config()
    config.parallel_config.tensor_parallel_size = 4
    validate_dflash_config(config)


def test_block_size_comes_from_dflash_config():
    config = _lilicorr_config()
    del config.speculative_config.draft_model_config.hf_config.dflash_config["block_size"]
    config.speculative_config.draft_model_config.hf_config.block_size = 16
    validate_dflash_config(config)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"conv_kernel_size": 3, "conv_group_size": 64}, "grouped-convolution"),
        ({"conv_kernel_size": 3, "conv_group_size": 0}, "grouped-convolution"),
        ({"lilicorr_enabled": False}, "lilicorr_enabled"),
        ({"lilicorr_candidate_topk": 3}, "candidate_topk"),
        ({"lilicorr_candidate_topk": 32}, "candidate_topk"),
        ({}, "candidate_topk"),
        ({"lilicorr_num_heads": 3}, "num_heads"),
        ({"lilicorr_hidden_size": 64, "lilicorr_num_heads": 2}, "multiple of 64"),
        ({"lilicorr_hidden_size": 1024, "lilicorr_num_heads": 1}, "64..512"),
    ],
)
def test_lilicorr_geometry_is_validated(overrides, match):
    if not overrides:
        config = _lilicorr_config()
        del config.speculative_config.draft_model_config.hf_config.dflash_config[
            "lilicorr_candidate_topk"
        ]
    else:
        config = _lilicorr_config(**overrides)
    with pytest.raises(ValueError, match=match):
        validate_dflash_config(config)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("rejection_sample_method", "block", "rejection_sample_method"),
        ("enable_adaptive_verification", True, "adaptive verification"),
        ("draft_lm_head_quantization", "modelopt", "unquantized draft LM head"),
        ("draft_sample_method", "sampled", "greedy proposals"),
        ("num_speculative_tokens", 7, "block size"),
    ],
)
def test_lilicorr_unsupported_options_are_rejected(field, value, match):
    config = _lilicorr_config()
    setattr(config.speculative_config, field, value)
    with pytest.raises(ValueError, match=match):
        validate_dflash_config(config)


def test_lilicorr_model_name_infers_the_dflash_method():
    args = NS(speculative_config={"model": "org/model-lilicorr-b16"}, enable_prefix_caching=None)
    normalize_speculative_config(args)
    assert args.speculative_config["method"] == "dflash"
    assert args.speculative_config["disable_padded_drafter_batch"] is True
    assert args.enable_prefix_caching is False
