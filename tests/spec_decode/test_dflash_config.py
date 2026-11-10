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
    MAX_DFLASH_BATCH_SIZE,
    normalize_speculative_config,
    validate_dflash_config,
)
from spyre_inference.v1.spec_decode.dflash import accepted_context_length


@pytest.mark.parametrize(
    "scheduled,drafts,sampled,expected",
    [(64, 0, 0, 64), (5, 0, 1, 5), (16, 15, 1, 1), (16, 15, 4, 4), (16, 15, 16, 16)],
)
def test_context_excludes_rejected_rows_and_uncomputed_correction(
    scheduled, drafts, sampled, expected
):
    assert accepted_context_length(scheduled, drafts, sampled) == expected


def test_dflash_bridge_disables_padding_and_only_the_default_prefix_cache():
    original = dict(method="dflash", model="checkpoint", num_speculative_tokens=15, xpress_topc=128)
    args = NS(speculative_config=original, enable_prefix_caching=None)
    normalize_speculative_config(args)
    assert original == dict(
        method="dflash", model="checkpoint", num_speculative_tokens=15, xpress_topc=128
    )
    assert args.speculative_config == original | {"disable_padded_drafter_batch": True}
    assert args.enable_prefix_caching is False
    args.speculative_config = original
    args.enable_prefix_caching = True
    normalize_speculative_config(args)
    assert args.enable_prefix_caching is True


def test_other_methods_pass_through():
    args = NS(speculative_config=dict(method="ngram"), enable_prefix_caching=None)
    normalize_speculative_config(args)
    assert args.speculative_config == dict(method="ngram")
    assert args.enable_prefix_caching is None


@pytest.mark.parametrize("method", [{}, {"method": None}])
@pytest.mark.parametrize("prefix_cache", [None, False, True])
def test_inferred_dflash_normalizes_before_vllm_cache_defaults(method, prefix_cache):
    original = dict(model="z-lab/Qwen3-8B-DFlash", **method)
    args = NS(speculative_config=original, enable_prefix_caching=prefix_cache)
    normalize_speculative_config(args)
    assert args.speculative_config == dict(
        model=original["model"], method="dflash", disable_padded_drafter_batch=True
    )
    assert args.enable_prefix_caching is (prefix_cache is True)
    assert original == dict(model="z-lab/Qwen3-8B-DFlash", **method)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"model": "draft"},
        {"model": "eagle-DFlash"},
        {"model": "EAGLE3-DFlash"},
        {"model": "DFlash", "method": "eagle3"},
        {"model": "DFlash", "method": "draft_model"},
    ],
)
def test_dflash_inference_preserves_other_methods(spec):
    args = NS(speculative_config=spec, enable_prefix_caching=None)
    normalize_speculative_config(args)
    assert args.speculative_config is spec
    assert args.enable_prefix_caching is None


@pytest.fixture
def config():
    target = NS(model_type="qwen3", hidden_size=4096, vocab_size=151936, num_hidden_layers=36)
    draft = NS(
        architectures=["DFlashQwen3XPressModel"],
        hidden_size=4096,
        vocab_size=151936,
        layer_types=["full_attention"] * 5,
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
        xpress_block_size=16,
        xpress_rank=256,
        sample_from_anchor=False,
    )
    return NS(
        speculative_config=NS(
            use_dflash=lambda: True,
            draft_model_config=NS(
                hf_config=draft,
                architecture="DFlashQwen3XPressModel",
                quantization=None,
                dtype=torch.float16,
            ),
            num_speculative_tokens=15,
            max_num_new_slots_for_drafting=15,
            draft_sample_method="greedy",
        ),
        model_config=NS(
            hf_text_config=target,
            is_multimodal_model=False,
            dtype=torch.float16,
            enforce_eager=False,
            quantization=None,
        ),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=64),
        cache_config=NS(enable_prefix_caching=False),
    )


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_published_contract_is_accepted(config, batch):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = 128
    validate_dflash_config(config)
    assert config.speculative_config.disable_padded_drafter_batch
    assert config.speculative_config.draft_model_config.hf_config.xpress_topc == 128


@pytest.mark.parametrize("checkpoint,expected", [(None, 128), (1, 1), (512, 512), (1024, 1024)])
def test_spyre_topc_default_applies_only_when_unset(config, checkpoint, expected):
    # Upstream writes --xpress-topc onto the draft config before this runs.
    draft = config.speculative_config.draft_model_config.hf_config
    draft.xpress_topc = checkpoint
    validate_dflash_config(config)
    assert draft.xpress_topc == expected


@pytest.mark.parametrize("topc", [0, 1025, 2048])
def test_unsupported_shortlists_fail_before_model_loading(config, topc):
    config.speculative_config.draft_model_config.hf_config.xpress_topc = topc
    with pytest.raises(ValueError, match="requires 1 to 1024 candidates"):
        validate_dflash_config(config)


@pytest.mark.parametrize("topc", [128, 1024, 2048])
def test_shortlist_width_is_normalized_to_the_real_vocabulary(config, topc):
    draft = config.speculative_config.draft_model_config.hf_config
    draft.xpress_topc = topc
    draft.vocab_size = config.model_config.hf_text_config.vocab_size = 73
    validate_dflash_config(config)
    assert draft.xpress_topc == 73


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_refinement_passes_fail_before_model_loading(config, value):
    config.speculative_config.draft_model_config.hf_config.xpress_num_passes = value
    with pytest.raises(ValueError, match="xpress_num_passes must be a nonnegative integer"):
        validate_dflash_config(config)


def test_plain_dflash_is_not_given_refiner_defaults(config):
    draft = config.speculative_config.draft_model_config.hf_config
    draft.architectures = ["DFlashDraftModel"]
    config.speculative_config.draft_model_config.architecture = "DFlashDraftModel"
    del draft.xpress_rank
    validate_dflash_config(config)
    assert not hasattr(draft, "xpress_topc")


@pytest.mark.parametrize("architecture", [None, "DFlash2DraftModel", "MuseGlimmerAssistantModel"])
def test_unsupported_draft_architecture_fails_before_model_loading(config, architecture):
    config.speculative_config.draft_model_config.architecture = architecture
    config.speculative_config.draft_model_config.hf_config = NS()
    with pytest.raises(ValueError, match="Unsupported Spyre DFlash/XPress draft architecture"):
        validate_dflash_config(config)


def test_draft_validation_uses_the_architecture_vllm_selected(config):
    draft = config.speculative_config.draft_model_config.hf_config
    draft.architectures = ["UnknownDraftModel", "DFlashDraftModel", "Qwen3XPressModel"]
    config.speculative_config.draft_model_config.architecture = "DFlashDraftModel"
    del draft.xpress_rank
    validate_dflash_config(config)


@pytest.mark.parametrize("nested", [False, True])
def test_shifted_anchor_layout_is_rejected(config, nested):
    draft = config.speculative_config.draft_model_config.hf_config
    if nested:
        draft.dflash_config["sample_from_anchor"] = True
    else:
        draft.sample_from_anchor = True
    with pytest.raises(ValueError, match="fixed anchor"):
        validate_dflash_config(config)


@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_both_runners_are_accepted(config, monkeypatch, runner):
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", runner)
    validate_dflash_config(config)


@pytest.mark.parametrize(
    "invalid",
    [
        "block",
        "taps",
        "vocab",
        "missing_refiner",
        "prefix_cache",
        "batch",
        "bf16",
        "draft_bf16",
        "quantized",
        "eager",
    ],
)
def test_incompatible_checkpoint_or_serving_configuration_fails_early(config, invalid):
    draft = config.speculative_config.draft_model_config.hf_config
    if invalid == "block":
        config.speculative_config.num_speculative_tokens = 7
    elif invalid == "taps":
        draft.dflash_config["target_layer_ids"] = [1, 36]
    elif invalid == "vocab":
        draft.vocab_size = 100
    elif invalid == "missing_refiner":
        del draft.xpress_rank
    elif invalid == "prefix_cache":
        config.cache_config.enable_prefix_caching = True
    elif invalid == "batch":
        config.scheduler_config.max_num_seqs = MAX_DFLASH_BATCH_SIZE + 1
    elif invalid == "bf16":
        config.model_config.dtype = torch.bfloat16
    elif invalid == "draft_bf16":
        config.speculative_config.draft_model_config.dtype = torch.bfloat16
    elif invalid == "quantized":
        config.model_config.quantization = "fp8"
    elif invalid == "eager":
        config.model_config.enforce_eager = True
    with pytest.raises(ValueError):
        validate_dflash_config(config)


@pytest.mark.parametrize(
    "batch,budget", [(1, 15), (1, 16), (1, 30), (2, 31), (2, 61), (3, 64), (4, 63), (4, 123)]
)
def test_draft_batch_requires_enough_token_capacity(config, batch, budget):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = budget
    with pytest.raises(ValueError, match="per-request drafting reservation"):
        validate_dflash_config(config)


@pytest.mark.parametrize("batch,budget", [(1, 31), (2, 62), (3, 93), (4, 124)])
def test_token_budget_includes_the_vllm_drafting_reservation(config, batch, budget):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = budget
    validate_dflash_config(config)


def test_v1_keeps_upstreams_other_refusals_but_allows_xpress(monkeypatch):
    from vllm.config.vllm import VllmConfig

    from spyre_inference.v1.spec_decode.config import allow_xpress_on_v1

    monkeypatch.setattr(
        VllmConfig,
        "_get_v1_model_runner_unsupported_features",
        lambda self: ["prefill context parallel", "xpress speculative decoding"],
    )
    allow_xpress_on_v1()
    allow_xpress_on_v1()
    assert VllmConfig._get_v1_model_runner_unsupported_features(None) == [
        "prefill context parallel"
    ]
