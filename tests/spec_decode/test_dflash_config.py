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
    install_dflash_request_validation,
    normalize_speculative_config,
    validate_dflash_config,
)
from spyre_inference.v1.spec_decode.dflash import accepted_context_length


def test_random_target_is_rejected_before_input_processing(monkeypatch):
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    monkeypatch.setattr(InputProcessor, "_validate_params", InputProcessor._validate_params)
    install_dflash_request_validation()
    processor = object.__new__(InputProcessor)
    processor.speculative_config = NS(use_dflash=lambda: True)
    with pytest.raises(VLLMValidationError, match="temperature=0") as error:
        processor.process_inputs(
            "request",
            {"type": "token", "prompt_token_ids": [42]},
            SamplingParams(temperature=0.7),
            ("generate",),
        )
    assert error.value.parameter == "temperature"


def test_request_validation_is_scoped_to_each_engine_and_preserves_upstream(monkeypatch):
    from vllm.exceptions import VLLMValidationError
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    calls = []

    def original(processor, params, tasks):
        calls.append((processor, params, tasks))
        return False

    monkeypatch.setattr(InputProcessor, "_validate_params", original)
    install_dflash_request_validation()
    validate = InputProcessor._validate_params
    install_dflash_request_validation()
    assert InputProcessor._validate_params is validate
    draft = NS(speculative_config=NS(use_dflash=lambda: True))
    plain = NS(speculative_config=None)
    other = NS(speculative_config=NS(use_dflash=lambda: False))
    random = SamplingParams(temperature=0.7)
    greedy = SamplingParams(temperature=0.0)
    for processor, params in (
        (draft, greedy),
        (plain, random),
        (other, random),
        (draft, PoolingParams(task="embed")),
    ):
        assert validate(processor, params, ("generate",)) is False
        assert calls[-1] == (processor, params, ("generate",))
        with pytest.raises(VLLMValidationError, match="temperature=0"):
            validate(draft, random, ("generate",))
    assert len(calls) == 4


def test_platform_rejects_v2_before_constructing_a_worker(monkeypatch):
    from spyre_inference import threading_config
    from spyre_inference.platform import TorchSpyrePlatform

    config = NS(
        model_config=None,
        parallel_config=NS(
            data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1, world_size=1
        ),
        speculative_config=NS(use_dflash=lambda: True),
    )
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", "v2")
    monkeypatch.setattr(TorchSpyrePlatform, "log_server_boot", lambda config: None)
    monkeypatch.setattr(threading_config, "configure_threading", lambda world_size: None)
    with pytest.raises(ValueError, match="requires SPYRE_MODEL_RUNNER=v1"):
        TorchSpyrePlatform.check_and_update_config(config)


@pytest.mark.parametrize(
    "scheduled,drafts,sampled,expected",
    [(64, 0, 0, 64), (5, 0, 1, 5), (16, 15, 1, 1), (16, 15, 4, 4), (16, 15, 16, 16)],
)
def test_context_excludes_rejected_rows_and_uncomputed_correction(
    scheduled, drafts, sampled, expected
):
    assert accepted_context_length(scheduled, drafts, sampled) == expected


def test_xpress_bridge_preserves_checkpoint_and_disables_only_default_prefix_cache():
    original = dict(
        method="xpress", model="checkpoint", num_speculative_tokens=15, xpress_num_passes=0
    )
    args = NS(
        speculative_config=original, additional_config={"unrelated": 1}, enable_prefix_caching=None
    )
    normalize_speculative_config(args)
    assert original["method"] == "xpress"
    assert args.speculative_config == dict(
        method="dflash",
        model="checkpoint",
        num_speculative_tokens=15,
        disable_padded_drafter_batch=True,
    )
    assert args.additional_config == {"unrelated": 1, "spyre_xpress": {"num_passes": 0}}
    assert args.enable_prefix_caching is False
    args.speculative_config = original
    args.enable_prefix_caching = True
    normalize_speculative_config(args)
    assert args.enable_prefix_caching is True


def test_upstream_dflash_spelling_and_shortlist_override_are_preserved():
    args = NS(
        speculative_config=dict(method="dflash", model="checkpoint", xpress_topc=256),
        additional_config={"unrelated": 1},
        enable_prefix_caching=None,
    )
    normalize_speculative_config(args)
    normalize_speculative_config(args)
    assert args.speculative_config == dict(
        method="dflash", model="checkpoint", disable_padded_drafter_batch=True
    )
    assert args.additional_config == {"unrelated": 1, "spyre_xpress": {"topc": 256}}
    assert args.enable_prefix_caching is False


@pytest.mark.parametrize("name", ["xpress_topc", "xpress_num_passes"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "512"])
def test_invalid_cli_refiner_options_fail_before_vllm(name, value):
    args = NS(
        speculative_config={"method": "dflash", name: value},
        additional_config=None,
        enable_prefix_caching=None,
    )
    with pytest.raises(ValueError, match=name):
        normalize_speculative_config(args)


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
            draft_model_config=NS(hf_config=draft, quantization=None, dtype=torch.float16),
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
        additional_config={"spyre_xpress": {}},
    )


@pytest.mark.parametrize("batch", [1])
def test_published_contract_is_accepted(config, batch):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = 128
    validate_dflash_config(config)
    assert config.speculative_config.disable_padded_drafter_batch
    assert config.speculative_config.draft_model_config.hf_config.xpress_topc == 128


@pytest.mark.parametrize("architecture", ["Qwen3XPressModel", "DFlashQwen3XPressModel"])
@pytest.mark.parametrize(
    "checkpoint,override,expected",
    [(None, None, 128), (512, None, 512), (512, 128, 128), (128, 0, 0), (0, 256, 256)],
)
def test_checkpoint_detection_and_topc_precedence(
    config, architecture, checkpoint, override, expected
):
    draft = config.speculative_config.draft_model_config.hf_config
    draft.architectures = [architecture]
    draft.xpress_topc = checkpoint
    config.additional_config = {} if override is None else {"spyre_xpress": {"topc": override}}
    validate_dflash_config(config)
    assert draft.xpress_topc == expected


@pytest.mark.parametrize("value", [-1, True, 1.5, 151937])
def test_invalid_checkpoint_topc_is_rejected(config, value):
    config.speculative_config.draft_model_config.hf_config.xpress_topc = value
    with pytest.raises(ValueError, match="xpress_topc"):
        validate_dflash_config(config)


@pytest.mark.parametrize("source", ["checkpoint", "additional_config"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_refinement_passes_fail_before_model_loading(config, source, value):
    if source == "checkpoint":
        config.speculative_config.draft_model_config.hf_config.xpress_num_passes = value
    else:
        config.additional_config["spyre_xpress"]["num_passes"] = value
    with pytest.raises(ValueError, match="xpress_num_passes must be a nonnegative integer"):
        validate_dflash_config(config)


def test_plain_dflash_does_not_accept_xpress_options(config):
    draft = config.speculative_config.draft_model_config.hf_config
    draft.architectures = ["DFlashDraftModel"]
    del draft.xpress_rank
    with pytest.raises(ValueError, match="XPress options"):
        validate_dflash_config(config)
    config.additional_config = {}
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


def test_v2_runner_is_rejected_before_model_loading(config, monkeypatch):
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", "v2")
    with pytest.raises(ValueError, match="SPYRE_MODEL_RUNNER=v1"):
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
        config.scheduler_config.max_num_seqs = 2
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


@pytest.mark.parametrize("batch,budget", [(1, 15), (1, 16), (1, 30)])
def test_draft_batch_requires_enough_token_capacity(config, batch, budget):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = budget
    with pytest.raises(ValueError, match="draft block"):
        validate_dflash_config(config)


@pytest.mark.parametrize("batch,budget", [(1, 31)])
def test_token_budget_includes_the_vllm_drafting_reservation(config, batch, budget):
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = budget
    validate_dflash_config(config)
