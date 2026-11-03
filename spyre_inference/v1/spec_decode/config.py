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

# vLLM 0.32 predates XPress (vllm#54448); use upstream fields when its canary fails.
# Retain Spyre validation when removing this bridge; cleanup: #1417.

import torch


def normalize_speculative_config(engine_args) -> None:
    spec = engine_args.speculative_config
    if not isinstance(spec, dict) or spec.get("method") not in ("dflash", "xpress"):
        return
    spec = dict(spec)
    options = {}
    for name, key in (("xpress_num_passes", "num_passes"), ("xpress_topc", "topc")):
        value = spec.pop(name, None)
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            options[key] = value
    # additional_config carries the XPress options that vLLM 0.32 does not recognize.
    if spec["method"] == "xpress" or options:
        extra = dict(engine_args.additional_config or {})
        extra["spyre_xpress"] = dict(extra.get("spyre_xpress", {})) | options
        engine_args.additional_config = extra
    spec["method"] = "dflash"
    spec["disable_padded_drafter_batch"] = True
    engine_args.speculative_config = spec
    if engine_args.enable_prefix_caching is None:
        engine_args.enable_prefix_caching = False


def validate_dflash_config(config) -> None:
    spec = config.speculative_config
    if spec is None or not spec.use_dflash():
        return
    draft = spec.draft_model_config.hf_config
    target = config.model_config.hf_text_config
    parallel = config.parallel_config
    from spyre_inference import envs

    if envs.SPYRE_MODEL_RUNNER != "v1":
        raise ValueError("Spyre DFlash/XPress currently requires SPYRE_MODEL_RUNNER=v1")
    if config.model_config.dtype != torch.float16 or spec.draft_model_config.dtype != torch.float16:
        raise ValueError("Spyre DFlash/XPress currently requires float16 serving weights")
    if config.model_config.enforce_eager:
        raise ValueError("Spyre DFlash/XPress currently requires compiled execution")
    if (
        config.model_config.quantization is not None
        or spec.draft_model_config.quantization is not None
    ):
        raise ValueError(
            "Spyre DFlash/XPress currently requires unquantized target and draft weights"
        )
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or config.scheduler_config.max_num_seqs != 1
    ):
        raise ValueError("Spyre DFlash/XPress currently requires TP=1, PP=1 and max_num_seqs=1")
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            "Disable prefix caching for Spyre DFlash/XPress: draft context needs every target state"
        )
    if target.model_type != "qwen3" or config.model_config.is_multimodal_model:
        raise ValueError("Spyre DFlash/XPress currently supports the Qwen3 text target")
    if spec.draft_sample_method != "greedy":
        raise ValueError("Spyre DFlash/XPress currently uses greedy proposals")
    block = spec.num_speculative_tokens + 1
    minimum_budget = block + spec.max_num_new_slots_for_drafting
    if config.scheduler_config.max_num_batched_tokens < minimum_budget:
        raise ValueError(
            "max_num_batched_tokens must accommodate a complete draft block "
            "and target verification with vLLM's drafting reservation "
            f"(at least {minimum_budget} tokens)"
        )
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError("The draft must share the target hidden size and full vocabulary")
    if getattr(draft, "draft_vocab_size", draft.vocab_size) not in (None, target.vocab_size):
        raise ValueError("Spyre DFlash/XPress requires the full target vocabulary")
    if getattr(draft, "logit_scale", 1.0) != 1.0 or getattr(draft, "attention_bias", False):
        raise ValueError("Spyre DFlash/XPress requires unscaled logits and no attention bias")
    if getattr(draft, "is_causal", False) or draft.dflash_config.get("causal", False):
        raise ValueError("Spyre DFlash/XPress requires non-causal draft attention")
    if draft.dflash_config.get("use_swa", False) or draft.dflash_config.get(
        "attention_sink_bias", False
    ):
        raise ValueError(
            "Spyre DFlash/XPress does not support sliding attention or attention sinks"
        )
    if any(t != "full_attention" for t in getattr(draft, "layer_types", [])):
        raise ValueError(
            "Spyre DFlash currently supports full non-causal attention in every draft layer"
        )
    layers = draft.dflash_config.get("target_layer_ids", [])
    if (
        not layers
        or layers != sorted(set(layers))
        or not all(0 <= i < target.num_hidden_layers for i in layers)
    ):
        raise ValueError("The draft checkpoint must identify ordered, distinct target layers")
    block_size = getattr(draft, "xpress_block_size", getattr(draft, "block_size", None))
    if block_size != spec.num_speculative_tokens + 1:
        raise ValueError("num_speculative_tokens must equal the checkpoint block size minus one")
    options = config.additional_config.get("spyre_xpress", {})
    xpress = any(
        arch in ("Qwen3XPressModel", "DFlashQwen3XPressModel") for arch in draft.architectures
    )
    if "spyre_xpress" in config.additional_config and not xpress:
        raise ValueError(
            "XPress options require an XPress checkpoint, including its refiner weights"
        )
    if xpress:
        if not hasattr(draft, "xpress_rank"):
            raise ValueError("The XPress checkpoint is missing its refiner configuration")
        for name, key, default in (
            ("xpress_num_passes", "num_passes", 6),
            ("xpress_topc", "topc", 128),
        ):
            value = options.get(key, getattr(draft, name, None))
            if value is None:
                value = default
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            if name == "xpress_topc" and value > draft.vocab_size:
                raise ValueError("xpress_topc must not exceed the draft vocabulary size")
            setattr(draft, name, value)
    if getattr(draft, "sample_from_anchor", False) or draft.dflash_config.get(
        "sample_from_anchor", False
    ):
        raise ValueError("Spyre XPress requires a fixed anchor in slot zero")
    spec.disable_padded_drafter_batch = True
