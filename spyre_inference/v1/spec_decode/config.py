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

import functools

import torch

MAX_DFLASH_BATCH_SIZE = 4
_SPYRE_V1_FEATURES = {"xpress speculative decoding", "mixed sliding/full dflash drafts"}


def allow_xpress_on_v1() -> None:
    """Allow XPress and mixed draft attention through Spyre's V1 proposer."""
    from vllm.config.vllm import VllmConfig

    original = VllmConfig._get_v1_model_runner_unsupported_features
    if getattr(original, "_spyre_patched", False):
        return

    @functools.wraps(original)
    def without_spyre_features(self) -> list[str]:
        return [f for f in original(self) if f not in _SPYRE_V1_FEATURES]

    without_spyre_features._spyre_patched = True
    VllmConfig._get_v1_model_runner_unsupported_features = without_spyre_features  # ty: ignore[invalid-assignment]


def normalize_speculative_config(engine_args) -> None:
    spec = engine_args.speculative_config
    if not isinstance(spec, dict):
        return
    method = spec.get("method")
    if method is None:
        model = str(spec.get("model", "")).lower()
        # Match vLLM's name-based detection before it applies cache defaults.
        if "dflash" in model and "eagle-" not in model and "eagle3" not in model:
            method = "dflash"
    if method != "dflash":
        return
    # Set before VllmConfig picks async scheduling, which this flag disables on V1.
    engine_args.speculative_config = spec | {
        "method": "dflash",
        "disable_padded_drafter_batch": True,
    }
    if engine_args.enable_prefix_caching is None:
        engine_args.enable_prefix_caching = False


def validate_dflash_config(config) -> None:
    spec = config.speculative_config
    if spec is None or not spec.use_dflash():
        return
    draft = spec.draft_model_config.hf_config
    architecture = spec.draft_model_config.architecture
    supported_drafts = {"DFlashDraftModel", "Qwen3XPressModel", "DFlashQwen3XPressModel"}
    if architecture not in supported_drafts:
        raise ValueError(
            f"Unsupported Spyre DFlash/XPress draft architecture: {architecture}. "
            f"Supported architectures: {', '.join(sorted(supported_drafts))}"
        )
    target = config.model_config.hf_text_config
    if spec.max_model_len is None and spec.draft_shares_target_positions():
        # The platform's default length cap must not stop drafts inside the target context.
        spec.draft_model_config.max_model_len = config.model_config.max_model_len
    parallel = config.parallel_config
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
        or not 1 <= config.scheduler_config.max_num_seqs <= MAX_DFLASH_BATCH_SIZE
    ):
        raise ValueError(
            "Spyre DFlash/XPress currently requires TP=1, PP=1 and "
            f"max_num_seqs<={MAX_DFLASH_BATCH_SIZE}"
        )
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            "Disable prefix caching for Spyre DFlash/XPress: draft context needs every target state"
        )
    if target.model_type not in {"qwen3", "gemma4", "gemma4_text"} or (
        config.model_config.is_multimodal_model
    ):
        raise ValueError("Spyre DFlash/XPress supports Qwen3 and Gemma 4 text targets")
    if spec.draft_sample_method != "greedy":
        raise ValueError("Spyre DFlash/XPress currently uses greedy proposals")
    max_batch = config.scheduler_config.max_num_seqs
    block = spec.num_speculative_tokens + 1
    minimum_budget = max_batch * (block + spec.max_num_new_slots_for_drafting)
    if config.scheduler_config.max_num_batched_tokens < minimum_budget:
        raise ValueError(
            "max_num_batched_tokens must cover target verification "
            "and vLLM's per-request drafting reservation "
            f"(at least {minimum_budget} tokens)"
        )
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError("The draft must share the target hidden size and full vocabulary")
    if getattr(draft, "draft_vocab_size", draft.vocab_size) not in (None, target.vocab_size):
        raise ValueError("Spyre DFlash/XPress requires the full target vocabulary")
    if getattr(draft, "logit_scale", 1.0) != 1.0 or getattr(draft, "attention_bias", False):
        raise ValueError("Spyre DFlash/XPress requires unscaled logits and no attention bias")
    if draft.dflash_config.get("attention_sink_bias", False):
        raise ValueError("Spyre DFlash/XPress does not support attention sinks")
    layer_types = getattr(draft, "layer_types", None)
    if layer_types is not None and (
        len(layer_types) != draft.num_hidden_layers
        or any(t not in {"full_attention", "sliding_attention"} for t in layer_types)
    ):
        raise ValueError("DFlash requires one full or sliding attention type per draft layer")
    if draft.dflash_config.get("use_swa", False) or "sliding_attention" in (layer_types or []):
        window = draft.dflash_config.get("swa_window_size", getattr(draft, "sliding_window", None))
        if type(window) is not int or window <= 0:
            raise ValueError("DFlash sliding attention requires a positive window size")
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
    if architecture in ("Qwen3XPressModel", "DFlashQwen3XPressModel"):
        if not hasattr(draft, "xpress_rank"):
            raise ValueError("The XPress checkpoint is missing its refiner configuration")
        # SpeculativeConfig has already range-checked xpress_topc and written it here.
        if getattr(draft, "xpress_topc", None) is None:
            draft.xpress_topc = 128
        draft.xpress_topc = min(draft.xpress_topc, draft.vocab_size)
        if not 1 <= draft.xpress_topc <= 1024:
            raise ValueError(
                "Spyre XPress requires 1 to 1024 candidates: "
                "min(xpress_topc, vocab_size) must be in this range; "
                "full-vocabulary refinement (xpress_topc=0) is unsupported"
            )
        passes = getattr(draft, "xpress_num_passes", None)
        if passes is None:
            draft.xpress_num_passes = 6
        elif type(passes) is not int or passes < 0:
            raise ValueError("xpress_num_passes must be a nonnegative integer")
    if getattr(draft, "sample_from_anchor", False) or draft.dflash_config.get(
        "sample_from_anchor", False
    ):
        raise ValueError("Spyre XPress requires a fixed anchor in slot zero")
    spec.disable_padded_drafter_batch = True
