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

"""Tests for the BharatGen Param -> native Llama config mapping.

Param is served by rewriting its config to LlamaForCausalLM before ModelConfig is
built, so these pin the rewrite itself: the fields that must be translated, the
fields that must be left alone, and the two cases that must not load silently
(a custom_mlp_ratio disagreeing with intermediate_size, and pretraining_tp > 1).

Needs no network and no card: the configs are built by hand, matching the real
Param-1-5B values.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.models.param import (
    _param_llama_override,
    force_llama_architecture,
)

# Param-1-5B's real config values.
PARAM_1_5B = {
    "model_type": "parambharatgen",
    "architectures": ["ParamBharatGenForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "custom_mlp_ratio": 3.5,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "vocab_size": 256000,
    "tie_word_embeddings": False,
    "pretraining_tp": 1,
}


def _param_config(**overrides):
    return SimpleNamespace(**{**PARAM_1_5B, **overrides})


def test_rewrites_architecture_to_llama():
    cfg = _param_llama_override(_param_config())
    assert cfg.architectures == ["LlamaForCausalLM"]
    assert cfg.model_type == "llama"


def test_sets_head_dim_explicitly():
    """Param's config carries no head_dim; its remote code derives hidden // heads."""
    cfg = _param_llama_override(_param_config())
    assert cfg.head_dim == 128  # 4096 // 32


def test_preserves_an_explicit_head_dim():
    cfg = _param_llama_override(_param_config(head_dim=64))
    assert cfg.head_dim == 64


def test_leaves_shared_fields_untouched():
    """Everything Llama and Param spell the same way must survive verbatim."""
    cfg = _param_llama_override(_param_config())
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "hidden_act",
        "vocab_size",
        "tie_word_embeddings",
    ):
        assert getattr(cfg, field) == PARAM_1_5B[field], field


def test_folds_custom_mlp_ratio_into_intermediate_size():
    """ParamBharatGenMLP prefers custom_mlp_ratio; Llama reads only intermediate_size.

    They agree on Param-1-5B, so the value must come out unchanged.
    """
    cfg = _param_llama_override(_param_config())
    assert cfg.intermediate_size == 14336  # 4096 * 3.5


def test_disagreeing_mlp_ratio_follows_the_ratio_and_warns(caplog):
    """The ratio is what the MLP is built at, so it wins over intermediate_size."""
    with caplog.at_level("WARNING"):
        cfg = _param_llama_override(_param_config(custom_mlp_ratio=4.0, intermediate_size=14336))
    assert cfg.intermediate_size == 16384  # 4096 * 4.0, not the declared 14336
    assert "custom_mlp_ratio" in caplog.text


def test_absent_mlp_ratio_keeps_intermediate_size():
    cfg = _param_config()
    del cfg.custom_mlp_ratio
    assert _param_llama_override(cfg).intermediate_size == 14336


@pytest.mark.parametrize("tp", [2, 4])
def test_rejects_pretraining_tp_slicing(tp):
    """vLLM's Llama has no equivalent of the remote code's weight slicing."""
    with pytest.raises(ValueError, match="pretraining_tp"):
        _param_llama_override(_param_config(pretraining_tp=tp))


@pytest.mark.parametrize("attr", ["rope_scaling", "rope_parameters"])
def test_clears_synthesised_default_rope_scaling(attr):
    """transformers >=5 synthesises {"rope_type": "default"}; it means no scaling."""
    cfg = _param_llama_override(
        _param_config(**{attr: {"rope_type": "default", "rope_theta": 10000.0}})
    )
    assert getattr(cfg, attr) is None


def test_keeps_real_rope_scaling():
    """A Param config declaring actual scaling must never be silently un-scaled."""
    scaling = {"rope_type": "linear", "factor": 2.0}
    cfg = _param_llama_override(_param_config(rope_scaling=dict(scaling)))
    assert cfg.rope_scaling == scaling


def test_override_survives_vllms_dummy_config_probe():
    """vLLM applies hf_overrides twice, the first time to a near-empty config.

    ``get_config`` calls it on ``PretrainedConfig(architectures=[""],
    model_type=...)`` purely to read back the rewritten model_type, so every field
    other than those two has to be optional here.
    """
    from transformers import PretrainedConfig

    dummy = PretrainedConfig(architectures=[""], model_type="dummy_parambharatgen")
    assert _param_llama_override(dummy).model_type == "llama"


def test_force_llama_architecture_skips_user_hf_overrides():
    sentinel = {"already": "set"}
    engine_args = SimpleNamespace(hf_overrides=sentinel)
    force_llama_architecture(engine_args)
    assert engine_args.hf_overrides is sentinel


def test_force_llama_architecture_ignores_non_param_models(monkeypatch):
    """Detection is by config model_type, so a llama checkpoint must pass through."""
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *a, **k: SimpleNamespace(model_type="llama"),
    )
    engine_args = _engine_args()
    force_llama_architecture(engine_args)
    assert engine_args.hf_overrides is None


def test_force_llama_architecture_sets_the_override_for_param(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *a, **k: SimpleNamespace(model_type="parambharatgen"),
    )
    engine_args = _engine_args()
    force_llama_architecture(engine_args)
    assert engine_args.hf_overrides is _param_llama_override


def test_force_llama_architecture_defers_on_config_load_failure(monkeypatch):
    """ModelConfig loads the same config and raises the real error."""

    def boom(*a, **k):
        raise OSError("no such model")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", boom)
    engine_args = _engine_args()
    force_llama_architecture(engine_args)
    assert engine_args.hf_overrides is None


def _engine_args(**overrides):
    return SimpleNamespace(
        **{
            "hf_overrides": None,
            "model": "bharatgenai/Param-1-5B",
            "hf_config_path": None,
            "trust_remote_code": False,
            "revision": None,
            "code_revision": None,
            "config_format": "auto",
            "hf_token": None,
            **overrides,
        }
    )


def test_prelaunch_hook_invokes_the_param_override(monkeypatch):
    """The mapping is only reachable if apply_prelaunch_overrides calls it."""
    from spyre_inference import models

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *a, **k: SimpleNamespace(model_type="parambharatgen"),
    )
    engine_args = _engine_args()
    models.apply_prelaunch_overrides(engine_args)
    assert engine_args.hf_overrides is _param_llama_override
