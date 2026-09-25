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

"""Tests for the BharatGen Param config translation.

Param's architecture is registered straight to vLLM's LlamaForCausalLM
(_ALIASED_ARCHS), so what is left to pin is the config: the fields that must be
translated into the spelling Llama reads, the fields that must be left alone
(architectures and model_type above all, since not rewriting them is the point),
and the two cases that must not load silently (a custom_mlp_ratio disagreeing with
intermediate_size, and pretraining_tp > 1).

Needs no network and no card: the configs are built by hand, matching the real
Param-1-5B values.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.models.param import (
    _param_config_override,
    normalize_param_config,
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


def test_leaves_the_architecture_alone():
    """The architecture is registered, so Param is served under its own name.

    Rewriting it to LlamaForCausalLM would work too, but it erases Param from the
    logs and from /v1/models, which is exactly what registering the arch avoids.
    """
    cfg = _param_config_override(_param_config())
    assert cfg.architectures == ["ParamBharatGenForCausalLM"]
    assert cfg.model_type == "parambharatgen"


def test_sets_head_dim_explicitly():
    """Param's config carries no head_dim; its remote code derives hidden // heads.

    vLLM's Llama defaults it the same way, so this pins the value rather than
    supplying a missing one.
    """
    cfg = _param_config_override(_param_config())
    assert cfg.head_dim == 128  # 4096 // 32


def test_preserves_an_explicit_head_dim():
    cfg = _param_config_override(_param_config(head_dim=64))
    assert cfg.head_dim == 64


def test_leaves_shared_fields_untouched():
    """Everything Llama and Param spell the same way must survive verbatim."""
    cfg = _param_config_override(_param_config())
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
    cfg = _param_config_override(_param_config())
    assert cfg.intermediate_size == 14336  # 4096 * 3.5


def test_disagreeing_mlp_ratio_follows_the_ratio_and_warns(caplog):
    """The ratio is what the MLP is built at, so it wins over intermediate_size."""
    with caplog.at_level("WARNING"):
        cfg = _param_config_override(_param_config(custom_mlp_ratio=4.0, intermediate_size=14336))
    assert cfg.intermediate_size == 16384  # 4096 * 4.0, not the declared 14336
    assert "custom_mlp_ratio" in caplog.text


def test_absent_mlp_ratio_keeps_intermediate_size():
    cfg = _param_config()
    del cfg.custom_mlp_ratio
    assert _param_config_override(cfg).intermediate_size == 14336


@pytest.mark.parametrize("tp", [2, 4])
def test_rejects_pretraining_tp_slicing(tp):
    """vLLM's Llama has no equivalent of the remote code's weight slicing."""
    with pytest.raises(ValueError, match="pretraining_tp"):
        _param_config_override(_param_config(pretraining_tp=tp))


@pytest.mark.parametrize(
    "params",
    [
        {"rope_type": "default", "rope_theta": 10000.0},  # what transformers >=5 writes
        {"rope_type": "linear", "factor": 2.0},  # real scaling
        {"type": "linear", "factor": 2.0},  # the legacy spelling
        {"factor": 2.0},  # no type at all
    ],
    ids=["synthesised-default", "modern-scaling", "legacy-spelling", "unspelled"],
)
@pytest.mark.parametrize("attr", ["rope_scaling", "rope_parameters"])
def test_does_not_touch_rope_config(attr, params):
    """Rope is vLLM's to normalize, and it does so after this hook runs.

    ``patch_legacy_rope_type`` promotes the legacy ``type`` key and
    ``standardize_rope_params`` fills in a missing one, both on the config this
    returns. Setting ``rope_type`` here would make the legacy case carry both keys,
    which that same function rejects with ValueError whenever they disagree.
    """
    cfg = _param_config_override(_param_config(**{attr: dict(params)}))
    assert getattr(cfg, attr) == params


def test_override_survives_vllms_dummy_config_probe():
    """vLLM applies hf_overrides twice, the first time to a near-empty config.

    ``get_config`` calls it on ``PretrainedConfig(architectures=[""],
    model_type=...)`` purely to read the model_type back off it, so every field this
    touches has to be optional. An AttributeError here surfaces at ModelConfig
    construction, which a SimpleNamespace config cannot reproduce.
    """
    from transformers import PretrainedConfig

    dummy = PretrainedConfig(architectures=[""], model_type="dummy_parambharatgen")
    cfg = _param_config_override(dummy)
    # Returned unchanged: nothing to translate, and the model_type vLLM reads back
    # off this probe selects the config class, so it must survive verbatim.
    assert cfg.model_type == "dummy_parambharatgen"
    assert cfg.architectures == [""]


def test_normalize_param_config_skips_user_hf_overrides():
    sentinel = {"already": "set"}
    engine_args = SimpleNamespace(hf_overrides=sentinel)
    normalize_param_config(engine_args)
    assert engine_args.hf_overrides is sentinel


def test_normalize_param_config_ignores_non_param_models(monkeypatch):
    """Detection is by config model_type, so a llama checkpoint must pass through."""
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *a, **k: SimpleNamespace(model_type="llama"),
    )
    engine_args = _engine_args()
    normalize_param_config(engine_args)
    assert engine_args.hf_overrides is None


def test_normalize_param_config_sets_the_override_for_param(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *a, **k: SimpleNamespace(model_type="parambharatgen"),
    )
    engine_args = _engine_args()
    normalize_param_config(engine_args)
    assert engine_args.hf_overrides is _param_config_override


def test_normalize_param_config_defers_on_config_load_failure(monkeypatch):
    """ModelConfig loads the same config and raises the real error."""

    def boom(*a, **k):
        raise OSError("no such model")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", boom)
    engine_args = _engine_args()
    normalize_param_config(engine_args)
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
    assert engine_args.hf_overrides is _param_config_override
