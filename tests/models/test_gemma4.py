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

"""`force_text_backbone` must keep repairing plain text gemma-4 checkpoints (the
transformers>=5.16 heterogeneous-config issue) while leaving a real multimodal
Gemma4Config (populated vision_config/audio_config) alone, so vLLM's own
Gemma4ForConditionalGeneration can resolve normally. No card needed: this is pure
EngineArgs/config-object plumbing, mocked at the vllm.transformers_utils.config.get_config
seam `force_text_backbone` imports locally.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.models.gemma4 import (
    GEMMA4_TEXT_BACKBONE_OVERRIDE,
    force_text_backbone,
    repair_head_dim_access,
)


def _engine_args(**overrides):
    defaults = dict(
        hf_overrides=None,
        hf_config_path=None,
        model="google/gemma-4-26B-A4B",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_get_config(monkeypatch, hf_config):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *args, **kwargs: hf_config,
    )


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_force_text_backbone_applies_for_pure_text_checkpoint(monkeypatch, model_type):
    """A plain text checkpoint (no vision_config/audio_config) still gets the fix."""
    hf_config = SimpleNamespace(model_type=model_type, vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides == GEMMA4_TEXT_BACKBONE_OVERRIDE


def test_force_text_backbone_skips_for_vlm_checkpoint_with_vision_config(monkeypatch):
    """A vision checkpoint must keep its architectures -- forcing the text backbone would
    strip the tower. It needs no override at all: the head-dim repair runs from the
    platform hook."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=None,
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


def test_audio_only_checkpoint_is_rejected(monkeypatch):
    """Falling through to the text backbone would silently drop the tower, leaving a
    model that looks fine but ignores its audio inputs."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=None,
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    with pytest.raises(NotImplementedError, match="audio is not supported"):
        force_text_backbone(_engine_args())


def test_vision_plus_audio_checkpoint_is_allowed_for_its_vision_path(monkeypatch):
    """Both towers present: the vision path still works, so this warns rather than
    refusing to load."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


def test_repair_head_dim_access_reaches_the_nested_text_config():
    text_config = SimpleNamespace(model_type="gemma4_text")
    config = SimpleNamespace(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
    )

    repair_head_dim_access(config)

    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    assert config.allow_global_per_layer_attribute_access is True
    assert text_config.allow_global_per_layer_attribute_access is True


def test_repair_head_dim_access_ignores_an_unrelated_config():
    """It runs from `check_and_update_config`, i.e. for every model Spyre loads."""
    config = SimpleNamespace(model_type="llama")

    repair_head_dim_access(config)

    assert not hasattr(config, "allow_global_per_layer_attribute_access")


def test_force_text_backbone_ignores_unrelated_model_type(monkeypatch):
    hf_config = SimpleNamespace(model_type="llama", vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(model_type="gemma4", vision_config=object(), audio_config=None), True),
        # Audio is out of scope, so an audio-only config is not "multimodal" here.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=object()), False),
        # Text-only gemma-4 is already validated in fp16 here; don't move it.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="gemma4_text", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="llama", vision_config=None, audio_config=None), False),
    ],
)
def test_is_multimodal_gemma4_requires_a_vision_tower(hf_config, expected):
    """``force_text_backbone`` gates on this: a checkpoint with a tower keeps the
    multimodal architecture, one without it is pinned to the text backbone."""
    from spyre_inference.models.gemma4 import is_multimodal_gemma4

    assert is_multimodal_gemma4(hf_config) is expected


def test_multimodal_gemma4_delegates_its_text_half_through_the_model_registry():
    """Why ``Gemma4ForConditionalGeneration`` needs no ``_ADAPTED_ARCHS`` entry: it
    resolves its text half through ``ModelRegistry``, so it already lands on
    ``SpyreGemma4ForCausalLM``. If upstream ever built that model directly, the PLE and
    MoE adaptations would vanish silently -- hence a tripwire rather than a comment.
    """
    import inspect

    gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

    source = inspect.getsource(gemma4_mm.Gemma4ForConditionalGeneration.__init__)
    assert "init_vllm_registered_model" in source, (
        "Gemma4ForConditionalGeneration no longer resolves its text model through "
        "the vLLM registry; SpyreGemma4ForCausalLM's MoE/PLE adaptations are being "
        "bypassed. Register a Spyre subclass of it in models/_ADAPTED_ARCHS."
    )
    assert '"Gemma4ForCausalLM"' in source, (
        "Gemma4ForConditionalGeneration no longer names Gemma4ForCausalLM as its "
        "text architecture; check which architecture it resolves now and make sure "
        "spyre_models() adapts that one."
    )


def test_force_text_backbone_respects_user_supplied_hf_overrides(monkeypatch):
    """Skipped entirely (not even a get_config call) when the user already set
    hf_overrides. Safe because the head-dim repair no longer rides on this override:
    `repair_head_dim_access` runs from the platform hook, so a user override -- a dict
    such as GEMMA4_TEXT_BACKBONE_OVERRIDE included -- still gets it.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("get_config should not be called when hf_overrides is already set")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _boom)

    def _user_override(config):
        return config

    engine_args = _engine_args(hf_overrides=_user_override)
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _user_override
