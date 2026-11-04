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

"""The Muse Glimmer adapter: its norm swap and its text-only default.

Each upstream ``MuseGlimmerRMSNorm`` becomes a vLLM norm; these check that the
replacement computes what transformers does for every variant the model builds, run
through ``forward_native`` on CPU. The text-only default is checked through the real
``EngineArgs.create_model_config`` on a default config saved to disk, so it pins vLLM's
resulting limits rather than the hook's output.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models import muse_glimmer as mg
from vllm.transformers_utils.configs.muse_glimmer import MuseGlimmerConfig

from spyre_inference.models import apply_prelaunch_overrides
from spyre_inference.models.muse_glimmer import disable_vision
from spyre_inference.platform import TorchSpyrePlatform

hfm = pytest.importorskip("transformers.models.muse_glimmer.modeling_muse_glimmer")


@pytest.fixture
def glimmer_dir(tmp_path):
    MuseGlimmerConfig().save_pretrained(tmp_path)
    return str(tmp_path)


def _model_config(model, **kwargs):
    TorchSpyrePlatform.pre_register_and_update()
    return EngineArgs(model=model, skip_tokenizer_init=True, **kwargs).create_model_config()


@pytest.mark.parametrize("limits", [None, {"image": 0}, {"video": 0}])
def test_glimmer_checkpoint_serves_text_only(glimmer_dir, limits):
    """vLLM defaults an unnamed modality to 999, which would keep the tower."""
    mc = _model_config(glimmer_dir, limit_mm_per_prompt=limits)
    assert mc.multimodal_config.language_model_only
    for modality in ("image", "video"):
        assert mc.multimodal_config.get_limit_per_prompt(modality) == 0
    assert not mc._supports_multimodal_inputs()


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("value", [1, {"count": 2}, {}])
def test_requested_image_or_video_is_refused(glimmer_dir, modality, value):
    with pytest.raises(NotImplementedError, match=f"{modality} input is not supported"):
        _model_config(glimmer_dir, limit_mm_per_prompt={modality: value})


def _engine_args(**overrides):
    defaults = dict(language_model_only=False, limit_mm_per_prompt={})
    return SimpleNamespace(**(defaults | overrides))


def _glimmer_config(**fields):
    return SimpleNamespace(model_type="muse_glimmer", **fields)


def test_language_model_only_is_left_alone():
    engine_args = _engine_args(language_model_only=True, limit_mm_per_prompt={"image": 1})
    disable_vision(engine_args, _glimmer_config(vision_config=object()))
    assert engine_args.limit_mm_per_prompt == {"image": 1}


@pytest.mark.parametrize(
    "hf_config",
    [
        _glimmer_config(),
        _glimmer_config(vision_config=object(), has_vision=False),
        SimpleNamespace(model_type="gemma4", vision_config=object()),
        None,
    ],
    ids=["no-vision-config", "has-vision-false", "other-model", "config-load-failed"],
)
def test_checkpoints_without_a_glimmer_tower_are_left_alone(hf_config):
    engine_args = _engine_args()
    disable_vision(engine_args, hf_config)
    assert engine_args == _engine_args()


def test_a_config_load_failure_leaves_engine_args_to_model_config(monkeypatch, glimmer_dir):
    """The probe swallows the error, so ModelConfig raises the real one."""

    def _raise(*args, **kwargs):
        raise OSError("unreadable config")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _raise)
    engine_args = EngineArgs(model=glimmer_dir)
    before = vars(engine_args).copy()
    apply_prelaunch_overrides(engine_args)
    assert vars(engine_args) == before


def _close(norm, ref, x):
    torch.testing.assert_close(norm.forward_native(x), ref(x), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_weightless_norm_matches_transformers(default_vllm_config, dtype):
    """QK norm, embedding norm and perception norm: no weight, any width."""
    from spyre_inference.models.muse_glimmer import _spyre_rms_norm

    norm = _spyre_rms_norm(eps=1e-5, with_scale=False)
    assert isinstance(norm, RMSNorm) and not norm.has_weight
    x = (torch.randn(7, 128) * 30).to(dtype)
    _close(norm, hfm.MuseGlimmerRMSNorm(eps=1e-5, with_scale=False), x)


def test_offset_norm_matches_transformers_centered_norm(default_vllm_config):
    """The sandwich norms scale by ``1 + w``, at the post norms' tiny eps too."""
    from spyre_inference.models.muse_glimmer import _spyre_rms_norm

    norm = _spyre_rms_norm(256, eps=1e-8, weight_offset=1)
    assert isinstance(norm, GemmaRMSNorm)
    ref = hfm.MuseGlimmerTextCenteredRMSNorm(256, eps=1e-8)
    torch.nn.init.normal_(ref.weight, std=0.5)
    norm.weight.data.copy_(ref.weight)
    _close(norm, ref, torch.randn(5, 256) * 100)


def test_scaled_norm_matches_transformers(default_vllm_config):
    """The final norm scales by ``w``."""
    from spyre_inference.models.muse_glimmer import _spyre_rms_norm

    norm = _spyre_rms_norm(256, eps=1e-5)
    assert isinstance(norm, RMSNorm) and norm.has_weight
    ref = hfm.MuseGlimmerRMSNorm(256, eps=1e-5)
    torch.nn.init.normal_(ref.weight, mean=1.0, std=0.5)
    norm.weight.data.copy_(ref.weight)
    _close(norm, ref, torch.randn(5, 256) * 100)


def test_upstream_norm_is_restored_after_construction():
    """The swap is scoped to the model's own construction."""
    from spyre_inference.models.muse_glimmer import _spyre_norms

    original = mg.MuseGlimmerRMSNorm
    with _spyre_norms():
        assert mg.MuseGlimmerRMSNorm is not original
    assert mg.MuseGlimmerRMSNorm is original
