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
through ``forward_native`` on CPU. The text-only default is EngineArgs plumbing, mocked
at the ``get_config`` seam as in ``test_gemma4.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models import muse_glimmer as mg

from spyre_inference.models.muse_glimmer import force_language_model_only

hfm = pytest.importorskip("transformers.models.muse_glimmer.modeling_muse_glimmer")


def _engine_args(**overrides):
    defaults = dict(
        model="meta-models/Muse-Glimmer-30B",
        hf_config_path=None,
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
        language_model_only=False,
        limit_mm_per_prompt={},
    )
    return SimpleNamespace(**(defaults | overrides))


def _patch_get_config(monkeypatch, model_type):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *args, **kwargs: SimpleNamespace(model_type=model_type),
    )


def test_glimmer_checkpoint_defaults_to_text_only(monkeypatch):
    """The tower is then a placeholder: not loaded, and image requests are refused."""
    _patch_get_config(monkeypatch, "muse_glimmer")
    engine_args = _engine_args()
    force_language_model_only(engine_args)
    assert engine_args.language_model_only


def test_explicit_multimodal_limits_are_left_alone(monkeypatch):
    _patch_get_config(monkeypatch, "muse_glimmer")
    engine_args = _engine_args(limit_mm_per_prompt={"image": 1})
    force_language_model_only(engine_args)
    assert not engine_args.language_model_only


def test_other_models_are_left_alone(monkeypatch):
    _patch_get_config(monkeypatch, "gemma4")
    engine_args = _engine_args()
    force_language_model_only(engine_args)
    assert not engine_args.language_model_only


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
