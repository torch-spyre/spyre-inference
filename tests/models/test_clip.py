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

"""Tests for `spyre_inference/models/clip.py`. `get_config` is monkeypatched
to a stub -- no Spyre hardware, no real HF config fetch."""

from __future__ import annotations

import sys
import types

import pytest

from spyre_inference.models.clip import force_disable_chunked_prefill


def _fake_engine_args(model_type: str | None, enable_chunked_prefill=None):
    return types.SimpleNamespace(
        enable_chunked_prefill=enable_chunked_prefill,
        hf_config_path=None,
        model="some/model",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
        _model_type=model_type,
    )


def _patch_get_config(monkeypatch, model_type: str | None):
    import spyre_inference.models.clip as clip_mod

    def _fake_get_config(*args, **kwargs):
        return types.SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config", _fake_get_config, raising=False
    )
    return clip_mod


def test_disables_chunked_prefill_for_clip(monkeypatch):
    _patch_get_config(monkeypatch, "clip")
    args = _fake_engine_args("clip")

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is False


def test_leaves_non_clip_models_untouched(monkeypatch):
    _patch_get_config(monkeypatch, "bert")
    args = _fake_engine_args("bert")

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is None


def test_respects_an_explicit_user_choice(monkeypatch):
    _patch_get_config(monkeypatch, "clip")
    args = _fake_engine_args("clip", enable_chunked_prefill=True)

    force_disable_chunked_prefill(args)

    assert args.enable_chunked_prefill is True


def test_tolerates_config_load_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _raise, raising=False)
    args = _fake_engine_args(None)

    force_disable_chunked_prefill(args)  # must not raise

    assert args.enable_chunked_prefill is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
