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

"""Tests for `spyre_inference/custom_ops/clip_attn_type.py`.

No Spyre hardware: `Attention.__init__` needs a live `VllmConfig`, so it is
monkeypatched to a kwargs-capturing stub before `register()` builds its
subclass around it -- mirrors `tests/models/test_mistral.py`.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from spyre_inference.custom_ops import clip_attn_type


def _stub_init(captured):
    def init(self, *args, **kwargs):
        captured.update(kwargs)
        self.impl = SimpleNamespace()

    return init


@pytest.fixture(autouse=True)
def _reset_clip_attention(monkeypatch):
    """register() mutates the shared `clip` module; reset it before each test
    so tests don't leak a patched class into each other."""
    import vllm.model_executor.models.clip as clip_mod
    from vllm.model_executor.layers.attention import Attention

    monkeypatch.setattr(clip_mod, "Attention", Attention)
    yield


def test_register_defaults_to_causal_encoder_only(monkeypatch):
    import vllm.model_executor.models.clip as clip_mod
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    captured = {}
    monkeypatch.setattr(Attention, "__init__", _stub_init(captured))

    clip_attn_type.register()
    layer = clip_mod.Attention(8, 64, 1.0, prefix="text_model.encoder.layers.0.self_attn.attn")

    assert captured["attn_type"] is AttentionType.ENCODER_ONLY
    assert layer.impl.causal is True


def test_register_does_not_override_an_explicit_attn_type(monkeypatch):
    import vllm.model_executor.models.clip as clip_mod
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    captured = {}
    monkeypatch.setattr(Attention, "__init__", _stub_init(captured))

    clip_attn_type.register()
    layer = clip_mod.Attention(8, 64, 1.0, prefix="p", attn_type=AttentionType.DECODER)

    assert captured["attn_type"] is AttentionType.DECODER
    assert not hasattr(layer.impl, "causal")


def test_register_is_idempotent():
    import vllm.model_executor.models.clip as clip_mod

    clip_attn_type.register()
    patched_cls = clip_mod.Attention
    clip_attn_type.register()

    assert clip_mod.Attention is patched_cls


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
