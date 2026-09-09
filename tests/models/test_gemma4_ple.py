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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models.gemma4 import (
    Gemma4SelfDecoderLayers,
    _run_decoder_layers,
)

from spyre_inference.models.gemma4 import (
    SpyreGemma4SelfDecoderLayers,
    _PerLayerRows,
)


def test_upstream_decoder_uses_precomputed_per_layer_rows():
    class CaptureLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.per_layer_input = None

        def forward(self, positions, hidden_states, residual, *, per_layer_input, **kwargs):
            self.per_layer_input = per_layer_input
            return hidden_states, None

    layers = [CaptureLayer(), CaptureLayer()]
    ple = torch.zeros(3, 2, 4)
    rows = ple.as_subclass(_PerLayerRows)
    rows.spyre_rows = tuple(torch.full((3, 4), i + 1.0) for i in range(2))

    _run_decoder_layers(layers, 0, torch.arange(3), torch.zeros(3, 4), rows)

    assert all(layer.per_layer_input is row for layer, row in zip(layers, rows.spyre_rows)), (
        "vLLM changed its PLE row access; update _PerLayerRows before upgrading"
    )


def _decoder_with_masked_ple(*, compile_enabled: bool) -> SpyreGemma4SelfDecoderLayers:
    decoder = SpyreGemma4SelfDecoderLayers.__new__(SpyreGemma4SelfDecoderLayers)
    nn.Module.__init__(decoder)
    decoder.embed_tokens_per_layer = nn.Identity()
    decoder.vocab_size_per_layer_input = 8
    decoder.config = SimpleNamespace(vocab_size=16)
    decoder.spyre_compile_enabled = compile_enabled
    return decoder


def test_compiled_masked_ple_is_rejected():
    decoder = _decoder_with_masked_ple(compile_enabled=True)

    with pytest.raises(NotImplementedError, match="vocab_size_per_layer_input"):
        decoder.get_per_layer_inputs(torch.tensor([1]))


def test_eager_masked_ple_delegates_upstream(monkeypatch):
    sentinel = torch.tensor([42])
    monkeypatch.setattr(
        Gemma4SelfDecoderLayers,
        "get_per_layer_inputs",
        lambda self, input_ids: sentinel,
    )
    decoder = _decoder_with_masked_ple(compile_enabled=False)

    assert decoder.get_per_layer_inputs(torch.tensor([1])) is sentinel
