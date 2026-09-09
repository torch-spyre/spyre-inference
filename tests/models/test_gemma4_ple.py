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

"""Gemma-4's per-layer-embedding (PLE) adaptations.

Both are host-side and need no card: the row cut upstream's backbone loop takes out of the
projected PLE tensor, and the per-layer vocab width the Spyre path requires.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models.gemma4 import _run_decoder_layers

from spyre_inference.models.gemma4 import (
    SpyreGemma4SelfDecoderLayers,
    _PerLayerRows,
    reject_masked_per_layer_vocab,
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


def _decoder_with_masked_ple() -> SpyreGemma4SelfDecoderLayers:
    decoder = SpyreGemma4SelfDecoderLayers.__new__(SpyreGemma4SelfDecoderLayers)
    nn.Module.__init__(decoder)
    decoder.embed_tokens_per_layer = nn.Identity()
    decoder.vocab_size_per_layer_input = 8
    decoder.config = SimpleNamespace(vocab_size=16)
    return decoder


def test_masked_per_layer_vocab_is_rejected():
    """Unconditional: torch-spyre's eager dispatch lowers through Inductor too, so
    enforce_eager is not a way around the mask."""
    with pytest.raises(NotImplementedError, match="vocab_size_per_layer_input"):
        reject_masked_per_layer_vocab(_decoder_with_masked_ple())


def test_full_width_per_layer_vocab_is_accepted():
    decoder = _decoder_with_masked_ple()
    decoder.vocab_size_per_layer_input = decoder.config.vocab_size

    reject_masked_per_layer_vocab(decoder)


def test_checkpoint_without_per_layer_embeddings_is_accepted():
    decoder = _decoder_with_masked_ple()
    decoder.embed_tokens_per_layer = None

    reject_masked_per_layer_vocab(decoder)
