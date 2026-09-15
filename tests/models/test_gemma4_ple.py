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
from vllm.model_executor.models import gemma4 as upstream

from spyre_inference.models.gemma4 import (
    SpyreGemma4SelfDecoderLayers,
    _PerLayerRows,
    reject_masked_per_layer_vocab,
)


class _CaptureLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.per_layer_input: torch.Tensor | None = None

    def forward(self, positions, hidden_states, residual, *, per_layer_input, **kwargs):
        self.per_layer_input = per_layer_input
        return hidden_states, None


def _backbone(layers: list[nn.Module], rows: _PerLayerRows) -> upstream.Gemma4Model:
    model = upstream.Gemma4Model.__new__(upstream.Gemma4Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.norm = nn.Identity()
    model.fast_prefill_enabled = False
    model.project_per_layer_inputs = lambda inputs_embeds, per_layer_inputs: rows
    return model


def test_upstream_backbone_loop_uses_precomputed_per_layer_rows(monkeypatch):
    """The only PLE cut Spyre reaches is ``Gemma4Model.forward``'s inline
    ``per_layer_inputs[:, layer_idx, :]``: upstream's other one, in ``_run_decoder_layers``,
    is reachable only through ``fast_prefill_forward``, which the platform rejects for
    per-layer-embedding models.
    """
    monkeypatch.setattr(
        upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    layers = [_CaptureLayer(), _CaptureLayer()]
    rows = torch.zeros(3, len(layers), 4).as_subclass(_PerLayerRows)
    rows.spyre_rows = tuple(torch.full((3, 4), i + 1.0) for i in range(len(layers)))

    # Unbound: support_torch_compile's __call__ reads do_not_compile, which only __init__ sets.
    upstream.Gemma4Model.forward(
        _backbone(layers, rows),
        None,
        torch.arange(3),
        None,
        inputs_embeds=torch.zeros(3, 4),
    )

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
