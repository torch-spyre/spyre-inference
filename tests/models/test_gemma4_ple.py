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
from vllm.config import CompilationMode
from vllm.model_executor.models import gemma4 as upstream

from spyre_inference.custom_ops import lazy_compile
from spyre_inference.models import gemma4
from spyre_inference.models.gemma4 import _PerLayerRows


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
    monkeypatch.setattr(
        upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    layers = [_CaptureLayer(), _CaptureLayer()]
    rows = torch.zeros(3, len(layers), 4).as_subclass(_PerLayerRows)
    rows.spyre_rows = tuple(torch.full((3, 4), i + 1.0) for i in range(len(layers)))

    # Avoid support_torch_compile.__call__, whose state is initialized by the skipped constructor.
    upstream.Gemma4Model.forward(
        _backbone(layers, rows), None, torch.arange(3), None, inputs_embeds=torch.zeros(3, 4)
    )

    assert all(layer.per_layer_input is row for layer, row in zip(layers, rows.spyre_rows)), (
        "vLLM changed its PLE row access; update _PerLayerRows before upgrading"
    )


def _build(monkeypatch, *, per_layer_vocab: int, vocab: int = 16, ple: bool = True):
    def parent_init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(vocab_size=vocab)
        decoder = upstream.Gemma4SelfDecoderLayers.__new__(upstream.Gemma4SelfDecoderLayers)
        nn.Module.__init__(decoder)
        decoder.embed_tokens_per_layer = nn.Identity() if ple else None
        decoder.vocab_size_per_layer_input = per_layer_vocab
        self.model = SimpleNamespace(self_decoder=decoder, layers=[])

    monkeypatch.setattr(upstream.Gemma4ForCausalLM, "__init__", parent_init)
    # The retype arms compile state, which reads the live vLLM config.
    monkeypatch.setattr(
        lazy_compile, "get_cached_compilation_config", lambda: SimpleNamespace(mode=None)
    )
    return gemma4.SpyreGemma4ForCausalLM(vllm_config=None)


def test_ple_self_decoder_is_retyped(monkeypatch):
    model = _build(monkeypatch, per_layer_vocab=16)
    assert type(model.model.self_decoder) is gemma4.SpyreGemma4SelfDecoderLayers


def test_masked_per_layer_vocab_is_supported(monkeypatch):
    model = _build(monkeypatch, per_layer_vocab=8)
    assert type(model.model.self_decoder) is gemma4.SpyreGemma4SelfDecoderLayers


def test_checkpoint_without_per_layer_embeddings_keeps_upstream_decoder(monkeypatch):
    model = _build(monkeypatch, per_layer_vocab=8, ple=False)
    assert type(model.model.self_decoder) is upstream.Gemma4SelfDecoderLayers


@pytest.mark.parametrize("num_tokens", [4, 128])
def test_compiled_split_returns_each_layers_own_offset_zero_row(monkeypatch, num_tokens):
    """The real compiled split on the card, with values distinct per token, layer and
    channel so a wrong offset, stride or layer order cannot pass."""
    monkeypatch.setattr(
        lazy_compile,
        "get_cached_compilation_config",
        lambda: SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
    )
    num_layers, width = 3, 256
    decoder = gemma4.SpyreGemma4SelfDecoderLayers.__new__(gemma4.SpyreGemma4SelfDecoderLayers)
    nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(num_hidden_layers=num_layers)
    decoder.arm_outer_compile()
    ple = torch.arange(num_tokens * num_layers * width).reshape(num_tokens, num_layers, width)
    # Below 1024: the card's 16-bit format does not round-trip larger odd integers.
    ple = (ple % 1009).to(torch.float16)

    with torch.inference_mode():
        rows = decoder.split_per_layer_inputs(ple.to("spyre"))

    assert decoder.spyre_compiled_kernels
    assert len(rows) == num_layers
    for i, row in enumerate(rows):
        assert row.storage_offset() == 0
        torch.testing.assert_close(row.cpu(), ple[:, i, :], rtol=0, atol=0)


@pytest.fixture
def ple_decoder(monkeypatch):
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("Spyre hardware is required")

    def build(mode):
        monkeypatch.setattr(
            lazy_compile, "get_cached_compilation_config", lambda: SimpleNamespace(mode=mode)
        )
        decoder = gemma4.SpyreGemma4SelfDecoderLayers.__new__(gemma4.SpyreGemma4SelfDecoderLayers)
        nn.Module.__init__(decoder)
        decoder.config = SimpleNamespace(num_hidden_layers=3)
        decoder.vocab_size_per_layer_input = 64
        decoder.hidden_size_per_layer_input = 64
        decoder.embed_scale_per_layer = 2.0
        # Nonzero row zero distinguishes the upstream fallback row from zeroing outputs.
        weight = (torch.arange(64 * 192).reshape(64, 192) % 97 + 1).to(torch.float16)
        decoder.embed_tokens_per_layer = nn.Embedding.from_pretrained(weight).to("spyre")
        decoder.arm_outer_compile()
        return decoder, weight

    return build


@pytest.mark.parametrize("mode", [CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [1, 31, 32, 33, 63, 64, 65])
def test_compiled_lookup_keeps_upstream_mask(ple_decoder, mode, dtype, num_tokens):
    decoder, weight = ple_decoder(mode)
    ids = torch.tensor([-1, 0, 1, 63, 64, 65], dtype=dtype).repeat(num_tokens)[:num_tokens]
    safe_ids = ids.masked_fill((ids < 0) | (ids >= 64), 0).long()
    expected = (weight[safe_ids] * 2).reshape(num_tokens, 3, 64)

    actual = decoder.get_per_layer_inputs(ids.to("spyre"))

    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert len(decoder.spyre_compiled_kernels) == 1


@pytest.mark.parametrize("lookup_first", [False, True])
def test_lookup_and_split_cache_independent_kernels(ple_decoder, lookup_first):
    decoder, weight = ple_decoder(CompilationMode.STOCK_TORCH_COMPILE)
    ids = torch.tensor([1, 0, 63, 64, -1, 2, 65], dtype=torch.int64)
    safe_ids = torch.tensor([1, 0, 63, 0, 0, 2, 0])
    expected = (weight[safe_ids] * 2).reshape(7, 3, 64)

    if not lookup_first:
        decoder.split_per_layer_inputs(expected.to("spyre"))
    actual = decoder.get_per_layer_inputs(ids.to("spyre"))
    rows = decoder.split_per_layer_inputs(actual)
    cached = dict(decoder.spyre_compiled_kernels)

    assert len(cached) == 2
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    for i, row in enumerate(rows):
        assert row.storage_offset() == 0
        torch.testing.assert_close(row.cpu(), expected[:, i, :], atol=0, rtol=0)

    # New values and allocations must use their own inputs, not a prior call's binding.
    actual = decoder.get_per_layer_inputs(ids.flip(0).to("spyre"))
    rows = decoder.split_per_layer_inputs(actual)
    assert decoder.spyre_compiled_kernels == cached
    for i, row in enumerate(rows):
        torch.testing.assert_close(row.cpu(), expected.flip(0)[:, i, :], atol=0, rtol=0)


def test_enclosing_graph_absorbs_lookup_and_split(ple_decoder):
    decoder, weight = ple_decoder(CompilationMode.NONE)
    ids = torch.tensor([-1, 0, 63, 64], dtype=torch.int64)
    expected = (weight[torch.tensor([0, 0, 63, 0])] * 2).reshape(4, 3, 64)

    def lookup_and_split(ids):
        return decoder.split_per_layer_inputs(decoder.get_per_layer_inputs(ids))

    compiled = torch.compile(lookup_and_split, fullgraph=True, dynamic=False)
    rows = compiled(ids.to("spyre"))

    assert not decoder.spyre_compiled_kernels
    for i, row in enumerate(rows):
        torch.testing.assert_close(row.cpu(), expected[:, i, :], atol=0, rtol=0)


class _RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.variance_epsilon) * self.weight


class _Linear(nn.Module):
    """Spyre's linear form: a stored ``Wᵀ``, ``[in, out]``, and a plain matmul."""

    def __init__(self, weight_t: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight_t, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.weight)


def _projecting_decoder(cls, *, projection_t, norm_weight, num_layers, width):
    decoder = cls.__new__(cls)
    nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(num_hidden_layers=num_layers)
    decoder.hidden_size_per_layer_input = width
    decoder.per_layer_model_projection = _Linear(projection_t)
    decoder.per_layer_projection_norm = _RMSNorm(norm_weight, eps=1e-6)
    decoder.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)))
    decoder.register_buffer(
        "per_layer_projection_scale", torch.tensor(projection_t.shape[0] ** -0.5)
    )
    return decoder


@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_card_projection_is_one_graph_matching_upstream(monkeypatch, num_tokens):
    """The projection, scale, norm and combine run as one compiled graph (eagerly, the
    projection's TP all-gather takes the slow list-form path) and match upstream."""
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("Spyre hardware is required")
    monkeypatch.setattr(
        lazy_compile,
        "get_cached_compilation_config",
        lambda: SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
    )
    num_layers, width, hidden = 3, 256, 512
    g = torch.Generator().manual_seed(0)
    shape = dict(
        projection_t=torch.randn(hidden, num_layers * width, generator=g) * 0.05,
        norm_weight=torch.rand(width, generator=g) + 0.5,
        num_layers=num_layers,
        width=width,
    )
    embeds = torch.randn(num_tokens, hidden, generator=g)
    lookups = torch.randn(num_tokens, num_layers, width, generator=g)

    reference = _projecting_decoder(upstream.Gemma4SelfDecoderLayers, **shape)
    expected = reference.project_per_layer_inputs(embeds, lookups)

    decoder = _projecting_decoder(gemma4.SpyreGemma4SelfDecoderLayers, **shape)
    decoder.arm_outer_compile()
    decoder.to("spyre", torch.float16)
    with torch.inference_mode():
        rows = decoder.project_per_layer_inputs(
            embeds.half().to("spyre"), lookups.half().to("spyre")
        )

    names = {kernel.__name__ for kernel in decoder.spyre_compiled_kernels}
    assert "_spyre_project_per_layer_inputs" in names
    for i, row in enumerate(rows.spyre_rows):
        torch.testing.assert_close(row.cpu().float(), expected[:, i, :], atol=3e-2, rtol=3e-2)
