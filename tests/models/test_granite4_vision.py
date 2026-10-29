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

"""``SpyreGranite4VisionForConditionalGeneration``'s vision pipeline against upstream.

The window-major layout, the projectors on it, the unpad-and-newline packing and the deepstack
hand-off are each pinned to upstream's result on CPU, at a toy size: 16x16 patches per tile,
8x8 windows, 4x4 queries, so the same 2x2 merge and 2x2 windows per tile as the real model.
"""

import copy
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

blip2 = pytest.importorskip("vllm.model_executor.models.blip2")
granite4_vision = pytest.importorskip("vllm.model_executor.models.granite4_vision")

pytestmark = pytest.mark.granite4_vision

SIDE, WINDOW, QUERY, HIDDEN, LM_HIDDEN = 16, 8, 4, 64, 64
N = SIDE // WINDOW  # windows per tile side
TOKENS_PER_TILE = (N * QUERY) ** 2


def _config(levels_layerwise: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        vision_config=SimpleNamespace(image_size=SIDE, patch_size=1, hidden_size=HIDDEN),
        text_config=SimpleNamespace(hidden_size=LM_HIDDEN),
        downsample_rate=f"{QUERY}/{WINDOW}",
        projector_dropout=0.0,
        image_grid_pinpoints=[[SIDE, SIDE], [SIDE, 2 * SIDE], [2 * SIDE, SIDE]],
        deepstack_layer_map=[[-1, 0]] * levels_layerwise,
        spatial_target_layers=[1, 2, 3, 4],
    )


def _projector(spatial_offset: int | None) -> nn.Module:
    torch.manual_seed(0)
    proj = granite4_vision.WindowQFormerDownsampler(
        _config(), quant_config=None, cache_config=None, spatial_offset=spatial_offset
    ).eval()
    for p in proj.parameters():
        p.data.normal_(std=0.1)
    return proj


def test_window_major_order_is_a_permutation():
    from spyre_inference.models.granite4_vision import window_major_order

    tile, local = window_major_order(SIDE, WINDOW, QUERY)
    assert sorted(tile.tolist()) == list(range(SIDE * SIDE))
    assert sorted(local.tolist()) == list(range(WINDOW * WINDOW))


def test_unwindow_matches_upstream_unwin():
    from spyre_inference.models.granite4_vision import unwindow

    proj = _projector(None)
    tiles = 3
    windows = torch.randn(tiles * N * N, QUERY * QUERY, HIDDEN)
    got = unwindow(windows.reshape(-1, HIDDEN), N, QUERY)
    torch.testing.assert_close(got, proj._unwin(windows, n=N, win=QUERY), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [None, 0, 1, 2, 3])
def test_project_matches_upstream_projector(offset):
    """Window-major patches through ``_project`` equal upstream's raster projector, for the
    area-mean projector and for each spatial offset."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )
    from spyre_inference.models.granite4_vision import unwindow, window_major_order

    tile, local = window_major_order(SIDE, WINDOW, QUERY)
    reference = _projector(offset)
    mine = copy.deepcopy(reference)
    mine.image_positions.data = reference.image_positions.data[:, local]
    model = SimpleNamespace(patches_per_token=4, _query=QUERY, _window=WINDOW)

    tiles = 2
    torch.manual_seed(1)
    raster = torch.randn(tiles, SIDE * SIDE, HIDDEN)
    with torch.no_grad():
        want = reference(raster)
        got = Spyre._project(
            model, mine, raster[:, tile].reshape(-1, HIDDEN), _down_weights(offset)
        )
        got = unwindow(got, N, QUERY)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def _down_weights(offset: int | None) -> torch.Tensor:
    if offset is None:
        return torch.full((4, 1, HIDDEN), 0.25)
    down = torch.zeros(4, 1, HIDDEN)
    down[offset] = 1.0
    return down


def test_every_projector_shares_one_compiled_graph():
    """The eight projectors differ only in weights and downsample, both graph inputs, so one
    compiled ``_project`` serves them all instead of one graph per projector."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    model = SimpleNamespace(patches_per_token=4, _query=QUERY, _window=WINDOW)
    project = torch.compile(
        lambda proj, tap, down: Spyre._project(model, proj, tap, down),
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    tap = torch.randn(2 * SIDE * SIDE, HIDDEN)
    with torch.no_grad():
        for offset in (None, None, 0, 1, 2, 3):
            project(_projector(offset), tap, _down_weights(offset))
    assert len(graphs) == 1


def _postprocess_model(levels: int = 5) -> SimpleNamespace:
    """The attributes ``postprocess_encoder_output`` reads, without a full model."""
    config = _config()
    return SimpleNamespace(
        config=config,
        _side=SIDE,
        _window=WINDOW,
        _query=QUERY,
        _levels=levels,
        _image_newline_host=torch.randn(LM_HIDDEN),
        _image_sizes=lambda kw: kw["image_sizes"],
    )


def test_postprocess_matches_upstream_packing():
    """Per level, the host packing is upstream's ``_pack_and_unpad_image_features`` on raster
    tokens; the placeholder rows are the packed ones, not the encoder tokens."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    levels = 5
    model = _postprocess_model(levels)
    # A 1-tile image (base only) and a 2-tile image (base + one anyres tile) of size 16x32.
    image_sizes = torch.tensor([[SIDE, SIDE], [SIDE, 2 * SIDE]])
    tiles = [2, 3]  # base + the best resolution's tiles: 16x16 -> 1 + 1, 16x32 -> 1 + 2
    torch.manual_seed(2)
    raster = [torch.randn(t, TOKENS_PER_TILE, levels * LM_HIDDEN) for t in tiles]

    def window_major(r: torch.Tensor) -> torch.Tensor:
        t = r.shape[0]
        r = r.view(t, N, QUERY, N, QUERY, -1).permute(0, 1, 3, 2, 4, 5)
        return r.reshape(t * N * N * QUERY * QUERY, -1)

    encoded = torch.cat([window_major(r) for r in raster] + [torch.zeros(7, levels * LM_HIDDEN)])
    counts = [t * TOKENS_PER_TILE for t in tiles]
    dest: list[torch.Tensor | None] = [None, None]
    model_post = Spyre.postprocess_encoder_output
    model_post(
        model,
        {"default": encoded},
        [0, 1],
        counts,
        dest,
        batch_mm_kwargs={"image_sizes": image_sizes},
    )

    packer = SimpleNamespace(
        config=model.config,
        _downsample_rate=Fraction(model.config.downsample_rate),
        image_newline=model._image_newline_host,
    )
    unpack = granite4_vision.Granite4VisionForConditionalGeneration._pack_and_unpad_image_features
    want_levels = [
        unpack(
            packer, [r[..., lvl * LM_HIDDEN : (lvl + 1) * LM_HIDDEN] for r in raster], image_sizes
        )
        for lvl in range(levels)
    ]
    for i, got in enumerate(dest):
        assert got is not None
        want = torch.cat([lvl[i] for lvl in want_levels], dim=-1)
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_embed_input_ids_hands_each_level_to_the_decoder_buffers():
    """Image rows are zeroed in the text embedding, scaled like upstream's, and each level
    lands unscaled at its placeholder rows of its own buffer."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    levels, tokens, multiplier = 3, 12, 2.0
    inner = SimpleNamespace(
        embed_input_ids=lambda ids: ids.float()[:, None].expand(-1, LM_HIDDEN).contiguous(),
        config=SimpleNamespace(embedding_multiplier=multiplier),
    )
    model = nn.Module()
    model.language_model = SimpleNamespace(model=inner)
    model._ds_buffers = [torch.zeros(16, LM_HIDDEN) for _ in range(levels)]
    model._ds_layer_indices = list(range(levels))
    model._ds_num_tokens = 0
    model.vision_merge = lambda text, mask: torch.where(mask, torch.zeros_like(text), text)

    input_ids = torch.arange(100, 100 + tokens)
    is_multimodal = torch.zeros(tokens, dtype=torch.bool)
    is_multimodal[[2, 3, 4, 8]] = True
    rows = [torch.randn(3, levels * LM_HIDDEN), torch.randn(1, levels * LM_HIDDEN)]

    out = Spyre.embed_input_ids(model, input_ids, rows, is_multimodal=is_multimodal)

    text = input_ids.float()[:, None].expand(-1, LM_HIDDEN) * multiplier
    want = torch.where(is_multimodal[:, None], torch.zeros_like(text), text)
    torch.testing.assert_close(out, want, atol=0, rtol=0)
    flat = torch.cat(rows)
    for level, buf in enumerate(model._ds_buffers):
        expected = torch.zeros(16, LM_HIDDEN)
        expected[:tokens][is_multimodal] = flat[:, level * LM_HIDDEN : (level + 1) * LM_HIDDEN]
        torch.testing.assert_close(buf, expected, atol=0, rtol=0)
    assert model._ds_num_tokens == tokens


@pytest.mark.parametrize(
    "positions",
    [[0, 3, 7], [1, 2, 5, 6], [0, 1, 2, 3], [4, 5, 6, 7]],
    ids=["scattered", "two_pairs", "contiguous_prefix", "contiguous_suffix"],
)
def test_embed_input_ids_narrow_staging_matches_full_buffer_staging(positions):
    """Staging ``[tokens, H]`` and copying into ``[:tokens]`` fills the same live rows as
    staging the whole buffer, and leaves the rows past ``tokens`` alone."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    levels, tokens, capacity = 2, 8, 16
    inner = SimpleNamespace(
        embed_input_ids=lambda ids: ids.float()[:, None].expand(-1, LM_HIDDEN).contiguous(),
        config=SimpleNamespace(embedding_multiplier=1.0),
    )
    model = nn.Module()
    model.language_model = SimpleNamespace(model=inner)
    stale = 7.0
    model._ds_buffers = [torch.full((capacity, LM_HIDDEN), stale) for _ in range(levels)]
    model._ds_num_tokens = 0
    model.vision_merge = lambda text, mask: torch.where(mask, torch.zeros_like(text), text)

    is_multimodal = torch.zeros(tokens, dtype=torch.bool)
    is_multimodal[positions] = True
    rows = torch.randn(len(positions), levels * LM_HIDDEN)

    Spyre.embed_input_ids(model, torch.arange(tokens), [rows], is_multimodal=is_multimodal)

    for level, buf in enumerate(model._ds_buffers):
        full = torch.zeros(capacity, LM_HIDDEN)
        full[:tokens][is_multimodal] = rows[:, level * LM_HIDDEN : (level + 1) * LM_HIDDEN]
        torch.testing.assert_close(buf[:tokens], full[:tokens], atol=0, rtol=0)
        assert torch.all(buf[tokens:] == stale)


def test_embed_input_ids_count_mismatch_fails_like_upstream():
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    inner = SimpleNamespace(
        embed_input_ids=lambda ids: ids.float()[:, None].expand(-1, LM_HIDDEN).contiguous(),
        config=SimpleNamespace(embedding_multiplier=1.0),
    )
    model = nn.Module()
    model.language_model = SimpleNamespace(model=inner)
    model._ds_buffers = [torch.zeros(16, LM_HIDDEN)]
    model._ds_num_tokens = 0
    is_multimodal = torch.zeros(8, dtype=torch.bool)
    is_multimodal[:3] = True
    with pytest.raises(ValueError, match="2 multimodal tokens to 3 placeholders"):
        Spyre.embed_input_ids(
            model, torch.arange(8), [torch.ones(2, LM_HIDDEN)], is_multimodal=is_multimodal
        )


def test_vision_inputs_order_patches_window_major():
    """Patches come out in ``tile_order``, each flattened as ``(c, py, px)`` like the conv."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )
    from spyre_inference.models.granite4_vision import window_major_order

    patch = 2
    side = 4  # patches per side; window 2, query 1 keeps the order non-trivial
    tile_order, _ = window_major_order(side, 2, 1)
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=patch, image_size=side * patch),
            image_grid_pinpoints=[[side * patch, side * patch]],
        ),
        _side=side,
        _tile_order=tile_order,
        _num_tiles=lambda kw: [1],
        _tiles=lambda kw: [kw["pixel_values"]],
    )
    pixels = torch.randn(1, 3, side * patch, side * patch)
    got = Spyre.vision_inputs(model, {"pixel_values": pixels})["patches"]

    raster = (
        pixels.reshape(3, side, patch, side, patch).permute(1, 3, 0, 2, 4).reshape(side * side, -1)
    )
    torch.testing.assert_close(got, raster[tile_order].half(), atol=0, rtol=0)


# --- Q-Former attention ---

# HEAD_DIM must be stick-aligned (a multiple of 64) on Spyre, as Granite Vision's Q-Former is.
QF_HEAD_DIM, QF_HEADS = 64, 4
QF_HIDDEN = QF_HEADS * QF_HEAD_DIM

# The swap never changes the upstream class, so this is the stock forward whenever it is read.
_STOCK_QFORMER_FORWARD = blip2.Blip2QFormerMultiHeadAttention.forward


def _finish_weight_loading(module: nn.Module) -> None:
    """Run ``process_weights_after_loading`` on every vLLM linear in ``module``, so the Spyre
    linears have their transposed weight and ``spyre_row_padding`` set before any forward."""
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _qformer_config(cross: bool = False):
    kwargs = {"encoder_hidden_size": QF_HIDDEN} if cross else {}
    return blip2.Blip2QFormerConfig(
        hidden_size=QF_HIDDEN,
        num_attention_heads=QF_HEADS,
        attention_probs_dropout_prob=0.0,
        **kwargs,
    )


def _qformer_attention(cross: bool = False, seed: int = 0) -> nn.Module:
    """A real upstream ``Blip2QFormerMultiHeadAttention`` with deterministic weights."""
    attn = blip2.Blip2QFormerMultiHeadAttention(
        _qformer_config(cross), quant_config=None, cache_config=None, is_cross_attention=cross
    ).to(torch.float16)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for p in attn.parameters():
        p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
    _finish_weight_loading(attn)
    return attn


def _swapped(attn: nn.Module) -> nn.Module:
    """``attn`` as a model holds it after construction: a child module, swapped."""
    from spyre_inference.models.granite4_vision import use_spyre_qformer_attention

    parent = nn.Module()
    parent.attn = attn
    use_spyre_qformer_attention(parent)
    return parent.attn


def test_qformer_swap_target_still_exists():
    """The class the swap looks for must still exist upstream; otherwise the swap silently
    matches nothing."""
    assert getattr(blip2, "Blip2QFormerMultiHeadAttention", None) is not None, (
        "vllm.model_executor.models.blip2.Blip2QFormerMultiHeadAttention is gone, so "
        "use_spyre_qformer_attention swaps nothing and must be updated"
    )


def test_qformer_swap_replaces_only_the_given_modules():
    """The Spyre class goes onto the modules under ``root``, not onto the upstream class other
    models share. Parameters and their names stay put, so the checkpoint maps unchanged."""
    from spyre_inference.models.granite4_vision import (
        SpyreBlip2QFormerMultiHeadAttention,
        use_spyre_qformer_attention,
    )

    mine, other = nn.Module(), nn.Module()
    mine.attn, other.attn = _qformer_attention(), _qformer_attention()
    names = list(mine.state_dict())

    assert use_spyre_qformer_attention(mine) == 1
    assert use_spyre_qformer_attention(mine) == 0  # per-load re-entrancy

    assert type(mine.attn) is SpyreBlip2QFormerMultiHeadAttention
    assert type(other.attn) is blip2.Blip2QFormerMultiHeadAttention
    assert blip2.Blip2QFormerMultiHeadAttention.forward is _STOCK_QFORMER_FORWARD
    assert list(mine.state_dict()) == names


def test_qformer_swap_reaches_self_and_cross_attention():
    """A Q-Former layer holds one self-attention and one cross-attention; both are swapped."""
    from spyre_inference.models.granite4_vision import (
        SpyreBlip2QFormerMultiHeadAttention,
        use_spyre_qformer_attention,
    )

    layer = blip2.Blip2QFormerLayer(
        _qformer_config(cross=True), quant_config=None, cache_config=None, layer_idx=0
    )
    assert use_spyre_qformer_attention(layer) == 2
    assert type(layer.attention.attention) is SpyreBlip2QFormerMultiHeadAttention
    assert type(layer.crossattention.attention) is SpyreBlip2QFormerMultiHeadAttention


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_qformer_attention_matches_stock(tp_group, bsz):
    rng = torch.Generator(device="cpu").manual_seed(10 + bsz)
    hidden_states = torch.randn(bsz, 8, QF_HIDDEN, dtype=torch.float16, generator=rng)

    expected = _STOCK_QFORMER_FORWARD(_qformer_attention(), hidden_states)
    actual = _swapped(_qformer_attention())(hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_qformer_cross_attention_matches_stock(tp_group, bsz):
    rng = torch.Generator(device="cpu").manual_seed(20 + bsz)
    hidden_states = torch.randn(bsz, 8, QF_HIDDEN, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, QF_HIDDEN, dtype=torch.float16, generator=rng)

    expected = _STOCK_QFORMER_FORWARD(
        _qformer_attention(cross=True, seed=2), hidden_states, encoder_hidden_states
    )
    actual = _swapped(_qformer_attention(cross=True, seed=2))(hidden_states, encoder_hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_qformer_attention_matches_cpu_on_spyre(tp_group, bsz):
    if not spyre_available():
        pytest.skip("Spyre device not available")

    rng = torch.Generator(device="cpu").manual_seed(5 + bsz)
    hidden_states = torch.randn(bsz, 8, QF_HIDDEN, dtype=torch.float16, generator=rng)
    expected = _STOCK_QFORMER_FORWARD(_qformer_attention(), hidden_states)

    device = torch.device("spyre")
    actual = _swapped(_qformer_attention()).to(device)(hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_qformer_cross_attention_matches_cpu_on_spyre(tp_group, bsz):
    if not spyre_available():
        pytest.skip("Spyre device not available")

    rng = torch.Generator(device="cpu").manual_seed(7 + bsz)
    hidden_states = torch.randn(bsz, 8, QF_HIDDEN, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, QF_HIDDEN, dtype=torch.float16, generator=rng)
    expected = _STOCK_QFORMER_FORWARD(
        _qformer_attention(cross=True, seed=2), hidden_states, encoder_hidden_states
    )

    device = torch.device("spyre")
    attn = _swapped(_qformer_attention(cross=True, seed=2)).to(device)
    actual = attn(hidden_states.to(device), encoder_hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


# --- SigLIP head padding ---

SIG_ORIG, SIG_PAD, SIG_HEADS = 72, 128, 16  # so400m: head_dim 72, padded to the 64-wide stick
SIG_HIDDEN = SIG_HEADS * SIG_ORIG


class _TransposedLinear(nn.Module):
    """Stand-in for a Spyre vLLM linear after loading: weight stored as Wᵀ, forward returns
    ``(output, None)``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features, dtype=torch.float16))
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None
        )

    def forward(self, x: torch.Tensor):
        out = x @ self.weight
        return (out if self.bias is None else out + self.bias), None


def _siglip_tower(
    num_layers: int = 2, num_heads: int = SIG_HEADS, head_dim: int = SIG_ORIG, bias=False
):
    """A tower with ``encoder.layers[*].self_attn`` carrying what ``pad_siglip_heads`` reads."""
    hidden = num_heads * head_dim
    rng = torch.Generator(device="cpu").manual_seed(42)
    layers = nn.ModuleList()
    for _ in range(num_layers):
        attn = nn.Module()
        attn.head_dim = head_dim
        attn.num_heads_per_partition = num_heads
        attn.qkv_proj = _TransposedLinear(hidden, 3 * hidden, bias=bias)
        attn.out_proj = _TransposedLinear(hidden, hidden)
        for p in (*attn.qkv_proj.parameters(), *attn.out_proj.parameters()):
            p.data.normal_(std=0.02, generator=rng)
        # What MMEncoderAttention holds: the scale is fixed at construction, from the real dim.
        attn.attn = SimpleNamespace(head_size=head_dim, scale=head_dim**-0.5)
        layer = nn.Module()
        layer.self_attn = attn
        layers.append(layer)
    tower = nn.Module()
    tower.encoder = nn.Module()
    tower.encoder.layers = layers
    return tower


def test_pad_siglip_heads_widens_every_layer():
    from spyre_inference.models.granite4_vision import pad_siglip_heads

    tower = _siglip_tower(num_layers=3)
    pad_siglip_heads(tower)

    for layer in tower.encoder.layers:
        attn = layer.self_attn
        assert attn.qkv_proj.weight.shape == (SIG_HIDDEN, 3 * SIG_HEADS * SIG_PAD)
        assert attn.out_proj.weight.shape == (SIG_HEADS * SIG_PAD, SIG_HIDDEN)
        assert attn.attn.head_size == SIG_PAD
        assert attn.attn.scale == SIG_ORIG**-0.5, "the scale must stay at the real head dim"


def test_pad_siglip_heads_leaves_a_stick_aligned_head_dim_alone():
    from spyre_inference.models.granite4_vision import pad_siglip_heads

    tower = _siglip_tower(num_heads=4, head_dim=64)
    attn = tower.encoder.layers[0].self_attn
    qkv_shape, out_shape = attn.qkv_proj.weight.shape, attn.out_proj.weight.shape

    pad_siglip_heads(tower)

    assert attn.qkv_proj.weight.shape == qkv_shape
    assert attn.out_proj.weight.shape == out_shape
    assert attn.attn.head_size == 64


def test_pad_siglip_heads_is_idempotent():
    from spyre_inference.models.granite4_vision import pad_siglip_heads

    tower = _siglip_tower()
    pad_siglip_heads(tower)
    padded = tower.encoder.layers[0].self_attn.qkv_proj.weight.detach().clone()
    pad_siglip_heads(tower)  # would widen again if the first pass were not remembered

    assert tower.encoder.layers[0].self_attn.qkv_proj.weight.shape == padded.shape
    assert all(layer.self_attn._spyre_head_dim_padded for layer in tower.encoder.layers)


@pytest.mark.parametrize("with_bias", [False, True])
def test_pad_siglip_heads_keeps_the_original_values_in_place(with_bias):
    """Each head block of Q, K and V keeps its 72 values at its padded offset, followed by
    zeros, in the weight and the bias; the out weight pads its rows the same way."""
    from spyre_inference.models.granite4_vision import pad_siglip_heads

    tower = _siglip_tower(bias=with_bias)
    attn = tower.encoder.layers[0].self_attn
    if with_bias:
        attn.qkv_proj.bias.data.normal_(std=0.02)
    qkv_w = attn.qkv_proj.weight.data.clone()
    qkv_b = attn.qkv_proj.bias.data.clone() if with_bias else None
    out_w = attn.out_proj.weight.data.clone()

    pad_siglip_heads(tower)

    n, d, pd = SIG_HEADS, SIG_ORIG, SIG_PAD
    for block in range(3):
        for h in range(n):
            src, dst = block * n * d + h * d, block * n * pd + h * pd
            assert torch.equal(attn.qkv_proj.weight[:, dst : dst + d], qkv_w[:, src : src + d])
            assert not attn.qkv_proj.weight[:, dst + d : dst + pd].any()
            if with_bias:
                assert torch.equal(attn.qkv_proj.bias[dst : dst + d], qkv_b[src : src + d])
                assert not attn.qkv_proj.bias[dst + d : dst + pd].any()
    for h in range(n):
        src, dst = h * d, h * pd
        assert torch.equal(attn.out_proj.weight[dst : dst + d], out_w[src : src + d])
        assert not attn.out_proj.weight[dst + d : dst + pd].any()


def test_padded_siglip_attention_matches_unpadded():
    """Padded projections through ``padded_sdpa`` at the real scale are the unpadded attention:
    the zero columns add nothing to the scores, and the zero rows of the out weight drop the
    padded value lanes."""
    from spyre_inference.models.granite4_vision import pad_siglip_heads
    from spyre_inference.multimodal.utils import padded_sdpa

    tower = _siglip_tower(num_layers=1)
    attn = tower.encoder.layers[0].self_attn
    bsz, seq = 1, 64
    rng = torch.Generator(device="cpu").manual_seed(6)
    x = torch.randn(bsz, seq, SIG_HIDDEN, dtype=torch.float16, generator=rng)

    def run(dim: int) -> torch.Tensor:
        q, k, v = (
            t.view(bsz, seq, SIG_HEADS, dim).transpose(1, 2)
            for t in (x @ attn.qkv_proj.weight).chunk(3, dim=-1)
        )
        out = padded_sdpa(q, k, v, None, scale=SIG_ORIG**-0.5)
        return out.transpose(1, 2).reshape(bsz, seq, -1) @ attn.out_proj.weight

    q, k, v = (
        t.view(bsz, seq, SIG_HEADS, SIG_ORIG).transpose(1, 2)
        for t in (x @ attn.qkv_proj.weight).chunk(3, dim=-1)
    )
    want = F.scaled_dot_product_attention(q, k, v, scale=SIG_ORIG**-0.5)
    want = want.transpose(1, 2).reshape(bsz, seq, -1) @ attn.out_proj.weight

    pad_siglip_heads(tower)

    torch.testing.assert_close(run(SIG_PAD).float(), want.float(), atol=1e-3, rtol=1e-3)
