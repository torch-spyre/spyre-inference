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


def test_init_fails_clearly_without_spatial_sampling():
    """Upstream leaves ``spatial_projectors`` None when spatial sampling is off; the encoder
    iterates it, so construction refuses instead of failing later with a TypeError."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    config = _config()
    config.vision_config.num_channels = 3
    model = SimpleNamespace(config=config, spatial_projectors=None)
    with pytest.raises(NotImplementedError, match="spatial sampling"):
        Spyre._init_spyre_vision(model)


def test_compile_vision_graphs_leaves_vision_project_uncompiled(monkeypatch):
    """``vision_project`` stays Python and calls the shared ``_project`` graph, so it is
    neither compiled nor registered with the compile guard."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )
    from spyre_inference.v1.worker import compile_guard

    labels = []
    monkeypatch.setattr(compile_guard, "watch", lambda target, label: labels.append(label))
    model = SimpleNamespace(
        vision_embed=lambda values: values,
        vision_merge=lambda text, image, mask: text,
        vision_project=lambda x, taps: x,
        _project=lambda proj, tap, down: tap,
        vision_blocks=lambda: nn.ModuleList([nn.Linear(1, 1)]),
    )

    Spyre.compile_vision_graphs(model)

    assert model._spyre_vision_project is model.vision_project
    assert sorted(labels) == [
        "Linear (vision block)",
        "SimpleNamespace._project",
        "SimpleNamespace.vision_embed",
        "SimpleNamespace.vision_merge",
    ]


def test_vision_project_pairs_each_projector_with_its_tap_and_down():
    """Deepstack projectors each read their own tap through the area mean; the four spatial
    projectors all read the one tap after them, each through its own offset. The levels are
    packed on the last dimension in that order."""
    from spyre_inference.models.granite4_vision import (
        SpyreGranite4VisionForConditionalGeneration as Spyre,
    )

    layerwise, spatial = 2, 4
    calls = []

    def project(proj, tap, down):
        calls.append((proj, tap, down))
        return torch.full((3, 2), float(len(calls)))

    model = SimpleNamespace(
        layerwise_projectors=[object() for _ in range(layerwise)],
        spatial_projectors=[object() for _ in range(spatial)],
        _down_mean=torch.zeros(1),
        _project=project,
        **{f"_down_offset{k}": torch.full((1,), float(k)) for k in range(spatial)},
    )
    taps = [torch.full((1,), float(i)) for i in range(layerwise + 1)]

    got = Spyre.vision_project(model, torch.zeros(1), taps)

    assert [c[0] for c in calls] == [*model.layerwise_projectors, *model.spatial_projectors]
    assert all(c[1] is taps[i] for i, c in enumerate(calls[:layerwise]))
    assert all(c[1] is taps[layerwise] for c in calls[layerwise:])
    assert all(c[2] is model._down_mean for c in calls[:layerwise])
    for k, c in enumerate(calls[layerwise:]):
        assert c[2] is getattr(model, f"_down_offset{k}")
    want = torch.cat([torch.full((3, 2), float(n)) for n in range(1, layerwise + spatial + 1)], -1)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def _postprocess_model(levels: int = 5) -> SimpleNamespace:
    """The attributes ``postprocess_encoder_output`` reads, without a full model."""
    config = _config()
    return SimpleNamespace(
        config=config,
        _side=SIDE,
        _window=WINDOW,
        _query=QUERY,
        _levels=levels,
        _packer=SimpleNamespace(
            config=config,
            _downsample_rate=Fraction(config.downsample_rate),
            image_newline=torch.randn(LM_HIDDEN),
        ),
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
        image_newline=model._packer.image_newline,
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
