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

"""CLIP's vision tower on Spyre against upstream with the same weights.

``SpyreCLIPEncoderLayer`` against ``CLIPEncoderLayer``: packed images under the same-image mask
each match their own unpacked run. ``SpyreCLIPEmbeddingModel``'s encoder pipeline against
upstream's ``get_image_features``, for every pooling type. The on-card twins skip without a
device."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available
from transformers import CLIPVisionConfig
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.models.clip import (
    CLIPEmbeddingModel,
    CLIPEncoderLayer,
    CLIPVisionTransformer,
)

from spyre_inference.models._vision_encoder import (
    SEGMENT_ITEMS,
    one_hot_segments,
    segment_mask,
)
from spyre_inference.models.clip import (
    SpyreCLIPAttention,
    SpyreCLIPEmbeddingModel,
    SpyreCLIPEncoderLayer,
)

# Rows per image: 50 is ViT-B/32@224's grid**2 + 1; packed with 30 they fill 80 of 128 rows.
SIZES = (50, 30)
ROWS = 128


def _config() -> CLIPVisionConfig:
    return CLIPVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,  # head_dim 64
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
    )


def _finish_weight_loading(module: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _layers() -> tuple[CLIPEncoderLayer, SpyreCLIPEncoderLayer]:
    """Upstream's layer and the Spyre one, with the same seeded weights."""
    config = _config()
    torch.manual_seed(0)
    upstream = CLIPEncoderLayer(config, prefix="layer", attn_cls=MMEncoderAttention)
    for p in upstream.parameters():
        p.data.normal_(std=0.05)
    for name, p in upstream.named_parameters():
        if "layer_norm" in name and name.endswith("weight"):
            p.data.add_(1.0)
    spyre = SpyreCLIPEncoderLayer(config, prefix="layer", attn_cls=MMEncoderAttention)
    # Copied before loading finishes: the Spyre linears transpose `weight` at that step.
    spyre.load_state_dict(upstream.state_dict())
    _finish_weight_loading(upstream)
    _finish_weight_loading(spyre)
    return upstream.float(), spyre.float()


def _packed(
    images: list[torch.Tensor], pad: torch.Tensor, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[1, ROWS, D]`` rows of ``images`` back to back, ``pad`` in the tail, and their mask."""
    tail = ROWS - sum(img.shape[0] for img in images)
    ids = torch.cat(
        [torch.full((img.shape[0],), i) for i, img in enumerate(images)]
        + [torch.full((tail,), SEGMENT_ITEMS)]
    )
    x = torch.cat([*images, pad[:tail]])
    return x[None].to(dtype), segment_mask(one_hot_segments(ids), dtype)


def _images(hidden: int) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(n, hidden) for n in SIZES]


def test_swaps_in_the_spyre_attention_with_upstream_weight_names(tp_group):
    upstream, spyre = _layers()
    assert type(spyre.self_attn) is SpyreCLIPAttention
    assert not hasattr(spyre.self_attn, "attn")
    assert spyre.state_dict().keys() == upstream.state_dict().keys()


def test_one_image_matches_upstream(tp_group):
    upstream, spyre = _layers()
    torch.manual_seed(1)
    x = torch.randn(1, 64, _config().hidden_size)
    mask = segment_mask(one_hot_segments(torch.zeros(64, dtype=torch.int64)), torch.float32)
    assert not mask.any()

    with torch.inference_mode():
        torch.testing.assert_close(spyre(x, mask), upstream(x), atol=1e-4, rtol=1e-4)


def test_packed_images_each_match_their_own_run(tp_group):
    upstream, spyre = _layers()
    images = _images(_config().hidden_size)
    x, mask = _packed(images, torch.zeros(ROWS, _config().hidden_size))

    with torch.inference_mode():
        out = spyre(x, mask)[0]
        start = 0
        for img in images:
            want = upstream(img[None])[0]
            got = out[start : start + img.shape[0]]
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
            start += img.shape[0]


def test_padding_rows_never_change_real_rows(tp_group):
    _, spyre = _layers()
    hidden = _config().hidden_size
    images = _images(hidden)
    real = sum(SIZES)
    torch.manual_seed(2)

    with torch.inference_mode():
        zeros = spyre(*_packed(images, torch.zeros(ROWS, hidden)))[0, :real]
        noise = spyre(*_packed(images, 100 * torch.randn(ROWS, hidden)))[0, :real]
    torch.testing.assert_close(zeros, noise, atol=0, rtol=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_packed_images_match_upstream_on_spyre(tp_group, compiled):
    """The packed comparison in fp16 on the card, compiled as ``compile_vision_graphs`` does."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert

    upstream, spyre = _layers()
    spyre = spyre.half().to("spyre")
    if compiled:
        spyre.compile(backend="inductor", fullgraph=True, dynamic=False)
    hidden = _config().hidden_size
    images = _images(hidden)
    x, mask = _packed(images, torch.zeros(ROWS, hidden), torch.float16)

    with torch.inference_mode():
        out = spyre(convert(x, device="spyre"), convert(mask, device="spyre"))
        out = convert(out, device="cpu")[0]
        start = 0
        for img in images:
            want = upstream(img[None])[0]
            # Tight: the residual alone keeps cosine above 0.995 with attention dropped or
            # the mask zeroed.
            assert _cosine(out[start : start + img.shape[0]], want) > 0.9999
            start += img.shape[0]


# --- SpyreCLIPEmbeddingModel's encoder pipeline ---

# 56 / 7 = an 8x8 grid: 65 rows per image, so a "class" slot is 128 rows with 63 of padding.
# 3*7*7 = 147 values per patch, padded to 192.
IMAGE, PATCH, PROJECTION = 56, 7, 64
POOLING_TYPES = ["LAST", "CLS", "MEAN", "ALL"]


def _tower_config(num_layers: int = 2) -> CLIPVisionConfig:
    return CLIPVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,  # head_dim 64
        num_hidden_layers=num_layers,
        num_channels=3,
        image_size=IMAGE,
        patch_size=PATCH,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
    )


def _upstream_modules(config: CLIPVisionConfig) -> nn.ModuleDict:
    torch.manual_seed(0)
    mods = nn.ModuleDict(
        {
            "vision_model": CLIPVisionTransformer(config, prefix="vision_model"),
            "visual_projection": nn.Linear(config.hidden_size, PROJECTION, bias=False),
        }
    )
    for p in mods.parameters():
        p.data.normal_(std=0.05)
    for name, p in mods.named_parameters():
        if ("norm" in name or "layrnorm" in name) and name.endswith("weight"):
            p.data.add_(1.0)
    # Unfinished on purpose: the Spyre linears transpose `weight` when loading finishes.
    return mods


def _reference(config, pooling_type: str, images: list[torch.Tensor]) -> torch.Tensor:
    """Upstream ``CLIPEmbeddingModel.get_image_features``: ``[images, tokens, projection]``."""
    mods = _upstream_modules(config)
    _finish_weight_loading(mods)
    model = SimpleNamespace(
        vision_model=mods["vision_model"],
        visual_projection=mods["visual_projection"],
        pooler_config=SimpleNamespace(seq_pooling_type=pooling_type),
    )
    return CLIPEmbeddingModel.get_image_features(model, torch.stack(images))


def _spyre_model(config, pooling_type: str, dtype=torch.float16) -> SpyreCLIPEmbeddingModel:
    """A ``SpyreCLIPEmbeddingModel`` around the same (seeded) upstream modules as
    ``_reference``, with the steps weight loading performs."""
    model = SpyreCLIPEmbeddingModel.__new__(SpyreCLIPEmbeddingModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=config)
    model.pooler_config = SimpleNamespace(seq_pooling_type=pooling_type)
    model.projection_dim = PROJECTION
    for name, mod in _upstream_modules(config).items():
        setattr(model, name, mod)
    # The Spyre layers replace upstream's whole, so they get the same seeded weights back.
    states = [layer.state_dict() for layer in model.vision_model.encoder.layers]
    model._init_spyre_vision(prefix="")
    for layer, state in zip(model.vision_model.encoder.layers, states, strict=True):
        layer.load_state_dict(state)
    # The loader's finish steps, in its order: per-layer first, then the model hook.
    _finish_weight_loading(model)
    model.process_weights_after_loading()
    return model.to(dtype)


def _pixels(num_images: int) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return list(torch.randn(num_images, 3, IMAGE, IMAGE).unbind(0))


@pytest.mark.parametrize("num_images", [1, 3])
@pytest.mark.parametrize("pooling_type", POOLING_TYPES)
def test_pipeline_matches_upstream(tp_group, pooling_type, num_images):
    config = _tower_config()
    images = _pixels(num_images)

    with torch.inference_mode():
        expected = _reference(config, pooling_type, images)
        actual = _spyre_model(config, pooling_type).embed_multimodal(pixel_values=images)

    assert [tuple(t.shape) for t in actual] == [tuple(t.shape) for t in expected.unbind(0)]
    for got, want in zip(actual, expected.unbind(0), strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.999


@pytest.mark.parametrize("pooling_type", ["LAST", "MEAN"])
def test_budget_padding_does_not_leak_into_images(tp_group, pooling_type):
    """A larger budget only adds padding slots, which have their own segment."""
    model = _spyre_model(_tower_config(num_layers=1), pooling_type)
    images = _pixels(2)
    values = model.prepare_encoder_cudagraph_replay_buffers({"pixel_values": images}, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    # The smallest budget the manager builds: "full" packs 2 x 65 = 130 rows, never a budget.
    step = model.vision_patch_multiple
    patches = -(-next(iter(values.values())).shape[0] // step) * step

    def run(num_patches):
        bufs = model._vision_empty(num_patches)
        for key, src in values.items():
            padding[key](bufs[key], src)
        return model.encoder_cudagraph_forward(bufs)

    with torch.inference_mode():
        tight, loose = run(patches), run(4 * model.vision_patch_multiple)
    n = sum(model.vision_item_tokens({"pixel_values": images}))
    torch.testing.assert_close(tight[:n], loose[:n], atol=1e-3, rtol=1e-3)


def test_class_layout_gives_each_image_one_token_and_a_fixed_slot(tp_group):
    model = _spyre_model(_tower_config(num_layers=1), "LAST")
    images = _pixels(3)
    values = model.vision_inputs({"pixel_values": images})

    assert model.vision_item_tokens({"pixel_values": images}) == [1, 1, 1]
    assert model.patches_per_token == model.vision_patch_multiple == 128
    assert values["patches"].shape == (3 * 128, 192)
    # The class row has no patch: its embedding comes from the position table alone.
    assert not values["patches"].reshape(3, 128, -1)[:, 0].any()
    assert values["positions"].reshape(3, 128)[0, :65].tolist() == list(range(65))

    # Slots batch on dim 0 under one key-padding row: no cross-image mask, whose size would
    # grow with the square of the images per call.
    assert "segments" not in values and "segments" not in model.vision_buffers
    with torch.inference_mode():
        x, block_kwargs = model.vision_embed(values)
    assert x.shape == (3, 128, 128)
    mask = block_kwargs["mask"]
    assert mask.shape == (1, 1, 1, 128) and mask.dtype == x.dtype
    assert (mask[..., :65] == 0).all() and (mask[..., 65:] < 0).all()


def test_an_image_of_another_size_is_rejected(tp_group):
    """Upstream's input schema enforced the configured size; the mixin path bypasses it."""
    model = _spyre_model(_tower_config(num_layers=1), "LAST")
    with pytest.raises(ValueError, match="images, got"):
        model.vision_inputs({"pixel_values": [torch.randn(3, IMAGE, IMAGE // 2)]})


@pytest.mark.parametrize(
    ("pooling_type", "items", "expected"),
    [
        ("LAST", 8, (1, 8)),
        ("MEAN", 8, (64, 576)),
        ("LAST", 100, (1, SEGMENT_ITEMS)),
        ("MEAN", 100, (64, 4096)),
    ],
    ids=["class", "full", "class-capped", "full-capped"],
)
def test_budget_range_covers_a_whole_step(tp_group, pooling_type, items, expected):
    """Eight images per step: eight 1-token images, or eight 65-token ones on 64-row sticks.
    A larger step stops at the 63 items one call holds, as no budget past that is filled."""
    model = _spyre_model(_tower_config(num_layers=1), pooling_type)
    model.max_items_per_step = items
    assert model.get_encoder_cudagraph_budget_range(None) == expected
    assert model.vision_items_limited_by_tokens
    assert model.vision_max_items_per_call == SEGMENT_ITEMS


def test_loaded_vision_weights_are_reshaped_in_place(tp_group):
    """The patch conv becomes the graph's linear under its own name, so one copy reaches the
    card, and the class embedding folds into the position table's row 0."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    config = _tower_config(num_layers=1)
    model = _spyre_model(config, "LAST", dtype=torch.float32)
    emb = model.vision_model.embeddings
    assert not isinstance(emb.patch_embedding, SpyreConv2d), "the runner would tile a conv copy"
    assert emb.patch_embedding.weight.shape == (192, config.hidden_size)
    table = model.spyre_position_table.table
    torch.testing.assert_close(table[1:], emb.position_embedding.weight[1:])
    torch.testing.assert_close(table[0], emb.position_embedding.weight[0] + emb.class_embedding)


def test_image_limit_zero_keeps_the_placeholder(tp_group):
    from vllm.model_executor.models.utils import StageMissingLayer

    model = SpyreCLIPEmbeddingModel.__new__(SpyreCLIPEmbeddingModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=_tower_config(num_layers=1))
    model.pooler_config = SimpleNamespace(seq_pooling_type="LAST")
    model.projection_dim = PROJECTION
    model.vision_model = StageMissingLayer("image_tower")
    model._init_spyre_vision(prefix="")
    model.process_weights_after_loading()
    assert isinstance(model.vision_model, StageMissingLayer)
    assert not hasattr(model, "spyre_position_table")


def test_embed_input_ids_reaches_the_mixin_merge(monkeypatch):
    """Image rows arrive on the host and land at their placeholders; the step's mask is kept
    for ``forward``."""
    text = torch.full((4, PROJECTION), -1.0)
    monkeypatch.setattr(
        CLIPEmbeddingModel, "embed_input_ids", lambda self, ids, mm=None, **kw: text.clone()
    )
    model = SpyreCLIPEmbeddingModel.__new__(SpyreCLIPEmbeddingModel)
    nn.Module.__init__(model)
    mask = torch.tensor([True, False, False, False])
    rows = torch.ones(1, PROJECTION)

    out = model.embed_input_ids(torch.zeros(4, dtype=torch.int64), [rows], is_multimodal=mask)

    assert torch.equal(out[0], rows[0])
    assert torch.equal(out[1:], text[1:])
    assert torch.equal(model._spyre_mm_token_mask, mask)
    assert model._spyre_has_text_tokens


def test_registry_and_mro_reach_upstream_init_and_protocol():
    """``super().__init__`` must land on upstream's: a ``Protocol`` base ahead of it in the
    MRO would substitute its placeholder ``__init__`` and skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models import spyre_models

    assert spyre_models()["CLIPModel"] == "spyre_inference.models.clip:SpyreCLIPEmbeddingModel"
    mro = SpyreCLIPEmbeddingModel.__mro__[1:]
    assert next(cls for cls in mro if "__init__" in cls.__dict__) is CLIPEmbeddingModel
    assert supports_encoder_cudagraph(SpyreCLIPEmbeddingModel)
    assert SpyreCLIPEmbeddingModel._processor_factory is CLIPEmbeddingModel._processor_factory


def _padded_device_inputs(model, images, token_budget):
    from spyre_inference.custom_ops.utils import convert

    values = model.prepare_encoder_cudagraph_replay_buffers({"pixel_values": images}, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    return {k: convert(v, device="spyre") for k, v in bufs.items()}


@pytest.mark.parametrize("pooling_type", ["LAST", "MEAN"])
def test_compiled_pipeline_matches_upstream_at_every_budget_on_spyre(tp_group, pooling_type):
    """The compiled graphs at each budget of a three-image step, fed the way the manager
    feeds them; at the smallest one, also against the uncompiled pipeline on the same
    device buffers. Three images leave one budget per layout (``class`` 3, ``full`` 256,
    which pads 195 tokens): each budget is a compile, and the CPU padding test covers the
    larger ones."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    from spyre_inference.custom_ops.utils import convert

    config = _tower_config()
    images = _pixels(3)
    model = _spyre_model(config, pooling_type).to("spyre")
    model.max_items_per_step = 3
    tokens = sum(model.vision_item_tokens({"pixel_values": images}))
    budgets = EncoderCudaGraphManager._generate_budgets(
        *model.get_encoder_cudagraph_budget_range(None)
    )
    budgets = [b for b in budgets if b >= tokens]
    expected = _reference(config, pooling_type, images).flatten(0, 1)

    with torch.inference_mode():
        eager = model.encoder_cudagraph_forward(_padded_device_inputs(model, images, budgets[0]))
    eager = convert(eager, device="cpu")[:tokens]
    model.compile_vision_graphs()
    for budget in budgets:
        with torch.inference_mode():
            out = model.encoder_cudagraph_forward(_padded_device_inputs(model, images, budget))
        out = convert(out, device="cpu")[:tokens]
        assert _cosine(out, expected) > 0.995, budget
        if budget == budgets[0]:
            assert _cosine(out, eager) > 0.9999
            torch.testing.assert_close(out.float(), eager.float(), atol=2e-2, rtol=2e-2)
