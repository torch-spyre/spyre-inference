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

"""``SpyreIdefics3ForConditionalGeneration``'s vision pipeline against upstream Idefics3.

The pipeline reorders patches offset-major, folds the conv into a linear, turns the learned
position embedding into a gather table, splits the connector's weight per pixel-shuffle offset
and pads to a bucket; these tests pin each of those to the upstream result on CPU. The on-card
twins at the bottom skip without a device.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.multimodal.utils import STICK

idefics3 = pytest.importorskip("vllm.model_executor.models.idefics3")
idefics2_vision = pytest.importorskip("vllm.model_executor.models.idefics2_vision_model")

PATCH = 7  # 3*7*7 = 147 values per patch: exercises the patch-width padding to 192
SCALE = 2  # pixel-shuffle factor: 4 patches per image token
SIDE = 8  # patches per tile side
HIDDEN = 128
TEXT_HIDDEN = 96
IMAGE_SIZE = PATCH * SIDE
TILES_PER_IMAGE = [1, 2]  # three tiles in all, the last one partly padded


@pytest.fixture(autouse=True)
def _engine_ir_dispatch():
    """Trace vLLM IR ops inline, as the worker does under STOCK_TORCH_COMPILE; the wrapped
    op fails a layout check the engine never reaches."""
    import vllm.ir

    with vllm.ir.enable_torch_wrap(False):
        yield


def _config(num_layers: int = 2) -> SimpleNamespace:
    """The fields of ``Idefics3Config`` that the tower, the connector and the Spyre model read."""
    return SimpleNamespace(
        scale_factor=SCALE,
        image_token_id=10,
        text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN),
        vision_config=SimpleNamespace(
            hidden_size=HIDDEN,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=2,  # head_dim 64
            num_channels=3,
            image_size=IMAGE_SIZE,
            patch_size=PATCH,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
        ),
    )


class _Inner(nn.Module):
    """The slice of upstream's ``Idefics3Model`` that the vision path uses, with its methods."""

    get_patch_attention_mask = idefics3.Idefics3Model.get_patch_attention_mask
    get_position_ids = idefics3.Idefics3Model.get_position_ids
    image_pixels_to_features = idefics3.Idefics3Model.image_pixels_to_features

    def __init__(self, config, mods: nn.ModuleDict) -> None:
        super().__init__()
        self.config = config
        self.vision_model = mods["vision_model"]
        self.connector = mods["connector"]
        vision = config.vision_config
        self.image_seq_len = int((vision.image_size // vision.patch_size) ** 2 / SCALE**2)


def _finish_weight_loading(module: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _upstream_modules(config) -> nn.ModuleDict:
    torch.manual_seed(0)
    mods = nn.ModuleDict(
        {
            "vision_model": idefics3.Idefics3VisionTransformer(
                config.vision_config, quant_config=None, prefix="vision_model"
            ),
            "connector": idefics3.Idefics3Connector(config, quant_config=None, prefix="connector"),
        }
    )
    for p in mods.parameters():
        p.data.normal_(std=0.05)
    for name, p in mods.named_parameters():
        if name.endswith(("layer_norm1.weight", "layer_norm2.weight", "post_layernorm.weight")):
            p.data.add_(1.0)
    # Unfinished on purpose: the Spyre linears transpose `weight` when loading finishes,
    # so each side copies raw weights first and finishes on its own.
    return mods


def _reference(config, mm_kwargs) -> list[torch.Tensor]:
    """Upstream ``Idefics3ForConditionalGeneration._process_image_input``, per image."""
    mods = _upstream_modules(config)
    _finish_weight_loading(mods)
    inner = _Inner(config, mods)
    feats = inner.image_pixels_to_features(
        mm_kwargs["pixel_values"],
        pixel_attention_mask=mm_kwargs["pixel_attention_mask"],
        remove_padding=False,
    )
    tokens = mods["connector"](feats).flatten(0, 1)
    sizes = [int(n) * inner.image_seq_len for n in mm_kwargs["num_patches"]]
    return list(torch.split(tokens, sizes))


def _spyre_model(config, dtype=torch.float16):
    """A ``SpyreIdefics3ForConditionalGeneration`` around the same (seeded) upstream modules
    as ``_reference``, with the steps weight loading performs."""
    from spyre_inference.models.idefics3 import SpyreIdefics3ForConditionalGeneration

    model = SpyreIdefics3ForConditionalGeneration.__new__(SpyreIdefics3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.model = _Inner(config, _upstream_modules(config))
    model._init_spyre_vision(prefix="")
    # The loader's finish steps, in its order: per-layer first, then the model hook.
    _finish_weight_loading(model)
    model.process_weights_after_loading()
    return model.to(dtype)


def _mm_kwargs(num_images: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    tiles = sum(TILES_PER_IMAGE[:num_images])
    pixel_values = torch.randn(tiles, 3, IMAGE_SIZE, IMAGE_SIZE)
    mask = torch.ones(tiles, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    if num_images > 1:
        # The last tile is padded below row 5 and right of column 6 (in patches), as the
        # processor pads its resized image tiles, so the position buckets are not uniform.
        mask[-1, 5 * PATCH :, :] = False
        mask[-1, :, 6 * PATCH :] = False
    return {
        "pixel_values": pixel_values * mask[:, None],
        "pixel_attention_mask": mask,
        "num_patches": torch.tensor(TILES_PER_IMAGE[:num_images]),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("num_images", [1, 2])
def test_pipeline_matches_upstream(tp_group, num_images):
    config = _config()
    mm_kwargs = _mm_kwargs(num_images)

    with torch.inference_mode():
        expected = _reference(config, mm_kwargs)
        actual = _spyre_model(config).embed_multimodal(**mm_kwargs)

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.999


def test_padding_does_not_leak_into_tiles(tp_group):
    """Bucket padding must not change any tile's embedding: padding slots have their own
    segment, and each offset group is padded separately."""
    model = _spyre_model(_config(num_layers=1))
    mm_kwargs = _mm_kwargs()
    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    config = model.get_encoder_cudagraph_config()

    def run(num_patches):
        bufs = model._vision_empty(num_patches)
        for key, src in values.items():
            config.padding_logics.get(key, lambda d, s: d[: s.shape[0]].copy_(s))(bufs[key], src)
        return model.encoder_cudagraph_forward(bufs)

    with torch.inference_mode():
        tight, loose = run(192), run(256)
    n = sum(model.vision_item_tokens(mm_kwargs))
    torch.testing.assert_close(tight[:n], loose[:n], atol=1e-3, rtol=1e-3)


def test_item_tokens_match_patches_per_token(tp_group):
    model = _spyre_model(_config(num_layers=1))
    mm_kwargs = _mm_kwargs()
    tokens = model.vision_item_tokens(mm_kwargs)
    patches = next(iter(model.vision_inputs(mm_kwargs).values())).shape[0]

    assert tokens == [n * (SIDE // SCALE) ** 2 for n in TILES_PER_IMAGE]
    assert sum(tokens) * model.patches_per_token == patches


@pytest.mark.parametrize("partial", [False, True])
def test_tile_position_ids_match_upstream(partial):
    from spyre_inference.models.idefics3 import tile_position_ids

    embeddings = idefics2_vision.Idefics2VisionEmbeddings(_config().vision_config)
    mask = torch.ones(3, SIDE, SIDE, dtype=torch.bool)
    if partial:
        mask[1, 5:, :] = False
        mask[1, :, 6:] = False
        mask[2, 3:, 2:] = False

    expected = embeddings.get_position_ids(mask)
    torch.testing.assert_close(tile_position_ids(mask, SIDE), expected, atol=0, rtol=0)


def test_group_projection_matches_pixel_shuffle_connector(tp_group):
    """Offset-major tokens times the per-offset weight equal ``pixel_shuffle`` plus the
    connector's projection."""
    from spyre_inference.models._vision_encoder import offset_major, split_weight_by_group

    config = _config()
    torch.manual_seed(2)
    connector = idefics3.Idefics3Connector(config, quant_config=None, prefix="connector")
    proj = connector.modality_projection.proj
    proj.weight.data.normal_(std=0.05)
    weight = split_weight_by_group(proj.weight.data.clone(), SCALE**2, group_major=True)
    _finish_weight_loading(connector)
    x = torch.randn(2, SIDE * SIDE, HIDDEN)
    expected = connector(x)

    for tile in range(2):
        blocks = offset_major(x[tile].reshape(SIDE, SIDE, HIDDEN), SCALE)
        torch.testing.assert_close((blocks @ weight).sum(0), expected[tile], atol=1e-5, rtol=1e-5)


def test_upstream_tower_contract_is_unchanged(tp_group):
    """What the Spyre model still takes from upstream's: the attention's constructor and
    projections, the submodules the weight hook rewrites and the model's attribute names.
    Diff it on a vLLM bump."""
    attention = idefics2_vision.Idefics2VisionAttention
    params = list(inspect.signature(attention.__init__).parameters)
    assert params == ["self", "config", "quant_config", "prefix"]
    layer = idefics2_vision.Idefics2EncoderLayer(_config().vision_config)
    for name in ("self_attn", "layer_norm1", "layer_norm2", "mlp"):
        assert hasattr(layer, name), f"upstream Idefics2EncoderLayer no longer has {name}"
    for name in ("qkv_proj", "out_proj", "num_heads_per_partition", "head_dim", "scale"):
        assert hasattr(layer.self_attn, name), f"upstream attention no longer has {name}"
    embeddings = idefics2_vision.Idefics2VisionEmbeddings(_config().vision_config)
    for name in ("patch_embedding", "position_embedding"):
        assert hasattr(embeddings, name), f"upstream embeddings no longer have {name}"

    init_src = inspect.getsource(idefics3.Idefics3Model.__init__)
    for name in ("vision_model", "connector", "text_model", "image_seq_len"):
        assert f"self.{name} =" in init_src, f"upstream Idefics3Model no longer builds {name}"
    assert "self.model = Idefics3Model(" in inspect.getsource(
        idefics3.Idefics3ForConditionalGeneration.__init__
    )


def test_registry_resolves_to_spyre_class_and_keeps_processor():
    from spyre_inference.models import spyre_models
    from spyre_inference.models.idefics3 import SpyreIdefics3ForConditionalGeneration

    assert (
        spyre_models()["Idefics3ForConditionalGeneration"]
        == "spyre_inference.models.idefics3:SpyreIdefics3ForConditionalGeneration"
    )
    upstream = idefics3.Idefics3ForConditionalGeneration
    assert SpyreIdefics3ForConditionalGeneration._processor_factory is upstream._processor_factory


def test_construction_reaches_upstream_init_and_protocol():
    """``super().__init__`` must land on upstream's: a ``Protocol`` base ahead of it in the
    MRO would substitute its placeholder ``__init__`` and skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models.idefics3 import SpyreIdefics3ForConditionalGeneration

    mro = SpyreIdefics3ForConditionalGeneration.__mro__[1:]
    first_init = next(cls for cls in mro if "__init__" in cls.__dict__)
    assert first_init is idefics3.Idefics3ForConditionalGeneration
    assert supports_encoder_cudagraph(SpyreIdefics3ForConditionalGeneration)


def test_spyre_attention_replaces_every_layers_forward(tp_group):
    from spyre_inference.models.idefics3 import SpyreIdefics2VisionAttention

    config = _config()
    model = _spyre_model(config)
    layers = model.vision_blocks()

    assert len(layers) == config.vision_config.num_hidden_layers
    assert all(isinstance(layer.self_attn, SpyreIdefics2VisionAttention) for layer in layers)


def test_loaded_vision_weights_are_reshaped_in_place(tp_group):
    """The patch-conv and projection weights become the graphs' layouts under their own
    names, and the position embedding moves to the gather table, so one copy of each reaches
    the card."""
    from spyre_inference.custom_ops.conv import SpyreConv2d
    from spyre_inference.models._vision_encoder import GatherTable

    config = _config()
    model = _spyre_model(config)
    embeddings = model.model.vision_model.embeddings
    patch_width = model.vision_buffers["patches"].width

    assert not isinstance(embeddings.patch_embedding, SpyreConv2d), "the runner would tile a conv"
    assert embeddings.patch_embedding.weight.shape == (patch_width, HIDDEN)
    assert embeddings.patch_embedding.bias.shape == (HIDDEN,)
    assert not hasattr(embeddings, "position_embedding")
    assert isinstance(model.spyre_position_table, GatherTable)
    assert model.spyre_position_table.table.shape == (SIDE * SIDE, HIDDEN)
    proj = model.model.connector.modality_projection.proj
    assert proj.weight.shape == (SCALE**2, HIDDEN, TEXT_HIDDEN)


def test_block_compile_is_honoured(tp_group):
    """``block.compile`` must install a graph that the layer's call runs: upstream's layer has
    no ``support_torch_compile`` wrapper, and the Spyre attention keeps it that way."""
    block = _spyre_model(_config(), dtype=torch.float32).vision_blocks()[0]
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    block.compile(backend=backend, fullgraph=True, dynamic=False)
    # As in the pipeline: a whole stick of patches and an additive mask at the input's dtype,
    # which `padded_sdpa` takes as is (any other length needs a bool keep-mask).
    x = torch.randn(1, STICK, HIDDEN)
    block(x, attention_mask=torch.zeros(1, 1, STICK, STICK))
    assert graphs


def test_pipeline_matches_upstream_on_spyre(tp_group):
    """Same comparison with the encoder on the card, uncompiled (the eager path)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    config = _config()
    mm_kwargs = _mm_kwargs()

    with torch.inference_mode():
        expected = _reference(config, mm_kwargs)
        actual = _spyre_model(config).to("spyre").embed_multimodal(**mm_kwargs)

    for got, want in zip(actual, expected, strict=True):
        assert _cosine(got, want) > 0.995


def _padded_device_inputs(model, mm_kwargs, token_budget):
    from spyre_inference.custom_ops.utils import convert

    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    return {k: convert(v, device="spyre") for k, v in bufs.items()}


@pytest.mark.parametrize("token_budget", [48, 64])
def test_compiled_pipeline_matches_upstream_on_spyre(tp_group, token_budget):
    """The compiled graphs at one bucket, fed the way the manager feeds them."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert

    config = _config()
    mm_kwargs = _mm_kwargs()
    model = _spyre_model(config).to("spyre")
    model.compile_vision_graphs()
    inputs = _padded_device_inputs(model, mm_kwargs, token_budget)

    with torch.inference_mode():
        expected = torch.cat(_reference(config, mm_kwargs))
        out = model.encoder_cudagraph_forward(inputs)
    actual = convert(out, device="cpu")[: expected.shape[0]]
    assert _cosine(actual, expected) > 0.995


def test_compiled_graphs_match_eager_on_the_same_inputs_on_spyre(tp_group):
    """The same padded device buffers through the uncompiled and the compiled pipeline."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    mm_kwargs = _mm_kwargs()
    model = _spyre_model(_config()).to("spyre")
    inputs = _padded_device_inputs(model, mm_kwargs, token_budget=64)
    tokens = sum(model.vision_item_tokens(mm_kwargs))

    with torch.inference_mode():
        eager = model.encoder_cudagraph_forward(inputs)[:tokens]
        model.compile_vision_graphs()
        compiled = model.encoder_cudagraph_forward(inputs)[:tokens]

    assert _cosine(compiled, eager) > 0.9999
    torch.testing.assert_close(compiled.float(), eager.float(), atol=2e-2, rtol=2e-2)
