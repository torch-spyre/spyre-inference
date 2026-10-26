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

"""``SpyreGemma4ForConditionalGeneration``'s vision pipeline against the stock tower."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
configuration_gemma4 = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

pytestmark = [pytest.mark.gemma4_vision]

# 26B-A4B's real vision head_dim; 72 is neither a 64-multiple nor evenly halvable
# onto the stick, which is the whole reason the padding exists.
ORIG_HEAD_DIM = 72
PADDED_HEAD_DIM = 128
NUM_HEADS = 4
NUM_PATCHES = 50
# Stick-wide, so the position tables gather rows-outermost on the card.
HIDDEN = 256
TEXT_HIDDEN = 64
PATCH = 4  # 3*4*4 = 48 values per patch: exercises the patch-width padding to 64
POOL = 3


def _vision_config(head_dim: int = ORIG_HEAD_DIM, num_heads: int = NUM_HEADS):
    return configuration_gemma4.Gemma4VisionConfig(
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
    )


def _tower_config(num_layers: int = 2, standardize: bool = False):
    return configuration_gemma4.Gemma4VisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=200,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=ORIG_HEAD_DIM,
        num_hidden_layers=num_layers,
        patch_size=PATCH,
        pooling_kernel_size=POOL,
        position_embedding_size=32,
        standardize=standardize,
    )


def _position_ids(num_patches: int = NUM_PATCHES) -> torch.Tensor:
    side = 8
    xs = torch.arange(side).repeat(side)[:num_patches]
    ys = torch.arange(side).repeat_interleave(side)[:num_patches]
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)


# `_pad_qk_linear`'s remap on activations, written out so the rope check is independent of it.
def _pad_activation_quarters(x: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    quarter = orig // 4
    half = padded // 2
    out = torch.zeros((*x.shape[:-1], padded), dtype=x.dtype)
    out[..., :quarter] = x[..., :quarter]
    out[..., quarter : 2 * quarter] = x[..., 2 * quarter : 3 * quarter]
    out[..., half : half + quarter] = x[..., quarter : 2 * quarter]
    out[..., half + quarter : half + 2 * quarter] = x[..., 3 * quarter :]
    return out


def _unpad_activation_quarters(padded: torch.Tensor, orig: int, padded_dim: int) -> torch.Tensor:
    quarter = orig // 4
    half = padded_dim // 2
    out = torch.zeros((*padded.shape[:-1], orig), dtype=padded.dtype)
    out[..., :quarter] = padded[..., :quarter]
    out[..., 2 * quarter : 3 * quarter] = padded[..., quarter : 2 * quarter]
    out[..., quarter : 2 * quarter] = padded[..., half : half + quarter]
    out[..., 3 * quarter :] = padded[..., half + quarter : half + 2 * quarter]
    return out


def _stock_modules(config) -> tuple[nn.Module, nn.Module]:
    # Seeded: two calls give identical weights.
    torch.manual_seed(0)
    tower = modeling_gemma4.Gemma4VisionModel(config).eval()
    # Without a named implementation `create_bidirectional_mask` early-exits to `None`,
    # and the stock forward would silently attend over the padding patches.
    tower.config._attn_implementation = "sdpa"
    embedder = modeling_gemma4.Gemma4MultimodalEmbedder(
        config, SimpleNamespace(hidden_size=TEXT_HIDDEN)
    ).eval()
    for module in (tower, embedder):
        for p in module.parameters():
            p.data.normal_(std=0.05)
    for name, p in tower.named_parameters():
        if name.endswith("norm.weight"):
            p.data.add_(1.0)
    if config.standardize:
        tower.std_bias.normal_()
        tower.std_scale.uniform_(0.5, 1.5)
    return tower, embedder


def _image(hp: int, wp: int, pad: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n = hp * wp
    pixel_values = torch.rand(n + pad, 3 * PATCH * PATCH, generator=generator)
    pixel_values[n:] = 0
    ys, xs = torch.meshgrid(torch.arange(hp), torch.arange(wp), indexing="ij")
    positions = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
    positions = torch.cat([positions, torch.full((pad, 2), -1)])
    return pixel_values, positions


def _images() -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Non-square, 54 + 18 = 72 valid patches: off the stick, so padding slots are live.
    return [_image(6, 9, pad=9, seed=1), _image(3, 6, pad=9, seed=2)]


def _mm_kwargs(images) -> dict:
    return {
        "pixel_values": [pv for pv, _ in images],
        "pixel_position_ids": [pos for _, pos in images],
    }


def _reference(config, images) -> list[torch.Tensor]:
    tower, embedder = _stock_modules(config)
    with torch.inference_mode():
        return [
            embedder(tower(pixel_values=pv[None], pixel_position_ids=pos[None]).last_hidden_state)
            for pv, pos in images
        ]


def _built_model(config):
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    model = SpyreGemma4ForConditionalGeneration.__new__(SpyreGemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=config, text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN)
    )
    model.model_dtype = torch.float16
    model.vision_tower, model.embed_vision = _stock_modules(config)
    weights = model.vision_tower.state_dict()
    model._init_spyre_vision()
    # As the checkpoint does, after construction and by name; strict, so a name has to match.
    model.vision_tower.load_state_dict(weights)
    return model


def _spyre_model(config, dtype=torch.float16):
    model = _built_model(config)
    model.process_weights_after_loading()
    return model.to(dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("standardize", [False, True], ids=["plain", "standardize"])
@pytest.mark.parametrize("num_images", [1, 2])
def test_pipeline_matches_stock(standardize, num_images):
    config = _tower_config(standardize=standardize)
    images = _images()[:num_images]

    with torch.inference_mode():
        expected = _reference(config, images)
        actual = _spyre_model(config).embed_multimodal(**_mm_kwargs(images))

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.995


def test_padding_does_not_leak_into_images():
    """Padding slots have their own segment, and each offset group is padded separately."""
    config = _tower_config(num_layers=1)
    model = _spyre_model(config)
    mm_kwargs = _mm_kwargs(_images())
    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics

    def run(num_patches):
        bufs = model._vision_empty(num_patches)
        for key, src in values.items():
            padding[key](bufs[key], src)
        return model.encoder_cudagraph_forward(bufs)

    with torch.inference_mode():
        tight, loose = run(576), run(1152)
    n = sum(model.vision_item_tokens(mm_kwargs))
    # The pooled sums reach ~16, where one fp16 ulp is 1/64 and a longer key axis reorders
    # the accumulation; a leak would move whole tokens, not single ulps.
    torch.testing.assert_close(tight[:n], loose[:n], atol=2e-2, rtol=1e-2)


def test_offset_major_sum_matches_the_stock_pooler():
    """Summed offset groups, scaled, equal stock's position-driven average."""
    from spyre_inference.models._vision_encoder import offset_major

    config = _tower_config()
    _, positions = _image(6, 9, pad=9, seed=0)
    torch.manual_seed(3)
    hidden = torch.randn(1, positions.shape[0], HIDDEN)
    padding = (positions == -1).all(dim=-1)[None]
    want, mask = modeling_gemma4.Gemma4VisionPooler(config)(
        hidden_states=hidden,
        pixel_position_ids=positions[None],
        padding_positions=padding,
        output_length=positions.shape[0] // POOL**2,
    )

    valid = positions[~padding[0]]
    grid = torch.empty(6, 9, HIDDEN)
    grid[valid[:, 1], valid[:, 0]] = hidden[0, ~padding[0]]
    got = offset_major(grid, POOL).sum(0) * (HIDDEN**0.5 / POOL**2)
    torch.testing.assert_close(got, want[mask], rtol=1e-5, atol=1e-5)


def test_item_tokens_count_valid_patches_only():
    model = _spyre_model(_tower_config(num_layers=1))
    assert model.vision_item_tokens(_mm_kwargs(_images())) == [6, 2]


def test_a_grid_the_kernel_does_not_divide_is_rejected():
    model = _spyre_model(_tower_config(num_layers=1))
    with pytest.raises(ValueError, match="pooling kernel"):
        model.vision_inputs(_mm_kwargs([_image(4, 6, pad=0, seed=0)]))


def test_video_is_rejected():
    model = _spyre_model(_tower_config(num_layers=1))
    with pytest.raises(NotImplementedError, match="video"):
        model.embed_multimodal(pixel_values_videos=torch.zeros(1, 9, 48))


def test_gqa_tower_is_rejected():
    config = _tower_config(num_layers=1)
    config.num_key_value_heads = 2
    with pytest.raises(NotImplementedError, match="GQA"):
        _spyre_model(config)


def test_tower_classes_stay_stock_and_weight_names_are_kept():
    """Only instances switch; names are checked as constructed, when the checkpoint loads."""
    from spyre_inference.multimodal.gemma4_vision import (
        SpyreGemma4RMSNorm,
        SpyreGemma4VisionAttention,
    )

    stock_norm = modeling_gemma4.Gemma4RMSNorm.forward
    stock_attn = modeling_gemma4.Gemma4VisionAttention.forward
    config = _tower_config(num_layers=1)
    names = [name for name, _ in _stock_modules(config)[0].named_parameters()]

    model = _built_model(config)

    assert modeling_gemma4.Gemma4RMSNorm.forward is stock_norm
    assert modeling_gemma4.Gemma4VisionAttention.forward is stock_attn
    layer = model.vision_tower.encoder.layers[0]
    assert type(layer.self_attn) is SpyreGemma4VisionAttention
    assert type(layer.input_layernorm) is SpyreGemma4RMSNorm
    assert [name for name, _ in model.vision_tower.named_parameters()] == names


def test_spyre_layers_are_constructed_and_keep_upstreams_linears():
    """Not a ``__class__`` swap: constructor state exists before weights load."""
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    config = _tower_config(num_layers=2)
    model = SpyreGemma4ForConditionalGeneration.__new__(SpyreGemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=config, text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN)
    )
    model.vision_tower, model.embed_vision = _stock_modules(config)
    layers = model.vision_tower.encoder.layers
    projections = [
        [getattr(layer.self_attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        for layer in layers
    ]

    model._init_spyre_vision()

    for layer, kept in zip(layers, projections, strict=True):
        attn = layer.self_attn
        assert [getattr(attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] == kept
        assert attn.spyre_rope_swap.shape == (PADDED_HEAD_DIM, PADDED_HEAD_DIM)
        assert "spyre_rope_swap" not in attn.state_dict()


def test_process_weights_after_loading_pads_every_layer():
    from spyre_inference.multimodal.utils import align_up

    model = _spyre_model(_tower_config(num_layers=3))
    for layer in model.vision_tower.encoder.layers:
        assert layer.self_attn.q_proj.linear.out_features == NUM_HEADS * PADDED_HEAD_DIM
        assert layer.mlp.gate_proj.linear.out_features == align_up(200)


def test_loaded_vision_weights_reach_the_card_once():
    """The stock position table and ``embed_vision`` live on in the gather tables and tail."""
    from spyre_inference.multimodal.utils import align_up

    model = _spyre_model(_tower_config(num_layers=1))
    embedder = model.vision_tower.patch_embedder

    assert not isinstance(embedder.input_proj, nn.Linear)
    assert embedder.input_proj.weight.shape == (align_up(3 * PATCH * PATCH), HIDDEN)
    assert not hasattr(embedder, "position_embedding_table")
    assert model.embed_vision is None
    names = {name for name, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    assert not any("spyre_patch_weight" in n or "embed_vision" in n for n in names)


def test_graph_inputs_take_the_model_dtype_not_the_towers():
    """Even from a bf16 tower: bf16 + fp16 in a graph promotes to fp32, which Spyre rejects."""
    model = _built_model(_tower_config(num_layers=1))
    model.vision_tower.to(torch.bfloat16)
    model.process_weights_after_loading()

    assert model.vision_tower.patch_embedder.input_proj.weight.dtype == torch.float16
    assert model.spyre_vision_x.table.dtype == torch.float16
    assert model.spyre_vision_y.table.dtype == torch.float16

    values = model._vision_empty(576)
    with torch.inference_mode():
        x, block_kwargs = model.vision_embed(values)
    cos, sin = block_kwargs["position_embeddings"]
    assert {x.dtype, cos.dtype, sin.dtype, block_kwargs["attention_mask"].dtype} == {torch.float16}


def test_tower_is_built_in_the_model_dtype(monkeypatch):
    """Set before upstream builds the tower, and transformers builds every parameter at it."""
    from transformers import AutoModel

    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    class _Built(Exception):
        pass

    seen = {}

    def upstream_init(self, *, vllm_config, prefix=""):
        seen["dtype"] = vllm_config.model_config.hf_config.vision_config.dtype
        raise _Built

    monkeypatch.setattr(gemma4_mm.Gemma4ForConditionalGeneration, "__init__", upstream_init)
    config = _tower_config(num_layers=1)
    config.dtype = torch.bfloat16
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16, hf_config=SimpleNamespace(vision_config=config)
        )
    )
    with pytest.raises(_Built):
        SpyreGemma4ForConditionalGeneration(vllm_config=vllm_config)
    assert seen["dtype"] == torch.float16

    tower = AutoModel.from_config(config=config)
    assert {p.dtype for p in tower.parameters()} == {torch.float16}


def test_upstream_tower_contract_is_unchanged():
    """The Spyre attention and the block call rely on these; diff them on a bump."""
    layer_src = inspect.getsource(modeling_gemma4.Gemma4VisionEncoderLayer.forward)
    assert "position_embeddings=position_embeddings" in layer_src
    assert "attention_mask=attention_mask" in layer_src
    tower_src = inspect.getsource(modeling_gemma4.Gemma4VisionModel.__init__)
    for name in ("patch_embedder", "encoder", "pooler"):
        assert f"self.{name} =" in tower_src, f"Gemma4VisionModel no longer builds {name}"
    init_src = inspect.getsource(gemma4_mm.Gemma4ForConditionalGeneration.__init__)
    for name in ("vision_tower", "embed_vision"):
        assert f"self.{name} =" in init_src, f"upstream Gemma 4 no longer builds {name}"


def test_registry_resolves_to_spyre_class_and_keeps_processor():
    from spyre_inference.models import spyre_models
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    assert (
        spyre_models()["Gemma4ForConditionalGeneration"]
        == "spyre_inference.models.gemma4_mm:SpyreGemma4ForConditionalGeneration"
    )
    upstream = gemma4_mm.Gemma4ForConditionalGeneration
    assert SpyreGemma4ForConditionalGeneration._processor_factory is upstream._processor_factory


def test_construction_reaches_upstream_init_and_protocol():
    """A ``Protocol`` base ahead of upstream in the MRO would skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    mro = SpyreGemma4ForConditionalGeneration.__mro__[1:]
    first_init = next(cls for cls in mro if "__init__" in cls.__dict__)
    assert first_init is gemma4_mm.Gemma4ForConditionalGeneration
    assert supports_encoder_cudagraph(SpyreGemma4ForConditionalGeneration)


def test_multimodal_patches_leave_the_spyre_model_alone():
    """The old class-name dispatch to ``multimodal/gemma4_vision.py`` must not reach it."""
    from spyre_inference import multimodal

    patched = (
        modeling_gemma4.Gemma4VisionEncoder,
        modeling_gemma4.Gemma4VisionPooler,
        modeling_gemma4.Gemma4VisionPatchEmbedder,
    )
    stock = {cls: dict(vars(cls)) for cls in patched}
    model = _spyre_model(_tower_config(num_layers=1))
    multimodal.apply_multimodal_patches(model, torch.device("cpu"))
    assert {cls: dict(vars(cls)) for cls in patched} == stock


@pytest.mark.rotary
def test_rope_tables_match_transformers_reference():
    """Catches an off-by-a-quarter interleave, a swapped axis or a sin sign error."""
    from spyre_inference.multimodal.gemma4_vision import gemma4_rope_tables
    from spyre_inference.multimodal.utils import rope_perm_matrix, rope_rotate_matmul

    torch.manual_seed(0)
    config = _vision_config()
    position_ids = _position_ids()
    x = torch.randn(1, NUM_PATCHES, NUM_HEADS, ORIG_HEAD_DIM, dtype=torch.float32)

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    cos, sin = rope(x, position_ids)
    want = modeling_gemma4.apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2)

    cos_t, sin_t = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    px, py = position_ids[0, :, 0], position_ids[0, :, 1]
    cos_rows = (cos_t[0, px] + cos_t[1, py])[None, :, None]
    sin_rows = (sin_t[0, px] + sin_t[1, py])[None, :, None]
    swap = rope_perm_matrix("half", PADDED_HEAD_DIM, torch.device("cpu"), torch.float32)
    x_padded = _pad_activation_quarters(x, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    got_padded = rope_rotate_matmul(x_padded, cos_rows, sin_rows, swap)
    got = _unpad_activation_quarters(got_padded, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.rotary
def test_rope_tables_padding_lanes_are_the_identity_rotation():
    """cos=1, sin=0 after both axes sum, so a nonzero padded channel cannot leak."""
    from spyre_inference.multimodal.gemma4_vision import gemma4_rope_tables

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(_vision_config())
    cos_t, sin_t = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    cos, sin = cos_t[0, 3] + cos_t[1, 5], sin_t[0, 3] + sin_t[1, 5]

    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    for lanes in (slice(2 * quarter, half), slice(half + 2 * quarter, PADDED_HEAD_DIM)):
        assert torch.all(cos[lanes] == 1.0)
        assert torch.all(sin[lanes] == 0.0)


def test_rope_tables_apply_attention_scaling_to_the_real_lanes_only():
    from spyre_inference.multimodal.gemma4_vision import gemma4_rope_tables

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(_vision_config())
    cos, sin = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    cos_scaled, sin_scaled = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM, 2.0)

    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    real = [slice(0, 2 * quarter), slice(half, half + 2 * quarter)]
    pad = [slice(2 * quarter, half), slice(half + 2 * quarter, PADDED_HEAD_DIM)]
    for lanes in real:
        torch.testing.assert_close(cos_scaled[..., lanes], 2.0 * cos[..., lanes])
        torch.testing.assert_close(sin_scaled[..., lanes], 2.0 * sin[..., lanes])
    for lanes in pad:
        assert torch.all(cos_scaled[0, :, lanes] == 1.0)
        assert torch.all(sin_scaled[..., lanes] == 0.0)


def test_pipeline_matches_stock_on_spyre():
    """Same comparison with the encoder on the card, uncompiled (the eager path)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    config = _tower_config()
    images = _images()

    with torch.inference_mode():
        expected = _reference(config, images)
        actual = _spyre_model(config).to("spyre").embed_multimodal(**_mm_kwargs(images))

    for got, want in zip(actual, expected, strict=True):
        assert _cosine(got, want) > 0.99


def test_compiled_pipeline_matches_stock_on_spyre():
    """The compiled graphs at one budget, fed as the manager feeds them, then the host tail."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops.utils import convert

    config = _tower_config()
    images = _images()
    model = _spyre_model(config).to("spyre")
    mm_kwargs = _mm_kwargs(images)
    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        64, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    tokens = model.vision_item_tokens(mm_kwargs)
    actual: list[torch.Tensor | None] = [None] * len(tokens)

    with torch.inference_mode():
        expected = _reference(config, images)
        model.compile_vision_graphs()
        out = model.encoder_cudagraph_forward(
            {k: convert(v, device="spyre") for k, v in bufs.items()}
        )
        model.postprocess_encoder_output({"default": out}, list(range(len(tokens))), tokens, actual)

    for got, want in zip(actual, expected, strict=True):
        assert got is not None
        assert _cosine(got, want) > 0.99
