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

"""``SpyrePixtralForConditionalGeneration``'s vision pipeline against upstream Pixtral.

The pipeline pads each image onto a square canvas, runs the patch conv and reorders its
output offset-major, permutes the merger weight and crops the canvas's padding off; these
tests pin each of those to the upstream result on CPU. The on-card twins at the bottom skip
without a device.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")

PATCH = 7
MERGE = 2


@pytest.fixture(autouse=True)
def _engine_ir_dispatch():
    """Trace vLLM IR ops such as ``RMSNorm`` inline, as the worker does under
    STOCK_TORCH_COMPILE; the wrapped op fails a layout check the engine never reaches."""
    import vllm.ir

    with vllm.ir.enable_torch_wrap(False):
        yield


def _args(num_layers: int = 2, image_size: int = 112) -> "pixtral.VisionEncoderArgs":
    return pixtral.VisionEncoderArgs(
        hidden_size=128,
        num_channels=3,
        image_size=image_size,
        patch_size=PATCH,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=2,  # head_dim 64
        rope_theta=10000.0,
        image_token_id=10,
        adapter_bias=True,
        spatial_merge_size=MERGE,
        add_pre_mm_projector_layer_norm=True,
        mm_projector_id="patch_merge",
    )


def _finish_weight_loading(module: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _upstream_modules(args) -> nn.ModuleDict:
    from vllm.model_executor.layers.layernorm import RMSNorm

    torch.manual_seed(0)
    mods = nn.ModuleDict(
        {
            "vision_encoder": pixtral.VisionTransformer(args),
            "pre_mm_projector_norm": RMSNorm(args.hidden_size, eps=1e-5),
            "patch_merger": pixtral.PatchMerger(args.hidden_size, MERGE, use_mlp_bias=False),
            "vision_language_adapter": pixtral.VisionLanguageAdapter(args, dim=args.hidden_size),
        }
    )
    for p in mods.parameters():
        p.data.normal_(std=0.05)
    for name, p in mods.named_parameters():
        if name.endswith("norm.weight"):
            p.data.add_(1.0)
    # Unfinished on purpose: the Spyre linears transpose `weight` when loading finishes,
    # so each side copies raw weights first and finishes on its own.
    return mods


def _reference(args, images: list[torch.Tensor]) -> list[torch.Tensor]:
    """Upstream ``PixtralForConditionalGeneration._process_image_input``."""
    mods = _upstream_modules(args)
    _finish_weight_loading(mods)
    feats = mods["vision_encoder"](images)
    sizes = [f.shape[0] // MERGE**2 for f in feats]
    x = mods["pre_mm_projector_norm"](torch.cat(feats))
    dims = [(img.shape[1] // PATCH, img.shape[2] // PATCH) for img in images]
    x = mods["patch_merger"](x, image_sizes=dims)
    return list(torch.split(mods["vision_language_adapter"](x), sizes))


def _spyre_model(args, dtype=torch.float16):
    """A ``SpyrePixtralForConditionalGeneration`` around the same (seeded) upstream
    modules as ``_reference``, with the steps weight loading performs."""
    from spyre_inference.models.pixtral import SpyrePixtralForConditionalGeneration

    model = SpyrePixtralForConditionalGeneration.__new__(SpyrePixtralForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_args = args
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=args.hidden_size))
    for name, mod in _upstream_modules(args).items():
        setattr(model, name, mod)
    # The Spyre blocks replace upstream's whole, so they get the same seeded weights back.
    states = [layer.state_dict() for layer in model.vision_encoder.transformer.layers]
    model._init_spyre_vision(prefix="")
    for layer, state in zip(model.vision_encoder.transformer.layers, states, strict=True):
        layer.load_state_dict(state)
    # The loader's finish steps, in its order: per-layer first, then the model hook.
    _finish_weight_loading(model)
    model.process_weights_after_loading()
    return model.to(dtype)


def _images() -> list[torch.Tensor]:
    torch.manual_seed(1)
    # Non-square, so each canvas has padding patches: 8x12 on a 16x16 canvas, 6x4 on 8x8.
    return [torch.randn(3, 8 * PATCH, 12 * PATCH), torch.randn(3, 6 * PATCH, 4 * PATCH)]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("num_images", [1, 2])
def test_pipeline_matches_upstream(tp_group, num_images):
    args = _args()
    images = _images()[:num_images]

    with torch.inference_mode():
        expected = _reference(args, images)
        actual = _spyre_model(args).embed_multimodal(images=images)

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.999


def test_budgets_are_one_per_canvas(tp_group):
    """Every budget is one square canvas, and an image's tokens are its canvas's."""
    args = _args()
    model = _spyre_model(args)
    # image_size 112 / patch 7 = 16 patches a side: canvases 8 and 16, 16 and 64 tokens.
    assert model.vision_token_budgets == [16, 64]
    assert model.vision_max_items_per_call == 1
    assert model.vision_item_tokens({"images": _images()}) == [64, 16]
    tower = model.vision_encoder
    assert tower.patch_conv.weight.shape == (args.hidden_size, 3, PATCH, PATCH)
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        64, 1, 1, torch.device("cpu"), torch.float16
    )
    assert bufs.values["image"].shape == (3, 16 * PATCH, 16 * PATCH)
    assert bufs.values["positions"].shape == (256,)


def test_top_canvas_outside_the_table_matches_upstream(tp_group):
    """37 patches a side rounds up to a 40-patch top canvas, which ``CANVAS_SIDES`` lacks, as
    Ministral-3's 110 round up to 112."""
    args = _args(image_size=37 * PATCH)
    model = _spyre_model(args)
    assert model.vision_token_budgets == [16, 64, 144, 256, 400]
    torch.manual_seed(4)
    image = torch.randn(3, 34 * PATCH, 20 * PATCH)

    with torch.inference_mode():
        (expected,) = _reference(args, [image])
        (actual,) = model.embed_multimodal(images=[image])

    assert actual.shape == expected.shape
    assert _cosine(actual, expected) > 0.999


def test_image_above_image_size_is_rejected(tp_group):
    model = _spyre_model(_args())
    with pytest.raises(ValueError, match="16-patch image_size limit"):
        model.vision_item_tokens({"images": [torch.randn(3, 18 * PATCH, 2 * PATCH)]})


def test_canvas_padding_does_not_leak_into_the_image(tp_group):
    """The same image on a larger canvas gives the same tokens: padding patches have their own
    segment."""
    model = _spyre_model(_args(num_layers=1))
    image = _images()[1]

    with torch.inference_mode():
        (tight,) = model.embed_multimodal(images=[image])
        model._canvas_sides = [16]
        (loose,) = model.embed_multimodal(images=[image])
    torch.testing.assert_close(tight, loose, atol=1e-3, rtol=1e-3)


def test_offset_major_merge_matches_upstream_merger(tp_group):
    """Offset-major patches times the per-offset weight equal ``unfold`` plus the original."""
    from spyre_inference.models._vision_encoder import offset_major
    from spyre_inference.models.pixtral import offset_major_merge_weight

    torch.manual_seed(2)
    hp, wp, d = 6, 4, 16
    x = torch.randn(hp * wp, d)
    merger = pixtral.PatchMerger(d, MERGE, use_mlp_bias=False)
    merger.merging_layer.weight.data.normal_()
    weight = offset_major_merge_weight(merger.merging_layer.weight.data, MERGE)
    _finish_weight_loading(merger)
    expected = merger(x, image_sizes=[(hp, wp)])

    blocks = offset_major(x.reshape(hp, wp, d), MERGE)
    torch.testing.assert_close((blocks @ weight).sum(0), expected, atol=1e-5, rtol=1e-5)


def test_rope_tables_match_upstream_complex_rope():
    from spyre_inference.models.pixtral import pixtral_rope_tables
    from spyre_inference.multimodal.utils import rope_perm_matrix, rope_rotate_matmul

    head_dim, side = 64, 16
    cos, sin = pixtral_rope_tables(head_dim, side, 10000.0)
    torch.manual_seed(3)
    rows, cols = torch.randint(0, side, (2, 40))
    q, k = torch.randn(2, 1, 40, 2, head_dim).unbind(0)
    freqs = torch.view_as_real(pixtral.precompute_freqs_cis_2d(head_dim, side, side, 10000.0))
    freqs = freqs[rows, cols]
    want_q, want_k = pixtral.apply_rotary_emb_vit(q, k, freqs_cis=freqs)

    flat = rows * side + cols
    perm = rope_perm_matrix("pair", head_dim, torch.device("cpu")).float()
    c, s = cos[flat][None, :, None], sin[flat][None, :, None]
    torch.testing.assert_close(rope_rotate_matmul(q, c, s, perm), want_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rope_rotate_matmul(k, c, s, perm), want_k, atol=1e-5, rtol=1e-5)


def test_upstream_tower_contract_is_unchanged():
    """What the Spyre blocks still take from upstream's: the constructors, the submodules the
    residual path calls and the attention's projections. Diff it on a vLLM bump."""
    params = list(inspect.signature(pixtral.Attention.__init__).parameters)
    assert params == ["self", "args", "quant_config", "prefix", "disable_tp"]
    # TransformerBlock's signature is support_torch_compile's wrapper; construct it instead.
    block = pixtral.TransformerBlock(_args(), quant_config=None, prefix="", disable_tp=True)
    for name in ("attention", "attention_norm", "feed_forward", "ffn_norm"):
        assert hasattr(block, name), f"upstream TransformerBlock no longer has {name}"
    for name in ("qkv_proj", "o_proj", "n_heads", "head_dim"):
        assert hasattr(block.attention, name), f"upstream Attention no longer has {name}"
    init_src = inspect.getsource(pixtral.PixtralForConditionalGeneration.__init__)
    for name in (
        "vision_encoder",
        "pre_mm_projector_norm",
        "patch_merger",
        "vision_language_adapter",
    ):
        assert f"self.{name} =" in init_src, f"upstream Pixtral no longer builds {name}"


def test_registry_resolves_to_spyre_class_and_keeps_processor():
    from spyre_inference.models import spyre_models
    from spyre_inference.models.pixtral import SpyrePixtralForConditionalGeneration

    assert (
        spyre_models()["PixtralForConditionalGeneration"]
        == "spyre_inference.models.pixtral:SpyrePixtralForConditionalGeneration"
    )
    upstream = pixtral.PixtralForConditionalGeneration
    assert SpyrePixtralForConditionalGeneration._processor_factory is upstream._processor_factory


def test_construction_reaches_upstream_init_and_protocol():
    """``super().__init__`` must land on upstream's: a ``Protocol`` base ahead of it in the
    MRO would substitute its placeholder ``__init__`` and skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models.pixtral import SpyrePixtralForConditionalGeneration

    mro = SpyrePixtralForConditionalGeneration.__mro__[1:]
    first_init = next(cls for cls in mro if "__init__" in cls.__dict__)
    assert first_init is pixtral.PixtralForConditionalGeneration
    assert supports_encoder_cudagraph(SpyrePixtralForConditionalGeneration)


def test_loaded_vision_weights_are_reshaped_in_place(tp_group):
    """The patch-conv weight stays ``[D, C, p, p]`` in a plain holder and the merging weight
    becomes the graph's layout, each under its own name, so one copy of each reaches the card."""
    args = _args()
    model = _spyre_model(args)
    tower, merger = model.vision_encoder, model.patch_merger
    from spyre_inference.custom_ops.conv import SpyreConv2d

    assert not isinstance(tower.patch_conv, SpyreConv2d), "the runner would tile a conv copy"
    assert tower.patch_conv.weight.shape == (args.hidden_size, 3, PATCH, PATCH)
    assert merger.merging_layer.weight.shape == (MERGE**2, args.hidden_size, args.hidden_size)
    names = {name for name, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    assert not any("spyre_patch_weight" in n or "spyre_merge_weight" in n for n in names)


def test_pipeline_matches_upstream_on_spyre(tp_group):
    """Same comparison with the encoder on the card, uncompiled (the eager path)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    args = _args()
    images = _images()

    with torch.inference_mode():
        expected = _reference(args, images)
        actual = _spyre_model(args).to("spyre").embed_multimodal(images=images)

    for got, want in zip(actual, expected, strict=True):
        assert _cosine(got, want) > 0.995


@pytest.mark.parametrize("image_index", [0, 1], ids=["canvas16", "canvas8"])
def test_compiled_pipeline_matches_upstream_on_spyre(tp_group, image_index):
    """The compiled graphs at one canvas's budget, fed and cropped the way the manager does."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    args = _args()
    image = _images()[image_index]
    batch = {"images": [image]}
    model = _spyre_model(args).to("spyre")
    model.compile_vision_graphs()
    (tokens,) = model.vision_item_tokens(batch)

    with torch.inference_mode():
        (expected,) = _reference(args, [image])
        out = model.encoder_cudagraph_forward(_padded_device_inputs(model, [image], tokens))
    dest: list[torch.Tensor | None] = [None]
    model.postprocess_encoder_output({"default": out}, [0], [tokens], dest, batch_mm_kwargs=batch)
    assert dest[0] is not None and dest[0].shape == expected.shape
    assert _cosine(dest[0], expected) > 0.995


def _padded_device_inputs(model, images, token_budget):
    from spyre_inference.custom_ops.utils import convert

    values = model.prepare_encoder_cudagraph_replay_buffers({"images": images}, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    return {k: convert(v, device="spyre") for k, v in bufs.items()}


def test_compiled_graphs_match_eager_on_the_same_inputs_on_spyre(tp_group):
    """The same padded device buffers through the uncompiled and the compiled pipeline."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    args = _args()
    images = _images()[:1]
    model = _spyre_model(args).to("spyre")
    (tokens,) = model.vision_item_tokens({"images": images})
    inputs = _padded_device_inputs(model, images, token_budget=tokens)

    with torch.inference_mode():
        eager = model.encoder_cudagraph_forward(inputs)[:tokens]
        model.compile_vision_graphs()
        compiled = model.encoder_cudagraph_forward(inputs)[:tokens]

    assert _cosine(compiled, eager) > 0.9999
    torch.testing.assert_close(compiled.float(), eager.float(), atol=2e-2, rtol=2e-2)


def test_gather_table_gathers_exact_rows_on_spyre():
    """Rows gathered on the card, eager and inside a compiled graph, are exactly the table's
    rows. Compared with the table as the card holds it: uploading fp16 rounds to the card's
    16-bit format, so the host values are not the reference."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.models._vision_encoder import GatherTable

    torch.manual_seed(0)
    host = torch.randn(256, 128).half()
    # Off the stick and out of order, with repeats, like Pixtral's positions.
    index = torch.randint(0, 256, (120,))
    table = GatherTable(host.clone()).to("spyre")
    device_index = convert(index, device="spyre")

    with torch.inference_mode():
        rows = convert(table.table, device="cpu")[index]
        eager = convert(table.gather(device_index), device="cpu")
        compiled_gather = torch.compile(table.gather, backend="inductor", dynamic=False)
        compiled = convert(compiled_gather(device_index), device="cpu")

    assert torch.equal(eager, rows)
    assert torch.equal(compiled, rows)
    torch.testing.assert_close(rows, host[index], atol=4e-3, rtol=1e-3)


def test_block_compile_is_honoured(tp_group):
    """Upstream's ``support_torch_compile`` ``__call__`` must not bypass the graph that
    ``block.compile`` installs."""
    block = _spyre_model(_args(), dtype=torch.float32).vision_encoder.transformer.layers[0]
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    block.compile(backend=backend, fullgraph=True, dynamic=False)
    head_dim = block.attention.head_dim
    x = torch.randn(1, 4, block.attention.n_heads * head_dim)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    block(x, mask, (torch.ones(4, head_dim), torch.zeros(4, head_dim)))
    assert graphs
