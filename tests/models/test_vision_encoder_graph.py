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

"""The ``SpyreVisionEncoderMixin`` contract and ``SpyreEncoderGraphManager``, on CPU with a
fake model: what a model author can rely on without touching any other file."""

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin, VisionBuffer
from spyre_inference.v1.worker.spyre_encoder_graph import SpyreEncoderGraphManager

MERGE_PATCHES = 4  # 2x2 merge


class _Upstream(nn.Module):
    """Stands in for the upstream model class the mixin sits ahead of."""

    def get_input_modality(self, mm_kwargs):
        return mm_kwargs.get("modality", "image")

    def embed_multimodal(self, **kwargs):
        return ("upstream",)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        """Text rows ``[T, 8]`` holding each token id; records what upstream was handed."""
        self.text_calls = [*getattr(self, "text_calls", []), (multimodal_embeddings, is_multimodal)]
        return input_ids.to(torch.float16)[:, None].expand(-1, 8).contiguous()


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, offset):
        return x + offset


class _FakeVisionModel(SpyreVisionEncoderMixin, _Upstream):
    """Item ``i`` is ``tokens[i]`` patches per offset group, all holding ``i + 1``."""

    def __init__(self, max_tokens: int = 300, num_blocks: int = 1) -> None:
        super().__init__()
        self.patches_per_token = MERGE_PATCHES
        self.max_tokens_per_item = max_tokens
        self.out_hidden_size = 8
        self.vision_buffers = {"patches": VisionBuffer(8), "segments": VisionBuffer(None, pad=-1.0)}
        self.blocks = nn.ModuleList([_Block() for _ in range(num_blocks)])
        self.forward_patches: list[int] = []
        self.seen_taps: list[list[torch.Tensor]] = []

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return list(mm_kwargs["tokens"])

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return {"tokens": [mm_kwargs["tokens"][i] for i in indices], "ids": indices}

    def vision_inputs(self, mm_kwargs):
        ids = mm_kwargs.get("ids", range(len(mm_kwargs["tokens"])))
        seg = torch.cat([torch.full((t,), float(i + 1)) for i, t in zip(ids, mm_kwargs["tokens"])])
        seg = seg.repeat(MERGE_PATCHES).to(torch.float16)
        return {"patches": seg[:, None].expand(-1, 8).contiguous(), "segments": seg}

    def vision_embed(self, values):
        return values["patches"], {"offset": 0.0}

    def vision_blocks(self):
        return self.blocks

    def vision_project(self, x, taps):
        self.forward_patches.append(x.shape[0])
        self.seen_taps.append(taps)
        return x.reshape(MERGE_PATCHES, -1, x.shape[-1]).mean(dim=0)


def _vllm_config(budgets=None, tp_mode="weights", videos=0):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=budgets or [],
            encoder_cudagraph_max_vision_items_per_batch=0,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda modality: videos if modality == "video" else 4,
                mm_encoder_tp_mode=tp_mode,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


def _manager(model, **kwargs) -> SpyreEncoderGraphManager:
    return SpyreEncoderGraphManager(
        _vllm_config(**kwargs), torch.device("cpu"), torch.float16, model
    )


def test_budgets_are_stick_aligned_and_cover_the_largest_item():
    manager = _manager(_FakeVisionModel(max_tokens=300))
    assert manager.token_budgets == [16, 32, 64, 128, 256, 304]
    assert all(b * MERGE_PATCHES % 64 == 0 for b in manager.token_budgets)


def test_user_budgets_below_the_largest_item_fail_at_startup():
    with pytest.raises(ValueError, match="largest encoder budget"):
        _manager(_FakeVisionModel(max_tokens=300), budgets=[16, 64])


def test_encoder_data_parallel_is_rejected():
    with pytest.raises(NotImplementedError, match="mm_encoder_tp_mode"):
        _manager(_FakeVisionModel(), tp_mode="data")


def test_execute_packs_pads_per_block_and_splits_per_item():
    model = _FakeVisionModel(max_tokens=300)
    manager = _manager(model)
    manager.capture(graph_pool=None)
    assert sorted(model.forward_patches) == sorted(b * MERGE_PATCHES for b in manager.token_budgets)

    model.forward_patches.clear()
    outputs = manager.execute({"tokens": [10, 30]})

    # Upstream splits rather than pad both items into 64: the 16 and 32 budgets sum to less.
    assert model.forward_patches == [16 * MERGE_PATCHES, 32 * MERGE_PATCHES]
    assert [o.shape[0] for o in outputs] == [10, 30]
    for i, out in enumerate(outputs):
        assert out.device.type == "cpu"
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))


def test_unrecorded_budget_is_a_miss():
    manager = _manager(_FakeVisionModel())
    assert manager._run_budget_graph({"tokens": [10]}, 16) is None


def test_inputs_must_match_the_declared_buffers():
    """An upstream ``prepare_encoder_cudagraph_replay_buffers`` with other keys would
    otherwise replay stale padding silently."""
    model = _FakeVisionModel()
    model.vision_inputs = lambda mm_kwargs: {"patches": torch.zeros(4, 8, dtype=torch.float16)}
    with pytest.raises(ValueError, match="vision_buffers declares"):
        model.prepare_encoder_cudagraph_replay_buffers({"tokens": [1]}, 0, 0)


def test_inputs_must_be_fp16_with_one_patch_count():
    model = _FakeVisionModel()
    model.vision_inputs = lambda mm_kwargs: {
        "patches": torch.zeros(4, 8),
        "segments": torch.zeros(4, dtype=torch.float16),
    }
    with pytest.raises(ValueError, match="declared dtypes"):
        model.prepare_encoder_cudagraph_replay_buffers({"tokens": [1]}, 0, 0)


def test_items_per_call_cap_splits_the_batch():
    model = _FakeVisionModel()
    type(model).vision_max_items_per_call = 1
    try:
        manager = _manager(model)
        manager.capture(graph_pool=None)
        model.forward_patches.clear()
        outputs = manager.execute({"tokens": [10, 30]})
    finally:
        del type(model).vision_max_items_per_call
    assert manager.max_batch_size == 1
    assert len(model.forward_patches) == 2
    assert [o.shape[0] for o in outputs] == [10, 30]


def test_items_per_call_cap_also_caps_the_frames():
    """Upstream sizes the frame limit from the uncapped item count before the cap applies."""
    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, max_frames_per_video=8)
    type(model).vision_max_items_per_call = 2
    try:
        manager = _manager(model, videos=1)
    finally:
        del type(model).vision_max_items_per_call
    assert manager.max_batch_size == 2
    assert manager.max_frames_per_batch == 2 * 8


def test_taps_reach_vision_project_in_declared_order():
    model = _FakeVisionModel(num_blocks=3)
    type(model).vision_taps = (-1, 0, 2)
    try:
        bufs = model._vision_empty(64)
        bufs["patches"][:] = 1.0
        model.blocks[0].forward = lambda x, offset: x + 1
        model.blocks[1].forward = lambda x, offset: x + 10
        model.blocks[2].forward = lambda x, offset: x + 100
        model.encoder_cudagraph_forward(bufs)
    finally:
        del type(model).vision_taps
    (taps,) = model.seen_taps
    assert [float(t[0, 0]) for t in taps] == [112.0, 1.0, 12.0]


def test_postprocess_override_gets_host_output_and_shapes_each_item():
    """A model reshapes its embeddings through upstream's ``postprocess_encoder_output``."""
    model = _FakeVisionModel()
    seen = []

    def postprocess(outputs, indices, tokens, dest, clone=False, batch_mm_kwargs=None):
        out = outputs["default"]
        seen.append(out.device.type)
        offset = 0
        for i in indices:
            dest[i] = out[offset + 1 : offset + tokens[i]]
            offset += tokens[i]

    model.postprocess_encoder_output = postprocess
    embeds = model.embed_multimodal(tokens=[10, 30])
    assert seen == ["cpu"]
    assert [e.shape[0] for e in embeds] == [9, 29]


def test_budget_range_override_sets_the_largest_budget():
    """A model whose processor count is not in encoder tokens overrides upstream's hook."""
    model = _FakeVisionModel(max_tokens=300)
    model.get_encoder_cudagraph_budget_range = lambda vllm_config: (16, 500)
    manager = _manager(model)
    assert manager.token_budgets[-1] >= 500


def test_other_modalities_go_to_upstream():
    assert _FakeVisionModel().embed_multimodal(modality="audio") == ("upstream",)


def test_integer_buffers_keep_their_dtype_and_padding():
    """Index buffers (e.g. positions) are padded and validated in their declared dtype."""
    model = _FakeVisionModel()
    model.vision_buffers = {
        **model.vision_buffers,
        "positions": VisionBuffer(None, pad=0, dtype=torch.int64),
    }
    bufs = model._vision_empty(64)
    assert bufs["positions"].dtype == torch.int64

    inputs = model.vision_inputs

    def with_positions(mm_kwargs):
        values = inputs(mm_kwargs)
        return {**values, "positions": torch.arange(values["patches"].shape[0])}

    model.vision_inputs = with_positions
    values = model.prepare_encoder_cudagraph_replay_buffers({"tokens": [10]}, 0, 0).values
    model.get_encoder_cudagraph_config().padding_logics["positions"](
        bufs["positions"], values["positions"]
    )
    assert bufs["positions"].view(MERGE_PATCHES, -1)[:, 10:].eq(0).all()


def test_gather_table_gathers_rows_on_the_host():
    from spyre_inference.models._vision_encoder import GatherTable

    table = GatherTable(torch.arange(12.0).reshape(4, 3))
    torch.testing.assert_close(
        table.gather(torch.tensor([2, 0])), torch.tensor([[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]])
    )
    assert table.half().table.dtype == torch.float16


def test_out_of_range_tap_fails_instead_of_aliasing():
    """A tap past the last layer would otherwise wrap onto an earlier one by modulo."""
    model = _FakeVisionModel(num_blocks=2)
    type(model).vision_taps = (3,)
    try:
        with pytest.raises(ValueError, match="out of range for 3 hidden states"):
            model.encoder_cudagraph_forward(model._vision_empty(64))
    finally:
        del type(model).vision_taps


def test_compile_vision_graphs_twice_keeps_the_first_compile():
    """A retried warmup must not re-wrap the graphs it already compiled."""
    model = _FakeVisionModel(num_blocks=2)
    model.compile_vision_graphs()
    embed, project = model._spyre_vision_embed, model._spyre_vision_project
    blocks = [b._compiled_call_impl for b in model.blocks]
    model.compile_vision_graphs()
    assert model._spyre_vision_embed is embed
    assert model._spyre_vision_project is project
    assert [b._compiled_call_impl for b in model.blocks] == blocks


def test_item_and_frame_caps_together_through_capture_and_execute():
    """Both caps reach capture, and replay still splits the batch at the item cap."""
    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, max_frames_per_video=8)
    capture_limits = []
    prepare = model.prepare_encoder_cudagraph_capture_inputs

    def recording_prepare(budget, max_batch_size, max_frames_per_batch, *args, **kwargs):
        capture_limits.append((max_batch_size, max_frames_per_batch))
        return prepare(budget, max_batch_size, max_frames_per_batch, *args, **kwargs)

    model.prepare_encoder_cudagraph_capture_inputs = recording_prepare
    type(model).vision_max_items_per_call = 2
    try:
        manager = _manager(model, videos=1)
        manager.capture(graph_pool=None)
        model.forward_patches.clear()
        outputs = manager.execute({"tokens": [5, 10, 15, 20, 25]})
    finally:
        del type(model).vision_max_items_per_call
    assert set(capture_limits) == {(2, 16)}
    assert len(model.forward_patches) == 3
    assert [o.shape[0] for o in outputs] == [5, 10, 15, 20, 25]
    for i, out in enumerate(outputs):
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))


def test_each_path_compiles_and_replays_from_its_own_staging():
    """A model declaring two upstream paths gets one staging set per path and budget."""
    from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphPathConfig, EncoderItemSpec

    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    paths = {"local": EncoderCudaGraphPathConfig(), "global": EncoderCudaGraphPathConfig()}
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(config, paths=paths)
    model.get_encoder_cudagraph_item_specs = lambda mm_kwargs: [
        EncoderItemSpec(t * MERGE_PATCHES, t, {"local": t, "global": t})
        for t in mm_kwargs["tokens"]
    ]
    forward = model.encoder_cudagraph_forward
    seen_paths = []

    def recording_forward(inputs, path="default"):
        seen_paths.append(path)
        return forward(inputs) + (100.0 if path == "global" else 0.0)

    model.encoder_cudagraph_forward = recording_forward
    model.postprocess_encoder_output = lambda outputs, indices, tokens, dest, **kw: [
        dest.__setitem__(i, outputs["local"][:t] + outputs["global"][:t])
        for i, t in zip(indices, [tokens[i] for i in indices])
    ]

    manager = _manager(model)
    manager.capture(graph_pool=None)
    assert set(manager._staging) == {"local", "global"}
    assert all(set(s) == set(manager.token_budgets) for s in manager._staging.values())

    seen_paths.clear()
    (out,) = manager.execute({"tokens": [10]})
    assert sorted(seen_paths) == ["global", "local"]
    torch.testing.assert_close(out, torch.full_like(out, 1.0 + 101.0))


def test_runner_warmup_builds_the_manager_and_serves_through_it():
    """The runner path: _record_vision_graphs compiles the model, builds the Spyre manager
    from the processor's largest item, captures every budget and warms the merge."""
    from vllm.config import CompilationMode

    from spyre_inference.v1.worker.spyre_model_runner import (
        TorchSpyreModelRunner,
        _SpyreModelWrapper,
    )

    model = _FakeVisionModel()
    compiled = []
    model.compile_vision_graphs = lambda: compiled.append(True)
    vllm_config = _vllm_config()
    vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    vllm_config.compilation_config.inductor_compile_config = {}
    vllm_config.model_config.enforce_eager = False
    vllm_config.model_config.dtype = torch.float16

    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.vllm_config = vllm_config
    runner.compilation_config = vllm_config.compilation_config
    runner.model_config = vllm_config.model_config
    runner.supports_mm_inputs = True
    runner.model = _SpyreModelWrapper(
        model,
        torch.device("cpu"),
        model_dtype=torch.float16,
        inputs_embeds_buffer=torch.zeros(16, 8, dtype=torch.float16),
    )
    runner.spyre_shape_bucketer = SimpleNamespace(bucket_sizes=[1, 4, 16])
    merged = []
    merge = model.vision_merge
    model.vision_merge = lambda text, image, mask: merged.append(text.shape[0]) or merge(
        text, image, mask
    )
    runner.mm_budget = SimpleNamespace(mm_max_toks_per_item={"image": 300})
    runner._spyre_device = torch.device("cpu")
    runner.encoder_cudagraph_manager = None

    runner._record_vision_graphs()

    manager = runner.encoder_cudagraph_manager
    assert compiled == [True]
    assert isinstance(manager, SpyreEncoderGraphManager)
    assert model.max_tokens_per_item == 300
    assert manager.token_budgets[-1] == 304
    assert set(manager._staging["default"]) == set(manager.token_budgets)
    outputs = manager.execute({"tokens": [10, 30]})
    assert [o.shape[0] for o in outputs] == [10, 30]
    assert manager.graph_hits == 2 and manager.graph_misses == 0
    # Every bucket an image step can reach; the one-token bucket is never one.
    assert merged == [4, 16]


def _prompt(mask_rows: list[int], num_tokens: int = 12):
    is_multimodal = torch.zeros(num_tokens, dtype=torch.bool)
    is_multimodal[mask_rows] = True
    return torch.arange(100, 100 + num_tokens), is_multimodal


def test_merge_matches_upstream_merge_for_several_images():
    """Two images' rows, out of order in the prompt, land where upstream's merge puts them."""
    from vllm.model_executor.models.utils import _merge_multimodal_embeddings

    model = _FakeVisionModel()
    input_ids, is_multimodal = _prompt([1, 2, 3, 7, 8])
    images = [torch.randn(3, 8).half(), torch.randn(2, 8).half()]

    got = model.embed_input_ids(input_ids, images, is_multimodal=is_multimodal)
    text = input_ids.to(torch.float16)[:, None].expand(-1, 8).clone()
    want = _merge_multimodal_embeddings(text, images, is_multimodal)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_upstream_text_chain_runs_without_image_tokens():
    """Upstream gets ``is_multimodal`` (out-of-vocabulary placeholders) but no image tokens, so
    its own merge never runs."""
    model = _FakeVisionModel()
    input_ids, is_multimodal = _prompt([0])
    model.embed_input_ids(input_ids, [torch.ones(1, 8).half()], is_multimodal=is_multimodal)
    ((passed_images, passed_mask),) = model.text_calls
    assert passed_images == ()
    assert passed_mask is is_multimodal


def test_text_only_steps_skip_the_merge():
    model = _FakeVisionModel()
    model.vision_merge = lambda *args: pytest.fail("merged a step with no image tokens")
    input_ids, _ = _prompt([])
    out = model.embed_input_ids(input_ids, (), is_multimodal=None)
    assert out.shape == (12, 8)


def test_count_mismatch_fails_like_upstream():
    model = _FakeVisionModel()
    input_ids, is_multimodal = _prompt([1, 2, 3])
    with pytest.raises(ValueError, match="2 multimodal tokens to 3 placeholders"):
        model.embed_input_ids(input_ids, [torch.ones(2, 8).half()], is_multimodal=is_multimodal)


def test_vision_merge_override_receives_the_placed_rows():
    model = _FakeVisionModel()
    seen = []

    def merge(text, image, mask):
        seen.append((image.clone(), mask.clone()))
        return text

    model.vision_merge = merge
    input_ids, is_multimodal = _prompt([4, 5])
    model.embed_input_ids(input_ids, [torch.full((2, 8), 3.0).half()], is_multimodal=is_multimodal)
    ((image, mask),) = seen
    assert mask.shape == (12, 1) and mask[:, 0].tolist() == is_multimodal.tolist()
    assert image[4:6].eq(3).all() and image[mask[:, 0].logical_not()].eq(0).all()


def test_upstream_returns_text_alone_when_handed_no_image_tokens():
    """The mixin relies on upstream's ``embed_input_ids`` returning before its merge for an
    empty embedding list; if that changes, image tokens would merge twice."""
    from vllm.model_executor.models import utils as vllm_utils
    from vllm.model_executor.models.interfaces import SupportsMultiModal

    text = torch.randn(4, 8)
    host = SimpleNamespace(
        _has_oov_mm_tokens=False,
        get_language_model=lambda: SimpleNamespace(embed_input_ids=lambda ids: text),
        _embed_text_input_ids=lambda ids,
        embed,
        is_multimodal: SupportsMultiModal._embed_text_input_ids(
            host, ids, embed, is_multimodal=is_multimodal
        ),
    )
    original = vllm_utils._merge_multimodal_embeddings
    vllm_utils._merge_multimodal_embeddings = lambda *a, **k: pytest.fail("upstream merged")
    try:
        out = SupportsMultiModal.embed_input_ids(
            host,
            torch.zeros(4, dtype=torch.long),
            (),
            is_multimodal=torch.ones(4, dtype=torch.bool),
        )
    finally:
        vllm_utils._merge_multimodal_embeddings = original
    assert out is text


def test_wrapper_stages_an_image_step_at_two_rows_and_keeps_the_mask_on_the_host():
    """A one-token image step (a chunk ending on a placeholder) stages two rows, so neither the
    staging copy nor upstream's copy is a single row (torch-spyre#4883)."""
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    model = _FakeVisionModel()
    seen = []
    merge = model.vision_merge

    def recording_merge(text, image, mask):
        seen.append(text.shape[0])
        return merge(text, image, mask)

    model.vision_merge = recording_merge
    buf = torch.zeros(16, 8, dtype=torch.float16)
    wrapper = _SpyreModelWrapper(
        model, torch.device("cpu"), model_dtype=torch.float16, inputs_embeds_buffer=buf
    )
    out = wrapper.embed_input_ids(
        torch.tensor([7]), [torch.full((1, 8), 5.0).half()], is_multimodal=torch.tensor([True])
    )
    assert seen == [2]
    assert out.shape == (1, 8) and out.data_ptr() == buf.data_ptr()
    assert out.eq(5).all()
    ((_, mask),) = model.text_calls
    assert mask.device.type == "cpu"


@pytest.mark.parametrize("num_tokens", [1, 2, 63, 64, 65, 128])
def test_compiled_merge_through_the_wrapper_on_spyre(monkeypatch, num_tokens):
    """Text rows on the card, image rows placed on the host, merged by the compiled graph and
    staged through the wrapper's buffer: the rows match a host merge exactly."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.utils import convert
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    hidden = 128
    torch.manual_seed(0)
    text_host = torch.randn(256, hidden).half()
    # Upstream's text embedding: rows of a device table, one per prompt position.
    monkeypatch.setattr(
        _Upstream,
        "embed_input_ids",
        lambda self, ids, mm=None, *, is_multimodal=None: convert(
            text_host[: ids.shape[0]], device="spyre"
        ),
    )
    model = _FakeVisionModel()
    model.out_hidden_size = hidden
    model.compile_vision_graphs()
    buf = convert(torch.zeros(256, hidden, dtype=torch.float16), device="spyre")
    wrapper = _SpyreModelWrapper(
        model, torch.device("spyre"), model_dtype=torch.float16, inputs_embeds_buffer=buf
    )
    is_multimodal = torch.zeros(num_tokens, dtype=torch.bool)
    is_multimodal[num_tokens // 3 :: 2] = True
    image = torch.randn(int(is_multimodal.sum()), hidden).half()

    with torch.inference_mode():
        out = wrapper.embed_input_ids(
            torch.arange(num_tokens), [image], is_multimodal=is_multimodal
        )
        got = convert(out, device="cpu")

    want = text_host[:num_tokens].clone()
    want[is_multimodal] = image
    # Against the card's rounding of both sources (see the GatherTable test on Spyre).
    want = convert(convert(want, device="spyre"), device="cpu")
    assert got.shape == (num_tokens, hidden)
    assert torch.equal(got, want)


def test_overrides_accept_everything_upstream_passes():
    """Upstream calls these with keywords that grow across vLLM releases (``axis_keys`` in
    0.31); an override that misses one fails at warmup."""
    import inspect

    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    def params(fn):
        return set(inspect.signature(fn).parameters)

    for name in ("_capture_budget_graph", "_run_budget_graph"):
        upstream = getattr(EncoderCudaGraphManager, name)
        assert params(upstream) <= params(getattr(SpyreEncoderGraphManager, name)), name
    for name, upstream in vars(SupportsEncoderCudaGraph).items():
        ours = getattr(SpyreVisionEncoderMixin, name, None)
        if callable(upstream) and callable(ours) and not name.startswith("__"):
            assert params(upstream) <= params(ours), name


def test_capture_axes_are_rejected():
    model = _FakeVisionModel()
    config = model.get_encoder_cudagraph_config()
    model.get_encoder_cudagraph_config = lambda: dataclasses.replace(
        config, capture_axes=(("a", "b"),)
    )
    with pytest.raises(NotImplementedError, match="capture_axes"):
        _manager(model)


def test_eager_path_respects_the_items_per_call_cap():
    """Without the manager (enforce_eager), embed_multimodal splits the step's items at the
    cap itself, so no call carries more items than the layout can tell apart."""
    model = _FakeVisionModel()
    type(model).vision_max_items_per_call = 2
    try:
        outputs = model.embed_multimodal(tokens=[5, 10, 15, 20, 25])
    finally:
        del type(model).vision_max_items_per_call
    assert len(model.forward_patches) == 3
    assert [o.shape[0] for o in outputs] == [5, 10, 15, 20, 25]
    for i, out in enumerate(outputs):
        torch.testing.assert_close(out, torch.full_like(out, float(i + 1)))
