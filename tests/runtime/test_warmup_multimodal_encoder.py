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

"""Warmup compiles every vision-encoder batch width a tower can see.

Nothing else traces the tower before the compile guard arms (``profile_run`` never
runs on Spyre), and each image-batch width is its own graph, so a width warmup skips
compiles on the first request that reaches it (spyre-inference#1182).
"""

from __future__ import annotations

import types

import torch
from vllm.config import CompilationMode

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


class _Bucketer:
    def __init__(self) -> None:
        self.bucket_sizes = [8]
        self.warmed_up = False

    def mark_warmed_up(self) -> None:
        self.warmed_up = True


class _Vision:
    def __init__(self) -> None:
        self.batches: list[tuple[str, int]] = []

    def embed_multimodal(self, **kwargs) -> None:
        self.batches.append((kwargs["modality"], kwargs["n"]))

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        return None


def _runner(
    runner_type: str,
    *,
    multimodal: bool,
    max_items: dict[str, int] | None,
    compiled_tower_blocks: int = 0,
):
    compilation_config = types.SimpleNamespace(
        compile_sizes=[8],
        inductor_compile_config={},
        static_forward_context={},
        mode=CompilationMode.NONE,
    )
    model_config = types.SimpleNamespace(
        runner_type=runner_type,
        enforce_eager=False,
        is_encoder_decoder=False,
        multimodal_config=types.SimpleNamespace(mm_encoder_only=True) if multimodal else None,
    )
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model_config = model_config
    runner.supports_mm_inputs = multimodal
    runner._compiled_tower_blocks = compiled_tower_blocks
    runner.vllm_config = types.SimpleNamespace(
        model_config=model_config,
        compilation_config=compilation_config,
    )
    runner.compilation_config = compilation_config
    runner._spyre_device = torch.device("cpu")
    runner._encoder_budget = 8
    runner._encoder_rectangles = []
    runner._spyre_kv_caches = []
    # load_model sets this. The fake runner skips load_model, and the compiled
    # pooling path reads it before deciding whether to warm the embed gather.
    runner._pooling_on_spyre = False
    runner.spyre_shape_bucketer = _Bucketer()
    runner.mm_budget = (
        types.SimpleNamespace(mm_max_items_per_batch=max_items) if max_items is not None else None
    )
    vision = _Vision()
    runner.model = vision

    def dummy_run(size, *args, **kwargs):
        return torch.zeros(size, 4), torch.zeros(size, 4)

    runner._dummy_run = dummy_run
    runner._dummy_pooler_run = lambda hidden_states: None
    runner._warm_pooler_row_widths = lambda hidden_states: None
    runner._warm_encoder_unpack = lambda hidden_states: None
    runner._get_mm_dummy_batch = lambda modality, n: {"modality": modality, "n": n}
    return runner, vision


def test_eager_pooling_does_not_run_the_vision_tower():
    """An eager pooling load does not run the vision tower: only a compiled tower is
    warmed, and an eager one compiles per op on first use regardless."""
    runner, vision = _runner("pooling", multimodal=True, max_items={"image": 4})
    runner.model_config.enforce_eager = True
    runner.max_num_reqs = 4
    runner.scheduler_config = types.SimpleNamespace(max_num_batched_tokens=64)

    runner.warming_up_model()

    assert vision.batches == []
    assert runner.spyre_shape_bucketer.warmed_up


def test_pooling_warmup_runs_every_image_batch_width(monkeypatch):
    monkeypatch.setattr(
        "spyre_inference.v1.worker.spyre_model_runner.encoder_group_shapes",
        lambda config: [],
    )
    runner, vision = _runner("pooling", multimodal=True, max_items={"image": 4})

    runner.warming_up_model()

    assert vision.batches == [("image", n) for n in range(1, 5)]
    assert runner.spyre_shape_bucketer.warmed_up


def test_pooling_warmup_covers_each_modality():
    runner, vision = _runner("pooling", multimodal=True, max_items={"image": 2, "audio": 1})

    runner._warmup_multimodal_encoder()

    assert vision.batches == [("image", 1), ("image", 2), ("audio", 1)]


def test_generative_vlm_with_an_eager_tower_is_not_enumerated():
    runner, vision = _runner("generate", multimodal=True, max_items={"image": 4})

    runner._warmup_multimodal_encoder()

    assert vision.batches == []


def test_generative_warmup_runs_a_compiled_tower_at_every_width():
    """The compile guard watches a marked tower, so the bucketed decoder warmup has to
    reach it too, not only pooling's."""
    runner, vision = _runner(
        "generate", multimodal=True, max_items={"image": 3}, compiled_tower_blocks=2
    )
    runner.max_num_reqs = 4
    runner._dummy_sampler_run = lambda hidden_states: None
    runner._record_attention_graphs = lambda: None

    runner.warming_up_model()

    assert vision.batches == [("image", n) for n in range(1, 4)]


def test_generative_warmup_walks_images_only():
    runner, vision = _runner(
        "generate",
        multimodal=True,
        max_items={"video": 1, "image": 2, "audio": 1},
        compiled_tower_blocks=2,
    )

    runner._warmup_multimodal_encoder()

    assert vision.batches == [("image", 1), ("image", 2)]


def test_text_only_pooling_skips_the_vision_tower():
    runner, vision = _runner("pooling", multimodal=False, max_items=None)

    runner._warmup_multimodal_encoder()

    assert vision.batches == []
