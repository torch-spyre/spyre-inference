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

"""Pooling warmup compiles every vision-encoder batch width, not just the widest.

``profile_run`` traces ``embed_multimodal`` once, at ``mm_max_items_per_batch``.
CLIP's vision tower stays eager, so each narrower image batch otherwise compiles
on the first request (spyre-inference#1182).
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


def _runner(runner_type: str, *, multimodal: bool, max_items: dict[str, int] | None):
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
    runner.vllm_config = types.SimpleNamespace(
        model_config=model_config,
        compilation_config=compilation_config,
    )
    runner.compilation_config = compilation_config
    runner._spyre_device = torch.device("cpu")
    runner._encoder_budget = 8
    runner._encoder_rectangles = []
    runner._spyre_kv_caches = []
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


def test_generative_vlm_is_not_enumerated():
    runner, vision = _runner("generate", multimodal=True, max_items={"image": 4})

    runner._warmup_multimodal_encoder()

    assert vision.batches == []


def test_text_only_pooling_skips_the_vision_tower():
    runner, vision = _runner("pooling", multimodal=False, max_items=None)

    runner._warmup_multimodal_encoder()

    assert vision.batches == []
