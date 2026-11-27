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

"""Tests for TorchSpyreScheduler's partial-prefill cap and sliced vision passes.

The cap lowers ``max_num_active_reqs`` around the upstream waiting loop, so these
drive that real loop and count the prefills it admits per step. Stubbing ``schedule``
would leave the cap free to degrade into a no-op under an upstream change.

The vision-pass tests (SPYRE_VISION_BLOCKS_PER_STEP) drive the same loop with the slicing
connector installed for a text model and an encoder cache installed by hand, and read the
plan each step ships as its connector metadata.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    CacheConfig,
    CompilationMode,
    ECTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from spyre_inference import envs
from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin
from spyre_inference.v1.core.scheduler import TorchSpyreScheduler
from spyre_inference.v1.core.vision_slices import (
    SpyreVisionSliceConnector,
    VisionSlices,
    VisionSlicing,
    vision_slicing,
)

# A 200-token prompt against a 256-token budget leaves 56 tokens over, which is
# exactly the leftover upstream tops the batch up with.
PROMPT_LEN = 200
MAX_NUM_BATCHED_TOKENS = 256
BLOCK_SIZE = 64
NUM_BLOCKS = 512


def _scheduler(
    runner_type: str = "generate",
    max_num_seqs: int = 4,
    ec_transfer_config: ECTransferConfig | None = None,
) -> TorchSpyreScheduler:
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=2048,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    object.__setattr__(model_config, "runner_type", runner_type)
    cache_config = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        ec_transfer_config=ec_transfer_config,
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    return TorchSpyreScheduler(
        vllm_config=vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
        ),
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )


def _prefills_per_step(scheduler: TorchSpyreScheduler, num_requests: int = 4) -> list[int]:
    """Admit `num_requests` prompts and report how many prefilled in each step."""
    for i in range(num_requests):
        scheduler.add_request(
            Request(
                request_id=f"r{i}",
                prompt_token_ids=list(range(PROMPT_LEN)),
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
            )
        )

    counts = []
    for _ in range(num_requests + 1):
        computed = {rid: req.num_computed_tokens for rid, req in scheduler.requests.items()}
        scheduled = scheduler.schedule().num_scheduled_tokens
        if not scheduled:
            break
        counts.append(
            sum(computed[rid] < scheduler.requests[rid].num_prompt_tokens for rid in scheduled)
        )
        # Stand in for execute_model + update_from_output so the loop progresses.
        for rid, num_tokens in scheduled.items():
            scheduler.requests[rid].num_computed_tokens += num_tokens
    return counts


def test_default_admits_one_prefill_per_step():
    assert _prefills_per_step(_scheduler()) == [1, 1, 1, 1]


def test_disabled_cap_lets_upstream_top_up_the_batch(monkeypatch):
    """The contrast that proves the cap is not a no-op: leftover budget draws a second."""
    monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "0")
    assert max(_prefills_per_step(_scheduler())) > 1


def test_negative_cap_reads_as_disabled(monkeypatch):
    monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "-1")
    assert max(_prefills_per_step(_scheduler())) > 1


def test_higher_cap_admits_more_prefills(monkeypatch):
    monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "2")
    assert max(_prefills_per_step(_scheduler())) == 2


def test_pooling_runner_is_exempt():
    """Pooling never decodes, so serialising its prefills only gives up batching."""
    scheduler = _scheduler(runner_type="pooling")
    assert scheduler.max_num_partial_prefills == 0
    assert max(_prefills_per_step(scheduler)) > 1


def test_cap_never_admits_beyond_max_num_seqs():
    scheduler = _scheduler(max_num_seqs=2)
    _prefills_per_step(scheduler)
    assert len(scheduler.running) <= 2


@pytest.mark.parametrize("num_streaming", [0, 2])
def test_streaming_slots_counted_as_occupancy(monkeypatch, num_streaming):
    """Paused streaming sessions hold a runner slot without sitting in `running`.

    Upstream's gate adds them to `len(running)`, so the cap has to as well; they are
    awkward to reach through the real loop, so read the ceiling the loop would see.
    """
    seen = []
    monkeypatch.setattr(
        Scheduler, "schedule", lambda self, *a, **kw: seen.append(self.max_num_active_reqs)
    )
    scheduler = _scheduler(max_num_seqs=8)
    scheduler.running = [
        SimpleNamespace(num_computed_tokens=PROMPT_LEN, num_prompt_tokens=PROMPT_LEN)
    ]
    scheduler.num_waiting_for_streaming_input = num_streaming
    scheduler.schedule()
    # One decode in `running`, the streaming slots, and one free prefill slot.
    assert seen == [2 + num_streaming]


# --- Sliced vision passes ---

CACHE_SIZE = 1000
IMAGE_TOKENS = 40


def _sliced_scheduler(max_num_seqs: int = 4, steps: int = 3) -> TorchSpyreScheduler:
    """Two blocks a step; every pass but the last slice runs ahead of its prefill."""
    sched = _scheduler(
        max_num_seqs=max_num_seqs,
        ec_transfer_config=ECTransferConfig(
            ec_connector=SpyreVisionSliceConnector.__name__,
            ec_connector_module_path=SpyreVisionSliceConnector.__module__,
            ec_role="ec_consumer",
            ec_connector_extra_config={
                "blocks_per_step": 2,
                "steps": steps,
                "modalities": ["image"],
            },
        ),
    )
    sched.encoder_cache_manager = EncoderCacheManager(cache_size=CACHE_SIZE)
    sched.max_num_encoder_input_tokens = CACHE_SIZE
    return sched


@pytest.fixture
def scheduler():
    return _sliced_scheduler()


def _text(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )


def _image(
    request_id: str, *mm_hashes: str, prompt_len: int = PROMPT_LEN, offset: int = 10
) -> Request:
    """One image per hash (``h`` if none), each ``IMAGE_TOKENS`` long, 10 tokens apart."""
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({}),
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(
                    offset=offset + i * (IMAGE_TOKENS + 10), length=IMAGE_TOKENS
                ),
            )
            for i, mm_hash in enumerate(mm_hashes or ("h",))
        ],
    )


def _step(scheduler) -> tuple[SchedulerOutput, VisionSlices]:
    """One schedule() plus the token update_from_output would append."""
    output = scheduler.schedule()
    # schedule() has already advanced num_computed_tokens; a request that has caught up
    # samples a token, which is what keeps it decoding.
    for rid in output.num_scheduled_tokens:
        request = scheduler.requests[rid]
        if request.num_computed_tokens >= request.num_tokens:
            request.append_output_token_ids(0)
    assert isinstance(output.ec_connector_metadata, VisionSlices)
    return output, output.ec_connector_metadata


def _with_running_decode(scheduler) -> None:
    scheduler.add_request(_text("t"))
    _step(scheduler)
    assert [r.request_id for r in scheduler.running] == ["t"]


def test_pass_spreads_over_steps_and_the_prefill_waits_for_it(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))

    out, slices = _step(scheduler)
    assert [s.mm_hash for s in slices.start] == ["h"]
    assert slices.run == {"h": 2}
    assert "img" not in out.num_scheduled_tokens

    out, slices = _step(scheduler)
    assert not slices.start
    assert slices.run == {"h": 2}
    assert "img" not in out.num_scheduled_tokens

    # Upstream allocates the entry as an external load, and the last slice runs in that
    # step: the worker caches before it gathers.
    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "img" in out.num_scheduled_tokens
    assert "img" not in out.scheduled_encoder_inputs

    out, slices = _step(scheduler)
    assert slices == VisionSlices()


def test_with_nothing_running_upstream_encodes_in_one_step(scheduler):
    scheduler.add_request(_image("img"))
    out, slices = _step(scheduler)
    assert slices == VisionSlices()
    assert out.scheduled_encoder_inputs == {"img": [0]}


def test_a_step_with_nothing_else_scheduled_runs_the_rest_of_the_pass():
    scheduler = _sliced_scheduler(steps=4)
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)

    scheduler.finish_requests("t", RequestStatus.FINISHED_ABORTED)
    out, slices = _step(scheduler)
    assert slices.run == {"h": 4}
    assert not out.num_scheduled_tokens

    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "img" in out.num_scheduled_tokens


def test_abort_mid_pass_cancels_it(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)

    scheduler.finish_requests("img", RequestStatus.FINISHED_ABORTED)
    out, slices = _step(scheduler)

    assert slices.cancel == ["h"]
    assert not slices.run
    # The entry is allocated only once the pass is done, so there is nothing to free.
    manager = scheduler.encoder_cache_manager
    assert "h" not in manager.cached
    assert manager.num_free_slots == CACHE_SIZE


def test_the_next_request_with_an_aborted_image_gets_a_pass_of_its_own(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    _step(scheduler)
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_image("b"))

    out, slices = _step(scheduler)
    assert slices.cancel == ["h"]
    assert [s.mm_hash for s in slices.start] == ["h"]
    assert "b" not in out.num_scheduled_tokens


def test_a_pass_outlives_its_first_request_while_another_waits_on_it(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    scheduler.add_request(_image("b"))
    _step(scheduler)

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    out, slices = _step(scheduler)
    assert not slices.cancel
    assert slices.run == {"h": 2}


def test_the_same_image_waits_for_the_running_pass(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    scheduler.add_request(_image("b"))

    starts = []
    scheduled = set()
    for _ in range(6):
        out, slices = _step(scheduler)
        starts += slices.start
        if slices.run.get("h") == 2:
            assert not {"a", "b"} & set(out.num_scheduled_tokens)
        scheduled |= set(out.num_scheduled_tokens)
        assert "b" not in out.scheduled_encoder_inputs
    assert len(starts) == 1
    assert {"a", "b"} <= scheduled


def test_passes_run_one_at_a_time_in_waiting_order(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a", "ha"))
    scheduler.add_request(_image("b", "hb"))

    runs = [_step(scheduler)[1].run for _ in range(6)]
    assert runs == [{"ha": 2}, {"ha": 2}, {"ha": None}, {"hb": 2}, {"hb": 2}, {"hb": None}]


def test_a_deferred_request_keeps_its_place_in_the_waiting_queue(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img", "h"))
    scheduler.add_request(_image("img2", "h2"))

    out, slices = _step(scheduler)
    assert slices.run == {"h": 2}
    assert not {"img", "img2"} & set(out.num_scheduled_tokens)
    assert [r.request_id for r in scheduler.waiting] == ["img", "img2"]


def test_images_that_never_fit_the_cache_together_go_to_upstream(scheduler):
    """Deferred, they would wait forever for space, and so would every image request behind
    them."""
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=60)
    scheduler.max_num_encoder_input_tokens = 60
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img", "ha", "hb"))

    out, slices = _step(scheduler)
    assert slices == VisionSlices()
    assert out.scheduled_encoder_inputs == {"img": [0]}


def test_a_request_holding_kv_blocks_waits_for_the_pass_of_its_image(scheduler):
    """Stands in for a streaming session whose next chunk brings an image already in a pass."""
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    _step(scheduler)
    session = _image("session")
    scheduler.requests[session.request_id] = session
    scheduler.kv_holding_waiting.add_request(session)

    out, slices = _step(scheduler)
    assert slices.run == {"h": 2}
    assert "session" not in out.num_scheduled_tokens

    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "session" in out.num_scheduled_tokens
    assert "session" not in out.scheduled_encoder_inputs


def test_a_running_prefill_with_an_image_mid_pass_ahead_waits_for_the_pass(monkeypatch):
    """Upstream defers it from its running loop; without that it would encode the image in
    one step once its prefill got there."""
    monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "2")
    scheduler = _sliced_scheduler()
    # Short chunks leave budget for the image request; the image sits far into the prompt.
    scheduler.scheduler_config.long_prefill_token_threshold = 100
    scheduler.add_request(_image("long", prompt_len=1000, offset=700))
    _step(scheduler)
    scheduler.add_request(_image("img"))

    out, slices = _step(scheduler)
    assert [s.mm_hash for s in slices.start] == ["h"]
    assert "long" in out.num_scheduled_tokens

    out, slices = _step(scheduler)
    assert "long" not in out.num_scheduled_tokens

    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "long" in out.num_scheduled_tokens


def test_reset_encoder_cache_cancels_the_pass_and_starts_it_again(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)

    scheduler.reset_encoder_cache()
    out, slices = _step(scheduler)
    assert slices.cancel == ["h"]
    assert [s.mm_hash for s in slices.start] == ["h"]
    assert slices.run == {"h": 2}
    assert "img" not in out.num_scheduled_tokens


def test_text_requests_are_not_held_behind_a_pass(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)
    scheduler.add_request(_text("t2"))
    out, _ = _step(scheduler)
    assert "t2" in out.num_scheduled_tokens


def test_off_by_default():
    assert _scheduler().ec_connector is None


class _SlicedVisionModel(SpyreVisionEncoderMixin):
    pass


@pytest.fixture
def slicing_config(monkeypatch):
    """A config ``vision_slicing`` accepts: K=6 over a 24-block compiled tower."""
    monkeypatch.setattr(envs, "SPYRE_VISION_BLOCKS_PER_STEP", 6, raising=False)
    monkeypatch.setattr(
        "spyre_inference.models.spyre_model_cls", lambda architecture: _SlicedVisionModel
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            is_multimodal_model=True,
            architecture="SlicedVisionModel",
            multimodal_config=SimpleNamespace(mm_processor_cache_type="lru"),
            hf_config=SimpleNamespace(vision_config=SimpleNamespace(num_hidden_layers=24)),
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        lora_config=None,
        ec_transfer_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
    )


def test_vision_slicing_spreads_the_tower_over_ceil_depth_by_k_steps(slicing_config):
    assert vision_slicing(slicing_config) == VisionSlicing(6, 4, ("image",))


@pytest.mark.parametrize(
    ("feature", "turn_on"),
    [
        ("async scheduling", lambda c: setattr(c.scheduler_config, "async_scheduling", True)),
        (
            "mm_processor_cache_type='shm'",
            lambda c: setattr(c.model_config.multimodal_config, "mm_processor_cache_type", "shm"),
        ),
        ("LoRA", lambda c: setattr(c, "lora_config", object())),
        (
            "another encoder cache connector",
            lambda c: setattr(c, "ec_transfer_config", SimpleNamespace(ec_connector="Other")),
        ),
        (
            "custom encoder cache manager",
            lambda c: setattr(c.ec_manager_config, "get_encoder_cache_manager_obj", lambda: object),
        ),
    ],
)
def test_vision_slicing_refuses_features_that_break_the_plan(slicing_config, feature, turn_on):
    turn_on(slicing_config)
    with pytest.raises(ValueError, match=feature):
        vision_slicing(slicing_config)


def test_vision_slicing_accepts_its_own_connector(slicing_config):
    """The platform's hook can run again on a config it already installed it in."""
    slicing_config.ec_transfer_config = SimpleNamespace(
        ec_connector=SpyreVisionSliceConnector.__name__
    )
    assert vision_slicing(slicing_config) == VisionSlicing(6, 4, ("image",))


def test_vision_slicing_needs_the_tower_depth(slicing_config):
    slicing_config.model_config.hf_config.vision_config = SimpleNamespace()
    with pytest.raises(ValueError, match="num_hidden_layers"):
        vision_slicing(slicing_config)


def test_vision_slicing_is_off_for_an_eager_tower_or_a_one_step_k(slicing_config, monkeypatch):
    slicing_config.compilation_config.mode = CompilationMode.NONE
    assert vision_slicing(slicing_config) is None
    slicing_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    monkeypatch.setattr(envs, "SPYRE_VISION_BLOCKS_PER_STEP", 24, raising=False)
    assert vision_slicing(slicing_config) is None
