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

The vision-pass tests (SPYRE_VISION_BLOCKS_PER_STEP) drive the same loop, with the slicing
gate patched on for a text model and an encoder cache installed by hand, and read the plan
each step attaches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from spyre_inference.v1.core import scheduler as scheduler_module
from spyre_inference.v1.core.scheduler import TorchSpyreScheduler
from spyre_inference.v1.core.vision_slices import VISION_SLICES_ATTR, VisionSlicing

# A 200-token prompt against a 256-token budget leaves 56 tokens over, which is
# exactly the leftover upstream tops the batch up with.
PROMPT_LEN = 200
MAX_NUM_BATCHED_TOKENS = 256
BLOCK_SIZE = 64
NUM_BLOCKS = 512


def _scheduler(runner_type: str = "generate", max_num_seqs: int = 4) -> TorchSpyreScheduler:
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


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "vision_slicing",
        lambda vllm_config: VisionSlicing(blocks_per_step=2, steps=3, modalities=("image",)),
    )
    sched = _scheduler()
    sched.encoder_cache_manager = EncoderCacheManager(cache_size=CACHE_SIZE)
    sched.max_num_encoder_input_tokens = CACHE_SIZE
    return sched


def _text(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )


def _image(request_id: str, mm_hash: str = "h") -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({}),
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=10, length=IMAGE_TOKENS),
            )
        ],
    )


def _step(scheduler):
    """One schedule() plus the bookkeeping execute_model + update_from_output would do."""
    output = scheduler.schedule()
    for rid, num_tokens in output.num_scheduled_tokens.items():
        scheduler.requests[rid].num_computed_tokens += num_tokens
    return output, getattr(output, VISION_SLICES_ATTR, None)


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

    # The last slice and the prefill share a step: the runner caches before it gathers.
    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "img" in out.num_scheduled_tokens
    assert "img" not in out.scheduled_encoder_inputs

    out, slices = _step(scheduler)
    assert slices is None


def test_with_nothing_running_upstream_encodes_in_one_step(scheduler):
    scheduler.add_request(_image("img"))
    out, slices = _step(scheduler)
    assert slices is None
    assert out.scheduled_encoder_inputs == {"img": [0]}


def test_pass_finishes_at_once_when_the_running_requests_drain(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)

    scheduler.finish_requests("t", RequestStatus.FINISHED_ABORTED)
    out, slices = _step(scheduler)
    assert slices.run == {"h": None}
    assert "img" in out.num_scheduled_tokens


def test_abort_mid_pass_cancels_it_and_frees_its_cache_entry(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)
    manager = scheduler.encoder_cache_manager
    assert "h" in manager.cached

    scheduler.finish_requests("img", RequestStatus.FINISHED_ABORTED)
    out, slices = _step(scheduler)

    assert slices.cancel == ["h"]
    assert not slices.run
    assert "h" not in manager.cached
    assert manager.num_free_slots == CACHE_SIZE
    assert "h" in out.free_encoder_mm_hashes


def test_an_aborted_pass_never_reads_as_cached_to_the_same_image(scheduler):
    """The next request with that image gets a pass of its own, not a missing embedding."""
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    _step(scheduler)
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_image("b"))

    out, slices = _step(scheduler)
    assert slices.cancel == ["h"]
    assert [s.mm_hash for s in slices.start] == ["h"]
    assert "b" not in out.num_scheduled_tokens


def test_the_same_image_waits_for_the_running_pass(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a"))
    scheduler.add_request(_image("b"))

    starts = []
    scheduled = set()
    for _ in range(6):
        out, slices = _step(scheduler)
        if slices is not None:
            starts += slices.start
        if slices is not None and slices.run.get("h") == 2:
            assert not {"a", "b"} & set(out.num_scheduled_tokens)
        scheduled |= set(out.num_scheduled_tokens)
        assert "b" not in out.scheduled_encoder_inputs
    assert len(starts) == 1
    assert {"a", "b"} <= scheduled


def test_passes_run_one_at_a_time_in_waiting_order(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("a", "ha"))
    scheduler.add_request(_image("b", "hb"))

    plans = [_step(scheduler)[1] for _ in range(6)]
    runs = [p.run for p in plans if p is not None]
    assert runs == [{"ha": 2}, {"ha": 2}, {"ha": None}, {"hb": 2}, {"hb": 2}, {"hb": None}]


def test_text_requests_are_not_held_behind_a_pass(scheduler):
    _with_running_decode(scheduler)
    scheduler.add_request(_image("img"))
    _step(scheduler)
    scheduler.add_request(_text("t2"))
    out, _ = _step(scheduler)
    assert "t2" in out.num_scheduled_tokens


def test_off_by_default():
    sched = _scheduler()
    assert sched._vision_slicing is None
