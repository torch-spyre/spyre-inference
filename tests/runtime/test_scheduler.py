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

"""Tests for TorchSpyreScheduler's partial-prefill cap.

The cap lowers ``max_num_active_reqs`` around the upstream waiting loop, so these
drive that real loop and count the prefills it admits per step. Stubbing ``schedule``
would leave the cap free to degrade into a no-op under an upstream change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from spyre_inference.v1.core.scheduler import TorchSpyreScheduler

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


def _engine_request(*items: tuple[str, int]) -> EngineCoreRequest:
    """A request carrying one multimodal item per (modality, num_embeds) pair."""
    mm_features = [
        MultiModalFeatureSpec(
            data=None,
            modality=modality,
            identifier=f"{modality}-{i}",
            mm_position=PlaceholderRange(offset=0, length=tokens),
        )
        for i, (modality, tokens) in enumerate(items)
    ]
    return EngineCoreRequest(
        request_id="r0",
        prompt_token_ids=list(range(PROMPT_LEN)),
        mm_features=mm_features or None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def gemma4_scheduler(monkeypatch):
    """gemma-4 as served on Spyre: video capped at 0 by the prelaunch override, images
    up to the 280 tokens the encoder compiled for."""
    from spyre_inference.platform import TorchSpyrePlatform

    scheduler = _scheduler()
    # After _scheduler(): building its VllmConfig resets the limits for the text model.
    monkeypatch.setattr(TorchSpyrePlatform, "_vision_token_limits", {"image": 280, "video": 280})
    object.__setattr__(
        scheduler.vllm_config.model_config,
        "multimodal_config",
        MultiModalConfig(limit_per_prompt={"video": 0}),
    )
    return scheduler


@pytest.mark.parametrize("items", [(), (("image", 280),), (("image", 64), ("image", 256))])
def test_validate_request_accepts_what_the_worker_runs(gemma4_scheduler, items):
    gemma4_scheduler.validate_request(_engine_request(*items))


def test_validate_request_rejects_an_image_above_the_compiled_budget(gemma4_scheduler):
    with pytest.raises(VLLMValidationError, match="encoder_cudagraph_token_budgets") as err:
        gemma4_scheduler.validate_request(_engine_request(("image", 64), ("image", 281)))
    assert err.value.value == 281


def test_validate_request_rejects_a_disabled_modality(gemma4_scheduler):
    with pytest.raises(VLLMValidationError, match="does not accept video input"):
        gemma4_scheduler.validate_request(_engine_request(("video", 16)))


def test_validate_request_rejects_any_item_on_a_text_only_model():
    with pytest.raises(VLLMValidationError, match="does not accept image input"):
        _scheduler().validate_request(_engine_request(("image", 16)))


def test_mm_embeds_pass_a_zero_limit(gemma4_scheduler):
    gemma4_scheduler.vllm_config.model_config.multimodal_config.enable_mm_embeds = True
    gemma4_scheduler.validate_request(_engine_request(("video", 16)))


def test_engine_core_validates_before_admission(gemma4_scheduler):
    """The check runs in preprocess_add_request, whose VLLMValidationError EngineCoreProc
    turns into a per-request FinishReason.ERROR instead of a fatal engine error."""
    engine_core = SimpleNamespace(
        scheduler=gemma4_scheduler,
        mm_receiver_cache=None,
        request_block_hasher=None,
    )
    with pytest.raises(VLLMValidationError, match="video"):
        EngineCore.preprocess_add_request(engine_core, _engine_request(("video", 16)))

    request, _ = EngineCore.preprocess_add_request(engine_core, _engine_request(("image", 256)))
    assert isinstance(request, Request)
    assert request.request_id == "r0"
