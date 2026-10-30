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

"""Tests for TorchSpyreScheduler's partial-prefill cap and draft padding.

The cap lowers ``max_num_active_reqs`` around the upstream waiting loop, so these
drive that real loop and count the prefills it admits per step. Stubbing ``schedule``
would leave the cap free to degrade into a no-op under an upstream change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds
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


NUM_SPEC_TOKENS = 3


def _decoding(scheduler: TorchSpyreScheduler, rid: str = "d0", prompt_len: int = 16) -> Request:
    """A request past its prefill, holding one sampled token and no drafts yet."""
    request = Request(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
    )
    scheduler.add_request(request)
    scheduler.schedule()  # the whole prompt in one chunk; advances num_computed_tokens
    assert not request.is_prefill_chunk
    request.append_output_token_ids(5)
    return request


def _speculative(scheduler: TorchSpyreScheduler, pad: bool = True) -> TorchSpyreScheduler:
    # The pad gate itself is batched_decode_query_len's, covered in the bucketer tests.
    scheduler.num_spec_tokens = NUM_SPEC_TOKENS
    scheduler._pad_drafts = pad
    return scheduler


@pytest.mark.parametrize(
    "drafts,expected",
    [
        ([], [-1, -1, -1]),
        ([7], [7, -1, -1]),
        ([7, 8, 9], [7, 8, 9]),
    ],
    ids=["none", "short", "full"],
)
def test_drafts_padded_to_k(drafts, expected):
    scheduler = _speculative(_scheduler())
    request = _decoding(scheduler)
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [list(drafts)]))
    assert request.spec_token_ids == expected


def test_drafts_left_alone_without_batched_spec_decode():
    scheduler = _speculative(_scheduler(), pad=False)
    request = _decoding(scheduler)
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[7]]))
    assert request.spec_token_ids == [7]


def test_prefill_chunk_gets_no_padding():
    """Upstream drops drafts for a prefill chunk; padding must not bring them back."""
    scheduler = _speculative(_scheduler())
    request = Request(
        request_id="p0",
        prompt_token_ids=list(range(PROMPT_LEN + MAX_NUM_BATCHED_TOKENS)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    scheduler.add_request(request)
    scheduler.schedule()
    assert request.is_prefill_chunk
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[7]]))
    assert request.spec_token_ids == []


def test_padded_verification_is_scheduled_whole_and_kept_out_of_stats():
    scheduler = _speculative(_scheduler())
    request = _decoding(scheduler)
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[7]]))

    output = scheduler.schedule()

    rid = request.request_id
    assert output.num_scheduled_tokens[rid] == 1 + NUM_SPEC_TOKENS
    assert output.scheduled_spec_decode_tokens[rid] == [7, -1, -1]
    assert output.num_invalid_spec_tokens == {rid: 2}


def test_unpadded_verification_reports_no_invalid_drafts():
    scheduler = _speculative(_scheduler())
    request = _decoding(scheduler)
    scheduler.update_draft_token_ids(DraftTokenIds([request.request_id], [[7, 8, 9]]))
    assert not scheduler.schedule().num_invalid_spec_tokens


def test_padding_only_step_is_not_counted_as_a_draft():
    scheduler = _speculative(_scheduler())
    scheduler.log_stats = True
    stats = scheduler.make_spec_decoding_stats(None, 3, 2, None, "real")
    stats = scheduler.make_spec_decoding_stats(stats, 3, 0, {"pad": 3}, "pad")
    stats = scheduler.make_spec_decoding_stats(stats, 3, 1, {"part": 2}, "part")
    assert stats.num_drafts == 2
    assert stats.num_draft_tokens == 3 + 1
    assert stats.num_accepted_tokens == 2 + 1
