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

"""Token-level pooling: ``SpyreAllPool`` gathers ladder-bucketed rows, ``SpyreTokenPooler`` trims.

The bucketed gather keeps ``index_select``'s shape off the request's real token count,
clamping the pad rows onto the last real one; the trim then drops them after the head,
at the D2H every token-pooling output makes anyway.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates

from spyre_inference.v1.pool.spyre_pooler import (
    SpyreAllPool,
    SpyreTokenPooler,
    patch_pooler_for_spyre,
)

HIDDEN = 8
LADDER = [2, 4, 8]


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _metadata(lens: list[int]) -> PoolingMetadata:
    lens_t = torch.tensor(lens, dtype=torch.int64)
    ends = torch.cumsum(lens_t, dim=0)
    cursor = PoolingCursor(
        first_token_indices_gpu=ends - lens_t,
        last_token_indices_gpu=ends - 1,
        prompt_lens_cpu=lens_t,
        seq_lens_cpu=lens_t,
        num_scheduled_tokens_cpu=lens_t,
        partial_prefill=False,
        finished_mask=[True] * len(lens),
    )
    return PoolingMetadata(
        prompt_lens=lens_t,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="embed") for _ in lens],
        pooling_states=[PoolingStates() for _ in lens],
        pooling_cursor=cursor,
    )


class _RecordingHead:
    """Stand-in for a token head: records the chunks it was handed, returns them."""

    def __init__(self):
        self.chunks: list[torch.Tensor] = []

    def __call__(self, pooled, pooling_metadata):
        self.chunks = list(pooled)
        return pooled


def _token_pooler(ladder: list[int] = LADDER, head=None) -> TokenPooler:
    # The platform runs pooling models without chunked prefill; the default test
    # config leaves it on, which the token pooler rejects.
    pooling = AllPool()
    pooling.enable_chunked_prefill = False
    pooler = TokenPooler(pooling=pooling, head=head)
    num_patched, unsupported = patch_pooler_for_spyre(pooler, ladder)
    assert (num_patched, unsupported) == (1, [])
    return pooler


def test_patch_swaps_the_pooler_and_its_all_pool_together(default_vllm_config):
    pooler = TokenPooler(pooling=AllPool())
    chunked = pooler.pooling.enable_chunked_prefill
    num_patched, unsupported = patch_pooler_for_spyre(pooler, LADDER)
    assert (num_patched, unsupported) == (1, [])
    assert type(pooler) is SpyreTokenPooler
    assert type(pooler.pooling) is SpyreAllPool
    # Bucketing the gather is only safe when something trims afterwards.
    assert pooler.pooling.defer_trim
    assert pooler.pooling.len_ladder == LADDER
    assert pooler.pooling.enable_chunked_prefill == chunked


def test_the_head_sees_ladder_bucketed_rows_and_the_output_is_trimmed(default_vllm_config):
    head = _RecordingHead()
    pooler = _token_pooler(head=head)
    lens = [3, 5]
    hidden = torch.arange(sum(lens) * HIDDEN, dtype=torch.float32).reshape(sum(lens), HIDDEN)

    out = pooler(hidden, _metadata(lens))

    # 3 rounds to 4 and 5 to 8: the head runs at the bucketed widths, with the
    # pad rows clamped onto each request's last real row ...
    assert [chunk.shape[0] for chunk in head.chunks] == [4, 8]
    assert torch.equal(head.chunks[0], hidden[[0, 1, 2, 2]])
    assert torch.equal(head.chunks[1], hidden[[3, 4, 5, 6, 7, 7, 7, 7]])
    # ... and the trim returns exactly the real rows.
    assert [chunk.shape[0] for chunk in out] == lens
    assert torch.equal(out[0], hidden[:3])
    assert torch.equal(out[1], hidden[3:8])


def test_a_length_past_every_ladder_entry_stick_aligns(default_vllm_config):
    """No ladder entry fits a 9-token request, so the gather stick-aligns to 64."""
    head = _RecordingHead()
    pooler = _token_pooler(head=head)
    out = pooler(torch.zeros(9, HIDDEN), _metadata([9]))
    assert head.chunks[0].shape[0] == 64
    assert out[0].shape[0] == 9


def test_a_zero_token_request_takes_the_plain_path(default_vllm_config):
    """n == 0 has no last real row to clamp onto, so it yields an empty chunk."""
    head = _RecordingHead()
    pooler = _token_pooler(head=head)
    out = pooler(torch.zeros(3, HIDDEN), _metadata([0, 3]))
    assert head.chunks[0].shape[0] == 0
    assert [chunk.shape[0] for chunk in out] == [0, 3]


def test_chunked_prefill_is_rejected(default_vllm_config):
    pooler = _token_pooler()
    pooler.pooling.enable_chunked_prefill = True
    with pytest.raises(NotImplementedError, match="chunked prefill"):
        pooler(torch.zeros(2, HIDDEN), _metadata([2]))


def test_bucketed_gather_and_trim_on_spyre(default_vllm_config, spyre_device):
    """The gather runs on device and the trim's D2H brings each chunk back whole."""
    pooler = _token_pooler()
    lens = [3, 5]
    hidden = torch.randn(sum(lens), HIDDEN, dtype=torch.float16)

    out = pooler(hidden.to(spyre_device), _metadata(lens))

    assert [chunk.shape[0] for chunk in out] == lens
    for chunk in out:
        assert chunk.device.type == "cpu"
    torch.testing.assert_close(out[0], hidden[:3], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out[1], hidden[3:8], atol=1e-3, rtol=1e-3)
