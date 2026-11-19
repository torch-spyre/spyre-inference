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

"""The pooler row sweep covers the width the poolers round up to, not just the limit.

``SpyreAllPool`` rounds each request's token count up to the declared length ladder
before its gather, and the token head runs at that bucketed width, so both must be
warmed at every ladder entry serving can reach.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead

from spyre_inference.v1.pool import warmup
from spyre_inference.v1.pool.spyre_pooler import SpyreAllPool, SpyreTokenPooler
from spyre_inference.v1.pool.warmup import warm_pooler_row_widths, warm_token_pool_widths

ROWS = 64
HIDDEN = 8


def _swept_widths(monkeypatch, max_num_seqs: int) -> list[int]:
    widths: list[int] = []
    monkeypatch.setattr(
        warmup,
        "select_rows",
        lambda hidden_states, row_indices: widths.append(int(row_indices.numel())),
    )
    # No classifier/embed head, so the real-width sweep (which needs one to warm)
    # stays a no-op and only the plain gather below runs.
    warm_pooler_row_widths(
        torch.nn.Module(), torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [], max_num_seqs
    )
    return widths


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        pytest.param(6, [1, 2, 4, 8], id="six_rounds_up_to_eight"),
        pytest.param(24, [1, 2, 4, 8, 16, 32], id="twenty_four_rounds_up_to_thirty_two"),
        pytest.param(4, [1, 2, 4], id="a_power_of_two_is_unchanged"),
        pytest.param(1, [1], id="one_sequence"),
    ],
)
def test_sweep_reaches_the_rounded_up_width(monkeypatch, max_num_seqs, expected):
    assert _swept_widths(monkeypatch, max_num_seqs) == expected


def test_sweep_never_exceeds_the_available_rows(monkeypatch):
    """A body smaller than the rounded-up width has nothing to gather from."""
    assert max(_swept_widths(monkeypatch, ROWS * 4)) <= ROWS


def _token_pooler(ladder: list[int], head=None) -> SpyreTokenPooler:
    """A ``SpyreTokenPooler`` as ``patch_pooler_for_spyre`` leaves it: bucketed
    ``SpyreAllPool`` gather with the trim deferred past the head."""
    return SpyreTokenPooler(SpyreAllPool(False, defer_trim=True, len_ladder=ladder), head)


def _recording_select_rows(monkeypatch, lengths: list[int]) -> None:
    monkeypatch.setattr(
        warmup,
        "select_rows",
        lambda hidden_states, row_indices: lengths.append(int(row_indices.numel())),
    )


def test_token_pool_sweep_gathers_every_declared_length(monkeypatch):
    """Each ladder entry is gathered once, sorted, deduplicated; an entry past the
    body has no rows to gather and is skipped."""
    lengths: list[int] = []
    _recording_select_rows(monkeypatch, lengths)
    warm_token_pool_widths(
        _token_pooler([128, 8, 2, 2, 3]), torch.zeros(ROWS, HIDDEN, dtype=torch.float16)
    )
    assert lengths == [2, 3, 8]


def test_token_pool_sweep_gathers_nothing_without_a_ladder(monkeypatch):
    """No declared lengths means plain stick alignment, which this sweep cannot
    enumerate; gathering anything here would miss the real widths anyway."""
    monkeypatch.setattr(warmup, "select_rows", lambda *args: pytest.fail("gathered with no ladder"))
    warm_token_pool_widths(_token_pooler([]), torch.zeros(ROWS, HIDDEN, dtype=torch.float16))


def test_token_pool_sweep_runs_the_head_at_every_declared_length():
    """The head sees the bucketed width, not the request's real length, so it must
    be warmed at each ladder entry too."""
    seen: list[tuple[int, ...]] = []

    def activation(logits):
        seen.append(tuple(logits.shape))
        return logits

    head = TokenClassifierPoolerHead(
        classifier=nn.Linear(HIDDEN, 4, dtype=torch.float16), activation=activation
    )
    warm_token_pool_widths(
        _token_pooler([2, 4], head), torch.randn(ROWS, HIDDEN, dtype=torch.float16)
    )
    assert seen == [(2, 4), (4, 4)]


def test_the_token_pooler_behind_the_dispatch_dict_is_swept(monkeypatch):
    """``DispatchPooler.poolers_by_task`` is a plain dict ``modules()`` never
    reaches, so the sweep has to walk it to find the token pooler."""
    pooler = DispatchPooler({"token_embed": _token_pooler([2, 8])})
    assert not any(isinstance(m, SpyreTokenPooler) for m in pooler.modules())
    lengths: list[int] = []
    _recording_select_rows(monkeypatch, lengths)
    warm_pooler_row_widths(pooler, torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [], 1)
    assert lengths == [1, 2, 8]  # the plain one-row sweep, then the ladder


@pytest.mark.skipif(not spyre_available(), reason="needs a Spyre card")
def test_token_pool_sweep_gathers_real_rows_on_spyre(monkeypatch):
    """The sweep's gathers run for real on the card and pick the rows the bucketed
    forward will ask for: the first ``length`` rows of the body."""
    hidden = (torch.arange(ROWS * 64) % 512).reshape(ROWS, 64).to(torch.float16)
    real_select_rows = warmup.select_rows
    chunks: list[torch.Tensor] = []

    def recording_select_rows(hidden_states, row_indices):
        chunks.append(real_select_rows(hidden_states, row_indices))

    monkeypatch.setattr(warmup, "select_rows", recording_select_rows)
    warm_token_pool_widths(_token_pooler([1, 4, 64]), hidden.to("spyre"))
    assert [chunk.shape[0] for chunk in chunks] == [1, 4, 64]
    for chunk in chunks:
        assert chunk.device.type == "spyre"
        torch.testing.assert_close(chunk.cpu(), hidden[: chunk.shape[0]], atol=0, rtol=0)
