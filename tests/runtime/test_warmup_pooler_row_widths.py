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

"""The pooler row sweep covers the width the poolers round up to, not just the limit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler

from spyre_inference.v1.pool import warmup
from spyre_inference.v1.pool.spyre_pooler import patch_pooler_for_spyre
from spyre_inference.v1.pool.warmup import (
    warm_pooler_row_widths,
    warm_token_pool_widths,
)

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


def _token_pooler(ladder: list[int], head) -> TokenPooler:
    pooler = TokenPooler(pooling=AllPool(), head=head)
    num_patched, unsupported = patch_pooler_for_spyre(pooler, ladder)
    assert (num_patched, unsupported) == (1, [])
    return pooler


def _recording_head(calls: list[tuple[int, ...]]) -> SimpleNamespace:
    def _record(x):
        calls.append(tuple(x.shape))
        return x

    return SimpleNamespace(classifier=_record, projector=_record, activation=_record)


def test_token_sweep_reaches_every_ladder_entry_the_body_holds(default_vllm_config):
    """The first request landing on an unseen ladder length must not compile mid-serve."""
    calls: list[tuple[int, ...]] = []
    pooler = _token_pooler([4, 128, 2], _recording_head(calls))

    warm_token_pool_widths(pooler, torch.zeros(ROWS, HIDDEN, dtype=torch.float16))

    # Sorted sweep, 128 left out: it exceeds the 64-row body. Each length warms
    # the classifier, the projector and the activation at the gathered width.
    assert calls == [(2, HIDDEN)] * 3 + [(4, HIDDEN)] * 3


def test_token_sweep_gathers_without_a_head(default_vllm_config, monkeypatch):
    widths: list[int] = []

    def _record(hidden_states, row_indices):
        widths.append(int(row_indices.numel()))
        return hidden_states

    monkeypatch.setattr(warmup, "select_rows", _record)
    pooler = _token_pooler([2, 4], None)

    warm_token_pool_widths(pooler, torch.zeros(ROWS, HIDDEN, dtype=torch.float16))

    assert widths == [2, 4]


def test_token_sweep_skips_a_pooler_without_spyre_all_pool(default_vllm_config, monkeypatch):
    widths: list[int] = []

    def _record(hidden_states, row_indices):
        widths.append(int(row_indices.numel()))
        return hidden_states

    monkeypatch.setattr(warmup, "select_rows", _record)
    pooler = TokenPooler(pooling=lambda hidden_states, metadata: [])

    warm_token_pool_widths(pooler, torch.zeros(ROWS, HIDDEN, dtype=torch.float16))

    assert widths == []
