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

"""Pooling warmup runs the pooler at every request count, not only the widest.

A seqwise pooler gathers one row per request, and both the index length and the source
row count are compiled shape axes on Spyre, so every request count is a shape.
``_dummy_pooler_run`` derives its count from the token count and only reaches
``max_num_seqs``. The pooling twin of ``test_warmup_logits_widths.py``.
"""

from __future__ import annotations

import types
from typing import cast

import torch

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

TASKS = ["embed", "token_embed"]


def _runner(max_num_reqs: int = 8, tasks: list[str] | None = None, on_spyre: bool = True):
    """Stub self. The method's whole surface is these five attributes."""
    traced: list[tuple[int, str, int | None]] = []
    runner = types.SimpleNamespace(
        _pooling_on_spyre=on_spyre,
        max_num_reqs=max_num_reqs,
        _pooler_row_widths_done=set(),
        get_supported_pooling_tasks=lambda: list(TASKS if tasks is None else tasks),
        _dummy_pooler_run_task=lambda hidden, task, num_reqs=None: traced.append(
            (hidden.shape[0], task, num_reqs)
        ),
    )
    return runner, traced


def _sweep(runner, rows: int) -> None:
    TorchSpyreModelRunner._warmup_pooler_row_widths(
        cast(TorchSpyreModelRunner, runner), torch.zeros(rows, 4, dtype=torch.float16)
    )


def test_every_request_count_is_traced_for_each_task():
    runner, traced = _runner(max_num_reqs=8)
    _sweep(runner, 512)

    for task in TASKS:
        counts = sorted(num_reqs for _rows, seen, num_reqs in traced if seen == task)
        assert counts == [1, 2, 3, 4, 5, 6, 7, 8]


def test_widest_first():
    """Inductor's caches warm on the widest shape, as with the body buckets."""
    runner, traced = _runner(max_num_reqs=4)
    _sweep(runner, 512)

    assert [num_reqs for _rows, task, num_reqs in traced if task == "embed"] == [4, 3, 2, 1]


def test_a_row_count_is_swept_once():
    """Cells padding to the same body bucket must not re-pay the sweep."""
    runner, traced = _runner(max_num_reqs=4)
    _sweep(runner, 256)
    first = len(traced)
    _sweep(runner, 256)
    assert len(traced) == first

    _sweep(runner, 512)
    assert len(traced) == 2 * first, "a different body bucket is a different shape"


def test_the_request_count_cannot_exceed_the_rows_available():
    """``num_reqs > rows`` would make ``num_tokens // num_reqs`` zero."""
    runner, traced = _runner(max_num_reqs=8)
    _sweep(runner, 3)

    assert max(num_reqs for _rows, _task, num_reqs in traced) == 3


def test_a_cpu_pooler_is_not_swept():
    """An unpatched pooler runs on the host, where no shape compiles."""
    runner, traced = _runner(on_spyre=False)
    _sweep(runner, 512)

    assert traced == []
