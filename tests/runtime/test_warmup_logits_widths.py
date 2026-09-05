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

"""Warmup projects the lm_head at the sampled-row widths, not the body bucket sizes."""

from __future__ import annotations

import types

import torch

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

HIDDEN = 8
BODY_BUCKETS = [1, 2, 4, 8, 16, 32, 512]
MAX_NUM_REQS = 32


class _Bucketer:
    def __init__(self, bucket_sizes):
        self.bucket_sizes = bucket_sizes
        self.warmed_up = False

    def mark_warmed_up(self):
        self.warmed_up = True


def _runner(bucket_sizes=BODY_BUCKETS, max_num_reqs=MAX_NUM_REQS):
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model_config = types.SimpleNamespace(runner_type="generate")
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=types.SimpleNamespace(
            compile_sizes=list(bucket_sizes),
            inductor_compile_config={},
        ),
    )
    runner.spyre_shape_bucketer = _Bucketer(list(bucket_sizes))
    runner.max_num_reqs = max_num_reqs

    body_rows: list[int] = []
    projected_rows: list[int] = []

    def dummy_run(size, *args, **kwargs):
        body_rows.append(size)
        return None, torch.zeros(size, HIDDEN, dtype=torch.float16)

    def dummy_sampler_run(hidden_states):
        projected_rows.append(hidden_states.shape[0])
        return torch.tensor([])

    runner._dummy_run = dummy_run
    runner._dummy_sampler_run = dummy_sampler_run
    return runner, body_rows, projected_rows


def test_every_body_bucket_is_still_warmed():
    runner, body_rows, _ = _runner()
    runner.warming_up_model()

    assert sorted(body_rows) == BODY_BUCKETS


def test_projection_widths_are_the_row_buckets_not_the_body_buckets():
    runner, _, projected_rows = _runner()
    runner.warming_up_model()

    assert sorted(projected_rows) == [1, 2, 4, 8, 16, 32]


def test_the_prefill_bucket_is_never_projected():
    """512 packed tokens sample at most max_num_reqs rows, so compiling 512 is waste."""
    runner, _, projected_rows = _runner()
    runner.warming_up_model()

    assert max(projected_rows) <= MAX_NUM_REQS
    assert 512 not in projected_rows


def test_no_width_is_projected_twice():
    runner, _, projected_rows = _runner()
    runner.warming_up_model()

    assert len(projected_rows) == len(set(projected_rows))


def test_a_max_num_reqs_below_every_bucket_still_warms_one_width():
    runner, _, projected_rows = _runner(bucket_sizes=[64, 512], max_num_reqs=4)
    runner.warming_up_model()

    assert projected_rows == [4]


def test_warmup_marks_the_bucketer_warmed():
    runner, _, _ = _runner()
    runner.warming_up_model()

    assert runner.spyre_shape_bucketer.warmed_up
