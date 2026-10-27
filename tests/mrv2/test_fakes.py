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

"""Smoke tests for fakes.py: each factory builds, and the values it
produces are internally consistent (e.g. the fake InputBatch's per-request
token counts actually sum to num_tokens). fakes.py is frozen after #1236,
so nothing else in this PR exercises it.
"""

import torch

from .fakes import (
    FakeModel,
    make_fake_attn_groups,
    make_fake_input_batch,
    make_fake_kv_cache_config,
    make_fake_scheduler_output,
)


def test_fake_model_runs_forward() -> None:
    model = FakeModel(hidden_size=8, num_blocks=2)
    out = model(torch.zeros(3, 8))
    assert out.shape == (3, 8)


def test_fake_scheduler_output_totals_match() -> None:
    output = make_fake_scheduler_output({"req_0": 3, "req_1": 5})
    assert output.total_num_scheduled_tokens == 8


def test_fake_kv_cache_config_has_one_group_per_layer() -> None:
    config = make_fake_kv_cache_config(num_layers=2, num_blocks=4)
    assert len(config.kv_cache_groups) == 2
    assert len(config.kv_cache_tensors) == 2


def test_fake_attn_groups_one_per_window() -> None:
    groups = make_fake_attn_groups(num_windows=2, layers_per_window=3)
    assert len(groups) == 2
    assert all(len(g.layer_names) == 3 for g in groups)


def test_fake_input_batch_token_counts_sum_to_num_tokens() -> None:
    batch = make_fake_input_batch(num_reqs=2, num_tokens=5)
    assert batch.num_scheduled_tokens.sum().item() == 5
    assert batch.num_scheduled_tokens.tolist() == [3, 2]
