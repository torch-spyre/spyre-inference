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

"""Upstream's ``hidden_states[logits_indices]`` gathers through ``select_rows`` on a bucket."""

import types

import torch

from spyre_inference.v1.worker import spyre_model_runner as mr

ROWS, HIDDEN = 16, 8


def _hidden(monkeypatch, buckets):
    widths: list[int] = []

    def fake_select_rows(hidden_states, row_indices):
        assert type(hidden_states) is torch.Tensor
        widths.append(int(row_indices.numel()))
        return torch.index_select(hidden_states, 0, row_indices)

    monkeypatch.setattr(mr, "select_rows", fake_select_rows)
    plain = torch.randn(ROWS, HIDDEN, dtype=torch.float16)
    return plain, mr._SpyreHiddenStates.wrap(plain, buckets), widths


def test_index_gathers_the_requested_rows_on_a_warmed_width(monkeypatch):
    plain, hidden, widths = _hidden(monkeypatch, [1, 2, 4, 8])
    idx = torch.tensor([3, 9, 15])

    rows = hidden[idx]

    assert widths == [4]
    assert type(rows) is torch.Tensor
    torch.testing.assert_close(rows, plain[idx])
    assert rows._spyre_padded_rows.shape[0] == 4


def test_an_exact_width_is_returned_unpadded(monkeypatch):
    plain, hidden, widths = _hidden(monkeypatch, [1, 2, 4])
    idx = torch.tensor([0, 5])

    rows = hidden[idx]

    assert widths == [2]
    assert not hasattr(rows, "_spyre_padded_rows")
    torch.testing.assert_close(rows, plain[idx])


def test_slices_and_ops_return_plain_tensors(monkeypatch):
    plain, hidden, widths = _hidden(monkeypatch, [1, 2, 4])

    for out in (hidden[:4], hidden[2:6], hidden + 1, torch.rand_like(hidden)):
        assert type(out) is torch.Tensor
    torch.testing.assert_close(hidden[2:6], plain[2:6])
    assert widths == []


def _runner():
    projected: list[int] = []
    model = types.SimpleNamespace(compute_logits=lambda rows: projected.append(rows.shape[0]))
    return types.SimpleNamespace(model=model), projected


def test_warmup_compiles_every_row_bucket_a_body_bucket_holds(monkeypatch):
    """The lm_head sees the gather's output, so it compiles on the gathered layout."""
    _, hidden, widths = _hidden(monkeypatch, [1, 2, 4, 32])
    runner, projected = _runner()

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(runner, hidden, [1, 2, 4, 32])

    assert widths == [1, 2, 4]
    assert projected == [1, 2, 4]


def test_warmup_skips_hidden_states_left_on_the_host(monkeypatch):
    _, _, widths = _hidden(monkeypatch, [1, 2])
    runner, projected = _runner()

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(
        runner, torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [1, 2]
    )

    assert widths == []
    assert projected == []
