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

"""Sampled hidden-state rows are gathered on Spyre before output D2H."""

import torch
import torch.nn as nn

from spyre_inference.v1.worker import spyre_model_runner as mr
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

HIDDEN = 8


class _Body(nn.Module):
    def forward(self, input_ids=None, **kwargs):
        rows = input_ids.shape[0]
        return torch.arange(rows, dtype=torch.float16).unsqueeze(1).expand(rows, HIDDEN)


def _wrapper(monkeypatch, *, buckets, copied):
    def fake_convert(t, device=None, dtype=None):
        if device is not None and str(device) == "cpu" and t.dim() == 2:
            copied.append(t.shape[0])
        return t

    monkeypatch.setattr(mr, "convert", fake_convert)
    model = _Body()
    return _SpyreModelWrapper(
        model,
        torch.device("cpu"),
        logits_row_buckets=buckets,
    )


def _run(wrapper, num_tokens, rows=None):
    ids = torch.zeros(num_tokens, dtype=torch.int64)
    if rows is not None:
        object.__setattr__(wrapper, "_sample_rows", torch.tensor(rows, dtype=torch.int64))
    return wrapper(input_ids=ids)


def test_copies_only_the_sampled_rows(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    out = _run(wrapper, 512, rows=[511])

    assert copied == [1]
    assert out.shape == (1, HIDDEN)
    torch.testing.assert_close(out[0], torch.full((HIDDEN,), 511.0, dtype=torch.float16))


def test_sampled_rows_are_returned_in_order(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 2, 4], copied=copied)

    out = _run(wrapper, 512, rows=[127, 300, 511])

    assert copied == [4]
    assert out.shape == (3, HIDDEN)
    for result, row in zip(out, (127, 300, 511)):
        torch.testing.assert_close(result, torch.full((HIDDEN,), float(row), dtype=torch.float16))


def test_unarmed_call_copies_the_whole_body(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    _run(wrapper, 512)

    assert copied == [512]


def test_arming_does_not_leak_into_the_next_call(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 4], copied=copied)

    _run(wrapper, 512, rows=[511])
    _run(wrapper, 512)

    assert copied == [1, 512]


def test_full_copy_when_the_padded_gather_is_as_wide_as_the_body(monkeypatch):
    copied: list[int] = []
    wrapper = _wrapper(monkeypatch, buckets=[1, 8], copied=copied)

    _run(wrapper, 8, rows=list(range(8)))

    assert copied == [8]
