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

"""Generation D2Hs only the sampled hidden-state rows when that pays off."""

import pytest
import torch
import torch.nn as nn

from spyre_inference.v1.worker import spyre_model_runner as mr
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

# fp16 at 4096 is 8 KiB a row: a 512-row body clears the byte floor, a 4-row one does not.
HIDDEN = 4096


class _Body(nn.Module):
    """Row ``i`` of the output is filled with ``i``."""

    def forward(self, input_ids=None, **kwargs):
        rows = input_ids.shape[0]
        return torch.arange(rows, dtype=torch.float16).unsqueeze(1).expand(rows, HIDDEN)


class _Bucketer:
    def find_bucket(self, num_tokens):
        return next((b for b in (1, 4, 512) if b >= num_tokens), None)


def _wrapper(monkeypatch):
    """A wrapper whose gathered D2H row counts land in the returned list."""
    copied: list[int] = []
    real_convert = mr.convert

    def recording_convert(t, device=None, **kwargs):
        copied.append(t.shape[0])
        return real_convert(t, device=device, **kwargs)

    monkeypatch.setattr(mr, "convert", recording_convert)
    wrapper = _SpyreModelWrapper(
        _Body(),
        torch.device("cpu"),
        logits_row_buckets=[1, 4],
        shape_bucketer=_Bucketer(),
        model_dtype=torch.float16,
    )
    return wrapper, copied


def _run(wrapper, num_tokens, rows=None):
    if rows is not None:
        object.__setattr__(wrapper, "_sample_rows", torch.tensor(rows))
    return wrapper(input_ids=torch.zeros(num_tokens, dtype=torch.int64))


def test_only_the_sampled_rows_are_copied_and_land_at_their_offsets(monkeypatch):
    wrapper, copied = _wrapper(monkeypatch)
    rows = torch.tensor([3, 255, 511])

    out = _run(wrapper, 512, rows=rows)

    assert copied == [4], "three rows gathered at the warmed width 4"
    assert out.shape == (512, HIDDEN), "upstream indexes the full body"
    torch.testing.assert_close(out[rows], rows.to(torch.float16).unsqueeze(1).expand(3, HIDDEN))


# Too small for the gather to pay off, and past the largest bucket (no compiled gather).
@pytest.mark.parametrize("num_tokens", [4, 600])
def test_a_body_the_gather_cannot_serve_is_copied_whole(monkeypatch, num_tokens):
    wrapper, copied = _wrapper(monkeypatch)
    last = num_tokens - 1

    out = _run(wrapper, num_tokens, rows=[last])

    assert copied == []
    torch.testing.assert_close(out[last], torch.full((HIDDEN,), float(last), dtype=torch.float16))


def test_rows_are_consumed_by_one_forward(monkeypatch):
    wrapper, copied = _wrapper(monkeypatch)
    _run(wrapper, 512, rows=[511])
    copied.clear()

    out = _run(wrapper, 512)

    assert copied == [], "a forward without rows copies the whole body"
    torch.testing.assert_close(out[7], torch.full((HIDDEN,), 7.0, dtype=torch.float16))
