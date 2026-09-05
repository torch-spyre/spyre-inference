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

"""Device-free tests for RoBERTa position offset and model-wrapper I/O."""

from __future__ import annotations

import torch
import torch.nn as nn

from spyre_inference.models.roberta import offset_roberta_position_ids
from spyre_inference.v1.worker import spyre_model_runner as mr


def test_offset_roberta_position_ids_runs_on_cpu(monkeypatch):
    seen_devices: list[object] = []

    def fake_convert(t, device=None, dtype=None):
        seen_devices.append(device)
        out = t
        if dtype is not None:
            out = out.to(dtype)
        if device is not None:
            # Stay on CPU in this unit test; only record the requested device.
            pass
        return out

    monkeypatch.setattr(
        "spyre_inference.models.roberta.convert", fake_convert
    )

    pos = torch.tensor([0, 1, 2], dtype=torch.int64)
    out = offset_roberta_position_ids(pos, padding_idx=1, device=torch.device("meta"))
    assert seen_devices[0] == "cpu"
    assert seen_devices[1] == torch.device("meta")
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([2, 3, 4], dtype=torch.int64))


def test_wrapper_converts_ints_to_int64(monkeypatch):
    seen: list[torch.dtype | None] = []

    def fake_convert(t, device=None, dtype=None):
        seen.append(dtype)
        return t if dtype is None else t.to(dtype)

    monkeypatch.setattr(mr, "convert", fake_convert)

    class _Capture(nn.Module):
        def forward(self, input_ids=None, positions=None, **kwargs):
            return {"input_ids": input_ids, "positions": positions}

    wrapper = mr._SpyreModelWrapper(
        _Capture(), torch.device("cpu"), keep_outputs_on_device=True
    )
    out = wrapper(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int32),
    )
    assert seen == [torch.int64, torch.int64]
    assert out["input_ids"].dtype == torch.int64
    assert out["positions"].dtype == torch.int64
