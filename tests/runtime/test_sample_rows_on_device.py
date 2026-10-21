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

"""Sampled-row selection, output device contracts, and bucketed logits projection."""

import types

import pytest
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


@pytest.mark.parametrize("num_rows", [1, 3, 4])
def test_decode_reuses_the_body_bucket_without_a_gather(monkeypatch, num_rows):
    plain, _, widths = _hidden(monkeypatch, [1, 2, 4, 8, 16])
    size = 1 << (num_rows - 1).bit_length()
    hidden = mr._SpyreHiddenStates.wrap(plain[:size], [1, 2, 4, 8, 16])

    rows = hidden[torch.arange(num_rows)]

    assert widths == []
    assert rows.data_ptr() == plain.data_ptr()
    torch.testing.assert_close(rows, plain[:num_rows])
    assert getattr(rows, "_spyre_padded_rows", rows).shape[0] == size


@pytest.mark.parametrize("indices", [[2, 0, 1], [0, 0, 2], [1, 2, 3]])
def test_a_decode_sized_nonidentity_index_still_gathers(monkeypatch, indices):
    plain, _, widths = _hidden(monkeypatch, [1, 2, 4])
    hidden = mr._SpyreHiddenStates.wrap(plain[:4], [1, 2, 4])
    idx = torch.tensor(indices)

    rows = hidden[idx]

    assert widths == [4]
    torch.testing.assert_close(rows, plain[idx])


def test_empty_index_does_not_try_to_repeat_a_missing_row(monkeypatch):
    _, hidden, widths = _hidden(monkeypatch, [1, 2, 4])

    rows = hidden[torch.empty(0, dtype=torch.int64)]

    assert rows.shape == (0, HIDDEN)
    assert widths == []


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
    _, hidden, widths = _hidden(monkeypatch, [1, 2, 4, 16, 32])
    runner, projected = _runner()

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(runner, hidden, [1, 2, 4, 16, 32])

    assert widths == [1, 2, 4, 16]
    assert projected == [1, 2, 4, 16, 16]


def test_warmup_skips_hidden_states_left_on_the_host(monkeypatch):
    _, _, widths = _hidden(monkeypatch, [1, 2])
    runner, projected = _runner()

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(
        runner, torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [1, 2]
    )

    assert widths == []
    assert projected == []


def test_cpu_fallback_pooling_dummy_keeps_the_host_output_contract(monkeypatch):
    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("spyre")

    plain = torch.randn(4, HIDDEN)
    device_hidden = plain.as_subclass(DeviceTensor)
    copies = []

    def convert(t, device=None, dtype=None):
        copies.append(device)
        return plain

    monkeypatch.setattr("spyre_inference.custom_ops.utils.convert", convert)
    monkeypatch.setattr(mr.attn_layer, "publish_null_slots", lambda _: None)
    monkeypatch.setenv("SPYRE_SAMPLE_ROWS_ON_DEVICE", "1")
    model = lambda **kwargs: device_hidden
    wrapper = mr._SpyreModelWrapper(model, torch.device("spyre"), model_dtype=torch.float16)

    def dummy_run(self, *args, **kwargs):
        assert kwargs["force_attention"]
        hidden = self.model()
        return hidden, hidden[:1]

    monkeypatch.setattr(mr.GPUModelRunner, "_dummy_run", dummy_run)
    runner = object.__new__(mr.TorchSpyreModelRunner)
    runner.model = wrapper
    runner.model_config = types.SimpleNamespace(runner_type="pooling")

    hidden, _ = runner._dummy_run(4)

    assert copies == ["cpu"]
    assert type(hidden) is torch.Tensor
    assert hidden.device.type == "cpu"


@pytest.mark.parametrize(
    "offset,num_rows,stride",
    [(0, 1, 1), (0, 3, 1), (3, 3, 1), (4, 4, 1), (31, 33, 1), (1, 65, 1), (0, 4, 2), (3, 3, 2)],
)
def test_device_logits_materialize_views_without_unnecessary_host_copies(
    monkeypatch, offset, num_rows, stride
):
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("requires Spyre hardware")

    from spyre_inference.custom_ops.utils import convert

    torch.manual_seed(42)
    plain = torch.randn(128, 64 * stride, dtype=torch.float16)
    weight = torch.randn(64, 2048, dtype=torch.float16) * 0.01
    device_weight = convert(weight, "spyre")

    @torch.compile(backend="inductor", dynamic=False, fullgraph=True)
    def project(hidden):
        return hidden @ device_weight

    model = types.SimpleNamespace(compute_logits=lambda hidden: convert(project(hidden), "cpu"))
    wrapper = mr._SpyreModelWrapper(
        model, torch.device("spyre"), logits_row_buckets=[1, 2, 4, 8], model_dtype=torch.float16
    )
    hidden = convert(plain, "spyre")[offset : offset + num_rows, ::stride]

    copies = []

    def track_conversion(t, device=None, **kwargs):
        if t.device.type == "spyre" and str(device) == "cpu":
            copies.append("cpu")
        return convert(t, device=device, **kwargs)

    monkeypatch.setattr(mr, "convert", track_conversion)
    actual = wrapper.compute_logits(hidden)

    assert copies == (["cpu"] if stride != 1 else [])
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual, plain[offset : offset + num_rows, ::stride] @ weight, atol=0.01, rtol=0.03
    )
