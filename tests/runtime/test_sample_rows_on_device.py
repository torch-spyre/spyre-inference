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


def _wrapper(buckets, projected):
    model = types.SimpleNamespace(
        compute_logits=lambda rows: projected.append(rows.shape[0]) or rows
    )
    return mr._SpyreModelWrapper(
        model,
        torch.device("cpu"),
        logits_row_buckets=buckets,
        model_dtype=torch.float16,
        sample_rows_on_device=True,
    )


@pytest.mark.parametrize(
    "indices,padded", [([3, 9, 15], [3, 9, 15, 15]), ([0, 5], [0, 5]), ([], [])]
)
def test_the_sampled_index_is_padded_onto_a_row_bucket(indices, padded):
    wrapper = _wrapper([1, 2, 4, 8], [])

    assert wrapper.pad_sampled_rows(torch.tensor(indices, dtype=torch.int64)).tolist() == padded


def test_compute_logits_projects_the_padded_gather_and_trims_it():
    projected: list[int] = []
    wrapper = _wrapper([1, 2, 4, 8], projected)
    hidden = torch.randn(ROWS, HIDDEN, dtype=torch.float16)

    logits = wrapper.compute_logits(hidden[wrapper.pad_sampled_rows(torch.tensor([3, 9, 15]))])

    assert projected == [4]
    torch.testing.assert_close(logits, hidden[[3, 9, 15]])


def test_warmup_compiles_every_row_bucket_a_body_bucket_holds():
    """The lm_head sees the gather's output, so it compiles on the gathered layout."""
    projected: list[int] = []
    runner = types.SimpleNamespace(model=_wrapper([1, 2, 4, 16, 32], projected))
    hidden = torch.randn(ROWS, HIDDEN, dtype=torch.float16)

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(runner, hidden, [1, 2, 4, 16, 32])

    assert projected == [1, 2, 4, 16, 16]


@pytest.mark.parametrize("num_tokens,skips", [(3, True), (40, False)])
def test_only_decode_on_its_own_row_bucket_skips_the_gather(monkeypatch, num_tokens, skips):
    runner = object.__new__(mr.TorchSpyreModelRunner)
    runner.model = _wrapper([1, 2, 4], [])
    runner.spyre_shape_bucketer = types.SimpleNamespace(
        warmed_bucket=lambda n: next(b for b in (1, 2, 4, 512) if b >= n)
    )
    monkeypatch.setattr(
        mr.GPUModelRunner, "_prepare_inputs", lambda *a, **k: (torch.tensor([0, 1, 2]), None, 1)
    )
    scheduler_output = types.SimpleNamespace(total_num_scheduled_tokens=num_tokens)

    index, *_ = runner._prepare_inputs(scheduler_output)

    assert (index == slice(None)) is skips


def test_warmup_skips_a_runner_that_samples_on_the_host():
    projected: list[int] = []
    runner = types.SimpleNamespace(model=_wrapper([1, 2], projected))
    object.__setattr__(runner.model, "_sample_rows_on_device", False)

    mr.TorchSpyreModelRunner._warm_sample_row_gathers(
        runner, torch.zeros(ROWS, HIDDEN, dtype=torch.float16), [1, 2]
    )

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
def test_device_views_take_the_host_pad_path(monkeypatch, offset, num_rows, stride):
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

    assert copies == ["cpu"]
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual, plain[offset : offset + num_rows, ::stride] @ weight, atol=0.01, rtol=0.03
    )
