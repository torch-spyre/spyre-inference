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
from unittest import mock

import pytest
import torch
from vllm.config import CompilationMode

from spyre_inference.v1.worker.spyre_model_runner import (
    TorchSpyreModelRunner,
    _SpyreModelWrapper,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import logits_row_buckets

HIDDEN = 8
BODY_BUCKETS = [1, 2, 4, 8, 16, 32, 512]
MAX_NUM_REQS = 32


class _Bucketer:
    def __init__(self, bucket_sizes):
        self.bucket_sizes = bucket_sizes
        self.warmed_up = False

    def mark_warmed_up(self):
        self.warmed_up = True

    def find_bucket(self, num_tokens):
        return next((size for size in self.bucket_sizes if size >= num_tokens), None)


def _runner(bucket_sizes=BODY_BUCKETS, max_num_reqs=MAX_NUM_REQS):
    # NONE keeps warmup off the attention recorder, which this file does not
    # exercise and which needs a real KV cache.
    compilation_config = types.SimpleNamespace(
        compile_sizes=list(bucket_sizes),
        inductor_compile_config={},
        static_forward_context={},
        mode=CompilationMode.NONE,
    )
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model_config = types.SimpleNamespace(
        runner_type="generate",
        dtype=torch.float16,
        get_hidden_size=lambda: HIDDEN,
    )
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=compilation_config,
    )
    runner.compilation_config = compilation_config
    runner._spyre_device = torch.device("cpu")
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


class TestOutputGatherWarmup:
    """The output row gather is warmed on every reachable (body, rows) pair.

    ``select_rows`` reaches eager ``index_select``, whose torch-spyre kernel
    specializes on both operand shapes. Warming only ``(widest_body, rows)`` leaves
    every narrower body to compile mid-request, measured at ~0.5-1.5s per pair.
    """

    HIDDEN = 4096

    def _armed_runner(
        self,
        monkeypatch,
        bucket_sizes=(32, 64, 128, 512),
        max_num_reqs=4,
        dtype=torch.float16,
    ):
        runner, body_rows, _ = _runner(bucket_sizes=list(bucket_sizes), max_num_reqs=max_num_reqs)
        # Real hidden width: the savings guard is a byte threshold, so HIDDEN=8
        # would put every pair under the floor and warm nothing.
        hidden = self.HIDDEN
        runner.model_config = types.SimpleNamespace(
            runner_type="generate",
            dtype=dtype,
            get_hidden_size=lambda: hidden,
        )

        def dummy_run(size, *args, **kwargs):
            # Mirror upstream: the second value is hidden_states[logit_indices],
            # already reduced to the sampled rows and left on CPU for generation.
            # A fake returning [size, hidden] would hide a warmup that keys off
            # the wrong tensor -- which is exactly the defect this covers.
            body_rows.append(size)
            sampled = min(size, max_num_reqs)
            return None, torch.zeros(sampled, hidden, dtype=dtype, device="cpu")

        runner._dummy_run = dummy_run
        runner.model = _SpyreModelWrapper(
            torch.nn.Identity(),
            torch.device("cpu"),
            logits_row_buckets=logits_row_buckets(list(bucket_sizes), max_num_reqs),
        )
        runner._can_trim_output_d2h = lambda: True

        gathered: list[tuple[int, int]] = []
        original = _SpyreModelWrapper._d2h_sampled_rows

        def recording(wrapper, hidden_states, rows):
            gathered.append((hidden_states.shape[0], rows.numel()))
            return original(wrapper, hidden_states, rows)

        monkeypatch.setattr(_SpyreModelWrapper, "_d2h_sampled_rows", recording)
        runner.warming_up_model()
        return gathered

    def test_every_reachable_body_is_warmed_not_only_the_widest(self, monkeypatch):
        gathered = self._armed_runner(monkeypatch)

        bodies = {body for body, _ in gathered}
        # 32 is excluded by the savings guard: trimming 32->4 rows of 4096 fp16
        # saves 224 KiB, under the 256 KiB floor.
        assert bodies == {64, 128, 512}, bodies

    def test_each_body_is_warmed_at_every_row_width(self, monkeypatch):
        # Small buckets are what make row_widths wider than one entry:
        # min(size, max_num_reqs) over [1, 2, 4, ...] yields [1, 2, 4].
        gathered = self._armed_runner(monkeypatch, bucket_sizes=(1, 2, 4, 64, 512))

        for body in (64, 512):
            widths = {rows for b, rows in gathered if b == body}
            assert widths == {1, 2, 4}, (body, widths)

    def test_pairs_the_runtime_would_skip_are_not_warmed(self, monkeypatch):
        gathered = self._armed_runner(monkeypatch, bucket_sizes=(4, 8, 16))

        # Every body here is too small for the gather to pay off.
        assert gathered == []

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [(torch.float16, {64}), (torch.float32, {32, 64})],
    )
    def test_the_savings_filter_uses_the_model_dtype_itemsize(self, monkeypatch, dtype, expected):
        """The warmup filter must not be stricter than the runtime's.

        ``_build_attention_metadata`` arms on ``model_config.dtype.itemsize``. A
        warmup hardcoded to 2 bytes would skip the 32-row body at fp32 -- which the
        runtime then arms, paying the mid-request compile this warmup removes.
        Body 32 trims to 4 rows of 4096: 224 KiB at fp16 (under the floor), 448 KiB
        at fp32 (over it).
        """
        gathered = self._armed_runner(monkeypatch, bucket_sizes=(32, 64), dtype=dtype)

        assert {body for body, _ in gathered} == expected, gathered

    def test_warmed_bodies_are_body_buckets_on_the_spyre_device(self, monkeypatch):
        """Regression: the warmup must not key off ``_dummy_run``'s second value.

        That value is ``hidden_states[logit_indices]`` -- already reduced to at most
        ``max_num_reqs`` rows and left on CPU. Deriving the warmup bodies from it
        made every pair fail ``_worth_trimming`` (nothing was warmed) and allocated
        CPU dummies, which compile no Spyre kernel. Assert both properties directly.
        """
        seen: list[tuple[int, str]] = []

        def recording(wrapper, hidden_states, rows):
            seen.append((hidden_states.shape[0], hidden_states.device.type))

        monkeypatch.setattr(_SpyreModelWrapper, "_d2h_sampled_rows", recording)
        runner, _, _ = _runner(bucket_sizes=[1, 2, 4, 512], max_num_reqs=4)
        runner._spyre_device = torch.device("meta")
        hidden = self.HIDDEN
        runner.model_config = types.SimpleNamespace(
            runner_type="generate",
            dtype=torch.float16,
            get_hidden_size=lambda: hidden,
        )
        runner._dummy_run = lambda size, *a, **k: (
            None,
            torch.zeros(min(size, 4), hidden, dtype=torch.float16, device="cpu"),
        )
        runner.model = _SpyreModelWrapper(
            torch.nn.Identity(),
            runner._spyre_device,
            logits_row_buckets=logits_row_buckets([1, 2, 4, 512], 4),
        )
        runner._can_trim_output_d2h = lambda: True
        runner.warming_up_model()

        # 512 is the only body bucket wide enough to clear the byte floor.
        assert {body for body, _ in seen} == {512}, seen
        # Allocated on the runner's Spyre device, not wherever the sample tensor sat.
        assert {dev for _, dev in seen} == {runner._spyre_device.type}, seen


class TestArming:
    """The ``logits_indices`` rewrite happens only when the trim is certain.

    The rewrite is destructive: it replaces the real body offsets with
    ``arange(n)``. If the wrapper then declined to trim, upstream's
    ``hidden_states[logits_indices]`` would select the first ``n`` rows of the
    full body instead of the sampled ones, silently sampling wrong tokens. These
    pin the invariant that the decision is made once, here, before the rewrite.
    """

    HIDDEN = 4096

    def _runner_for_arming(self, *, bucket_sizes=(1, 2, 4, 8, 512), max_num_reqs=4, **gates):
        runner, _, _ = _runner(bucket_sizes=list(bucket_sizes), max_num_reqs=max_num_reqs)
        runner.model = _SpyreModelWrapper(
            torch.nn.Identity(),
            torch.device("cpu"),
            logits_row_buckets=logits_row_buckets(list(bucket_sizes), max_num_reqs),
        )
        runner.model_config = types.SimpleNamespace(
            runner_type="generate", dtype=torch.float16, get_hidden_size=lambda: self.HIDDEN
        )
        runner.is_pooling_model = gates.get("is_pooling_model", False)
        runner.num_prompt_logprobs = gates.get("num_prompt_logprobs", {})
        runner.use_aux_hidden_state_outputs = gates.get("use_aux_hidden_state_outputs", False)
        runner.speculative_config = gates.get("speculative_config")
        runner.cache_config = types.SimpleNamespace(
            kv_sharing_fast_prefill=gates.get("kv_sharing_fast_prefill", False)
        )
        return runner

    def _build(self, runner, num_tokens, rows):
        idx = torch.tensor(rows, dtype=torch.int64)
        with mock.patch.object(
            TorchSpyreModelRunner.__bases__[0],
            "_build_attention_metadata",
            return_value=object(),
        ):
            runner._build_attention_metadata(num_tokens=num_tokens, logits_indices=idx)
        return idx, runner.model._sample_rows

    def test_arms_and_rewrites_when_the_trim_is_worthwhile(self):
        runner = self._runner_for_arming()

        idx, stashed = self._build(runner, 512, [0, 255, 511])

        assert stashed.tolist() == [0, 255, 511], "real offsets must be preserved"
        assert idx.tolist() == [0, 1, 2], "indices must become an identity"

    def test_does_not_rewrite_when_the_trim_is_not_worthwhile(self):
        """The regression: a small body must keep its real offsets."""
        runner = self._runner_for_arming()

        idx, stashed = self._build(runner, 4, [3])

        assert stashed is None, "must not arm a trim that will not happen"
        assert idx.tolist() == [3], "real offsets must survive so upstream samples row 3"

    def test_mixed_batch_keeps_real_offsets(self):
        """Decodes plus a short prefill: non-identity rows, body too small to trim."""
        runner = self._runner_for_arming()

        idx, stashed = self._build(runner, 7, [0, 1, 2, 6])

        assert stashed is None
        assert idx.tolist() == [0, 1, 2, 6]

    def test_uses_the_padded_body_bucket_for_the_savings_guard(self):
        runner = self._runner_for_arming(bucket_sizes=(64, 128), max_num_reqs=64)

        rows = [*range(63), 127]
        idx, stashed = self._build(runner, 65, rows)

        assert stashed.tolist() == rows
        assert idx.tolist() == list(range(64))

    def test_uses_the_model_dtype_itemsize_for_the_savings_guard(self):
        runner = self._runner_for_arming(bucket_sizes=(64, 128), max_num_reqs=64)
        runner.model_config.dtype = torch.float32
        runner.model_config.get_hidden_size = lambda: 1024

        idx, stashed = self._build(runner, 65, list(range(64)))

        assert stashed.tolist() == list(range(64))
        assert idx.tolist() == list(range(64))

    @pytest.mark.parametrize(
        "gate",
        [
            {"is_pooling_model": True},
            {"num_prompt_logprobs": {"req": 3}},
            {"use_aux_hidden_state_outputs": True},
            {"speculative_config": object()},
            {"kv_sharing_fast_prefill": True},
        ],
    )
    def test_gated_modes_never_arm(self, gate):
        runner = self._runner_for_arming(**gate)

        idx, stashed = self._build(runner, 512, [0, 255, 511])

        assert stashed is None, f"must not arm with {gate}"
        assert idx.tolist() == [0, 255, 511]

    def test_num_tokens_passed_positionally_arms_the_trim(self):
        """Upstream may pass its first ``num_tokens`` parameter positionally."""
        runner = self._runner_for_arming()
        idx = torch.tensor([0, 255, 511], dtype=torch.int64)

        with mock.patch.object(
            TorchSpyreModelRunner.__bases__[0],
            "_build_attention_metadata",
            return_value=object(),
        ):
            runner._build_attention_metadata(512, logits_indices=idx)

        assert runner.model._sample_rows.tolist() == [0, 255, 511]
        assert idx.tolist() == [0, 1, 2]
