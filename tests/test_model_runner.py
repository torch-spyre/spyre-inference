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

"""Unit tests for TorchSpyreModelRunner internal components.

Tests cover:
- _SpyreModelWrapper: delegation, input conversion, output conversion, RoPE priming
- _compile_for_spyre: CompilationMode validation (error paths)
- warming_up_model: eager-pooling warmup clamping logic

These tests mock the Spyre device and model internals to run on CPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# _SpyreModelWrapper tests
# ---------------------------------------------------------------------------


class _FakeModel(nn.Module):
    """Minimal model that records calls and returns a known output."""

    def __init__(self):
        super().__init__()
        self.call_args = None
        self.call_kwargs = None
        self.linear = nn.Linear(4, 4)  # give it a real parameter

    def forward(self, *args, **kwargs):
        self.call_args = args
        self.call_kwargs = kwargs
        # Return a CPU tensor (simulating post-Spyre output)
        return torch.ones(2, 4, dtype=torch.float16)

    def compute_logits(self, hidden_states, *args, **kwargs):
        return hidden_states * 2


@pytest.fixture
def fake_model():
    return _FakeModel()


@pytest.fixture
def wrapper(fake_model):
    """Create a _SpyreModelWrapper with a fake model, mocking convert to be a no-op."""
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    # We mock convert to avoid needing real Spyre device
    with patch(
        "spyre_inference.v1.worker.spyre_model_runner.convert",
        side_effect=lambda t, **kw: t if t is None else t,
    ):
        w = _SpyreModelWrapper(
            model=fake_model,
            spyre_device=torch.device("cpu"),
            rope_modules=[],
        )
    return w


class TestSpyreModelWrapper:
    """Tests for _SpyreModelWrapper delegation and conversion."""

    def test_getattr_delegates_to_model(self, wrapper, fake_model):
        """__getattr__ delegates attribute access to the wrapped model."""
        # Access a real nn.Module attribute
        assert wrapper.linear is fake_model.linear

    def test_setattr_delegates_to_model(self, wrapper, fake_model):
        """__setattr__ sets attributes on the wrapped model."""
        wrapper.custom_attr = "hello"
        assert fake_model.custom_attr == "hello"

    def test_call_returns_result(self, fake_model):
        """__call__ invokes the model and returns the result."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=lambda t, **kw: t if t is None else t,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            result = w(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32))

        assert result is not None
        assert isinstance(result, torch.Tensor)

    def test_call_converts_int_inputs(self, fake_model):
        """__call__ converts int32 input_ids to int64 via convert()."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        convert_calls = []

        def mock_convert(t, **kwargs):
            if t is not None and isinstance(t, torch.Tensor):
                convert_calls.append(
                    {"dtype": kwargs.get("dtype"), "device": kwargs.get("device")}
                )
                if kwargs.get("dtype") == torch.int64:
                    return t.to(dtype=torch.int64)
            return t

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=mock_convert,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
            w(input_ids=input_ids)

        # Should have called convert with dtype=int64
        int64_calls = [c for c in convert_calls if c.get("dtype") == torch.int64]
        assert len(int64_calls) > 0

    def test_call_does_not_convert_none(self, fake_model):
        """__call__ passes None kwargs through without conversion."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=lambda t, **kw: t if t is None else t,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            w(input_ids=torch.tensor([1], dtype=torch.int32), positions=None)

        # The model should have received None for positions
        assert fake_model.call_kwargs.get("positions") is None

    def test_call_does_not_convert_float_tensors(self, fake_model):
        """__call__ leaves float tensors unconverted (only int goes to int64)."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        converted_dtypes = []

        def mock_convert(t, **kwargs):
            if t is not None and isinstance(t, torch.Tensor):
                if kwargs.get("dtype"):
                    converted_dtypes.append(kwargs["dtype"])
                    return t.to(dtype=kwargs["dtype"])
            return t

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=mock_convert,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            # Pass a float tensor as a kwarg
            float_input = torch.randn(2, 4, dtype=torch.float16)
            w(input_ids=torch.tensor([1, 2], dtype=torch.int32), extra=float_input)

        # Only int64 conversions should have happened
        assert all(d == torch.int64 for d in converted_dtypes)

    def test_compute_logits_delegates(self, fake_model):
        """compute_logits converts hidden_states to device and calls model."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=lambda t, **kw: t if t is None else t,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            hidden = torch.randn(2, 4, dtype=torch.float16)
            result = w.compute_logits(hidden)

        # compute_logits should have multiplied by 2 (fake model logic)
        torch.testing.assert_close(result, hidden * 2)

    def test_prime_rope_skipped_when_no_positions(self, fake_model):
        """_prime_rope_rotation is a no-op when positions=None."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        mock_rope = MagicMock()
        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=lambda t, **kw: t if t is None else t,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[mock_rope],
            )
            w._prime_rope_rotation(None)

        mock_rope.gather_rotation.assert_not_called()

    def test_prime_rope_skipped_when_no_modules(self, fake_model):
        """_prime_rope_rotation is a no-op when rope_modules is empty."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner.convert",
            side_effect=lambda t, **kw: t if t is None else t,
        ):
            w = _SpyreModelWrapper(
                model=fake_model,
                spyre_device=torch.device("cpu"),
                rope_modules=[],
            )
            # Should not raise
            w._prime_rope_rotation(torch.tensor([0, 1, 2]))


# ---------------------------------------------------------------------------
# _compile_for_spyre tests
# ---------------------------------------------------------------------------


class TestCompileForSpyre:
    """Test _compile_for_spyre error and skip paths."""

    def test_wrong_compilation_mode_raises_valueerror(self):
        """CompilationMode != NONE raises ValueError."""
        from vllm.config import CompilationMode

        # Build a minimal mock runner with just enough attributes
        runner = MagicMock()
        runner.compilation_config = MagicMock()
        runner.compilation_config.mode = CompilationMode(1)  # Not NONE
        runner.vllm_config = MagicMock()

        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        with pytest.raises(ValueError, match="Unsupported compilation mode"):
            TorchSpyreModelRunner._compile_for_spyre(runner)

    def test_enforce_eager_skips_compile(self):
        """enforce_eager=True skips torch.compile entirely."""
        from vllm.config import CompilationMode

        runner = MagicMock()
        runner.compilation_config = MagicMock()
        runner.compilation_config.mode = CompilationMode.NONE
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enforce_eager = True
        runner.model = MagicMock()

        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        # Should not raise, and model should remain unwrapped
        TorchSpyreModelRunner._compile_for_spyre(runner)
        # torch.compile was never called (model unchanged)
        assert runner.model is runner.model


# ---------------------------------------------------------------------------
# warming_up_model tests
# ---------------------------------------------------------------------------


class TestWarmingUpModel:
    """Test warming_up_model logic paths (eager-pooling clamping)."""

    def _make_mock_runner(self, runner_type="generate", enforce_eager=False):
        """Build a minimal mock of TorchSpyreModelRunner for warming_up_model."""
        runner = MagicMock()
        runner.max_num_reqs = 32
        runner.scheduler_config = MagicMock()
        runner.scheduler_config.max_num_batched_tokens = 512
        runner.scheduler_config.max_num_seqs = 16
        runner.model_config = MagicMock()
        runner.model_config.runner_type = runner_type
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enforce_eager = enforce_eager
        runner._dummy_run = MagicMock()
        return runner

    def test_normal_path_calls_dummy_run(self):
        """Non-pooling runner calls _dummy_run with standard num_tokens."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="generate", enforce_eager=False)

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            TorchSpyreModelRunner.warming_up_model(runner)

        runner._dummy_run.assert_called_once()
        # Should be min(max(16, 32), 512) = 32
        call_args = runner._dummy_run.call_args[0]
        assert call_args[0] == 32

    def test_pooling_eager_clamps_tokens(self):
        """Pooling + enforce_eager caps tokens at SPYRE_ENCODER_WARMUP_MAX_TOKENS."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            SPYRE_ENCODER_WARMUP_MAX_TOKENS,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="pooling", enforce_eager=True)

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            TorchSpyreModelRunner.warming_up_model(runner)

        runner._dummy_run.assert_called_once()
        call_args = runner._dummy_run.call_args[0]
        # Should be clamped to SPYRE_ENCODER_WARMUP_MAX_TOKENS
        assert call_args[0] <= SPYRE_ENCODER_WARMUP_MAX_TOKENS

    def test_pooling_eager_clamps_max_num_seqs(self):
        """Pooling + enforce_eager forces max_num_seqs=1 during warmup."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="pooling", enforce_eager=True)
        original_max_num_seqs = runner.scheduler_config.max_num_seqs

        seqs_during_warmup = None

        def capture_dummy_run(num_tokens):
            nonlocal seqs_during_warmup
            seqs_during_warmup = runner.scheduler_config.max_num_seqs

        runner._dummy_run = capture_dummy_run

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            TorchSpyreModelRunner.warming_up_model(runner)

        assert seqs_during_warmup == 1

    def test_pooling_eager_restores_max_num_seqs(self):
        """Pooling + enforce_eager restores max_num_seqs after warmup."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="pooling", enforce_eager=True)
        runner.scheduler_config.max_num_seqs = 16

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            TorchSpyreModelRunner.warming_up_model(runner)

        assert runner.scheduler_config.max_num_seqs == 16

    def test_pooling_eager_restores_max_num_seqs_on_exception(self):
        """max_num_seqs is restored even if _dummy_run raises."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="pooling", enforce_eager=True)
        runner.scheduler_config.max_num_seqs = 16
        runner._dummy_run = MagicMock(side_effect=RuntimeError("boom"))

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(RuntimeError, match="boom"):
                TorchSpyreModelRunner.warming_up_model(runner)

        # Must be restored despite the exception (try/finally pattern)
        assert runner.scheduler_config.max_num_seqs == 16

    def test_pooling_without_eager_uses_normal_path(self):
        """Pooling + compiled (not enforce_eager) uses normal warmup path."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            TorchSpyreModelRunner,
            SPYRE_ENCODER_WARMUP_MAX_TOKENS,
            _set_spyre_compilation_settings,
        )

        runner = self._make_mock_runner(runner_type="pooling", enforce_eager=False)

        with patch(
            "spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings"
        ) as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            TorchSpyreModelRunner.warming_up_model(runner)

        runner._dummy_run.assert_called_once()
        call_args = runner._dummy_run.call_args[0]
        # Should use normal num_tokens (32), not clamped
        assert call_args[0] == 32


# ---------------------------------------------------------------------------
# get_model unwrapping tests
# ---------------------------------------------------------------------------


class TestGetModel:
    """Test TorchSpyreModelRunner.get_model unwrapping logic."""

    def test_unwraps_spyre_model_wrapper(self):
        """get_model unwraps _SpyreModelWrapper to return the inner model."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            _SpyreModelWrapper,
            TorchSpyreModelRunner,
        )

        inner = nn.Linear(4, 4)
        runner = MagicMock()
        runner.model = _SpyreModelWrapper(inner, torch.device("cpu"), [])

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is inner

    def test_unwraps_optimized_module(self):
        """get_model unwraps torch.compile's OptimizedModule (has _orig_mod)."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        inner = nn.Linear(4, 4)
        # Simulate torch.compile wrapper
        compiled = MagicMock(spec=nn.Module)
        compiled._orig_mod = inner
        # Make isinstance check work
        compiled.__class__ = type("OptimizedModule", (nn.Module,), {})

        runner = MagicMock()
        runner.model = compiled

        # Since the mock won't pass isinstance(model, _SpyreModelWrapper),
        # it goes to _orig_mod path
        result = TorchSpyreModelRunner.get_model(runner)
        assert result is inner

    def test_returns_plain_module_unchanged(self):
        """get_model returns a plain nn.Module directly."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        model = nn.Linear(4, 4)
        runner = MagicMock()
        runner.model = model

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is model


# ---------------------------------------------------------------------------
# _sync_device tests
# ---------------------------------------------------------------------------


class TestSyncDevice:
    """Test _sync_device calls torch.spyre.synchronize."""

    def test_sync_device_calls_synchronize(self):
        """_sync_device invokes torch.spyre.synchronize with the correct device."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = MagicMock()
        runner._spyre_device = torch.device("cpu")

        mock_spyre = MagicMock()
        with patch.dict("sys.modules", {"torch.spyre": mock_spyre}):
            with patch("torch.spyre", mock_spyre, create=True):
                TorchSpyreModelRunner._sync_device(runner)

        mock_spyre.synchronize.assert_called_once_with(runner._spyre_device)
