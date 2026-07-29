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

"""Unit tests for Spyre compilation settings and _compile_for_spyre.

Tests target:
- `_set_spyre_compilation_settings` context manager
- `TorchSpyreModelRunner._compile_for_spyre` error validation
- Compilation mode enforcement

All tests run on CPU without requiring the Spyre runtime.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestSetSpyreCompilationSettings:
    """Tests for _set_spyre_compilation_settings context manager."""

    def test_restores_freezing_on_exit(self):
        """Context manager restores inductor freezing config on exit."""
        import torch._inductor.config as torch_inductor_config
        from spyre_inference.v1.worker.spyre_model_runner import (
            _set_spyre_compilation_settings,
        )

        original_freezing = torch_inductor_config.freezing

        mock_config = MagicMock()
        mock_config.compilation_config.inductor_compile_config = {}

        with _set_spyre_compilation_settings(mock_config):
            pass

        assert torch_inductor_config.freezing == original_freezing

    def test_enables_freezing_when_max_autotune(self):
        """When max_autotune=True, freezing is set to True."""
        import torch._inductor.config as torch_inductor_config
        from spyre_inference.v1.worker.spyre_model_runner import (
            _set_spyre_compilation_settings,
        )

        original_freezing = torch_inductor_config.freezing

        mock_config = MagicMock()
        mock_config.compilation_config.inductor_compile_config = {
            "max_autotune": True
        }

        with _set_spyre_compilation_settings(mock_config):
            assert torch_inductor_config.freezing is True

        # Restored after exit
        assert torch_inductor_config.freezing == original_freezing

    def test_no_freezing_change_without_max_autotune(self):
        """Without max_autotune, freezing stays unchanged."""
        import torch._inductor.config as torch_inductor_config
        from spyre_inference.v1.worker.spyre_model_runner import (
            _set_spyre_compilation_settings,
        )

        original_freezing = torch_inductor_config.freezing

        mock_config = MagicMock()
        mock_config.compilation_config.inductor_compile_config = {
            "max_autotune": False
        }

        with _set_spyre_compilation_settings(mock_config):
            assert torch_inductor_config.freezing == original_freezing

    def test_restores_on_exception(self):
        """Freezing is restored even when the body raises."""
        import torch._inductor.config as torch_inductor_config
        from spyre_inference.v1.worker.spyre_model_runner import (
            _set_spyre_compilation_settings,
        )

        original_freezing = torch_inductor_config.freezing

        mock_config = MagicMock()
        mock_config.compilation_config.inductor_compile_config = {
            "max_autotune": True
        }

        with pytest.raises(RuntimeError):
            with _set_spyre_compilation_settings(mock_config):
                assert torch_inductor_config.freezing is True
                raise RuntimeError("test error")

        assert torch_inductor_config.freezing == original_freezing


class TestCompileForSpyre:
    """Tests for TorchSpyreModelRunner._compile_for_spyre."""

    def _make_runner(self, enforce_eager=False, mode=None):
        """Create a minimal mock runner for testing _compile_for_spyre."""
        from vllm.config import CompilationMode
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)

        mock_compilation_config = MagicMock()
        mock_compilation_config.mode = mode or CompilationMode.NONE
        runner.compilation_config = mock_compilation_config

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.enforce_eager = enforce_eager
        runner.vllm_config = mock_vllm_config

        runner.model = MagicMock()
        return runner

    def test_raises_on_unsupported_compilation_mode(self):
        """Unsupported CompilationMode raises ValueError."""
        from vllm.config import CompilationMode
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        # Use a mode other than NONE (e.g., if INDUCTOR exists)
        for mode in CompilationMode:
            if mode == CompilationMode.NONE:
                continue
            runner = self._make_runner(mode=mode)
            with pytest.raises(ValueError, match="Unsupported compilation mode"):
                runner._compile_for_spyre()

    def test_enforce_eager_skips_compile(self):
        """enforce_eager=True skips torch.compile entirely."""
        runner = self._make_runner(enforce_eager=True)

        with patch("torch.compile") as mock_compile:
            runner._compile_for_spyre()
            mock_compile.assert_not_called()

    def test_compiles_with_inductor_backend(self):
        """Normal path calls torch.compile with backend='inductor'."""
        runner = self._make_runner(enforce_eager=False)

        with patch("torch.compile") as mock_compile:
            mock_compile.return_value = runner.model
            runner._compile_for_spyre()
            mock_compile.assert_called_once_with(
                runner.model,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )

    def test_compile_updates_model_attribute(self):
        """After compilation, runner.model is the compiled model."""
        runner = self._make_runner(enforce_eager=False)
        compiled_sentinel = MagicMock()

        with patch("torch.compile", return_value=compiled_sentinel):
            runner._compile_for_spyre()
            assert runner.model is compiled_sentinel


class TestFuncWrapper:
    """Tests for the _FuncWrapper grid-launch syntax adapter."""

    def test_getitem_returns_underlying_function(self):
        """_FuncWrapper[(grid,)](...) passes through to the wrapped func."""
        from spyre_inference.v1.worker.spyre_model_runner import _FuncWrapper

        def my_func(a, b):
            return a + b

        wrapper = _FuncWrapper(my_func)
        # Grid subscript returns the underlying function
        kernel = wrapper[(1,)]
        assert kernel is my_func
        assert kernel(3, 4) == 7

    def test_any_grid_value_works(self):
        """Any grid value (tuple, int, etc.) passes through."""
        from spyre_inference.v1.worker.spyre_model_runner import _FuncWrapper

        def my_func():
            return "ok"

        wrapper = _FuncWrapper(my_func)
        assert wrapper[(32, 32)]() == "ok"
        assert wrapper[1]() == "ok"
        assert wrapper[None]() == "ok"


class TestModelRunnerConstants:
    """Tests for model runner module-level constants."""

    def test_dma_token_limit_invariant(self):
        """SPYRE_ENCODER_WARMUP_MAX_TOKENS < SPYRE_ENCODER_DMA_TOKEN_LIMIT."""
        from spyre_inference.v1.worker.spyre_model_runner import (
            SPYRE_ENCODER_DMA_TOKEN_LIMIT,
            SPYRE_ENCODER_WARMUP_MAX_TOKENS,
        )

        assert SPYRE_ENCODER_WARMUP_MAX_TOKENS < SPYRE_ENCODER_DMA_TOKEN_LIMIT

    def test_pad_slot_id_is_negative(self):
        """_PAD_SLOT_ID must be negative (invalid cache index)."""
        from spyre_inference.v1.worker.spyre_model_runner import _PAD_SLOT_ID

        assert _PAD_SLOT_ID < 0
