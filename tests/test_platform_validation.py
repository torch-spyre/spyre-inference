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

"""Unit tests for platform.py validation guards and config defaults.

These tests cover the safety checks in check_and_update_config (dtype, DP>1)
and the defaults set by apply_config_platform_defaults (enforce_eager,
CompilationMode, dtype). These are critical guards: a non-float16 dtype
produces NaN/Inf on Spyre, and DP>1 produces incorrect rank assignment.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.compilation import CompilationConfig


# ---------------------------------------------------------------------------
# apply_config_platform_defaults
# ---------------------------------------------------------------------------


class TestApplyConfigPlatformDefaults:
    """Tests for TorchSpyrePlatform.apply_config_platform_defaults."""

    def _make_vllm_config(self, dtype=torch.float32, enforce_eager=False):
        """Create a minimal VllmConfig for testing defaults application."""
        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        compilation_config = CompilationConfig(custom_ops=["all"])
        return VllmConfig(
            model_config=model_config,
            compilation_config=compilation_config,
        )

    def test_forces_dtype_to_float16(self):
        """apply_config_platform_defaults forces dtype to float16."""
        from spyre_inference.platform import TorchSpyrePlatform

        vllm_config = self._make_vllm_config(dtype=torch.float32)
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.model_config.dtype == torch.float16

    def test_forces_enforce_eager_true(self):
        """apply_config_platform_defaults forces enforce_eager=True."""
        from spyre_inference.platform import TorchSpyrePlatform

        vllm_config = self._make_vllm_config(enforce_eager=False)
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.model_config.enforce_eager is True

    def test_forces_compilation_mode_none(self):
        """apply_config_platform_defaults forces CompilationMode.NONE."""
        from vllm.config import CompilationMode
        from spyre_inference.platform import TorchSpyrePlatform

        vllm_config = self._make_vllm_config()
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.compilation_config.mode == CompilationMode.NONE

    def test_preserves_other_model_config(self):
        """apply_config_platform_defaults does not alter unrelated model settings."""
        from spyre_inference.platform import TorchSpyrePlatform

        vllm_config = self._make_vllm_config()
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        # max_model_len should remain unchanged
        assert vllm_config.model_config.max_model_len == 128


# ---------------------------------------------------------------------------
# check_and_update_config — dtype guard
# ---------------------------------------------------------------------------


class TestDtypeValidation:
    """Tests for the dtype != float16 ValueError guard in check_and_update_config."""

    def test_non_float16_dtype_raises_value_error(self):
        """check_and_update_config raises ValueError when dtype != float16."""
        from spyre_inference.platform import TorchSpyrePlatform

        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        # Force dtype to float32 after construction (bypassing defaults)
        model_config.dtype = torch.float32

        cache_config = CacheConfig(block_size=64)
        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )

        with pytest.raises(ValueError, match="torch.float16"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_float16_dtype_passes_validation(self):
        """check_and_update_config accepts dtype=float16 without error."""
        from spyre_inference.platform import TorchSpyrePlatform

        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        cache_config = CacheConfig(block_size=64)
        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )

        # Should not raise
        TorchSpyrePlatform.check_and_update_config(vllm_config)
        assert vllm_config.model_config.dtype == torch.float16


# ---------------------------------------------------------------------------
# check_and_update_config — data_parallel_size > 1 guard
# ---------------------------------------------------------------------------


class TestDataParallelValidation:
    """Tests for the DP>1 ValueError guard in check_and_update_config."""

    def test_dp_greater_than_1_raises_value_error(self):
        """check_and_update_config raises ValueError for data_parallel_size > 1."""
        from spyre_inference.platform import TorchSpyrePlatform

        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        cache_config = CacheConfig(block_size=64)
        compilation_config = CompilationConfig(custom_ops=["all"])
        parallel_config = ParallelConfig(data_parallel_size=2)

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
            parallel_config=parallel_config,
        )

        with pytest.raises(ValueError, match="data_parallel_size"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_dp_equal_1_passes_validation(self):
        """check_and_update_config accepts data_parallel_size=1."""
        from spyre_inference.platform import TorchSpyrePlatform

        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        cache_config = CacheConfig(block_size=64)
        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )

        # data_parallel_size defaults to 1 — should not raise
        TorchSpyrePlatform.check_and_update_config(vllm_config)


# ---------------------------------------------------------------------------
# check_and_update_config — worker class assignment
# ---------------------------------------------------------------------------


class TestWorkerClassAssignment:
    """Tests for the worker_cls auto-detection in check_and_update_config."""

    def test_auto_worker_resolved_to_spyre_worker(self):
        """worker_cls='auto' should be replaced with the Spyre worker path."""
        from spyre_inference.platform import TorchSpyrePlatform

        model_config = ModelConfig(
            model="Qwen/Qwen3-0.6B",
            max_model_len=128,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        cache_config = CacheConfig(block_size=64)
        compilation_config = CompilationConfig(custom_ops=["all"])

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )

        TorchSpyrePlatform.check_and_update_config(vllm_config)

        expected_worker = "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker"
        assert vllm_config.parallel_config.worker_cls == expected_worker


# ---------------------------------------------------------------------------
# check_max_model_len — cap to _DEFAULT_DERIVED_MAX_MODEL_LEN
# ---------------------------------------------------------------------------


class TestCheckMaxModelLen:
    """Tests for TorchSpyrePlatform.check_max_model_len capping."""

    def test_caps_large_max_model_len(self):
        """Large model-derived max_model_len is capped to 2048."""
        from spyre_inference.platform import TorchSpyrePlatform

        result = TorchSpyrePlatform.check_max_model_len(131072)
        assert result == TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN

    def test_preserves_small_max_model_len(self):
        """Small model-derived max_model_len (< 2048) is not changed."""
        from spyre_inference.platform import TorchSpyrePlatform

        result = TorchSpyrePlatform.check_max_model_len(512)
        assert result == 512

    def test_equal_to_cap_stays_unchanged(self):
        """max_model_len equal to the cap stays at 2048."""
        from spyre_inference.platform import TorchSpyrePlatform

        result = TorchSpyrePlatform.check_max_model_len(2048)
        assert result == 2048
