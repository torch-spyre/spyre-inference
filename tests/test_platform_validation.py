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

"""Unit tests for platform.py validation logic beyond block_size.

Tests cover: dtype validation, data_parallel rejection, num_gpu_blocks_override
formula, check_max_model_len default, and apply_config_platform_defaults.
"""

import math

import pytest
import torch

from vllm.config import (
    VllmConfig,
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.config.compilation import CompilationConfig, CompilationMode

from spyre_inference.platform import TorchSpyrePlatform


def _make_vllm_config(
    dtype=torch.float16,
    max_model_len=128,
    block_size=64,
    data_parallel_size=1,
    max_num_seqs=4,
    num_gpu_blocks_override=None,
):
    """Helper to build a VllmConfig with the specified parameters."""
    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )
    cache_config = CacheConfig(block_size=block_size)
    if num_gpu_blocks_override is not None:
        cache_config.num_gpu_blocks_override = num_gpu_blocks_override

    compilation_config = CompilationConfig(custom_ops=["all"])

    parallel_config = ParallelConfig(
        data_parallel_size=data_parallel_size,
    )

    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs)

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )
    return vllm_config


class TestDtypeValidation:
    """Tests for the dtype==float16 requirement."""

    def test_float16_accepted(self):
        """float16 dtype passes validation without error."""
        vllm_config = _make_vllm_config(dtype=torch.float16)
        # Should not raise
        TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_float32_rejected(self):
        """float32 dtype raises ValueError."""
        vllm_config = _make_vllm_config(dtype=torch.float32)
        with pytest.raises(ValueError, match="float16"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_bfloat16_rejected(self):
        """bfloat16 dtype raises ValueError."""
        vllm_config = _make_vllm_config(dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="float16"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)


class TestDataParallelValidation:
    """Tests for DP>1 rejection."""

    def test_dp1_accepted(self):
        """data_parallel_size=1 passes validation."""
        vllm_config = _make_vllm_config(data_parallel_size=1)
        # Should not raise
        TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_dp2_rejected(self):
        """data_parallel_size=2 raises ValueError."""
        vllm_config = _make_vllm_config(data_parallel_size=2)
        with pytest.raises(ValueError, match="[Dd]ata.parallel"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)

    def test_dp4_rejected(self):
        """data_parallel_size=4 raises ValueError."""
        vllm_config = _make_vllm_config(data_parallel_size=4)
        with pytest.raises(ValueError, match="[Dd]ata.parallel"):
            TorchSpyrePlatform.check_and_update_config(vllm_config)


class TestNumGpuBlocksOverride:
    """Tests for the num_gpu_blocks_override calculation formula."""

    def test_formula_basic(self):
        """num_gpu_blocks_override = max_num_seqs * ceil(max_model_len / block_size)."""
        vllm_config = _make_vllm_config(
            max_model_len=128,
            block_size=64,
            max_num_seqs=4,
        )
        TorchSpyrePlatform.check_and_update_config(vllm_config)

        blocks_per_seq = math.ceil(128 / 64)  # = 2
        expected = 4 * blocks_per_seq  # = 8
        assert vllm_config.cache_config.num_gpu_blocks_override == expected

    def test_formula_non_divisible(self):
        """Non-divisible max_model_len rounds up blocks_per_seq."""
        vllm_config = _make_vllm_config(
            max_model_len=100,
            block_size=64,
            max_num_seqs=2,
        )
        TorchSpyrePlatform.check_and_update_config(vllm_config)

        blocks_per_seq = math.ceil(100 / 64)  # = 2
        expected = 2 * blocks_per_seq  # = 4
        assert vllm_config.cache_config.num_gpu_blocks_override == expected

    def test_user_override_not_overwritten(self):
        """User-specified num_gpu_blocks_override is preserved."""
        vllm_config = _make_vllm_config(
            max_model_len=128,
            block_size=64,
            max_num_seqs=4,
            num_gpu_blocks_override=100,
        )
        TorchSpyrePlatform.check_and_update_config(vllm_config)

        # User's value should be preserved
        assert vllm_config.cache_config.num_gpu_blocks_override == 100


class TestApplyConfigPlatformDefaults:
    """Tests for apply_config_platform_defaults."""

    def test_sets_compilation_mode_none(self):
        """Compilation mode is forced to NONE."""
        vllm_config = _make_vllm_config()
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.compilation_config.mode == CompilationMode.NONE

    def test_sets_enforce_eager(self):
        """enforce_eager is set to True."""
        vllm_config = _make_vllm_config()
        vllm_config.model_config.enforce_eager = False
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.model_config.enforce_eager is True

    def test_sets_dtype_float16(self):
        """dtype is forced to float16."""
        vllm_config = _make_vllm_config()
        TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
        assert vllm_config.model_config.dtype == torch.float16


class TestCheckMaxModelLen:
    """Tests for check_max_model_len default cap."""

    def test_caps_large_value(self):
        """Values larger than _DEFAULT_DERIVED_MAX_MODEL_LEN are capped."""
        result = TorchSpyrePlatform.check_max_model_len(100000)
        assert result == TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN

    def test_preserves_small_value(self):
        """Values smaller than default are preserved."""
        result = TorchSpyrePlatform.check_max_model_len(512)
        assert result == 512

    def test_exact_default_value(self):
        """Value equal to default is preserved."""
        result = TorchSpyrePlatform.check_max_model_len(
            TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN
        )
        assert result == TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN


class TestWorkerAndSchedulerConfig:
    """Tests for worker and scheduler class configuration."""

    def test_worker_cls_set_to_spyre(self):
        """worker_cls is set to TorchSpyreWorker when 'auto'."""
        vllm_config = _make_vllm_config()
        TorchSpyrePlatform.check_and_update_config(vllm_config)
        assert "TorchSpyreWorker" in vllm_config.parallel_config.worker_cls

    def test_scheduler_cls_set(self):
        """scheduler_cls is set to the vLLM default scheduler."""
        vllm_config = _make_vllm_config()
        TorchSpyrePlatform.check_and_update_config(vllm_config)
        assert "Scheduler" in vllm_config.scheduler_config.scheduler_cls
