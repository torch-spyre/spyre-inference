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

"""Unit tests for platform.py configuration logic."""

import torch

from vllm.config import VllmConfig, ModelConfig, CacheConfig
from vllm.config.compilation import CompilationConfig


def _round_up_to_multiple_of_64(value: int) -> int:
    """Helper: the exact rounding formula used in platform.py."""
    return ((value + 63) // 64) * 64


def test_block_size_override_formula():
    """Test the round-up formula used for block_size override.

    This isolates the core logic: ((value + 63) // 64) * 64
    """
    # Values that need rounding up
    assert _round_up_to_multiple_of_64(1) == 64
    assert _round_up_to_multiple_of_64(16) == 64
    assert _round_up_to_multiple_of_64(32) == 64
    assert _round_up_to_multiple_of_64(63) == 64
    assert _round_up_to_multiple_of_64(65) == 128
    assert _round_up_to_multiple_of_64(100) == 128
    assert _round_up_to_multiple_of_64(127) == 128

    # Values already aligned (should stay the same)
    assert _round_up_to_multiple_of_64(64) == 64
    assert _round_up_to_multiple_of_64(128) == 128
    assert _round_up_to_multiple_of_64(256) == 256


def test_block_size_override_default():
    """Test that check_and_update_config overrides block_size when not user-specified.

    The platform should round up non-64-aligned block sizes to the nearest
    multiple of 64 when user_specified_block_size is False (default case).
    """
    from spyre_inference.platform import TorchSpyrePlatform

    # Default block_size=16 (not user-specified)
    cache_config = CacheConfig()
    assert not cache_config.user_specified_block_size
    assert cache_config.block_size == 16

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.block_size % 64 == 0


def test_block_size_override_non_default_value():
    """Test override with a non-standard block_size value.

    This simulates a scenario where block_size=100 should round to 128.
    """
    from spyre_inference.platform import TorchSpyrePlatform

    # Create config with block_size=None, then set to 100
    # This keeps user_specified_block_size=False
    cache_config = CacheConfig(block_size=None)
    assert not cache_config.user_specified_block_size

    object.__setattr__(cache_config, "block_size", 100)

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.block_size == 128


def test_block_size_override_user_specified():
    """Test that even user-specified block_size is overridden when invalid.

    Spyre has a hard requirement for block_size to be a multiple of 64.
    Even when the user (or test harness) explicitly passes an invalid value,
    the platform must correct it to avoid a later ValueError.
    """
    from spyre_inference.platform import TorchSpyrePlatform

    cache_config = CacheConfig(block_size=16)
    assert cache_config.user_specified_block_size, "Should be user-specified"
    assert cache_config.block_size == 16

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.block_size == 64, (
        f"User-specified block_size=16 should be overridden to 64, "
        f"got {vllm_config.cache_config.block_size}"
    )


def test_torch_accelerator_ops_are_noop():
    """Regression for #327: EngineCore shutdown must not crash on accelerator ops."""
    from spyre_inference.platform import _disable_torch_accelerator

    # The module applies the patch at import. A live spyre accelerator would make
    # the real empty_cache() return None too, so assert on identity here.
    assert torch.accelerator.empty_cache.__name__ == "_noop"
    assert torch.accelerator.synchronize.__name__ == "_noop"

    def _raise(*args, **kwargs):
        raise RuntimeError("Cannot access accelerator device when none is available.")

    saved_empty_cache = torch.accelerator.empty_cache
    saved_synchronize = torch.accelerator.synchronize
    try:
        torch.accelerator.empty_cache = _raise
        torch.accelerator.synchronize = _raise

        _disable_torch_accelerator()

        assert torch.accelerator.empty_cache() is None
        assert torch.accelerator.synchronize() is None
    finally:
        torch.accelerator.empty_cache = saved_empty_cache
        torch.accelerator.synchronize = saved_synchronize


def test_block_size_valid_no_override():
    """Test that valid block_size (multiple of 64) is not changed."""
    from spyre_inference.platform import TorchSpyrePlatform

    cache_config = CacheConfig(block_size=128)

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.block_size == 128
