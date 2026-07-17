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

import math
from types import SimpleNamespace

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


def _fake_vllm_config(layer_types, use_text_config=True):
    """Minimal stand-in exposing the attribute path _is_hybrid_attention reads."""
    hf_config = SimpleNamespace(layer_types=layer_types)
    model_config = SimpleNamespace(hf_config=hf_config)
    if use_text_config:
        model_config.hf_text_config = hf_config
    return SimpleNamespace(model_config=model_config)


def test_is_hybrid_attention_true():
    """Interleaved (multiple distinct) layer_types → hybrid."""
    from spyre_inference.platform import TorchSpyrePlatform

    # Gemma-2 style interleaving of two attention types.
    layer_types = ["sliding_attention", "full_attention"] * 13
    assert TorchSpyrePlatform._is_hybrid_attention(_fake_vllm_config(layer_types))


def test_is_hybrid_attention_single_type():
    """A single distinct layer type is homogeneous, not hybrid."""
    from spyre_inference.platform import TorchSpyrePlatform

    assert not TorchSpyrePlatform._is_hybrid_attention(_fake_vllm_config(["full_attention"] * 32))


def test_is_hybrid_attention_missing_layer_types():
    """Models without layer_types (None or absent) are not hybrid."""
    from spyre_inference.platform import TorchSpyrePlatform

    assert not TorchSpyrePlatform._is_hybrid_attention(_fake_vllm_config(None))

    # hf_config with no layer_types attribute at all.
    model_config = SimpleNamespace(hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace())
    cfg = SimpleNamespace(model_config=model_config)
    assert not TorchSpyrePlatform._is_hybrid_attention(cfg)


def test_num_gpu_blocks_override_homogeneous():
    """Non-hybrid models get the plain seqs × blocks/seq pinned block count."""
    from spyre_inference.platform import TorchSpyrePlatform

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
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

    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    blocks_per_seq = math.ceil(
        vllm_config.model_config.max_model_len / vllm_config.cache_config.block_size
    )
    assert vllm_config.cache_config.num_gpu_blocks_override == max_num_seqs * blocks_per_seq


def test_num_gpu_blocks_override_skipped_for_hybrid():
    """Hybrid models leave num_gpu_blocks_override unset so vLLM sizes the cache.

    The single-group formula under-allocates for multi-group KV caches; the
    real requirement depends on vLLM's internal group packing, which isn't
    known at this stage. So we must NOT pin a block count for hybrid models.
    """
    from spyre_inference.platform import TorchSpyrePlatform

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    # Simulate a hybrid model by injecting interleaved layer_types onto the
    # HF config, mirroring Gemma-2/3's sliding/full split (2 distinct types).
    interleaved = ["sliding_attention", "full_attention"] * 13
    model_config.hf_config.layer_types = interleaved
    model_config.hf_text_config.layer_types = interleaved

    cache_config = CacheConfig(block_size=64)
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.num_gpu_blocks_override is None
