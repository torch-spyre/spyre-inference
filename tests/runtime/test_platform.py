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

import pytest

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
    """Non-hybrid models get seqs × blocks/seq pinned, plus the null block."""
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
    assert vllm_config.cache_config.num_gpu_blocks_override == max_num_seqs * blocks_per_seq + 1


def test_num_gpu_blocks_override_skipped_for_hybrid():
    """Hybrid models leave num_gpu_blocks_override unset so vLLM sizes the cache."""
    from spyre_inference.platform import TorchSpyrePlatform

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    # Simulate a hybrid model with two interleaved attention types.
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


def _fake_pad_config(head_dim=64, num_heads=8, *, transformers_backend=False, **rope_attrs):
    """Minimal vllm_config exposing everything _maybe_pad_head_dim touches.

    hf_config and hf_text_config share one object (the common case). Returns
    (vllm_config, hf_config, model_config) so tests can assert on mutations.
    """
    hf_config = SimpleNamespace(
        num_attention_heads=num_heads,
        hidden_size=head_dim * num_heads,
        head_dim=head_dim,
        **rope_attrs,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(head_size=head_dim),
        using_transformers_backend=lambda: transformers_backend,
    )
    return SimpleNamespace(model_config=model_config), hf_config, model_config


def test_pad_head_dim_full_rotary_pads():
    """Full neox rotary (rotary_dim == head_dim): head_dim 64 -> 128 as normal.

    transformers 5.x carries all RoPE config in ``rope_parameters``; absence of a
    partial-rotary factor there means full rotary.
    """
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config, hf, mc = _fake_pad_config(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}
    )
    TorchSpyrePlatform._maybe_pad_head_dim(vllm_config)

    assert hf.head_dim == 128
    assert hf._spyre_orig_head_dim == 64
    assert mc.model_arch_config.head_size == 128


def test_pad_head_dim_pads_on_the_transformers_backend():
    """Regression for #597: this used to return early for the Transformers backend."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config, hf, mc = _fake_pad_config(
        head_dim=4,
        num_heads=4,
        transformers_backend=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    TorchSpyrePlatform._maybe_pad_head_dim(vllm_config)

    assert hf.head_dim == 128
    assert hf._spyre_orig_head_dim == 4
    assert mc.model_arch_config.head_size == 128


def test_pad_head_dim_rejects_rope_dim():
    """An absolute rope_dim override won't scale to the padded width -> fail fast."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config, hf, mc = _fake_pad_config(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "rope_dim": 64},
    )
    with pytest.raises(NotImplementedError, match="rope_dim"):
        TorchSpyrePlatform._maybe_pad_head_dim(vllm_config)

    # Bail out before mutating anything.
    assert hf.head_dim == 64
    assert not hasattr(hf, "_spyre_orig_head_dim")
    assert mc.model_arch_config.head_size == 64


def test_pad_head_dim_rejects_partial_rotary_factor():
    """Partial rotary (GPTNeoX/Phi shape) lands in rope_parameters in 5.x -> fail fast."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config, hf, _ = _fake_pad_config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    with pytest.raises(NotImplementedError, match="partial_rotary_factor"):
        TorchSpyrePlatform._maybe_pad_head_dim(vllm_config)
    assert hf.head_dim == 64


def test_reduced_rotary_dim_reason_branches():
    """Unit-test the detector directly. All RoPE config lives in rope_parameters (5.x)."""
    from spyre_inference.custom_ops.head_pad import reduced_rotary_dim_reason

    ns = SimpleNamespace
    # Full rotary / no rope config -> not reduced.
    assert reduced_rotary_dim_reason(ns()) is None
    assert reduced_rotary_dim_reason(ns(rope_parameters={"rope_type": "default"})) is None
    assert reduced_rotary_dim_reason(ns(rope_parameters={"partial_rotary_factor": 1.0})) is None

    # Reductions below head_dim.
    assert "rope_dim" in reduced_rotary_dim_reason(ns(rope_parameters={"rope_dim": 64}))
    assert "partial_rotary_factor" in reduced_rotary_dim_reason(
        ns(rope_parameters={"partial_rotary_factor": 0.25})
    )


def test_pad_head_dim_aligned_model_with_rope_dim_not_rejected():
    """head_dim already 128-aligned (e.g. MLA rope_dim models): guard must not fire."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config, hf, _ = _fake_pad_config(
        head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "rope_dim": 64},
    )
    TorchSpyrePlatform._maybe_pad_head_dim(vllm_config)  # returns early, no raise

    assert hf.head_dim == 128
    assert not hasattr(hf, "_spyre_orig_head_dim")


def _defaults_config(enforce_eager: bool, mode) -> VllmConfig:
    """Minimal VllmConfig for exercising apply_config_platform_defaults."""
    from vllm.config.compilation import CompilationMode

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )
    compilation_config = CompilationConfig()
    if mode is not None:
        compilation_config.mode = getattr(CompilationMode, mode)

    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        compilation_config=compilation_config,
    )


def test_compile_default_is_stock_when_not_eager():
    """--enforce-eager off ⇒ default to STOCK_TORCH_COMPILE, keeping CustomOp dispatch."""
    from vllm.config.compilation import CompilationMode

    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
    assert "all" in vllm_config.compilation_config.custom_ops


def test_enforce_eager_forces_none():
    """--enforce-eager on ⇒ CompilationMode.NONE (everything eager)."""
    from vllm.config.compilation import CompilationMode

    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=True, mode=None)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.NONE


def test_enforce_eager_is_the_only_eager_switch():
    """An explicit mode=NONE without --enforce-eager is still overridden to STOCK."""
    from vllm.config.compilation import CompilationMode

    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode="NONE")
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE


def test_raise_dynamo_recompile_limits_survives_a_clobber():
    """torch_spyre's autoload lowers cache_size_limit to 1024; re-asserting must win."""
    import torch._dynamo

    from spyre_inference.platform import _raise_dynamo_recompile_limits

    saved = (
        torch._dynamo.config.cache_size_limit,
        torch._dynamo.config.accumulated_recompile_limit,
    )
    try:
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_recompile_limit = 256

        _raise_dynamo_recompile_limits()

        assert torch._dynamo.config.cache_size_limit == 100000
        assert torch._dynamo.config.accumulated_recompile_limit == 100000
    finally:
        (
            torch._dynamo.config.cache_size_limit,
            torch._dynamo.config.accumulated_recompile_limit,
        ) = saved


def test_worker_reasserts_recompile_limits_after_autoload():
    """The re-assert must come *after* torch_spyre._autoload(), or it is undone."""
    import inspect

    from spyre_inference.v1.worker import spyre_worker

    src = inspect.getsource(spyre_worker.TorchSpyreWorker.init_device)
    assert src.index("torch_spyre._autoload()") < src.index("_raise_dynamo_recompile_limits()")


def test_compile_sizes_default_generated():
    """When user doesn't set compile_sizes, the platform generates default buckets."""
    from spyre_inference.platform import SPYRE_MAX_PREFILL_CHUNK, TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    sizes = vllm_config.compilation_config.compile_sizes
    assert sizes, "compile_sizes should not be empty"
    assert sizes == sorted(sizes), "compile_sizes should be sorted ascending"
    assert sizes[0] == 1, "smallest bucket should be 1"
    assert max(sizes) <= SPYRE_MAX_PREFILL_CHUNK


def test_compile_sizes_high_tier_is_coarse_up_to_4k():
    """Above 512 the default buckets step coarsely (512-token stride) up to the
    4096 prefill-chunk cap, so raising the chunk cap adds only a few warmup
    buckets rather than hundreds. The <=512 tier stays dense (unchanged)."""
    from spyre_inference.platform import SPYRE_MAX_PREFILL_CHUNK, TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = []
    vllm_config.scheduler_config.max_num_batched_tokens = 4096

    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    sizes = vllm_config.compilation_config.compile_sizes

    assert max(sizes) == min(4096, SPYRE_MAX_PREFILL_CHUNK)
    assert vllm_config.scheduler_config.max_num_batched_tokens == max(sizes)
    high = [s for s in sizes if s > 512]
    # Coarse 512-stride tier; well under the ~220 buckets a 16-stride would give.
    assert high == [1024, 1536, 2048, 2560, 3072, 3584, 4096]
    # The <=512 tier is unchanged: dense 16-stride from 256.
    assert 512 in sizes and 496 in sizes


def test_compile_sizes_user_provided_respected():
    """User-supplied compile_sizes must not be overwritten."""
    from spyre_inference.platform import TorchSpyrePlatform

    user_sizes = [32, 64, 128]
    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = user_sizes

    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.compile_sizes == user_sizes


def test_compile_sizes_user_provided_caps_scheduler():
    """max_num_batched_tokens is capped to max(user-supplied compile_sizes)."""
    from spyre_inference.platform import TorchSpyrePlatform

    user_sizes = [16, 32, 64]
    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = user_sizes

    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.scheduler_config.max_num_batched_tokens == 64


def test_compile_sizes_default_caps_at_max_num_batched_tokens():
    """Default bucket generation respects max_num_batched_tokens as upper bound."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = []
    vllm_config.scheduler_config.max_num_batched_tokens = 32

    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    sizes = vllm_config.compilation_config.compile_sizes
    assert max(sizes) <= 32
    assert vllm_config.scheduler_config.max_num_batched_tokens == max(sizes)


def test_compile_sizes_not_set_when_eager():
    """--enforce-eager should skip compile_sizes generation entirely."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=True, mode=None)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert not vllm_config.compilation_config.compile_sizes
