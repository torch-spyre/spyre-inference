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
import os
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig
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
    assert torch.accelerator.empty_host_cache.__name__ == "_noop"

    def _raise(*args, **kwargs):
        raise RuntimeError("Cannot access accelerator device when none is available.")

    saved_empty_cache = torch.accelerator.empty_cache
    saved_synchronize = torch.accelerator.synchronize
    saved_empty_host_cache = torch.accelerator.empty_host_cache
    try:
        torch.accelerator.empty_cache = _raise
        torch.accelerator.synchronize = _raise
        torch.accelerator.empty_host_cache = _raise

        _disable_torch_accelerator()

        assert torch.accelerator.empty_cache() is None
        assert torch.accelerator.synchronize() is None
        assert torch.accelerator.empty_host_cache() is None
    finally:
        torch.accelerator.empty_cache = saved_empty_cache
        torch.accelerator.synchronize = saved_synchronize
        torch.accelerator.empty_host_cache = saved_empty_host_cache


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


def test_num_gpu_blocks_override_skipped_for_pooling():
    """Encoder/pooling models have no KV cache — do not invent a block count."""
    from spyre_inference.platform import TorchSpyrePlatform

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1024,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    object.__setattr__(model_config, "runner_type", "pooling")

    cache_config = CacheConfig(block_size=64)
    compilation_config = CompilationConfig(custom_ops=["all"])

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.num_gpu_blocks_override is None


def test_apply_config_sets_pooling_compile_sizes_from_token_cap():
    """Pooling body T lives on compile_sizes; attention L is independent."""
    from unittest.mock import MagicMock

    from vllm.config import CompilationMode

    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = MagicMock()
    vllm_config.model_config.enforce_eager = False
    vllm_config.model_config.runner_type = "pooling"
    vllm_config.model_config.max_model_len = 512
    vllm_config.scheduler_config.max_num_batched_tokens = 512
    vllm_config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    vllm_config.compilation_config.custom_ops = ["all"]
    # Empty list is falsy, so the platform generates pooling defaults.
    # A MagicMock here is truthy and would skip that path (#638).
    vllm_config.compilation_config.compile_sizes = []
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)
    assert vllm_config.compilation_config.compile_sizes == [64, 128, 256, 512]
    assert vllm_config.scheduler_config.max_num_batched_tokens == 512


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


def test_collectives_bypass_the_vllm_custom_op_wrappers():
    """Collectives must reach `SpyreCommunicator` directly, not via torch.ops.vllm.*."""
    from spyre_inference.platform import TorchSpyrePlatform

    assert TorchSpyrePlatform.use_custom_op_collectives() is False


@pytest.mark.parametrize("field", ["data_parallel_size", "pipeline_parallel_size"])
def test_only_tensor_parallelism_is_accepted(field):
    """DP and PP are rejected: the device collectives require TP group == world."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=True, mode=None)
    setattr(vllm_config.parallel_config, field, 2)

    with pytest.raises(ValueError, match="Spyre does not support"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


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
    """Defaults are powers of two up to max_num_seqs, plus one prefill bucket."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = []
    vllm_config.scheduler_config.max_num_seqs = 4
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.compile_sizes == [1, 2, 4, 512]


def test_compile_sizes_default_includes_non_power_of_two_max_num_seqs():
    """A max_num_seqs that is not a power of two still gets its own bucket."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=False, mode=None)
    vllm_config.compilation_config.compile_sizes = []
    vllm_config.scheduler_config.max_num_seqs = 6
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.compile_sizes == [1, 2, 4, 6, 512]


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
    vllm_config.scheduler_config.max_num_seqs = 4
    vllm_config.scheduler_config.max_num_batched_tokens = 32

    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert vllm_config.compilation_config.compile_sizes == [1, 2, 4, 32]
    assert vllm_config.scheduler_config.max_num_batched_tokens == 32


def test_compile_sizes_not_set_when_eager():
    """--enforce-eager should skip compile_sizes generation entirely."""
    from spyre_inference.platform import TorchSpyrePlatform

    vllm_config = _defaults_config(enforce_eager=True, mode=None)
    TorchSpyrePlatform.apply_config_platform_defaults(vllm_config)

    assert not vllm_config.compilation_config.compile_sizes


def test_get_cpu_count_num_cpus_override(monkeypatch):
    """SPYRE_NUM_CPUS takes precedence over any detection."""
    from spyre_inference.threading_config import get_cpu_count

    monkeypatch.setenv("SPYRE_NUM_CPUS", "6")
    count, message = get_cpu_count()
    assert count == 6.0
    assert "SPYRE_NUM_CPUS" in message


def _force_cpu_count(monkeypatch, value):
    """Pin get_cpu_count so threading tests don't depend on the host."""
    import spyre_inference.threading_config as tc

    monkeypatch.setattr(tc, "get_cpu_count", lambda use_logical_cpus=False: (value, "forced"))


def test_configure_threading_overrides_when_enabled(monkeypatch):
    """Enabled (the default) → every threading env is set to cpus/worker."""
    from spyre_inference.threading_config import THREADING_ENVS, configure_threading

    monkeypatch.delenv("SPYRE_UPDATE_THREAD_CONFIG", raising=False)  # default = on
    monkeypatch.setenv("OMP_NUM_THREADS", "128")  # the "wildly high" k8s default
    _force_cpu_count(monkeypatch, 8.0)

    configure_threading(worker_count=2)

    for env in THREADING_ENVS:
        assert os.environ[env] == "4", env  # ceil(8 / 2)


def test_configure_threading_single_worker_uses_full_count(monkeypatch):
    """TP=1 clamps to the detected budget, not the host core count."""
    from spyre_inference.threading_config import configure_threading

    monkeypatch.setenv("SPYRE_UPDATE_THREAD_CONFIG", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    _force_cpu_count(monkeypatch, 8.0)

    configure_threading(worker_count=1)

    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_configure_threading_warn_only_leaves_envs_untouched(monkeypatch):
    """Disabled → the env is left as-is (only a warning is logged)."""
    from spyre_inference.threading_config import configure_threading

    monkeypatch.setenv("SPYRE_UPDATE_THREAD_CONFIG", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    _force_cpu_count(monkeypatch, 8.0)

    configure_threading(worker_count=1)

    assert os.environ["OMP_NUM_THREADS"] == "128"


def test_configure_threading_raises_when_undetectable(monkeypatch):
    """Enabled but no CPU count detectable → fail loudly rather than guess."""
    from spyre_inference.threading_config import configure_threading

    monkeypatch.setenv("SPYRE_UPDATE_THREAD_CONFIG", "1")
    _force_cpu_count(monkeypatch, None)

    with pytest.raises(RuntimeError, match="SPYRE_NUM_CPUS"):
        configure_threading(worker_count=1)
