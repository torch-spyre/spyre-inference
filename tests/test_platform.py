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


# ---------------------------------------------------------------------------
# max_model_len / max_num_seqs defaulting hooks
# ---------------------------------------------------------------------------


def test_check_max_model_len_caps_large_derived_value():
    """The hook returns the smaller of the derived value and the Spyre cap."""
    from spyre_inference.platform import TorchSpyrePlatform

    cap = TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN
    assert TorchSpyrePlatform.check_max_model_len(131072) == cap
    assert TorchSpyrePlatform.check_max_model_len(cap + 1) == cap


def test_check_max_model_len_passes_through_small_values():
    """A model whose max_position_embeddings is already ≤ cap is untouched."""
    from spyre_inference.platform import TorchSpyrePlatform

    cap = TorchSpyrePlatform._DEFAULT_DERIVED_MAX_MODEL_LEN
    assert TorchSpyrePlatform.check_max_model_len(cap) == cap
    assert TorchSpyrePlatform.check_max_model_len(512) == 512


def test_pre_register_and_update_installs_and_is_idempotent():
    """`pre_register_and_update` should wrap `_set_default_max_num_seqs_...`
    exactly once and mark the wrapper so subsequent calls are no-ops."""
    from vllm.engine.arg_utils import EngineArgs

    from spyre_inference.platform import TorchSpyrePlatform

    original_attr_name = "_set_default_max_num_seqs_and_batched_tokens_args"
    pristine = getattr(EngineArgs, original_attr_name)
    try:
        # If a previous test already installed our wrapper, `functools.wraps`
        # stashed the true original on `__wrapped__`; reset to that so this
        # test observes a fresh install.
        setattr(EngineArgs, original_attr_name, getattr(pristine, "__wrapped__", pristine))

        TorchSpyrePlatform.pre_register_and_update()
        first = getattr(EngineArgs, original_attr_name)
        assert getattr(first, "_spyre_patched", False)

        TorchSpyrePlatform.pre_register_and_update()
        second = getattr(EngineArgs, original_attr_name)
        assert second is first, "second call should be a no-op"
    finally:
        setattr(EngineArgs, original_attr_name, pristine)


def test_pre_register_and_update_lowers_default_max_num_seqs():
    """The wrapper lowers `max_num_seqs` only when the user didn't supply it."""
    from types import SimpleNamespace

    from vllm.engine.arg_utils import EngineArgs

    from spyre_inference.platform import TorchSpyrePlatform

    original_attr_name = "_set_default_max_num_seqs_and_batched_tokens_args"
    pristine = getattr(EngineArgs, original_attr_name)

    def fake_original(self, usage_context, model_config, parallel_config):
        # Simulate vLLM's own defaulter filling `None` with 256.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256

    try:
        setattr(EngineArgs, original_attr_name, fake_original)
        TorchSpyrePlatform.pre_register_and_update()
        patched = getattr(EngineArgs, original_attr_name)

        unset = SimpleNamespace(max_num_seqs=None)
        patched(unset, None, None, None)
        assert unset.max_num_seqs == TorchSpyrePlatform._DEFAULT_MAX_NUM_SEQS

        user_set = SimpleNamespace(max_num_seqs=64)
        patched(user_set, None, None, None)
        assert user_set.max_num_seqs == 64
    finally:
        setattr(EngineArgs, original_attr_name, pristine)


# ---------------------------------------------------------------------------
# VFIO DMA-entry budget
# ---------------------------------------------------------------------------


def test_vfio_dma_entry_limit_env_override(monkeypatch):
    """SPYRE_VFIO_DMA_ENTRY_LIMIT wins over sysfs / fallback."""
    from spyre_inference.platform import TorchSpyrePlatform

    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "1234")
    assert TorchSpyrePlatform._vfio_dma_entry_limit() == 1234


def test_vfio_dma_entry_limit_bad_env_ignored(monkeypatch):
    """Non-integer env var falls through to sysfs / default."""
    from spyre_inference.platform import TorchSpyrePlatform

    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "not-a-number")
    result = TorchSpyrePlatform._vfio_dma_entry_limit()
    assert isinstance(result, int) and result > 0


def test_max_kv_blocks_scales_with_budget_and_layers(monkeypatch):
    """Budget/(2*num_layers) with a fixed limit and multiplier."""
    from spyre_inference.platform import TorchSpyrePlatform

    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "800")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "1.0")

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=64),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    num_layers = model_config.get_num_layers(vllm_config.parallel_config)

    expected = 800 // (2 * num_layers)
    assert TorchSpyrePlatform._max_kv_blocks_for_dma_budget(vllm_config) == expected


def test_max_kv_blocks_returns_at_least_one(monkeypatch):
    """A tiny budget still yields a usable (>= 1) num_blocks cap."""
    from spyre_inference.platform import TorchSpyrePlatform

    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "1")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "0.001")

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=64),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    assert TorchSpyrePlatform._max_kv_blocks_for_dma_budget(vllm_config) == 1


def test_max_kv_blocks_survives_mocked_get_num_layers(monkeypatch):
    """Upstream `tests/v1/attention/utils.py` monkey-patches `get_num_layers`
    with a zero-arg lambda; our budget math must not depend on
    `get_num_layers` (we use the public `get_total_num_hidden_layers`
    instead), so mocking it out has no effect on the returned budget.
    """
    import types

    from spyre_inference.platform import TorchSpyrePlatform

    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "800")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "1.0")

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=1,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=64),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    baseline = TorchSpyrePlatform._max_kv_blocks_for_dma_budget(vllm_config)

    # Install the upstream mock; the budget must not change.
    vllm_config.model_config.get_num_layers = types.MethodType(
        lambda self: 1, vllm_config.model_config
    )
    assert TorchSpyrePlatform._max_kv_blocks_for_dma_budget(vllm_config) == baseline


# ---------------------------------------------------------------------------
# DMA-budget clamp in check_and_update_config
# ---------------------------------------------------------------------------


def _make_config(max_model_len: int, max_num_seqs: int):
    """Minimal VllmConfig where max_num_seqs is settable post-hoc."""
    from vllm.config.scheduler import SchedulerConfig

    model_config = ModelConfig(
        model="Qwen/Qwen3-0.6B",
        max_model_len=max_model_len,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        max_num_batched_tokens=max(max_num_seqs, 8),
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=64),
        scheduler_config=scheduler_config,
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )


# `VllmConfig.__post_init__` (vllm/config/vllm.py:1335) already calls
# `check_and_update_config`, so the assertions below observe the resolved
# state after a single implicit pass — no explicit call is needed.


def test_dma_budget_no_clamp_when_under_budget(monkeypatch):
    """When desired num_blocks fits the DMA budget, both knobs stay put."""
    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "1_000_000")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "1.0")

    vllm_config = _make_config(max_model_len=512, max_num_seqs=8)

    assert vllm_config.model_config.max_model_len == 512
    assert vllm_config.scheduler_config.max_num_seqs == 8


def test_dma_budget_clamps_max_num_seqs_first(monkeypatch):
    """Over-budget with room for ≥1 seq: max_num_seqs shrinks, max_model_len untouched."""
    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "800")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "1.0")

    # max_model_len=512, block_size=64 → blocks_per_seq=8. 32 seqs → 256
    # blocks; well over 800/(2×num_layers) for any real num_layers ≥ 2.
    vllm_config = _make_config(max_model_len=512, max_num_seqs=32)

    assert vllm_config.model_config.max_model_len == 512
    assert vllm_config.scheduler_config.max_num_seqs < 32
    assert vllm_config.scheduler_config.max_num_seqs >= 1


def test_dma_budget_clamps_max_model_len_when_single_seq_too_big(monkeypatch):
    """Budget so small that even one seq can't fit: max_model_len is clamped and seqs=1."""
    # Budget=1 block; whatever block_size CpuPlatform picks, max_model_len
    # collapses to that single block and max_num_seqs drops to 1.
    monkeypatch.setenv("SPYRE_VFIO_DMA_ENTRY_LIMIT", "10")
    monkeypatch.setenv("SPYRE_KV_DMA_BLOCK_FACTOR", "1.0")

    vllm_config = _make_config(max_model_len=8192, max_num_seqs=4)

    assert vllm_config.scheduler_config.max_num_seqs == 1
    assert vllm_config.model_config.max_model_len < 8192
    # Post-clamp max_model_len is exactly one block's worth of tokens.
    assert vllm_config.model_config.max_model_len % vllm_config.cache_config.block_size == 0
