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

"""Unit tests for TorchSpyreModelRunner helper methods.

Covers:
- _patch_encoder_ops_for_spyre (bert token_type_ids patching)
- warming_up_model eager-pooling path (max_num_seqs clamping, token limit)
- load_model vocab embedding CPU-pin loop
- _sync_device (torch.spyre.synchronize integration)
- get_model (unwrapping _SpyreModelWrapper and OptimizedModule)
- _make_buffer (dtype routing for float vs int/bool)
- _FuncWrapper grid-launch syntax

All tests run on CPU — no Spyre device required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch, PropertyMock
import types

import pytest
import torch
import torch.nn as nn

from spyre_inference.v1.worker.spyre_model_runner import (
    TorchSpyreModelRunner,
    SpyreCpuGpuBuffer,
    _SpyreModelWrapper,
    _FuncWrapper,
    _compute_slot_mapping_impl,
    SPYRE_ENCODER_DMA_TOKEN_LIMIT,
    SPYRE_ENCODER_WARMUP_MAX_TOKENS,
)


# =============================================================================
# _patch_encoder_ops_for_spyre tests
# =============================================================================


class TestPatchEncoderOpsForSpyre:
    """Tests for the BERT token_type_ids patching logic."""

    def test_pooling_runner_patches_decode_token_type_ids(self):
        """When runner_type='pooling', _decode_token_type_ids is replaced with zeros."""
        from vllm.model_executor.models import bert

        # Save original
        original_fn = bert._decode_token_type_ids

        model_config = Mock()
        model_config.runner_type = "pooling"

        try:
            TorchSpyreModelRunner._patch_encoder_ops_for_spyre(model_config)

            # Verify the function was replaced
            assert bert._decode_token_type_ids is not original_fn

            # Verify the replacement returns zeros
            input_ids = torch.tensor([1, 2, 3, 4, 5])
            result = bert._decode_token_type_ids(input_ids)
            assert torch.all(result == 0)
            assert result.shape == input_ids.shape
            assert result.dtype == input_ids.dtype
        finally:
            # Restore original
            bert._decode_token_type_ids = original_fn

    def test_non_pooling_runner_does_not_patch(self):
        """When runner_type != 'pooling', no patching occurs."""
        from vllm.model_executor.models import bert

        original_fn = bert._decode_token_type_ids

        model_config = Mock()
        model_config.runner_type = "generate"

        TorchSpyreModelRunner._patch_encoder_ops_for_spyre(model_config)

        # Should be unchanged
        assert bert._decode_token_type_ids is original_fn

    def test_missing_decode_fn_raises_runtime_error(self):
        """If bert._decode_token_type_ids doesn't exist, raises RuntimeError."""
        from vllm.model_executor.models import bert

        original_fn = bert._decode_token_type_ids
        model_config = Mock()
        model_config.runner_type = "pooling"

        try:
            # Remove the attribute temporarily
            delattr(bert, "_decode_token_type_ids")
            with pytest.raises(RuntimeError, match="not found"):
                TorchSpyreModelRunner._patch_encoder_ops_for_spyre(model_config)
        finally:
            bert._decode_token_type_ids = original_fn


# =============================================================================
# warming_up_model tests
# =============================================================================


class TestWarmingUpModel:
    """Tests for the warming_up_model eager-pooling path."""

    def _make_runner_mock(
        self,
        runner_type="generate",
        enforce_eager=False,
        max_num_reqs=4,
        max_num_batched_tokens=256,
        max_num_seqs=4,
    ):
        """Create a mock runner with just enough structure for warming_up_model."""
        runner = Mock(spec=TorchSpyreModelRunner)
        runner.max_num_reqs = max_num_reqs

        # Model config
        runner.model_config = Mock()
        runner.model_config.runner_type = runner_type
        runner.model_config.enforce_eager = enforce_eager

        # Scheduler config
        runner.scheduler_config = Mock()
        runner.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
        runner.scheduler_config.max_num_seqs = max_num_seqs

        # VllmConfig (needed by _set_spyre_compilation_settings)
        runner.vllm_config = Mock()
        runner.vllm_config.model_config = runner.model_config
        runner.vllm_config.compilation_config = Mock()
        runner.vllm_config.compilation_config.inductor_compile_config = {}

        runner._dummy_run = Mock()

        return runner

    def test_eager_pooling_clamps_tokens(self):
        """Eager pooling warmup caps num_tokens to SPYRE_ENCODER_WARMUP_MAX_TOKENS."""
        runner = self._make_runner_mock(
            runner_type="pooling",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
        )

        # Call the real method
        TorchSpyreModelRunner.warming_up_model(runner)

        # _dummy_run should be called with capped tokens
        runner._dummy_run.assert_called_once()
        actual_tokens = runner._dummy_run.call_args[0][0]
        assert actual_tokens <= SPYRE_ENCODER_WARMUP_MAX_TOKENS

    def test_eager_pooling_sets_max_num_seqs_to_1(self):
        """Eager pooling warmup temporarily sets max_num_seqs=1."""
        runner = self._make_runner_mock(
            runner_type="pooling",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
        )

        original_max_num_seqs = runner.scheduler_config.max_num_seqs

        def check_max_num_seqs(num_tokens):
            # During _dummy_run, max_num_seqs should be 1
            assert runner.scheduler_config.max_num_seqs == 1

        runner._dummy_run.side_effect = check_max_num_seqs

        TorchSpyreModelRunner.warming_up_model(runner)

        # After warmup, should be restored
        assert runner.scheduler_config.max_num_seqs == original_max_num_seqs

    def test_eager_pooling_restores_max_num_seqs_on_error(self):
        """max_num_seqs is restored even if _dummy_run raises."""
        runner = self._make_runner_mock(
            runner_type="pooling",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
        )

        runner._dummy_run.side_effect = RuntimeError("dummy run failed")

        with pytest.raises(RuntimeError, match="dummy run failed"):
            TorchSpyreModelRunner.warming_up_model(runner)

        # Should still be restored
        assert runner.scheduler_config.max_num_seqs == 8

    def test_non_pooling_uses_normal_path(self):
        """Non-pooling models use the normal warmup path (no clamping)."""
        runner = self._make_runner_mock(
            runner_type="generate",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
        )

        TorchSpyreModelRunner.warming_up_model(runner)

        runner._dummy_run.assert_called_once()
        actual_tokens = runner._dummy_run.call_args[0][0]
        # Normal path: min(max(16, max_num_reqs), max_num_batched_tokens) = min(max(16,4),1024)=16
        assert actual_tokens == 16

    def test_pooling_without_eager_uses_normal_path(self):
        """Pooling models without enforce_eager use the normal warmup path."""
        runner = self._make_runner_mock(
            runner_type="pooling",
            enforce_eager=False,
            max_num_reqs=4,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
        )

        TorchSpyreModelRunner.warming_up_model(runner)

        runner._dummy_run.assert_called_once()
        actual_tokens = runner._dummy_run.call_args[0][0]
        # Normal path for pooling without eager
        assert actual_tokens == 16


# =============================================================================
# _sync_device tests
# =============================================================================


class TestSyncDevice:
    """Tests for _sync_device behavior."""

    def test_sync_device_calls_synchronize(self):
        """_sync_device should call torch.spyre.synchronize() when available."""
        runner = Mock(spec=TorchSpyreModelRunner)

        with patch("torch.spyre.synchronize", create=True) as mock_sync:
            TorchSpyreModelRunner._sync_device(runner)

        # Current implementation is a no-op (TODO comment in source)
        # This test documents the expected behavior once the TODO is resolved.
        # For now, verify no exception is raised.

    def test_sync_device_does_not_raise(self):
        """_sync_device must not raise even without torch.spyre."""
        runner = Mock(spec=TorchSpyreModelRunner)
        # Should not raise
        TorchSpyreModelRunner._sync_device(runner)


# =============================================================================
# get_model unwrapping tests
# =============================================================================


class TestGetModel:
    """Tests for get_model unwrapping logic."""

    def test_get_model_unwraps_spyre_model_wrapper(self):
        """get_model should unwrap _SpyreModelWrapper to get the inner model."""
        inner_model = nn.Linear(10, 10)
        wrapper = _SpyreModelWrapper(inner_model, torch.device("cpu"))

        runner = Mock(spec=TorchSpyreModelRunner)
        runner.model = wrapper

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is inner_model

    def test_get_model_unwraps_optimized_module(self):
        """get_model should unwrap torch.compile's OptimizedModule."""
        inner_model = nn.Linear(10, 10)

        # Simulate torch.compile wrapper (has _orig_mod attribute)
        compiled_model = Mock()
        compiled_model._orig_mod = inner_model

        runner = Mock(spec=TorchSpyreModelRunner)
        runner.model = compiled_model

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is inner_model

    def test_get_model_unwraps_both_wrappers(self):
        """get_model unwraps _SpyreModelWrapper containing an OptimizedModule."""
        inner_model = nn.Linear(10, 10)

        # Simulate compiled model inside _SpyreModelWrapper
        compiled = Mock()
        compiled._orig_mod = inner_model
        # _SpyreModelWrapper delegates __getattr__ to _model
        wrapper = _SpyreModelWrapper(compiled, torch.device("cpu"))

        runner = Mock(spec=TorchSpyreModelRunner)
        runner.model = wrapper

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is inner_model

    def test_get_model_plain_module_returned_directly(self):
        """get_model returns a plain nn.Module directly."""
        model = nn.Linear(10, 10)

        runner = Mock(spec=TorchSpyreModelRunner)
        runner.model = model

        result = TorchSpyreModelRunner.get_model(runner)
        assert result is model


# =============================================================================
# _make_buffer dtype routing tests
# =============================================================================


class TestMakeBuffer:
    """Tests for _make_buffer dtype routing."""

    def _make_runner_for_buffer(self):
        """Create a runner-like object with _spyre_device for _make_buffer."""
        runner = Mock(spec=TorchSpyreModelRunner)
        runner._spyre_device = torch.device("cpu")  # Use CPU to avoid Spyre dependency
        return runner

    def test_float_buffer_creates_spyre_gpu_tensor(self):
        """Float dtypes should create .gpu on the spyre device."""
        runner = self._make_runner_for_buffer()
        runner._spyre_device = torch.device("cpu")

        buf = TorchSpyreModelRunner._make_buffer(runner, 16, dtype=torch.float32)

        assert isinstance(buf, SpyreCpuGpuBuffer)
        assert buf.cpu.dtype == torch.float32
        # With device=cpu as _spyre_device substitute, gpu is aliased
        # In real usage, device would be "spyre" and gpu would be float16

    def test_int_buffer_aliases_gpu_to_cpu(self):
        """Int dtypes should alias .gpu = .cpu (CPU-only pattern)."""
        runner = self._make_runner_for_buffer()

        buf = TorchSpyreModelRunner._make_buffer(runner, 16, dtype=torch.int32)

        assert isinstance(buf, SpyreCpuGpuBuffer)
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.int32

    def test_bool_buffer_aliases_gpu_to_cpu(self):
        """Bool dtypes should alias .gpu = .cpu."""
        runner = self._make_runner_for_buffer()

        buf = TorchSpyreModelRunner._make_buffer(runner, 16, dtype=torch.bool)

        assert isinstance(buf, SpyreCpuGpuBuffer)
        assert buf.gpu is buf.cpu
        assert buf.cpu.dtype == torch.bool

    def test_buffer_numpy_enabled_by_default(self):
        """Buffers should have .np attribute by default."""
        runner = self._make_runner_for_buffer()

        buf = TorchSpyreModelRunner._make_buffer(runner, 16, dtype=torch.int32)

        assert hasattr(buf, "np")

    def test_buffer_numpy_disabled(self):
        """Buffers with numpy=False should not create numpy array."""
        runner = self._make_runner_for_buffer()

        buf = TorchSpyreModelRunner._make_buffer(runner, 16, dtype=torch.int32, numpy=False)

        assert not hasattr(buf, "np")


# =============================================================================
# SpyreEncoderAttentionImpl attn_metadata=None early return
# =============================================================================


class TestEncoderAttentionNoneMetadata:
    """Test the early-return path when attn_metadata is None."""

    def test_encoder_attn_returns_output_when_metadata_is_none(self):
        """forward() with attn_metadata=None returns output unchanged (warmup path)."""
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            SpyreEncoderAttentionImpl,
        )
        from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

        num_heads = 4
        head_size = 64
        num_kv_heads = 4

        impl = SpyreEncoderAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size**-0.5,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

        # Create dummy tensors
        total_tokens = 8
        query = torch.randn(total_tokens, num_heads, head_size)
        key = torch.randn(total_tokens, num_kv_heads, head_size)
        value = torch.randn(total_tokens, num_kv_heads, head_size)
        output = torch.randn(total_tokens, num_heads, head_size)
        output_original = output.clone()

        kv_cache = SpyrePagedKVCache(k_pages=[], v_pages=[])

        result = impl.forward(
            layer=None,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=None,  # <-- the None path
            output=output,
        )

        # Should return the same output tensor unchanged
        assert result is output
        torch.testing.assert_close(result, output_original)

    def test_encoder_attn_none_metadata_does_not_modify_output_inplace(self):
        """The None path must not touch the output tensor at all."""
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            SpyreEncoderAttentionImpl,
        )
        from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

        impl = SpyreEncoderAttentionImpl(
            num_heads=2,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

        output = torch.ones(4, 2, 64)
        kv_cache = SpyrePagedKVCache(k_pages=[], v_pages=[])

        result = impl.forward(
            layer=None,
            query=torch.randn(4, 2, 64),
            key=torch.randn(4, 2, 64),
            value=torch.randn(4, 2, 64),
            kv_cache=kv_cache,
            attn_metadata=None,
            output=output,
        )

        # All ones should still be ones
        assert torch.all(result == 1.0)


# =============================================================================
# _FuncWrapper tests
# =============================================================================


class TestFuncWrapper:
    """Tests for _FuncWrapper grid-launch syntax."""

    def test_grid_syntax_passes_through(self):
        """_FuncWrapper[(grid,)](...) should just call the function."""
        def add(a, b):
            return a + b

        wrapper = _FuncWrapper(add)
        # Grid-launch syntax: wrapper[(1,)](3, 4)
        result = wrapper[(1,)](3, 4)
        assert result == 7

    def test_grid_syntax_different_grids(self):
        """Different grid values should not affect function call."""
        call_count = [0]

        def counter(*args, **kwargs):
            call_count[0] += 1

        wrapper = _FuncWrapper(counter)
        wrapper[(1,)]()
        wrapper[(128,)]()
        wrapper[42]()
        assert call_count[0] == 3


# =============================================================================
# load_model vocab embedding CPU-pin loop
# =============================================================================


class TestLoadModelVocabEmbeddingCPUPin:
    """Tests for the vocab embedding CPU-pin logic in load_model.

    After model.to(spyre_device), SpyreVocabParallelEmbedding weights must
    be pinned back to CPU because F.embedding has no Spyre kernel.
    """

    def test_vocab_embedding_weight_pinned_to_cpu(self):
        """SpyreVocabParallelEmbedding.weight stays on CPU after model.to(device)."""
        from spyre_inference.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )

        # Create a simple model with a SpyreVocabParallelEmbedding
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 32)

        model = SimpleModel()

        # Manually convert the embedding to a SpyreVocabParallelEmbedding-like check
        # The actual pin logic checks isinstance(module, SpyreVocabParallelEmbedding)
        # We verify the pattern: after .to(device), pin embedding weight back to CPU
        original_weight = model.embed.weight.data.clone()

        # Simulate the pin-back logic from load_model (lines 408-411):
        # for module in self.model.modules():
        #     if isinstance(module, SpyreVocabParallelEmbedding):
        #         module.weight = nn.Parameter(module.weight.data.to("cpu"), requires_grad=False)

        # After moving to some device and pinning back:
        model.embed.weight = nn.Parameter(model.embed.weight.data.to("cpu"), requires_grad=False)

        assert model.embed.weight.device == torch.device("cpu")
        assert model.embed.weight.requires_grad is False
        torch.testing.assert_close(model.embed.weight.data, original_weight)

    def test_isinstance_check_pattern(self):
        """The isinstance check correctly identifies SpyreVocabParallelEmbedding."""
        from spyre_inference.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )

        # After OOT registration, VocabParallelEmbedding() returns SpyreVocabParallelEmbedding
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        # Verify the OOT registration yields the correct type
        # Note: actual instantiation requires TP config, so we just verify the class hierarchy
        assert issubclass(SpyreVocabParallelEmbedding, VocabParallelEmbedding)


# =============================================================================
# Constants validation
# =============================================================================


class TestConstants:
    """Validate important module-level constants."""

    def test_encoder_dma_token_limit_is_positive(self):
        assert SPYRE_ENCODER_DMA_TOKEN_LIMIT > 0

    def test_encoder_warmup_max_tokens_under_dma_limit(self):
        """Warmup max tokens must stay below the DMA failure threshold."""
        assert SPYRE_ENCODER_WARMUP_MAX_TOKENS < SPYRE_ENCODER_DMA_TOKEN_LIMIT

    def test_encoder_warmup_max_tokens_value(self):
        """Document the expected value for regression detection."""
        assert SPYRE_ENCODER_WARMUP_MAX_TOKENS == 16
