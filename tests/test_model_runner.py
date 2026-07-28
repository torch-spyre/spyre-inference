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

"""Unit tests for _SpyreModelWrapper and TorchSpyreModelRunner internals.

Coverage targets:
- _SpyreModelWrapper.__call__: int-conversion, output-to-CPU, RoPE priming
- _SpyreModelWrapper.compute_logits: H2D/D2H conversion around lm_head
- _SpyreModelWrapper attribute proxy (__getattr__/__setattr__)
- TorchSpyreModelRunner.warming_up_model: eager-pooling path logic
- TorchSpyreModelRunner._patch_encoder_ops_for_spyre: bert segment patch
- TorchSpyreModelRunner._sync_device: calls torch.spyre.synchronize
- SpyreEncoderAttentionImpl.forward(attn_metadata=None): early-return path
- _FuncWrapper: Triton grid-launch syntax emulation
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.modelrunner


class TestSpyreModelWrapperAttributeProxy:
    """Test attribute proxy behavior of _SpyreModelWrapper."""

    def test_getattr_delegates_to_model(self):
        """__getattr__ should delegate to the wrapped model."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = nn.Linear(4, 8)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        assert wrapper.weight is model.weight
        assert wrapper.bias is model.bias
        assert wrapper.in_features == 4
        assert wrapper.out_features == 8

    def test_setattr_delegates_to_model(self):
        """__setattr__ should set attributes on the wrapped model."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = nn.Linear(4, 8)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        wrapper.custom_attr = "test_value"
        assert model.custom_attr == "test_value"

    def test_internal_attributes_stored_on_wrapper(self):
        """_model and _spyre_device should be on the wrapper itself."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = nn.Linear(4, 8)
        device = torch.device("cpu")
        wrapper = _SpyreModelWrapper(model, device)

        assert object.__getattribute__(wrapper, "_model") is model
        assert object.__getattribute__(wrapper, "_spyre_device") == device


class TestSpyreModelWrapperCall:
    """Test the __call__ method of _SpyreModelWrapper."""

    def test_call_invokes_model_forward(self):
        """Calling the wrapper should invoke the wrapped model."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        wrapper(input_ids=input_ids)

        model.assert_called_once()

    def test_call_converts_int32_to_int64(self):
        """Int32 tensors should be converted to int64."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        int32_input = torch.tensor([1, 2, 3], dtype=torch.int32)
        wrapper(input_ids=int32_input)

        call_kwargs = model.call_args[1]
        assert call_kwargs["input_ids"].dtype == torch.int64

    def test_call_preserves_float_kwargs(self):
        """Float tensor kwargs should pass through unchanged."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        float_input = torch.randn(2, 4, dtype=torch.float16)
        wrapper(hidden_states=float_input)

        call_kwargs = model.call_args[1]
        assert call_kwargs["hidden_states"].dtype == torch.float16

    def test_call_handles_none_kwargs(self):
        """None values in kwargs should pass through without error."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        wrapper(input_ids=torch.tensor([1], dtype=torch.int64), positions=None)

        call_kwargs = model.call_args[1]
        assert call_kwargs["positions"] is None

    def test_call_output_on_cpu(self):
        """Model output should be converted to CPU."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        # Model returns a CPU tensor (simulating the convert() result)
        expected = torch.randn(3, 8)
        model.return_value = expected
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        result = wrapper(input_ids=torch.tensor([1, 2, 3], dtype=torch.int64))
        assert result.device.type == "cpu"


class TestSpyreModelWrapperComputeLogits:
    """Test _SpyreModelWrapper.compute_logits method."""

    def test_compute_logits_invokes_model_compute_logits(self):
        """compute_logits should delegate to the underlying model's compute_logits."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        # Return a CPU tensor (simulate lm_head returning CPU logits)
        logits = torch.randn(4, 100, dtype=torch.float16)
        model.compute_logits.return_value = logits
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        hidden_states = torch.randn(4, 16, dtype=torch.float16)
        result = wrapper.compute_logits(hidden_states)

        model.compute_logits.assert_called_once()

    def test_compute_logits_passes_args(self):
        """compute_logits should forward additional args/kwargs."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        logits = torch.randn(4, 100, dtype=torch.float16)
        model.compute_logits.return_value = logits
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"))

        hidden_states = torch.randn(4, 16, dtype=torch.float16)
        extra_arg = torch.randn(4, 8)
        wrapper.compute_logits(hidden_states, extra_arg, some_kwarg="value")

        # Verify args were passed through
        args, kwargs = model.compute_logits.call_args
        assert args[1] is extra_arg
        assert kwargs["some_kwarg"] == "value"


class TestSpyreModelWrapperRope:
    """Test RoPE priming in _SpyreModelWrapper."""

    def test_prime_rope_no_modules_is_noop(self):
        """With no RoPE modules, _prime_rope_rotation should be a no-op."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[])

        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        # Should not raise even without forward context
        wrapper(input_ids=torch.tensor([1, 2, 3], dtype=torch.int64), positions=positions)

    def test_prime_rope_none_positions_is_noop(self):
        """With None positions, _prime_rope_rotation should be a no-op."""
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        model = MagicMock()
        model.return_value = torch.randn(2, 4)

        rope_mock = MagicMock()
        wrapper = _SpyreModelWrapper(model, torch.device("cpu"), rope_modules=[rope_mock])

        wrapper(input_ids=torch.tensor([1], dtype=torch.int64), positions=None)
        rope_mock.gather_rotation.assert_not_called()


class TestWarmingUpModel:
    """Test warming_up_model logic.

    These tests exercise the branching logic in warming_up_model without
    requiring a full model load. We mock internal state and verify:
    - Normal path calls _dummy_run with correct token count
    - Eager pooling path clamps tokens and sets max_num_seqs=1
    - max_num_seqs is restored after pooling warmup
    """

    def _make_mock_runner(self, runner_type="generate", enforce_eager=False,
                          max_num_reqs=4, max_num_batched_tokens=256,
                          max_num_seqs=4):
        """Create a minimal mock TorchSpyreModelRunner for warmup testing."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = MagicMock(spec=TorchSpyreModelRunner)
        runner.warming_up_model = TorchSpyreModelRunner.warming_up_model.__get__(runner)

        # Set up config mocks
        runner.max_num_reqs = max_num_reqs

        scheduler_config = MagicMock()
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens
        scheduler_config.max_num_seqs = max_num_seqs
        runner.scheduler_config = scheduler_config

        model_config = MagicMock()
        model_config.runner_type = runner_type
        model_config.enforce_eager = enforce_eager
        runner.model_config = model_config

        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        runner.vllm_config = vllm_config

        runner._dummy_run = MagicMock()
        runner._spyre_device = torch.device("cpu")

        return runner

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_normal_path_calls_dummy_run(self, mock_ctx):
        """Normal (non-pooling) warmup should call _dummy_run with min(max(16, max_num_reqs), max_batched)."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="generate",
            enforce_eager=False,
            max_num_reqs=4,
            max_num_batched_tokens=256,
        )

        runner.warming_up_model()

        # num_tokens = min(max(16, 4), 256) = 16
        runner._dummy_run.assert_called_once_with(16)

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_normal_path_with_large_max_num_reqs(self, mock_ctx):
        """With large max_num_reqs, num_tokens should be capped by max_batched_tokens."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="generate",
            enforce_eager=True,
            max_num_reqs=1024,
            max_num_batched_tokens=256,
        )

        runner.warming_up_model()

        # num_tokens = min(max(16, 1024), 256) = 256
        runner._dummy_run.assert_called_once_with(256)

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_eager_pooling_path_clamps_tokens(self, mock_ctx):
        """Eager pooling path should clamp num_tokens to SPYRE_ENCODER_WARMUP_MAX_TOKENS."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="pooling",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=256,
            max_num_seqs=8,
        )

        runner.warming_up_model()

        # SPYRE_ENCODER_WARMUP_MAX_TOKENS = 16
        # num_tokens = min(min(max(16, 4), 256), 16) = 16
        runner._dummy_run.assert_called_once_with(16)

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_eager_pooling_path_sets_max_num_seqs_to_1(self, mock_ctx):
        """Eager pooling path should temporarily set max_num_seqs=1."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="pooling",
            enforce_eager=True,
            max_num_reqs=4,
            max_num_batched_tokens=256,
            max_num_seqs=8,
        )

        # Track the max_num_seqs value when _dummy_run is called
        captured_max_num_seqs = []

        def capture_dummy_run(n):
            captured_max_num_seqs.append(runner.scheduler_config.max_num_seqs)

        runner._dummy_run.side_effect = capture_dummy_run

        runner.warming_up_model()

        # During _dummy_run, max_num_seqs should be 1
        assert captured_max_num_seqs == [1]

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_eager_pooling_path_restores_max_num_seqs(self, mock_ctx):
        """Eager pooling path should restore max_num_seqs after warmup."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="pooling",
            enforce_eager=True,
            max_num_seqs=8,
        )

        runner.warming_up_model()

        # After warmup, max_num_seqs should be restored
        assert runner.scheduler_config.max_num_seqs == 8

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_eager_pooling_restores_on_exception(self, mock_ctx):
        """max_num_seqs should be restored even if _dummy_run raises."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="pooling",
            enforce_eager=True,
            max_num_seqs=8,
        )
        runner._dummy_run.side_effect = RuntimeError("test error")

        with pytest.raises(RuntimeError, match="test error"):
            runner.warming_up_model()

        # max_num_seqs should still be restored
        assert runner.scheduler_config.max_num_seqs == 8

    @patch("spyre_inference.v1.worker.spyre_model_runner._set_spyre_compilation_settings")
    def test_pooling_without_eager_uses_normal_path(self, mock_ctx):
        """Pooling with enforce_eager=False should use the normal warmup path."""
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        runner = self._make_mock_runner(
            runner_type="pooling",
            enforce_eager=False,
            max_num_reqs=4,
            max_num_batched_tokens=256,
            max_num_seqs=8,
        )

        runner.warming_up_model()

        # Should NOT have changed max_num_seqs (normal path)
        # num_tokens = min(max(16, 4), 256) = 16
        runner._dummy_run.assert_called_once_with(16)
        assert runner.scheduler_config.max_num_seqs == 8


class TestSyncDevice:
    """Test _sync_device method calls torch.spyre.synchronize."""

    def test_sync_device_calls_synchronize(self):
        """_sync_device should call torch.spyre.synchronize(device)."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = MagicMock(spec=TorchSpyreModelRunner)
        runner._spyre_device = torch.device("cpu")
        runner._sync_device = TorchSpyreModelRunner._sync_device.__get__(runner)

        with patch("torch.spyre.synchronize", create=True) as mock_sync:
            runner._sync_device()
            mock_sync.assert_called_once_with(runner._spyre_device)


class TestPatchEncoderOpsForSpyre:
    """Test _patch_encoder_ops_for_spyre static method."""

    def test_patches_token_type_ids_for_pooling(self):
        """For pooling runner_type, _decode_token_type_ids should be patched to zeros."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
        import vllm.model_executor.models.bert as bert_module

        model_config = MagicMock()
        model_config.runner_type = "pooling"

        # Save original to restore after test
        original_fn = bert_module._decode_token_type_ids
        try:
            TorchSpyreModelRunner._patch_encoder_ops_for_spyre(model_config)

            # After patching, the function should return zeros
            input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
            result = bert_module._decode_token_type_ids(input_ids)
            assert (result == 0).all()
            assert result.shape == input_ids.shape
        finally:
            # Restore original to avoid polluting other tests
            bert_module._decode_token_type_ids = original_fn

    def test_skips_patch_for_non_pooling(self):
        """For non-pooling runner_type, _decode_token_type_ids should NOT be patched."""
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
        import vllm.model_executor.models.bert as bert_module

        model_config = MagicMock()
        model_config.runner_type = "generate"

        original_fn = bert_module._decode_token_type_ids
        TorchSpyreModelRunner._patch_encoder_ops_for_spyre(model_config)

        # Should still be the original function (not patched)
        assert bert_module._decode_token_type_ids is original_fn


class TestEncoderAttentionMetadataNone:
    """Test SpyreEncoderAttentionImpl.forward returns output when attn_metadata=None."""

    def test_returns_output_when_metadata_is_none(self):
        """forward() with attn_metadata=None should return output unchanged."""
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            SpyreEncoderAttentionImpl,
        )
        from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

        impl = SpyreEncoderAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=64**-0.5,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

        # Create dummy tensors
        output = torch.randn(4, 8, 64, dtype=torch.float16)
        query = torch.randn(4, 8, 64, dtype=torch.float16)
        key = torch.randn(4, 8, 64, dtype=torch.float16)
        value = torch.randn(4, 8, 64, dtype=torch.float16)
        kv_cache = SpyrePagedKVCache(k_pages=[], v_pages=[])

        result = impl.forward(
            layer=None,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=None,
            output=output,
        )

        # Should return the same output tensor unchanged
        assert result is output


class TestFuncWrapper:
    """Test _FuncWrapper used for slot mapping kernel dispatch."""

    def test_func_wrapper_grid_syntax(self):
        """_FuncWrapper should support kernel[(grid,)](...) → kernel(...) syntax."""
        from spyre_inference.v1.worker.spyre_model_runner import _FuncWrapper

        call_count = 0

        def my_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return args, kwargs

        wrapped = _FuncWrapper(my_func)
        result = wrapped[(1,)](42, key="value")
        assert call_count == 1
        assert result == ((42,), {"key": "value"})

    def test_func_wrapper_different_grids(self):
        """Grid argument should be ignored (just returns the function)."""
        from spyre_inference.v1.worker.spyre_model_runner import _FuncWrapper

        def my_func(x):
            return x * 2

        wrapped = _FuncWrapper(my_func)
        assert wrapped[(1,)](5) == 10
        assert wrapped[(32, 32)](7) == 14
        assert wrapped[1](3) == 6
