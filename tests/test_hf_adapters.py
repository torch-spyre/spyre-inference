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

"""Unit tests for spyre_inference/hf_adapters.py.

Tests the helper functions and utilities in the hf_adapters module:
- _qk_expand_matrix: interleaved expand matrix for Q/K RoPE padding
- _SpyreRotaryEmbedding: wrapper for precomputed rotation matrices
- _make_spyre_apply_rotary: matmul-based RoPE replacement with optional padding
- _fix_generic_config: re-resolves generic PretrainedConfig

These are unit tests that exercise the logic on CPU without needing a full
model or Spyre device.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _qk_expand_matrix
# ---------------------------------------------------------------------------


class TestQkExpandMatrix:
    """Tests for the interleaved expand matrix used for stick-aligned RoPE."""

    def test_square_identity_when_same_size(self):
        """When orig_hd == padded_hd, result is the identity matrix."""
        from spyre_inference.hf_adapters import _qk_expand_matrix

        m = _qk_expand_matrix(128, 128)
        assert m.shape == (128, 128)
        torch.testing.assert_close(m, torch.eye(128))

    def test_correct_shape_when_padded(self):
        """Expand matrix has shape (orig_hd, padded_hd)."""
        from spyre_inference.hf_adapters import _qk_expand_matrix

        m = _qk_expand_matrix(96, 128)
        assert m.shape == (96, 128)

    def test_interleaved_structure(self):
        """Top-left and bottom-right blocks are identity submatrices.

        For orig_hd=4, padded_hd=8:
        - rows [0:2] map to cols [0:4] (first half → first padded-half)
        - rows [2:4] map to cols [4:8] (second half → second padded-half)
        """
        from spyre_inference.hf_adapters import _qk_expand_matrix

        m = _qk_expand_matrix(4, 8)
        assert m.shape == (4, 8)

        # First half: rows 0,1 should have 1s at cols 0,1 respectively
        assert m[0, 0] == 1.0
        assert m[1, 1] == 1.0
        assert m[0, 4] == 0.0
        assert m[1, 5] == 0.0

        # Second half: rows 2,3 should have 1s at cols 4,5 respectively
        assert m[2, 4] == 1.0
        assert m[3, 5] == 1.0
        assert m[2, 0] == 0.0
        assert m[3, 1] == 0.0

    def test_expansion_preserves_norm(self):
        """Multiplying a vector by the expand matrix preserves its L2 norm.

        The expand matrix is an isometric embedding: ||x @ M|| == ||x||.
        """
        from spyre_inference.hf_adapters import _qk_expand_matrix

        torch.manual_seed(42)
        m = _qk_expand_matrix(96, 128)
        x = torch.randn(1, 96)
        expanded = x @ m
        torch.testing.assert_close(
            torch.norm(expanded), torch.norm(x), atol=1e-5, rtol=1e-5
        )

    def test_expand_contract_roundtrip(self):
        """expand then contract (via M^T) is identity on the original space."""
        from spyre_inference.hf_adapters import _qk_expand_matrix

        torch.manual_seed(42)
        orig_hd, padded_hd = 96, 128
        m = _qk_expand_matrix(orig_hd, padded_hd)
        contract = m.t().contiguous()

        x = torch.randn(4, orig_hd)
        roundtrip = (x @ m) @ contract
        torch.testing.assert_close(roundtrip, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# _SpyreRotaryEmbedding
# ---------------------------------------------------------------------------


class TestSpyreRotaryEmbedding:
    """Tests for _SpyreRotaryEmbedding wrapper."""

    def test_forward_returns_rotation_and_none(self):
        """forward() returns (rotation_matrices, None) matching HF API."""
        from spyre_inference.hf_adapters import _SpyreRotaryEmbedding

        # Mock the precomputed rotary embedding
        fake_rotation = torch.randn(4, 2, 64, 64)
        mock_pre = MagicMock(return_value=fake_rotation)

        rope = _SpyreRotaryEmbedding(mock_pre)
        x = torch.randn(4, 2, 8, 64)
        position_ids = torch.arange(8).unsqueeze(0)

        result = rope(x, position_ids)

        assert isinstance(result, tuple)
        assert len(result) == 2
        torch.testing.assert_close(result[0], fake_rotation)
        assert result[1] is None
        mock_pre.assert_called_once_with(x, position_ids)

    def test_apply_is_noop(self):
        """_apply returns self without modifying the module."""
        from spyre_inference.hf_adapters import _SpyreRotaryEmbedding

        mock_pre = MagicMock()
        rope = _SpyreRotaryEmbedding(mock_pre)

        result = rope._apply(lambda t: t.cuda())
        assert result is rope


# ---------------------------------------------------------------------------
# _make_spyre_apply_rotary
# ---------------------------------------------------------------------------


class TestMakeSpyreApplyRotary:
    """Tests for the matmul-based RoPE replacement function."""

    def test_basic_rotation_without_expand(self):
        """Without qk_expand, wrapper calls apply_rope_matmul on q and k."""
        from spyre_inference.hf_adapters import _make_spyre_apply_rotary

        torch.manual_seed(42)
        original_fn = MagicMock()
        wrapper = _make_spyre_apply_rotary(original_fn, qk_expand=None)

        # The wrapper should have _spyre_patched marker
        assert wrapper._spyre_patched is True

        # Create fake q, k, cos (rotation matrices)
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)
        # cos should be rotation matrices of shape matching apply_rope_matmul input
        cos = torch.eye(64).unsqueeze(0).unsqueeze(0).expand(1, 1, 8, 64, 64)

        # The function internally calls apply_rope_matmul; it doesn't call original_fn
        result_q, result_k = wrapper(q, k, cos)

        # With identity rotation, output should equal input
        assert result_q.shape == q.shape
        assert result_k.shape == k.shape

    def test_patched_flag_set(self):
        """Wrapper function has _spyre_patched=True for guard detection."""
        from spyre_inference.hf_adapters import _make_spyre_apply_rotary

        wrapper = _make_spyre_apply_rotary(lambda *a, **kw: None, qk_expand=None)
        assert hasattr(wrapper, "_spyre_patched")
        assert wrapper._spyre_patched is True

    def test_expand_contract_roundtrip_in_wrapper(self):
        """With qk_expand, Q/K are padded, rotated, then contracted back."""
        from spyre_inference.hf_adapters import _qk_expand_matrix, _make_spyre_apply_rotary

        torch.manual_seed(42)
        orig_hd, padded_hd = 96, 128
        qk_expand = _qk_expand_matrix(orig_hd, padded_hd)

        original_fn = MagicMock()
        wrapper = _make_spyre_apply_rotary(original_fn, qk_expand=qk_expand)

        q = torch.randn(1, 4, 8, orig_hd)
        k = torch.randn(1, 4, 8, orig_hd)
        # Identity rotation in padded space
        cos = torch.eye(padded_hd).unsqueeze(0).unsqueeze(0).expand(1, 1, 8, padded_hd, padded_hd)

        result_q, result_k = wrapper(q, k, cos)

        # With identity rotation, expand->rotate->contract should be identity
        assert result_q.shape == (1, 4, 8, orig_hd)
        assert result_k.shape == (1, 4, 8, orig_hd)
        torch.testing.assert_close(result_q, q, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result_k, k, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# _fix_generic_config
# ---------------------------------------------------------------------------


class TestFixGenericConfig:
    """Tests for HfAdaptersForCausalLM._fix_generic_config."""

    def test_skips_non_pretrained_config(self):
        """If hf_config is not generic PretrainedConfig, _fix_generic_config is a no-op."""
        from spyre_inference.hf_adapters import HfAdaptersForCausalLM
        from transformers.models.llama.configuration_llama import LlamaConfig

        # A specific config (LlamaConfig) — should not be re-resolved
        hf_config = LlamaConfig(hidden_size=256, num_attention_heads=4, num_hidden_layers=2)

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.hf_text_config = hf_config
        model_config.trust_remote_code = False
        model_config.revision = None
        model_config.hf_config_path = None
        model_config.model = "test/model"

        load_config = MagicMock()
        load_config.load_format = "auto"

        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.load_config = load_config

        # Should not modify anything
        HfAdaptersForCausalLM._fix_generic_config(vllm_config)

        # hf_config should remain the same specific type
        assert isinstance(vllm_config.model_config.hf_config, LlamaConfig)

    def test_resolves_generic_pretrained_config(self):
        """If hf_config is generic PretrainedConfig, it should attempt re-resolution."""
        from spyre_inference.hf_adapters import HfAdaptersForCausalLM
        from transformers.configuration_utils import PretrainedConfig

        # A generic config (type(config) is PretrainedConfig, not a subclass)
        hf_config = PretrainedConfig()
        # Must be exactly PretrainedConfig, not a subclass
        assert type(hf_config) is PretrainedConfig

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.hf_text_config = hf_config
        model_config.trust_remote_code = False
        model_config.revision = None
        model_config.hf_config_path = None
        model_config.model = "nonexistent-model/does-not-exist"

        load_config = MagicMock()
        load_config.load_format = "auto"

        vllm_config = MagicMock()
        vllm_config.model_config = model_config
        vllm_config.load_config = load_config

        # Should attempt resolution (will fail for nonexistent model, but gracefully)
        HfAdaptersForCausalLM._fix_generic_config(vllm_config)

        # The config remains PretrainedConfig since resolution failed
        assert type(vllm_config.model_config.hf_config) is PretrainedConfig


# ---------------------------------------------------------------------------
# HfAdaptersForCausalLM class registration
# ---------------------------------------------------------------------------


class TestHfAdaptersRegistration:
    """Tests for the class registration and naming conventions."""

    def test_class_name_is_transformers_for_causal_lm(self):
        """HfAdaptersForCausalLM.__name__ must be 'TransformersForCausalLM'.

        vLLM's Transformers backend test checks model_cls.__name__ against
        'TransformersForCausalLM' to determine the backend.
        """
        from spyre_inference.hf_adapters import HfAdaptersForCausalLM

        assert HfAdaptersForCausalLM.__name__ == "TransformersForCausalLM"

    def test_inherits_from_transformers_for_causal_lm(self):
        """HfAdaptersForCausalLM must inherit from TransformersForCausalLM."""
        from spyre_inference.hf_adapters import HfAdaptersForCausalLM
        from vllm.model_executor.models.transformers import TransformersForCausalLM

        assert issubclass(HfAdaptersForCausalLM, TransformersForCausalLM)
