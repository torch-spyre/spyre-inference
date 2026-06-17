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

"""
Test SpyreLlama3RotaryEmbedding custom op correctness.
"""

import pytest
import torch


@pytest.mark.rotary
def test_llama3_rotary_oot_registration(default_vllm_config):
    """Verify OOT registration: get_rope with rope_type='llama3' returns SpyreLlama3RotaryEmbedding.

    This confirms the OOT registration actually resolves to our class, not the base
    Llama3RotaryEmbedding.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from spyre_inference.custom_ops.rotary_embedding import SpyreLlama3RotaryEmbedding

    rope = get_rope(
        head_size=128,
        max_position=2048,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 4096,
        },
        dtype=torch.float16,
    )

    assert isinstance(rope, SpyreLlama3RotaryEmbedding), (
        f"Expected SpyreLlama3RotaryEmbedding, got {type(rope).__name__}"
    )


@pytest.mark.rotary
def test_spyre_llama3_rotary_matches_reference(default_vllm_config):
    """SpyreLlama3RotaryEmbedding output matches CPU reference.

    Tests forward_oot() path: inputs on CPU, computation via Llama3RotaryEmbedding.forward_native.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

    torch.manual_seed(42)

    head_size = 128
    rotary_dim = 128
    max_position = 2048
    num_tokens = 32
    num_heads = 4

    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 4096,
        },
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)

    actual_query, actual_key = rope.forward_oot(positions, query, key)

    cpu_positions = positions.cpu()
    cpu_query = query.cpu()
    cpu_key = key.cpu()
    expected_query, expected_key = Llama3RotaryEmbedding.forward_native(
        rope, cpu_positions, cpu_query, cpu_key
    )

    torch.testing.assert_close(
        actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.rotary
def test_spyre_base_rotary_oot_registration(default_vllm_config):
    """Verify OOT registration: default get_rope returns SpyreRotaryEmbedding."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from spyre_inference.custom_ops.rotary_embedding import SpyreRotaryEmbedding

    rope = get_rope(
        head_size=128,
        max_position=2048,
        is_neox_style=True,
        dtype=torch.float16,
    )

    assert isinstance(rope, SpyreRotaryEmbedding), (
        f"Expected SpyreRotaryEmbedding, got {type(rope).__name__}"
    )


@pytest.mark.rotary
def test_spyre_base_rotary_matches_reference(default_vllm_config):
    """SpyreRotaryEmbedding output matches CPU reference.

    Tests forward_oot() path: inputs on CPU, computation via RotaryEmbedding.forward_native.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(42)

    head_size = 128
    rotary_dim = 128
    max_position = 2048
    num_tokens = 32
    num_heads = 4

    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)

    actual_query, actual_key = rope.forward_oot(positions, query, key)

    cpu_positions = positions.cpu()
    cpu_query = query.cpu()
    cpu_key = key.cpu()
    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, cpu_positions, cpu_query, cpu_key
    )

    torch.testing.assert_close(
        actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2
    )
