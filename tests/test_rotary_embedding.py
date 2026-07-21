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
Test Spyre RoPE custom op correctness against upstream CPU reference implementations.
"""

import pytest
import torch

LLAMA3_ROPE_PARAMS = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 4096,
}

# head_size values spanning both Spyre RoPE regimes: the 2x2 inner dim
# head_size//2 is stick-aligned (128->64, 256->128) or padded up to a stick (64->32).
HEAD_SIZES = [64, 128, 256]


def _prime_rope(rope, positions):
    """Mimic _SpyreModelWrapper: pre-gather the rotation slice and stash it in the
    forward context so a direct forward_oot can fetch it. Returns the slice (or None
    for CPU-fallback configs).

    Uses ``setdefault`` (not the single-shot dict rebuild of the production
    ``_prime_rope_rotation``) so multiple modules primed by successive calls
    accumulate under distinct ``_rope_key`` entries in one dict."""
    from vllm.forward_context import get_forward_context

    rot = rope.gather_rotation(positions, positions.device)
    if rot is not None:
        cache = get_forward_context().additional_kwargs.setdefault("spyre_rope_rot", {})
        cache[rope._rope_key] = rot
    return rot


def _make_qk(num_tokens, num_q_heads, num_kv_heads, head_size, flatten):
    """Build (query, key) on CPU as 2D [T, H*D] (production) or 3D [T, H, D]."""
    query = torch.randn(num_tokens, num_q_heads, head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float16)
    if flatten:
        query = query.reshape(num_tokens, num_q_heads * head_size)
        key = key.reshape(num_tokens, num_kv_heads * head_size)
    return query, key


@pytest.mark.rotary
def test_llama3_rotary_oot_registration(default_vllm_config):
    """Verify get_rope(rope_type='llama3') resolves to SpyreLlama3RotaryEmbedding."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from spyre_inference.custom_ops.rotary_embedding import SpyreLlama3RotaryEmbedding

    rope = get_rope(
        head_size=128,
        max_position=2048,
        is_neox_style=True,
        rope_parameters=LLAMA3_ROPE_PARAMS,
        dtype=torch.float16,
    )

    assert isinstance(rope, SpyreLlama3RotaryEmbedding), (
        f"Expected SpyreLlama3RotaryEmbedding, got {type(rope).__name__}"
    )


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_rotation_math_matches_reference_cpu(default_vllm_config, head_size):
    """CPU-only: gather_rotation + _rotate_neox_2x2 match forward_native without a
    Spyre device, so the core rotation formula is validated on dev laptops where the
    forward_oot tests skip. Stick-aligned inner dims (128->64, 256->128) exercise the
    pure-view path; head_size=64 (inner dim 32) exercises the pad-to-stick expand-matrix
    path."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from spyre_inference.custom_ops.rotary_embedding import _rotate_neox_2x2

    torch.manual_seed(11)
    max_position, num_tokens, num_heads = 2048, 32, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    rot = rope.gather_rotation(positions, torch.device("cpu"))
    assert rot is not None and rot.device.type == "cpu"
    actual_query = _rotate_neox_2x2(query, rot, head_size)
    actual_key = _rotate_neox_2x2(key, rot, head_size)

    expected_query, expected_key = RotaryEmbedding.forward_native(rope, positions, query, key)
    torch.testing.assert_close(actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("flatten", [True, False])
def test_rotary_forward_oot_on_spyre(
    default_vllm_config,
    head_size,
    num_q_heads,
    num_kv_heads,
    flatten,
):
    """forward_oot runs the 2x2 rotation on Spyre and matches forward_native across
    head_size (aligned 128/256, pad-to-stick 64), GQA, and 2D/3D layouts."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(42)
    max_position, num_tokens = 2048, 32

    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_q_heads, num_kv_heads, head_size, flatten)

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))

    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    assert actual_query.device.type == "spyre"
    # The rotation cache stays on CPU (no eager index_select); only the slice moves.
    assert rope._rotation_cache is not None and rope._rotation_cache.device.type == "cpu"
    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("flatten", [True, False])
def test_llama3_rotary_forward_oot_on_spyre(default_vllm_config, head_size, flatten):
    """Llama3 (scaled) rotation runs on Spyre and matches forward_native across head_size
    and 2D/3D layouts, confirming the 2x2 cache inherits llama3 frequency scaling via the
    MRO."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

    torch.manual_seed(42)
    max_position, num_tokens, num_heads = 2048, 32, 4
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=LLAMA3_ROPE_PARAMS,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_heads, num_heads, head_size, flatten)

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))
    expected_query, expected_key = Llama3RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_rotary_forward_oot_key_none_on_spyre(default_vllm_config, head_size):
    """forward_oot(..., key=None) returns (rotated_query, None) on Spyre."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(0)
    max_position, num_tokens, num_heads = 2048, 16, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), None)
    assert actual_key is None

    expected_query, _ = RotaryEmbedding.forward_native(rope, positions.cpu(), query.cpu(), None)
    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_rotary_sel_cache_isolated_across_layers(default_vllm_config, head_size):
    """Two distinct rope modules (different rope_theta -> different rotations) prime
    their own slices into one spyre_rope_rot dict under distinct _rope_key entries;
    each forward_oot fetches its own slice and matches its own reference. A key mixup
    would rotate with the wrong frequencies and fail the per-module assert_close."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(1)
    max_position, num_tokens, nh = 2048, 32, 4
    rope_a = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)
    rope_b = get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters={"rope_theta": 1000000.0},
        dtype=torch.float16,
    )
    assert rope_a._rope_key != rope_b._rope_key

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    qa = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)
    qb = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)

    _prime_rope(rope_a, positions)
    _prime_rope(rope_b, positions)

    aqa, _ = rope_a.forward_oot(positions, qa.to("spyre"))
    aqb, _ = rope_b.forward_oot(positions, qb.to("spyre"))

    eqa, _ = RotaryEmbedding.forward_native(rope_a, positions.cpu(), qa, None)
    eqb, _ = RotaryEmbedding.forward_native(rope_b, positions.cpu(), qb, None)
    torch.testing.assert_close(aqa.cpu().float(), eqa.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(aqb.cpu().float(), eqb.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_gather_rotation_returns_spyre_slice(default_vllm_config, head_size):
    """gather_rotation returns the per-token [T, 2, 2, round_up(rotary_dim//2)] slice
    on Spyre for a supported config."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.utils.math_utils import round_up
    from spyre_inference.custom_ops.rotary_embedding import _SPYRE_STICK

    max_position, num_tokens = 2048, 32
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    rot = rope.gather_rotation(positions, torch.device("spyre"))
    assert rot is not None
    assert rot.device.type == "spyre"
    assert tuple(rot.shape) == (num_tokens, 2, 2, round_up(rope.rotary_dim // 2, _SPYRE_STICK))


@pytest.mark.rotary
def test_gather_rotation_mrope_positions_returns_none(default_vllm_config):
    """Multi-dim (mrope/xdrope) positions have no Spyre rotation path: gather_rotation
    returns None so _prime_rope_rotation leaves the module unprimed."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rope = get_rope(128, 2048, is_neox_style=True, dtype=torch.float16)
    positions = torch.randint(0, 2048, (3, 8), dtype=torch.long)  # 2D -> mrope-style
    assert rope.gather_rotation(positions, torch.device("cpu")) is None


@pytest.mark.rotary
def test_rope_rot_op_unprimed_raises(default_vllm_config):
    """The spyre_rope_rot op body raises when a module's slice was never primed into
    the forward context (rather than silently returning stale/empty data)."""
    from spyre_inference.custom_ops.rotary_embedding import _rope_rot_op_func

    with pytest.raises(RuntimeError, match="not primed"):
        _rope_rot_op_func(torch.zeros(4, dtype=torch.long), "spyre_rope_never_primed", 128)


@pytest.mark.rotary
@pytest.mark.parametrize(
    "head_size,partial_rotary_factor",
    [
        (128, 0.5),  # partial AND unaligned: rotary_dim=64 -> inner dim 32
        (256, 0.5),  # partial but inner-aligned: rotary_dim=128 -> rejected for being partial
    ],
)
def test_rotary_partial_config_raises(default_vllm_config, head_size, partial_rotary_factor):
    """Partial rotary raises NotImplementedError at construction (no CPU fallback),
    whether or not its inner dim is stick-aligned."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    with pytest.raises(NotImplementedError):
        get_rope(
            head_size=head_size,
            max_position=2048,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": partial_rotary_factor},
            dtype=torch.float16,
        )


@pytest.mark.rotary
def test_rotary_non_neox_config_raises(default_vllm_config):
    """gptj/interleaved (is_neox_style=False) full rotary is rejected at construction:
    only the neox 2x2 kernel is implemented."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    with pytest.raises(NotImplementedError):
        get_rope(
            head_size=128,
            max_position=2048,
            is_neox_style=False,
            dtype=torch.float16,
        )
