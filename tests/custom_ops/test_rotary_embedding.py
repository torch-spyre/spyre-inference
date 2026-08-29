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

YARN_ROPE_PARAMS = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 2048,
    "extrapolation_factor": 1,
    "attn_factor": 1,
    "beta_fast": 32,
    "beta_slow": 1,
}

# head_size values with a stick-aligned 2x2 inner dim (128->64, 256->128). On the native
# path the platform pads head_dim to a 128-multiple before RoPE is built, so SpyreRoPE
# only ever sees stick-aligned inners.
HEAD_SIZES = [128, 256]


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
    """CPU-only: host gather + _rotate_neox_2x2 match forward_native without a
    Spyre device, so the core rotation formula is validated on dev laptops where the
    forward_oot tests skip. Stick-aligned inner dims (128->64, 256->128) exercise the
    pure-view rotation path."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    from spyre_inference.custom_ops.rotary_embedding import _rotate_neox_2x2

    torch.manual_seed(11)
    max_position, num_tokens, num_heads = 2048, 32, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    rot = rope._get_rotation_cache().index_select(0, positions).to(torch.float16)
    assert rot.device.type == "cpu"
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
    head_size (stick-aligned 128/256), GQA, and 2D/3D layouts."""
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

    rope.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_q_heads, num_kv_heads, head_size, flatten)

    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))

    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    assert actual_query.device.type == "spyre"
    # The CPU cache is the source; a device-resident copy is gathered on Spyre.
    assert rope._rotation_cache is not None and rope._rotation_cache.device.type == "cpu"
    assert (
        rope._device_rotation_cache is not None
        and rope._device_rotation_cache.device.type == "spyre"
    )
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
    rope.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_heads, num_heads, head_size, flatten)

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
    rope.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), None)
    assert actual_key is None

    expected_query, _ = RotaryEmbedding.forward_native(rope, positions.cpu(), query.cpu(), None)
    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_rotary_sel_cache_isolated_across_layers(default_vllm_config, head_size):
    """Two distinct rope modules (different rope_theta -> different rotations) each keep
    their own device rotation cache; each forward_oot gathers from its own cache and
    matches its own reference. A cache mixup would rotate with the wrong frequencies and
    fail the per-module assert_close."""
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
    rope_a.to("spyre")
    rope_b.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    qa = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)
    qb = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)

    aqa, _ = rope_a.forward_oot(positions, qa.to("spyre"))
    aqb, _ = rope_b.forward_oot(positions, qb.to("spyre"))

    eqa, _ = RotaryEmbedding.forward_native(rope_a, positions.cpu(), qa, None)
    eqb, _ = RotaryEmbedding.forward_native(rope_b, positions.cpu(), qb, None)
    torch.testing.assert_close(aqa.cpu().float(), eqa.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(aqb.cpu().float(), eqb.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_rope_device_cache_gather_returns_spyre_slice(default_vllm_config, head_size):
    """forward_oot's in-graph gather (index_select over the device-resident rotation
    cache) returns the per-token row on Spyre, viewable as [T, 2, 2, rotary_dim//2]."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    max_position, num_tokens = 2048, 32
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)
    rope.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    cache = rope._get_device_rotation_cache()
    rot = cache.index_select(0, positions.flatten())
    assert rot.device.type == "spyre"
    assert tuple(rot.shape) == (num_tokens, 2 * head_size)
    inner = rope.rotary_dim // 2
    assert tuple(rot.view(-1, 2, 2, inner).shape) == (num_tokens, 2, 2, inner)


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


@pytest.mark.rotary
def test_yarn_rotary_oot_registration(default_vllm_config):
    """Verify get_rope(rope_type='yarn') resolves to SpyreYaRNScalingRotaryEmbedding."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    from spyre_inference.custom_ops.rotary_embedding import SpyreYaRNScalingRotaryEmbedding

    rope = get_rope(
        head_size=128,
        max_position=8192,
        is_neox_style=True,
        rope_parameters=YARN_ROPE_PARAMS,
        dtype=torch.float16,
    )

    assert isinstance(rope, SpyreYaRNScalingRotaryEmbedding), (
        f"Expected SpyreYaRNScalingRotaryEmbedding, got {type(rope).__name__}"
    )


@pytest.mark.rotary
@pytest.mark.parametrize(
    "yarn_params",
    [
        YARN_ROPE_PARAMS,
        {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
        },
        {
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
            "attn_factor": 2.0,
            "beta_fast": 64,
            "beta_slow": 2,
        },
    ],
    ids=["factor4_defaults", "factor2_defaults", "factor8_custom_params"],
)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_yarn_rotation_math_matches_reference_cpu(default_vllm_config, yarn_params, head_size):
    """CPU-only: host gather + _rotate_neox_2x2 match forward_native for YaRN,
    validating that the scaled cos/sin cache produced by YaRN is correctly transformed
    into the 2x2 rotation matrix format across different scaling factors and parameters."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )

    from spyre_inference.custom_ops.rotary_embedding import _rotate_neox_2x2

    torch.manual_seed(77)
    max_position = int(yarn_params["original_max_position_embeddings"] * yarn_params["factor"])
    num_tokens, num_heads = 32, 4
    rope = get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters=yarn_params,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    rot = rope._get_rotation_cache().index_select(0, positions).to(torch.float16)
    assert rot.device.type == "cpu"
    actual_query = _rotate_neox_2x2(query, rot, head_size)
    actual_key = _rotate_neox_2x2(key, rot, head_size)

    expected_query, expected_key = YaRNScalingRotaryEmbedding.forward_native(
        rope, positions, query, key
    )
    torch.testing.assert_close(actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("flatten", [True, False])
def test_yarn_rotary_forward_oot_on_spyre(default_vllm_config, head_size, flatten):
    """YaRN rotation runs on Spyre and matches forward_native across head_size
    and 2D/3D layouts, confirming the 2x2 cache inherits YaRN frequency scaling
    (interpolation + extrapolation blending + mscale) via the MRO."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )

    torch.manual_seed(42)
    max_position, num_tokens, num_heads = 8192, 32, 4
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=YARN_ROPE_PARAMS,
        dtype=torch.float16,
    )
    rope.to("spyre")

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_heads, num_heads, head_size, flatten)

    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))
    expected_query, expected_key = YaRNScalingRotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_yarn_cos_sin_cache_has_mscale(default_vllm_config, head_size):
    """YaRN's cos_sin_cache incorporates the mscale factor, producing magnitudes
    distinct from base RoPE. This ensures the scaling is not silently lost."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    max_position = 8192
    rope_yarn = get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters=YARN_ROPE_PARAMS,
        dtype=torch.float16,
    )
    rope_base = get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )

    assert not torch.allclose(
        rope_yarn.cos_sin_cache[:2048], rope_base.cos_sin_cache[:2048], atol=1e-4
    ), "YaRN cos_sin_cache should differ from base RoPE due to mscale and freq blending"


def _fresh_rope(**kwargs):
    """get_rope memoizes instances globally; these tests need an unprimed one."""
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope

    _ROPE_DICT.clear()
    return get_rope(**kwargs)


@pytest.mark.rotary
@pytest.mark.parametrize("head_size", HEAD_SIZES)
def test_device_rotation_cache_is_2d_row_gathered(default_vllm_config, head_size):
    """The device cache is flattened to 2D so it can be stickified rows-outermost."""
    max_position = 2048
    rope = _fresh_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )
    rope.to("spyre")

    assert rope._rotation_cache.shape == (max_position, 2, 2, head_size // 2)
    assert rope._device_rotation_cache.shape == (max_position, 2 * head_size)
    assert rope._device_rotation_cache.device.type == "spyre"


@pytest.mark.rotary
def test_rotary_matches_native_at_high_positions(default_vllm_config):
    """Positions far above a served context still rotate correctly: the whole cache
    stays addressable, so no bound on max_model_len is implied."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(11)
    max_position, num_tokens, num_heads, head_size = 131072, 16, 4, 128
    rope = _fresh_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )
    rope.to("spyre")

    positions = torch.randint(max_position - 1024, max_position, (num_tokens,), dtype=torch.long)
    query, key = _make_qk(num_tokens, num_heads, num_heads, head_size, True)

    actual_query, actual_key = rope.forward_oot(
        positions.to("spyre"), query.to("spyre"), key.to("spyre")
    )
    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions, query.clone(), key.clone()
    )

    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)
