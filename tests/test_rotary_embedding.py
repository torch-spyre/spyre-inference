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

import pytest
import torch

LLAMA3_ROPE_PARAMS = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 4096,
}


def _spyre_available() -> bool:
    try:
        torch.randn(1, device=torch.device("spyre"))
        return True
    except Exception:
        return False


@pytest.fixture()
def requires_spyre():
    """Lazily skip when no Spyre device is present.

    Checked inside the fixture (not at import time) to avoid claiming the
    single-tenant device during collection.
    """
    if not _spyre_available():
        pytest.skip("Spyre device unavailable")


def _prime_rope(rope, positions):
    """Mimic _SpyreModelWrapper: pre-gather the rotation slice for ``positions`` and
    stash it in the forward context (keyed by the module's ``_layer_name``) so a
    subsequent ``forward_oot`` can fetch it via the ``spyre_rope_rot`` op.

    Direct ``forward_oot`` calls don't go through the wrapper, so tests must prime
    first (the op requires the slice to be present). No-op for CPU-fallback configs
    (``gather_rotation`` returns None). ``default_vllm_config`` already provides an
    active forward context. Returns the gathered slice (or None).
    """
    from vllm.forward_context import get_forward_context

    rot = rope.gather_rotation(positions, positions.device)
    if rot is not None:
        cache = get_forward_context().additional_kwargs.setdefault("spyre_rope_rot", {})
        cache[rope._layer_name] = rot
    return rot


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
def test_llama3_rotary_forward_matches_reference(default_vllm_config):
    """forward_oot output matches Llama3RotaryEmbedding.forward_native reference.

    Runs on CPU (default-toggle eager 2x2 path); validates the copied rotation
    math against the vLLM reference without hardware.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

    torch.manual_seed(42)

    head_size = 128
    max_position = 2048
    num_tokens = 32
    num_heads = 4

    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=LLAMA3_ROPE_PARAMS,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16)

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query, key)

    expected_query, expected_key = Llama3RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    torch.testing.assert_close(actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
def test_base_rotary_forward_matches_reference(default_vllm_config):
    """SpyreRotaryEmbedding.forward_oot matches RotaryEmbedding.forward_native (CPU)."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(42)

    head_size = 128
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

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query, key)

    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    torch.testing.assert_close(actual_query.float(), expected_query.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_key.float(), expected_key.float(), atol=1e-2, rtol=1e-2)


def _make_qk(num_tokens, num_q_heads, num_kv_heads, head_size, flatten):
    """Build (query, key) on CPU as 2D [T, H*D] (production) or 3D [T, H, D]."""
    query = torch.randn(num_tokens, num_q_heads, head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float16)
    if flatten:
        query = query.reshape(num_tokens, num_q_heads * head_size)
        key = key.reshape(num_tokens, num_kv_heads * head_size)
    return query, key


@pytest.mark.rotary
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("flatten", [True, False])
def test_rotary_forward_oot_on_spyre(
    requires_spyre,
    default_vllm_config,
    num_q_heads,
    num_kv_heads,
    flatten,
):
    """forward_oot runs the 2x2 rotation on Spyre and matches forward_native.

    head_size=128 (rotary_dim=128, stick-aligned) is the granite/llama target.
    Exercises GQA (MHA + grouped) and both the 2D [T, H*D] and 3D [T, H, D]
    layouts.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(42)
    head_size, max_position, num_tokens = 128, 2048, 32

    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        dtype=torch.float16,
    )
    assert rope._spyre_rope_supported  # confirms the on-device path is taken

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query, key = _make_qk(num_tokens, num_q_heads, num_kv_heads, head_size, flatten)

    _prime_rope(rope, positions)
    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))

    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    assert actual_query.device.type == "spyre"
    # The 2x2 rotation cache stays on CPU (Spyre has no eager index_select); only
    # the gathered per-token slice (built during priming) is moved to Spyre.
    assert rope._rotation_cache is not None and rope._rotation_cache.device.type == "cpu"
    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
@pytest.mark.parametrize(
    "head_size,partial_rotary_factor",
    [
        (64, 1.0),  # full but unaligned: rotary_dim=64 -> inner dim 32
        (128, 0.5),  # partial AND unaligned: rotary_dim=64 -> inner dim 32
        (256, 0.5),  # partial but inner-aligned: rotary_dim=128 -> gated off for being partial
    ],
)
def test_rotary_unsupported_falls_back_to_cpu(
    requires_spyre, default_vllm_config, head_size, partial_rotary_factor
):
    """Configs the on-device path can't handle use the CPU fallback and still
    return correct results on the requested device. Covers an unaligned inner
    dim (head_size 64) and partial rotary (rotary_dim < head_size) -- whose
    slicing has no Spyre kernel -- including a partial case whose inner dim IS
    stick-aligned (256, 0.5 -> rotary_dim 128), to confirm partial is gated off
    regardless of alignment."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(3)
    max_position, num_tokens, num_heads = 2048, 32, 4
    rope_parameters = (
        None if partial_rotary_factor == 1.0 else {"partial_rotary_factor": partial_rotary_factor}
    )
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=rope_parameters,
        dtype=torch.float16,
    )
    assert not rope._spyre_rope_supported  # gated off -> CPU fallback

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    actual_query, actual_key = rope.forward_oot(
        positions.to("spyre"), query.to("spyre"), key.to("spyre")
    )
    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )
    assert actual_query.device.type == "spyre"  # result still on the device
    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
def test_rotary_forward_oot_key_none_on_spyre(requires_spyre, default_vllm_config):
    """forward_oot(..., key=None) returns (rotated_query, None) on Spyre."""
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(0)
    head_size, max_position, num_tokens, num_heads = 128, 2048, 16, 4
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
@pytest.mark.parametrize("device", ["spyre", "cpu"])
def test_rope_device_toggle_parity(requires_spyre, default_vllm_config, monkeypatch, device):
    """SPYRE_INFERENCE_ROPE_DEVICE=spyre and =cpu both match forward_native.

    Proves the toggle dispatches correctly and the on-device 2x2 path is
    numerically equivalent to the CPU round-trip.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    monkeypatch.setenv("SPYRE_INFERENCE_ROPE_DEVICE", device)

    torch.manual_seed(7)
    head_size, max_position, num_tokens, num_heads = 128, 2048, 32, 8
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

    _prime_rope(rope, positions)  # no-op when device="cpu" (gather_rotation returns None)
    actual_query, actual_key = rope.forward_oot(positions, query.to("spyre"), key.to("spyre"))
    expected_query, expected_key = RotaryEmbedding.forward_native(
        rope, positions.cpu(), query.cpu(), key.cpu()
    )

    torch.testing.assert_close(
        actual_query.cpu().float(), expected_query.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(actual_key.cpu().float(), expected_key.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rotary
def test_llama3_rotary_forward_oot_on_spyre(requires_spyre, default_vllm_config):
    """Llama3 (scaled) rotation runs on Spyre and matches forward_native.

    Confirms the 2x2 cache inherits llama3 frequency scaling via the MRO.
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

    torch.manual_seed(42)
    head_size, max_position, num_tokens, num_heads = 128, 2048, 32, 4
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        is_neox_style=True,
        rope_parameters=LLAMA3_ROPE_PARAMS,
        dtype=torch.float16,
    )

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.float16)

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
def test_rotary_sel_cache_shared_across_layers(requires_spyre, default_vllm_config):
    """The rotation slice is gathered once (pre-forward) and stashed in the forward
    context; every attention layer fetches that shared slice via forward_oot, which
    stays correct for the differing q each layer passes.

    Mirrors what _SpyreModelWrapper does: prime once, then forward_oot reads the
    slice back through spyre_rope_rot. (default_vllm_config already runs the body
    inside a set_forward_context scope.)
    """
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    torch.manual_seed(1)
    head_size, max_position, num_tokens, nh = 128, 2048, 32, 4
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)
    assert rope._spyre_rope_supported

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long).to("spyre")
    q1 = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)
    q2 = torch.randn(num_tokens, nh * head_size, dtype=torch.float16)

    # Prime once (pre-forward); two attention layers then reuse the shared slice.
    rot = _prime_rope(rope, positions)
    assert rot is not None and rot.device.type == "spyre"

    aq1, _ = rope.forward_oot(positions, q1.to("spyre"))
    aq2, _ = rope.forward_oot(positions, q2.to("spyre"))

    eq1, _ = RotaryEmbedding.forward_native(
        rope, positions.cpu(), q1.view(num_tokens, nh, head_size), None
    )
    eq2, _ = RotaryEmbedding.forward_native(
        rope, positions.cpu(), q2.view(num_tokens, nh, head_size), None
    )
    torch.testing.assert_close(
        aq1.cpu().float().view(num_tokens, nh, head_size), eq1.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        aq2.cpu().float().view(num_tokens, nh, head_size), eq2.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.rotary
def test_gather_rotation_returns_spyre_slice(requires_spyre, default_vllm_config):
    """gather_rotation returns the per-token [T, 2, 2, rotary_dim//2] slice on
    Spyre for a supported config, and None for a config on the CPU fallback."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    head_size, max_position, num_tokens = 128, 2048, 32
    rope = get_rope(head_size, max_position, is_neox_style=True, dtype=torch.float16)
    assert rope._spyre_rope_supported

    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.long)
    rot = rope.gather_rotation(positions, torch.device("spyre"))
    assert rot is not None
    assert rot.device.type == "spyre"
    assert tuple(rot.shape) == (num_tokens, 2, 2, rope.rotary_dim // 2)

    # Unaligned head_size=64 is gated off -> CPU fallback -> gather_rotation is a no-op.
    rope_unsupported = get_rope(64, max_position, is_neox_style=True, dtype=torch.float16)
    assert not rope_unsupported._spyre_rope_supported
    assert rope_unsupported.gather_rotation(positions, torch.device("spyre")) is None
