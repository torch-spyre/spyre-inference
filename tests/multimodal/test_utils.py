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

"""Tests for `spyre_inference/multimodal/utils.py`: stick-aligned SDPA and the matmul RoPE.

The callers cover the padding algebra on CPU; what they cannot cover is why the no-mask
path exists at all, so the on-card case below is the one that matters.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.multimodal import utils
from spyre_inference.multimodal.utils import padded_sdpa


# CLIP ViT-B/32's 50 patches and SigLIP's 729, the stick-coprime lengths in use, and
# an aligned 64, which needs no mask at all. Batch 3 is a multi-image step: the
# key-pad row is [1, 1, 1, seq_pad] and must broadcast.
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq", [50, 64, 729])
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_padded_sdpa_without_mask_matches_cpu_on_spyre(batch, seq, mode):
    """`padded_sdpa(..., None)` on card must equal the same call on CPU.

    The key-pad row is built on the host because writing `-inf` into the tail of a
    stick is a sub-stick offset write, which has no offset-free alternative stick dim.
    Doing it on device instead fails to lower rather than returning wrong values.

    The `compile` leg is what the next tower to opt into per-block compile needs: the
    build must run once, at trace time, and never land in the graph.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    q, k, v = (torch.randn(batch, 2, seq, 64, dtype=torch.float16) for _ in range(3))
    expected = padded_sdpa(q, k, v, None)

    device = torch.device("spyre")
    q, k, v = (x.to(device) for x in (q, k, v))

    fn = padded_sdpa
    if mode == "compile":
        torch._dynamo.reset()
        utils._KEY_PAD_MASKS.clear()
        fn = torch.compile(padded_sdpa, dynamic=False, backend="inductor")

    actual = fn(q, k, v, None)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)

    if mode == "compile":
        if seq % 64 == 0:
            assert not utils._KEY_PAD_MASKS  # nothing to mask, nothing built
        else:
            (row,) = utils._KEY_PAD_MASKS.values()  # built once, at trace time
            fn(q, k, v, None)
            assert next(iter(utils._KEY_PAD_MASKS.values())) is row  # graph input, not rebuilt


def test_aligned_device_mask_passes_through():
    """An aligned, additive mask already on the target device is used as is: inside a
    compiled graph the host round trip of the general path cannot run."""
    q, k, v = (torch.randn(1, 2, 64, 64, dtype=torch.float16) for _ in range(3))
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.float16)
    mask[..., 32:] = torch.finfo(torch.float16).min / 2

    out = padded_sdpa(q, k, v, mask)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=64**-0.5
    )
    torch.testing.assert_close(out, expected)
    assert not hasattr(mask, utils._MASK_ATTR), "the pass-through must not build a padded copy"


# ---------------------------------------------------------------------------
# Matmul RoPE
# ---------------------------------------------------------------------------


def _reference_pair_swap(x: torch.Tensor) -> torch.Tensor:
    """Swap each `(2k, 2k+1)` element pair along the last dim."""
    out = x.clone()
    out[..., 0::2] = x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", [64, 128])
def test_pair_matrix_equals_pair_swap(head_dim):
    """`x @ M_pair` swaps each `(2k, 2k+1)` pair; the sign lives in the sin table."""
    torch.manual_seed(1)
    x = torch.randn(2, 7, head_dim, dtype=torch.float16)
    m = utils.rope_perm_matrix("pair", head_dim, torch.device("cpu"))
    assert m.shape == (head_dim, head_dim) and m.device.type == "cpu"
    torch.testing.assert_close(torch.matmul(x, m), _reference_pair_swap(x))


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_rotate_matmul_matches_rotation_formula(head_dim):
    torch.manual_seed(2)
    x = torch.randn(1, 4, 11, head_dim, dtype=torch.float16)
    angles = torch.randn(11, head_dim, dtype=torch.float16)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    m = utils.rope_perm_matrix("pair", head_dim, torch.device("cpu"))
    expected = x * cos + _reference_pair_swap(x) * sin
    torch.testing.assert_close(utils.rope_rotate_matmul(x, cos, sin, m), expected)


@pytest.mark.rotary
@pytest.mark.parametrize("num_patches", [64, 121])
def test_rope_rotate_matmul_matches_cpu_on_spyre(num_patches):
    """On card, the 4-D @ 2-D rotation must equal CPU: the shape family of torch-spyre#4155,
    which once returned confident garbage with no warning."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(29)
    x = torch.randn(1, 4, num_patches, 64, dtype=torch.float16)
    angles = torch.randn(num_patches, 64, dtype=torch.float16)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    m = utils.rope_perm_matrix("pair", 64, torch.device("cpu"))
    expected = utils.rope_rotate_matmul(x, cos, sin, m)

    device = torch.device("spyre")
    actual = utils.rope_rotate_matmul(x.to(device), cos.to(device), sin.to(device), m.to(device))
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Padded-mask cache (the general mask path)
# ---------------------------------------------------------------------------


def test_padded_mask_is_cached_across_layers():
    """Every layer gets the same mask object; the O(L²) padded copy is built once."""
    mask = torch.ones(67, 67, dtype=torch.bool).tril()
    args = (mask, 1, 67, 128, torch.float16, torch.device("cpu"))
    first = utils._padded_attn_mask(*args)
    assert all(utils._padded_attn_mask(*args) is first for _ in range(23))


def test_padded_mask_cache_misses_on_a_new_mask():
    tril = torch.ones(67, 67, dtype=torch.bool).tril()
    first = utils._padded_attn_mask(tril, 1, 67, 128, torch.float16, torch.device("cpu"))
    triu = torch.ones(67, 67, dtype=torch.bool).triu()
    second = utils._padded_attn_mask(triu, 1, 67, 128, torch.float16, torch.device("cpu"))
    assert second is not first
    assert not torch.equal(second, first)


def test_padded_mask_is_released_with_its_source_mask():
    """Cached on the mask, so the O(L²) copy dies with it, not at process exit."""
    import gc
    import weakref

    mask = torch.ones(67, 67, dtype=torch.bool).tril()
    padded = weakref.ref(
        utils._padded_attn_mask(mask, 1, 67, 128, torch.float16, torch.device("cpu"))
    )
    assert padded() is not None
    del mask
    gc.collect()
    assert padded() is None


@pytest.mark.parametrize("seq,seq_pad", [(64, 64), (67, 128)])
def test_padded_keys_are_masked_off(seq, seq_pad):
    """A full-attention bool keep-mask adds no masking: only the padded keys are cut."""
    source = torch.ones(seq, seq, dtype=torch.bool)
    m = utils._padded_attn_mask(source, 1, seq, seq_pad, torch.float16, torch.device("cpu"))
    assert m.shape == (1, 1, seq_pad, seq_pad)
    assert (m[:, :, :, seq:] == torch.finfo(torch.float16).min).all()
    assert (m[:, :, :, :seq] == 0).all()


def test_padded_sdpa_refuses_a_padded_mask_at_another_dtype():
    """A mask already padded to `seq_pad` keys is used as is, so a dtype miss has to say
    so instead of reaching `_padded_attn_mask`, which would reshape it to `[seq, seq]`."""
    q, k, v = (torch.randn(2, 2, 50, 64, dtype=torch.float16) for _ in range(3))
    mask = torch.zeros(2, 1, 1, 64, dtype=torch.float32)

    with pytest.raises(TypeError, match="float32"):
        padded_sdpa(q, k, v, mask)


def test_padded_sdpa_refuses_an_unpadded_float_mask():
    """Every caller passes a bool keep-mask or an additive mask already padded to the
    stick; an unpadded float mask is the ambiguous case and has to say so."""
    q, k, v = (torch.randn(2, 2, 50, 64, dtype=torch.float16) for _ in range(3))

    with pytest.raises(TypeError, match="bool keep-mask"):
        padded_sdpa(q, k, v, torch.zeros(50, 50, dtype=torch.float16))


# ---------------------------------------------------------------------------
# Block-diagonal mask from packed boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patch_embeds_list",
    [
        [16],  # one image: the whole mask is one block
        [16, 16],  # two images
        [9, 16, 25],  # three unequal images
    ],
)
def test_block_diagonal_mask_matches_upstream_blocks(patch_embeds_list):
    """The block-diagonal mask rebuilt from `cu_seqlens` must match the per-image
    blocks of transformers' `generate_block_attention_mask`."""
    from spyre_inference.multimodal.utils import block_diagonal_mask

    seq = sum(patch_embeds_list)
    cu_seqlens = torch.tensor([0, *torch.tensor(patch_embeds_list).cumsum(0).tolist()])

    got = block_diagonal_mask(cu_seqlens, seq)

    want = torch.zeros(seq, seq, dtype=torch.bool)
    start = 0
    for length in patch_embeds_list:
        want[start : start + length, start : start + length] = True
        start += length

    assert got.dtype == torch.bool, "padded_sdpa's mask contract is a bool keep-mask"
    assert got.device.type == "cpu", "built on CPU: strided sub-block writes are not stick-safe"
    assert torch.equal(got, want)


def test_block_diagonal_mask_blocks_cross_image_attention():
    """Two images must not attend to each other: the off-diagonal blocks stay False."""
    from spyre_inference.multimodal.utils import block_diagonal_mask

    mask = block_diagonal_mask(torch.tensor([0, 16, 32]), 32)

    assert mask[:16, :16].all()
    assert mask[16:, 16:].all()
    assert not mask[:16, 16:].any()
    assert not mask[16:, :16].any()


def test_block_diagonal_mask_is_cached_on_cu_seqlens():
    """The tower reuses one `cu_seqlens` across every layer; the O(L²) mask must be
    built once, so that each layer then hits the padded-mask cache too."""
    from spyre_inference.multimodal.utils import block_diagonal_mask

    cu_seqlens = torch.tensor([0, 16, 32])

    first = block_diagonal_mask(cu_seqlens, 32)
    assert all(block_diagonal_mask(cu_seqlens, 32) is first for _ in range(23))


def test_block_diagonal_mask_follows_a_new_partition():
    """`[0, 16, 32]` (two images) and `[0, 32]` (one) share a token count. vLLM builds
    a fresh `cu_seqlens` per tower forward, so each gets its own mask."""
    from spyre_inference.multimodal.utils import block_diagonal_mask

    assert not block_diagonal_mask(torch.tensor([0, 16, 32]), 32)[0, 16]
    assert block_diagonal_mask(torch.tensor([0, 32]), 32).all()


def test_block_diagonal_mask_rebuilds_after_an_in_place_repartition():
    """A tensor that tracks versions is re-read once it changes in place."""
    from spyre_inference.multimodal.utils import block_diagonal_mask

    with torch.inference_mode(False):
        cu_seqlens = torch.tensor([0, 16, 32])
        assert not block_diagonal_mask(cu_seqlens, 32)[0, 16]
        cu_seqlens[:] = torch.tensor([0, 32, 32])
        assert block_diagonal_mask(cu_seqlens, 32).all(), "stale mask after a repartition"


def test_block_diagonal_mask_cache_hit_reads_no_boundaries(monkeypatch):
    """`cu_seqlens` lives on the card, so only the first layer may copy it to the host."""
    from spyre_inference.multimodal import utils

    with torch.inference_mode():  # as vLLM builds it: no version counter to key on
        cu_seqlens = torch.tensor([0, 16, 32])
    utils.block_diagonal_mask(cu_seqlens, 32)
    reads = []
    monkeypatch.setattr(utils, "convert", lambda *a, **k: reads.append(a) or a[0])
    for _ in range(23):
        utils.block_diagonal_mask(cu_seqlens, 32)
    assert not reads
