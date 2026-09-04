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

"""Tests for `spyre_inference/multimodal/pixtral.py`.

The patches are guarded with `getattr(..., None)`, so a vLLM rename turns one into a
silent no-op — hence the staleness tripwires alongside the equivalence checks.
Section 4 repeats the numeric checks on the card and skips without a device.
"""

import sys

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")

HIDDEN_SIZE = 256
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64 — the case that motivated the rewrite
# 16x16 = 256 grid positions, so a 67-patch (stick-coprime) image still fits.
MAX_PATCHES_PER_SIDE = 16
ROPE_THETA = 10000.0

HEAD_DIMS = [64, 128]


def _finish_weight_loading(module: torch.nn.Module) -> None:
    """Run `process_weights_after_loading` on every linear in `module`.

    It establishes the transposed weight and `spyre_row_padding` the Spyre OOT linears
    read, so a test calling `forward` directly must run it as `load_model` does.
    """
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


@pytest.fixture(autouse=True)
def restore_pixtral(monkeypatch):
    """Undo the patches after each test: they mutate the shared `pixtral` module,
    and one test would otherwise leak a patched tower into the whole session."""
    monkeypatch.setattr(pixtral, "apply_rotary_emb_vit", pixtral.apply_rotary_emb_vit)
    monkeypatch.setattr(
        pixtral.VisionTransformer, "freqs_cis", pixtral.VisionTransformer.__dict__["freqs_cis"]
    )
    monkeypatch.setattr(pixtral.Attention, "forward", pixtral.Attention.forward)
    monkeypatch.setattr(pixtral.PatchMerger, "forward", pixtral.PatchMerger.forward)
    # The block-mask patch lives on transformers, not vllm.
    from transformers.models.pixtral import modeling_pixtral

    monkeypatch.setattr(
        modeling_pixtral,
        "generate_block_attention_mask",
        modeling_pixtral.generate_block_attention_mask,
    )
    yield


def _vision_args(spatial_merge_size: int = 1):
    return pixtral.VisionEncoderArgs(
        hidden_size=HIDDEN_SIZE,
        num_channels=3,
        image_size=128,
        patch_size=16,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        rope_theta=ROPE_THETA,
        image_token_id=10,
        spatial_merge_size=spatial_merge_size,
    )


# ---------------------------------------------------------------------------
# 1. RoPE kernels vs the reference formulas
# ---------------------------------------------------------------------------


def reference_pair_swap(x: torch.Tensor) -> torch.Tensor:
    """Swap each `(2k, 2k+1)` element pair along the last dim."""
    out = x.clone()
    out[..., 0::2] = x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_pair_matrix_equals_pair_swap(head_dim):
    """`x @ M_pair` swaps each `(2k, 2k+1)` pair (no negation — the sign lives
    in the `sin_signed` half of the packed freqs table)."""
    from spyre_inference.multimodal.pixtral import rope_perm_matrix

    torch.manual_seed(1)
    x = torch.randn(2, 7, head_dim, dtype=torch.float16)

    m = rope_perm_matrix("pair", head_dim, torch.device("cpu"))
    assert m.shape == (head_dim, head_dim)

    torch.testing.assert_close(torch.matmul(x, m), reference_pair_swap(x))


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_rope_rotate_matmul_matches_rotation_formula(head_dim):
    """`rope_rotate_matmul` is the rotation `x*cos + pair_swap(x)*sin`."""
    from spyre_inference.multimodal.pixtral import rope_perm_matrix, rope_rotate_matmul

    torch.manual_seed(2)
    x = torch.randn(1, 4, 11, head_dim, dtype=torch.float16)
    angles = torch.randn(11, head_dim, dtype=torch.float16)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    m = rope_perm_matrix("pair", head_dim, torch.device("cpu"))
    expected = x * cos + reference_pair_swap(x) * sin

    torch.testing.assert_close(rope_rotate_matmul(x, cos, sin, m), expected)


# ---------------------------------------------------------------------------
# 2. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize(
    "symbol",
    [
        "apply_rotary_emb_vit",
        "precompute_freqs_cis_2d",
        "VisionTransformer",
        "Attention",
        "PatchMerger",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the patches reach for must still exist. They `getattr(..., None)`
    and return silently, so this is the only place a rename is caught."""
    assert getattr(pixtral, symbol, None) is not None, (
        f"vllm.model_executor.models.pixtral.{symbol} is gone — the corresponding "
        "Spyre patch in multimodal/pixtral.py is now a silent no-op and must be updated"
    )


@pytest.mark.pixtral
def test_vision_rope_vit_patch_is_applied_and_idempotent():
    from spyre_inference.multimodal.pixtral import patch_vision_rope_vit

    original_prop = pixtral.VisionTransformer.__dict__["freqs_cis"]

    patch_vision_rope_vit()
    patched = pixtral.apply_rotary_emb_vit
    assert getattr(patched, "_spyre_patched", False) is True
    assert pixtral.VisionTransformer.__dict__["freqs_cis"] is not original_prop

    patch_vision_rope_vit()
    assert pixtral.apply_rotary_emb_vit is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_vision_attention_patch_is_applied_and_idempotent():
    from spyre_inference.multimodal.pixtral import patch_vision_attention

    patch_vision_attention()
    patched = pixtral.Attention.forward
    assert getattr(patched, "_spyre_patched", False) is True

    patch_vision_attention()
    assert pixtral.Attention.forward is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_patch_merger_patch_is_applied_and_idempotent():
    from spyre_inference.multimodal.pixtral import patch_patch_merger

    patch_patch_merger()
    patched = pixtral.PatchMerger.forward
    assert getattr(patched, "_spyre_patched", False) is True

    patch_patch_merger()
    assert pixtral.PatchMerger.forward is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_block_attention_mask_patch_is_applied_and_idempotent():
    from transformers.models.pixtral import modeling_pixtral

    from spyre_inference.multimodal.pixtral import patch_block_attention_mask

    patch_block_attention_mask()
    patched = modeling_pixtral.generate_block_attention_mask
    assert getattr(patched, "_spyre_patched", False) is True

    patch_block_attention_mask()
    assert modeling_pixtral.generate_block_attention_mask is patched, "second call must be a no-op"


@pytest.mark.pixtral
@pytest.mark.parametrize(
    "patch_embeds_list",
    [
        [16],  # one image: a single full-range write
        [16, 16],  # two images: strided sub-block writes, the unsafe case
        [9, 16, 25],  # three unequal images
    ],
)
def test_cpu_block_mask_matches_upstream(patch_embeds_list):
    """The CPU stand-in must reproduce upstream's mask exactly. A device tensor cannot
    be built here; the device path is covered by the two-image e2e test."""
    from transformers.models.pixtral import modeling_pixtral

    from spyre_inference.multimodal.pixtral import patch_block_attention_mask

    patch_block_attention_mask()
    seq = sum(patch_embeds_list)
    embeds = torch.zeros(1, seq, 8, dtype=torch.float16)

    # A CPU tensor takes the passthrough branch, which is upstream verbatim.
    got = modeling_pixtral.generate_block_attention_mask(patch_embeds_list, embeds)

    neg_inf = torch.finfo(torch.float16).min
    want = torch.full((seq, seq), neg_inf, dtype=torch.float16)
    start = 0
    for length in patch_embeds_list:
        want[start : start + length, start : start + length] = 0
        start += length

    assert got.device.type == "cpu"
    assert torch.equal(got[0, 0], want)


@pytest.mark.pixtral
def test_block_mask_blocks_cross_image_attention():
    """Two images must not attend to each other: the off-diagonal blocks stay -inf."""
    from transformers.models.pixtral import modeling_pixtral

    from spyre_inference.multimodal.pixtral import patch_block_attention_mask

    patch_block_attention_mask()
    embeds = torch.zeros(1, 32, 8, dtype=torch.float16)
    mask = modeling_pixtral.generate_block_attention_mask([16, 16], embeds)[0, 0]

    neg_inf = torch.finfo(torch.float16).min
    assert torch.equal(mask[:16, :16], torch.zeros(16, 16, dtype=torch.float16))
    assert torch.equal(mask[16:, 16:], torch.zeros(16, 16, dtype=torch.float16))
    assert torch.equal(mask[:16, 16:], torch.full((16, 16), neg_inf, dtype=torch.float16))
    assert torch.equal(mask[16:, :16], torch.full((16, 16), neg_inf, dtype=torch.float16))


@pytest.mark.pixtral
def test_apply_installs_rope_vit_before_attention():
    """`apply` must install the real rope before the attention patch: the patched
    forward resolves `apply_rotary_emb_vit` by name at call time, so the reverse
    order would leave it reaching upstream's complex rope (no ComplexFloat on Spyre)."""
    from spyre_inference.multimodal import pixtral as spyre_pixtral

    spyre_pixtral.apply(torch.nn.Module(), torch.device("cpu"))

    assert getattr(pixtral.apply_rotary_emb_vit, "_spyre_patched", False) is True
    assert getattr(pixtral.Attention.forward, "_spyre_patched", False) is True


# ---------------------------------------------------------------------------
# 3a. mistral-format 2D rope: real rewrite == upstream complex rope
# ---------------------------------------------------------------------------


class _FreqsStub:
    """Stands in for a `VisionTransformer`: the patched `freqs_cis` property reads
    only `args`, `max_patches_per_side`, `_freqs_cis` and `device`."""

    def __init__(self):
        self.args = _vision_args()
        self.max_patches_per_side = MAX_PATCHES_PER_SIDE
        self._freqs_cis = None
        self.device = torch.device("cpu")


def _positions(num_patches: int) -> torch.Tensor:
    """`[L, 2]` (row, col) patch positions inside the max-patch grid."""
    torch.manual_seed(7)
    rows = torch.randint(0, MAX_PATCHES_PER_SIDE, (num_patches,), dtype=torch.int64)
    cols = torch.randint(0, MAX_PATCHES_PER_SIDE, (num_patches,), dtype=torch.int64)
    return torch.stack([rows, cols], dim=-1)


@pytest.mark.pixtral
@pytest.mark.parametrize("num_patches", [1, 17, 67])
def test_real_rope_matches_upstream_complex_rope(num_patches):
    """The real (cos, sin_signed) table + pair-swap matmul reproduces upstream's
    complex rotation — where a sign flip or interleave error would hide."""
    from spyre_inference.multimodal.pixtral import patch_vision_rope_vit

    original_apply = pixtral.apply_rotary_emb_vit

    torch.manual_seed(11)
    xq = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    xk = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    positions = _positions(num_patches)

    # Upstream reference: complex table, advanced-index gather, complex multiply.
    complex_table = pixtral.precompute_freqs_cis_2d(
        dim=HEAD_DIM,
        height=MAX_PATCHES_PER_SIDE,
        width=MAX_PATCHES_PER_SIDE,
        theta=ROPE_THETA,
    )
    complex_gathered = complex_table[positions[:, 0], positions[:, 1]]
    expected_q, expected_k = original_apply(xq, xk, complex_gathered)

    # Spyre rewrite: real table, flat index_select gather, pair-swap matmul.
    patch_vision_rope_vit()
    table = pixtral.VisionTransformer.__dict__["freqs_cis"].fget(_FreqsStub())
    real_gathered = table[(positions[:, 0], positions[:, 1])]
    assert real_gathered.shape == (num_patches, 2, HEAD_DIM)
    assert not real_gathered.is_complex(), "the table must be real — Spyre has no complex dtype"

    actual_q, actual_k = pixtral.apply_rotary_emb_vit(xq, xk, real_gathered)

    torch.testing.assert_close(actual_q.float(), expected_q.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k.float(), expected_k.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.pixtral
def test_flat_index_gather_matches_2d_index():
    """The `_OnCardFreqsTable` wrapper folds `(row, col)` into `row*W + col` and
    uses `index_select` (Spyre has no `aten::index`). That flattening must agree
    with a plain 2-D advanced index."""
    from spyre_inference.multimodal.pixtral import patch_vision_rope_vit

    patch_vision_rope_vit()

    stub = _FreqsStub()
    table = pixtral.VisionTransformer.__dict__["freqs_cis"].fget(stub)
    flat = stub._freqs_cis.reshape(MAX_PATCHES_PER_SIDE, MAX_PATCHES_PER_SIDE, 2, HEAD_DIM)

    positions = _positions(23)
    gathered = table[(positions[:, 0], positions[:, 1])]

    torch.testing.assert_close(gathered, flat[positions[:, 0], positions[:, 1]])


# ---------------------------------------------------------------------------
# 3b. Padded on-card SDPA == stock SDPA
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize(
    "num_patches",
    [
        64,  # stick-aligned
        67,  # coprime with the 64 stick — the case the stock lowering rejects
    ],
)
@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_padded_vision_attention_matches_stock(tp_group, num_patches, mask_kind):
    """The pad-to-64 + `-inf` mask + crop SDPA must equal upstream's forward: padded
    keys contribute nothing and padded queries are cropped."""

    args = _vision_args()
    layer = pixtral.Attention(args, disable_tp=True).to(torch.float16)
    torch.manual_seed(13)
    for param in layer.parameters():
        param.data.normal_(std=0.02)
    _finish_weight_loading(layer)

    torch.manual_seed(17)
    x = torch.randn(1, num_patches, HIDDEN_SIZE, dtype=torch.float16)
    freqs_cis = pixtral.precompute_freqs_cis_2d(
        dim=HEAD_DIM,
        height=MAX_PATCHES_PER_SIDE,
        width=MAX_PATCHES_PER_SIDE,
        theta=ROPE_THETA,
    ).reshape(-1, HEAD_DIM // 2)[:num_patches]

    if mask_kind == "bool":
        mask = torch.ones(num_patches, num_patches, dtype=torch.bool).tril()
    else:
        mask = torch.zeros(num_patches, num_patches, dtype=torch.float16)
        mask[:, num_patches // 2 :] = torch.finfo(torch.float16).min

    from spyre_inference.multimodal.pixtral import patch_vision_attention

    expected = layer.forward(x, mask, freqs_cis)

    patch_vision_attention()
    actual = pixtral.Attention.forward(layer, x, mask, freqs_cis)

    assert actual.shape == expected.shape == (1, num_patches, HIDDEN_SIZE)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 3c. Padded-mask cache
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
def test_padded_mask_is_cached_across_layers():
    """The tower hands all 24 layers the same mask object; the O(L²) padded mask
    must be built and uploaded once, not per layer."""
    from spyre_inference.multimodal.pixtral import _padded_attn_mask

    mask = torch.ones(67, 67, dtype=torch.bool).tril()
    args = (mask, 1, 67, 128, torch.float16, torch.device("cpu"))

    first = _padded_attn_mask(*args)
    assert all(_padded_attn_mask(*args) is first for _ in range(23))


@pytest.mark.pixtral
def test_padded_mask_cache_misses_on_a_new_mask():
    """A second image brings a new mask object — the cache must not serve the
    previous image's mask."""
    from spyre_inference.multimodal.pixtral import _padded_attn_mask

    tril = torch.ones(67, 67, dtype=torch.bool).tril()
    first = _padded_attn_mask(tril, 1, 67, 128, torch.float16, torch.device("cpu"))

    triu = torch.ones(67, 67, dtype=torch.bool).triu()
    second = _padded_attn_mask(triu, 1, 67, 128, torch.float16, torch.device("cpu"))

    assert second is not first
    assert not torch.equal(second, first)


@pytest.mark.pixtral
def test_padded_mask_is_released_with_its_source_mask():
    """The padded copy is O(L²) — 296 MB at full resolution. Caching it on the mask
    means it dies when upstream drops the mask, not at process exit."""
    import gc
    import weakref

    from spyre_inference.multimodal.pixtral import _padded_attn_mask

    mask = torch.ones(67, 67, dtype=torch.bool).tril()
    padded = weakref.ref(_padded_attn_mask(mask, 1, 67, 128, torch.float16, torch.device("cpu")))
    assert padded() is not None

    del mask
    gc.collect()
    assert padded() is None


@pytest.mark.pixtral
@pytest.mark.parametrize("seq,seq_pad", [(64, 64), (67, 128)])
def test_padded_keys_are_masked_off(seq, seq_pad):
    """Padded key columns must be `-inf` and real ones must stay unmasked; a
    full-attention source mask (all-zero) must not add masking of its own."""
    from spyre_inference.multimodal.pixtral import _padded_attn_mask

    source = torch.zeros(seq, seq, dtype=torch.float16)
    m = _padded_attn_mask(source, 1, seq, seq_pad, torch.float16, torch.device("cpu"))

    assert m.shape == (1, 1, seq_pad, seq_pad)
    neg_inf = torch.finfo(torch.float16).min
    assert (m[:, :, :, seq:] == neg_inf).all(), "padded keys must be masked off"
    assert (m[:, :, :, :seq] == 0).all(), "real keys must be unmasked"


# ---------------------------------------------------------------------------
# 3d. Projector norm offload
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
def test_projector_norm_offload_installs_hooks():
    """A name miss is a silent skip, only visible as an Inductor failure on hardware,
    so the match is asserted here rather than discovered on the device."""
    from spyre_inference.multimodal.pixtral import offload_projector_norm

    model = torch.nn.Module()
    leaf = torch.nn.RMSNorm(8)
    model.add_module("pre_mm_projector_norm", leaf)

    offload_projector_norm(model, torch.device("cpu"))

    assert leaf._forward_pre_hooks, "missing the D2H pre-hook"
    assert leaf._forward_hooks, "missing the H2D post-hook"


# ---------------------------------------------------------------------------
# 3e. PatchMerger regroup on CPU == stock regroup
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize("image_size", [(4, 4), (6, 8)])
def test_patch_merger_cpu_regroup_matches_stock(tp_group, image_size):
    """Moving `permute` (unsupported `aten::im2col`) to CPU must not change the
    result; the `merging_layer` GEMM stays untouched."""
    from spyre_inference.multimodal.pixtral import patch_patch_merger

    spatial_merge_size = 2
    merger = pixtral.PatchMerger(
        vision_encoder_dim=HIDDEN_SIZE,
        spatial_merge_size=spatial_merge_size,
    ).to(torch.float16)
    torch.manual_seed(19)
    merger.merging_layer.weight.data.normal_(std=0.02)
    _finish_weight_loading(merger)

    h, w = image_size
    torch.manual_seed(23)
    x = torch.randn(h * w, HIDDEN_SIZE, dtype=torch.float16)

    expected = merger.forward(x, [image_size])

    patch_patch_merger()
    actual = pixtral.PatchMerger.forward(merger, x, [image_size])

    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# 4. On-card equivalence
#
# The CPU tests above prove the algebra but cannot prove the rewrites lower
# correctly, which is the bug these patches exist for: on-card SDPA at a
# stick-coprime patch count returns wrong values silently.
# ---------------------------------------------------------------------------


# The measured-bad vision shape: 154x154 at patch_size 14 gives an 11x11 grid.
CORRUPTING_PATCHES = 121


@pytest.mark.rotary
@pytest.mark.parametrize("num_patches", [64, CORRUPTING_PATCHES])
def test_rope_rotate_matmul_matches_cpu_on_spyre(num_patches):
    """The rope rotation on-card must equal the same rotation on CPU.

    `torch.matmul(x, m)` here is 4-D @ 2-D, the shape family that torch-spyre#4155
    computes wrongly for 3-D operands (confident garbage, no warning). Nothing
    else covers the 4-D case, so this is the only guard against a silently wrong
    vision rope.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.pixtral import rope_perm_matrix, rope_rotate_matmul

    torch.manual_seed(29)
    # Production layout: [B, patches, heads, head_dim], cos/sin broadcast over heads.
    x = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    angles = torch.randn(num_patches, HEAD_DIM, dtype=torch.float16)
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    m_cpu = rope_perm_matrix("pair", HEAD_DIM, torch.device("cpu"))
    expected = rope_rotate_matmul(x, cos, sin, m_cpu)

    device = torch.device("spyre")
    m_dev = rope_perm_matrix("pair", HEAD_DIM, device)
    actual = rope_rotate_matmul(x.to(device), cos.to(device), sin.to(device), m_dev)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.pixtral
@pytest.mark.parametrize("num_patches", [64, CORRUPTING_PATCHES])
def test_padded_vision_attention_matches_cpu_on_spyre(tp_group, num_patches):
    """The patched vision attention on-card must equal the same forward on CPU.

    At `CORRUPTING_PATCHES` the stock lowering is what `padded_sdpa` works around,
    so this is the test that actually exercises the fix. A regression makes it
    return wrong values rather than raise.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.pixtral import (
        patch_vision_attention,
        patch_vision_rope_vit,
    )

    args = _vision_args()
    layer = pixtral.Attention(args, disable_tp=True).to(torch.float16)
    torch.manual_seed(31)
    for param in layer.parameters():
        param.data.normal_(std=0.02)
    _finish_weight_loading(layer)

    # Both patches, in `apply()`'s order: upstream's complex rope cannot run on the
    # card, so the real table is required here.
    patch_vision_rope_vit()
    patch_vision_attention()

    positions = _positions(num_patches)
    table = pixtral.VisionTransformer.__dict__["freqs_cis"].fget(_FreqsStub())
    freqs_cis = table[(positions[:, 0], positions[:, 1])]

    torch.manual_seed(37)
    x = torch.randn(1, num_patches, HIDDEN_SIZE, dtype=torch.float16)
    # Kept on CPU: the padded mask is assembled host-side and Spyre has no bool.
    mask = torch.ones(num_patches, num_patches, dtype=torch.bool).tril()

    expected = pixtral.Attention.forward(layer, x, mask, freqs_cis)

    device = torch.device("spyre")
    layer = layer.to(device)
    actual = pixtral.Attention.forward(layer, x.to(device), mask, freqs_cis.to(device))

    assert actual.shape == expected.shape == (1, num_patches, HIDDEN_SIZE)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.pixtral
@pytest.mark.parametrize("image_size", [(4, 4), (6, 8)])
def test_patch_merger_matches_cpu_on_spyre(tp_group, image_size):
    """The patched merger on-card must equal the same forward on CPU.

    Only the regroup moves to CPU; the `merging_layer` GEMM runs on the card at a
    merged patch count that need not be stick-aligned."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.pixtral import patch_patch_merger

    merger = pixtral.PatchMerger(
        vision_encoder_dim=HIDDEN_SIZE,
        spatial_merge_size=2,
    ).to(torch.float16)
    torch.manual_seed(41)
    merger.merging_layer.weight.data.normal_(std=0.02)
    _finish_weight_loading(merger)

    h, w = image_size
    torch.manual_seed(43)
    x = torch.randn(h * w, HIDDEN_SIZE, dtype=torch.float16)

    patch_patch_merger()
    expected = pixtral.PatchMerger.forward(merger, x, [image_size])

    device = torch.device("spyre")
    merger = merger.to(device)
    actual = pixtral.PatchMerger.forward(merger, x.to(device), [image_size])

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
