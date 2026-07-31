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

"""Strict-xfail probes for torch-spyre primitives blocking CPU fallbacks.

Each test exercises a single primitive that the Granite 3.3 forward path
needs to run fully on-device. They are intentionally strict xfail: when a
primitive starts working in torch-spyre, the corresponding probe flips to
XPASS and we can remove the associated CPU detour in spyre-inference.

All tests run against the real Spyre device when available; otherwise they
skip silently (the same pattern used by test_spyre_attn.py).
"""

import pytest
import torch
import torch.nn.functional as F

from spyre_testing_plugin.pytest_plugin import spyre_available


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


# ---------------------------------------------------------------------------
# 1. Slicing / narrow / select
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre returns a non-contiguous last-dim slice whose values are "
        "correct, but using it as a binary-op operand silently produces "
        "wrong results (the second operand appears to ignore its storage "
        "offset). This blocks removing the CPU detour in SpyreSiluAndMul "
        "(fused gate|up slice) and SpyreParallelLMHead (unpad slice)."
    ),
)
def test_spyre_last_dim_slice(spyre_device):
    """Last-dim slice of a Spyre tensor (fused gate|up path)."""
    x = torch.randn(32, 8192, dtype=torch.float16, device=spyre_device)
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    out = F.silu(gate) * up
    expected = F.silu(x.cpu()[..., :d]) * x.cpu()[..., d:]
    torch.testing.assert_close(out.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre F.linear fails when the output dimension is not a multiple "
        "of 64 * (k * 32) due to a work-division limitation. The on-device "
        "unpad slice is exercised too, but the mismatch comes from the "
        "matmul path. Tracked by torch-spyre#1918."
    ),
)
def test_spyre_lm_head_unpadded_matmul_and_slice(spyre_device):
    """F.linear with non-aligned output dim + on-device unpad slice."""
    hidden = torch.randn(32, 4096, dtype=torch.float16, device=spyre_device)
    weight = torch.randn(32000, 4096, dtype=torch.float16, device=spyre_device)
    logits = F.linear(hidden, weight)
    logits = logits[:, :32000]
    expected = F.linear(hidden.cpu(), weight.cpu())[:, :32000]
    torch.testing.assert_close(logits.cpu(), expected, atol=1e-1, rtol=5e-2)


# ---------------------------------------------------------------------------
# 2. Scatter / index_select / embedding
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre cannot use a non-contiguous (strided) tensor as the source of "
        "an indexed scatter write. Historically this forced SpyreQKVParallelLinear "
        "to D2H its result before returning; the current Spyre path side-steps it "
        "by un-fusing the QKV weight after load. The probe is kept because the "
        "underlying torch-spyre limitation still gates attention's per-token "
        "KV-cache scatter and other rework."
    ),
)
def test_spyre_strided_scatter_source(spyre_device):
    """Scatter write whose source is a non-contiguous strided view.

    Failure path:
      1. qkv.split()        → strided 2D Spyre views
      2. v.view(-1, H, D)   → non-contiguous 3D Spyre tensor (Attention.forward)
      3. kv_cache[idx] = v  → scatter write with strided source
    """
    num_tokens = 16
    num_heads, num_kv_heads, head_size = 8, 2, 64
    q_size, kv_size = num_heads * head_size, num_kv_heads * head_size

    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size,
        dtype=torch.float16,
        device=spyre_device,
    )
    _, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    v = v.view(-1, num_kv_heads, head_size)

    num_blocks, block_size = 4, 8
    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float16,
        device=spyre_device,
    )
    block_indices = torch.zeros(num_tokens, dtype=torch.long, device=spyre_device)
    # Avoid aten.remainder on Spyre; compute offsets on CPU and copy.
    block_offsets = torch.arange(num_tokens, dtype=torch.long) % block_size
    block_offsets = block_offsets.to(spyre_device)
    kv_cache[block_indices, 1, block_offsets] = v


def test_spyre_index_select_for_rope(spyre_device):
    """index_select rows from a cache (RoPE cos/sin gather primitive).

    torch-spyre now has a multi-row index_select kernel, so this passes. The
    single-row case still SIGABRTs (torch-spyre#3418; see
    test_spyre_single_row_index_select), and single-token decode hits that case
    every step, so SpyreRotaryEmbedding.gather_rotation still gathers on CPU. Move
    it on-device once the single-row probe below flips to passing."""
    cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=spyre_device)
    positions = torch.arange(32, device=spyre_device)
    out = cos_sin_cache.index_select(0, positions)
    expected = cos_sin_cache.cpu().index_select(0, positions.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch-spyre's dsc codegen SIGABRTs on a single-row index_select "
        "(!allocNode->layoutDimOrder_.empty(), dsc2.cpp:4023) — the same crash as "
        "the single-row embedding gather (torch-spyre#3418). Single-token decode "
        "steps hit this every step, which is why the RoPE cos/sin gather "
        "(SpyreRotaryEmbedding.gather_rotation) stays on CPU rather than using the "
        "on-device index_select above. When this flips to passing, move that gather "
        "on-device."
    ),
)
def test_spyre_single_row_index_select(spyre_device):
    """A one-row index_select over the 4D RoPE rotation cache (single-token decode)."""
    cache = torch.randn(2048, 2, 2, 64, dtype=torch.float16, device=spyre_device)
    idx = torch.zeros(1, dtype=torch.int64, device=spyre_device)
    out = cache.index_select(0, idx)
    expected = cache.cpu().index_select(0, idx.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# Note: the embedding single-row probe lives in
# tests/test_vocab_parallel_embedding.py::test_single_token_embedding_on_device
# (xfail strict). It is intentionally not duplicated here.


# ---------------------------------------------------------------------------
# 3. Symbolic-offset in-place write
# ---------------------------------------------------------------------------


# Note: eager narrow().copy_() at a constant offset started working in a recent
# torch-spyre dependency bump, so it is no longer xfail here. A follow-up PR
# can remove the torch.ops.spyre.overwrite workaround in the attention backend.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Compiled narrow().copy_() at a data-dependent (SymInt) offset fails "
        "to lower ('shape error in scatter op, can not broadcast [.,1,.] to "
        "[.,u,.]'). This is why slot_mapping is copied to host int constants "
        "before the write instead of indexing pages on-device."
    ),
)
def test_spyre_compiled_narrow_copy_at_symbolic_offset(spyre_device):
    """Compiled row write at a tensor-derived (symbolic) offset fails to lower."""
    page = torch.zeros(2, 256, 64, dtype=torch.float16, device=spyre_device)
    tok = torch.randn(2, 1, 64, dtype=torch.float16, device=spyre_device)
    offset = torch.tensor(37, device=spyre_device)

    @torch.compile(dynamic=False)
    def write(page, tok, off):
        # capture_scalar_outputs keeps off.item() an unbacked SymInt, so the
        # narrow start is genuinely symbolic in the graph (not a constant).
        page.narrow(1, off.item(), 1).copy_(tok)
        return page

    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        write(page, tok, offset)

    expected = torch.zeros(2, 256, 64, dtype=torch.float16)
    expected[:, 37, :] = tok.cpu()[:, 0, :]
    torch.testing.assert_close(page.cpu(), expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4. In-place mul on non-contiguous tensor (LogitsProcessor)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "In-place multiplication on a non-contiguous Spyre tensor triggers "
        "a torch-spyre compile issue. This forces SpyreLogitsProcessor to "
        "call .contiguous() on the logits before downstream scaling."
    ),
)
def test_spyre_inplace_mul_noncontiguous(spyre_device):
    """In-place mul on a transposed/logit-shaped non-contiguous Spyre tensor."""
    logits = torch.randn(32, 32000, dtype=torch.float16, device=spyre_device).t()[:32]
    assert not logits.is_contiguous()
    expected = logits.cpu().clone() * (1.0 / 6.0)
    logits *= 1.0 / 6.0
    torch.testing.assert_close(logits.cpu(), expected, atol=1e-3, rtol=1e-3)
