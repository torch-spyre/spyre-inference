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

Each test exercises a single primitive that spyre-inference needs on-device
(decoder forward, encoder pack, pooling). They are intentionally strict
xfail: when a primitive starts working in torch-spyre, the corresponding
probe flips to XPASS and we can remove the associated workaround here.

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


@pytest.mark.parametrize(
    "mode",
    [
        "compile",
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                reason=(
                    "Spyre returns a non-contiguous last-dim slice whose values are "
                    "correct, but using it as a binary-op operand silently produces "
                    "wrong results (the second operand appears to ignore its storage "
                    "offset). This blocks removing the CPU detour in SpyreSiluAndMul "
                    "(fused gate|up slice) and SpyreParallelLMHead (unpad slice)."
                ),
            ),
        ),
    ],
)
def test_spyre_last_dim_slice(spyre_device, mode):
    """Last-dim slice of a Spyre tensor (fused gate|up path)."""
    x = torch.randn(32, 8192, dtype=torch.float16, device=spyre_device)

    def fn(x):
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return F.silu(gate) * up

    if mode == "compile":
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    expected = F.silu(x.cpu()[..., : x.shape[-1] // 2]) * x.cpu()[..., x.shape[-1] // 2 :]

    out = fn(x)

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
        "an indexed scatter write (torch-spyre#3508). Historically this forced "
        "SpyreQKVParallelLinear to D2H before return; we side-step that by "
        "un-fusing QKV after load. The same gap keeps encoder-only attention "
        "Q/K/V pack/unpack on CPU (spyre_encoder_attn.py). Once this probe "
        "passes, move encoder ragged→dense packing back onto Spyre."
    ),
)
def test_spyre_strided_scatter_source(spyre_device):
    """Scatter write whose source is a non-contiguous strided view.

    Failure path:
      1. qkv.split()        → strided 2D Spyre views
      2. v.view(-1, H, D)   → non-contiguous 3D Spyre tensor (Attention.forward)
      3. kv_cache[idx] = v  → scatter write with strided source

    Also blocks on-device encoder attention packing (torch-spyre#3508).
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

    torch-spyre has a multi-row index_select kernel. The single-row case now works
    too (torch-spyre#3418; see test_spyre_single_row_index_select), so
    SpyreRotaryEmbedding.gather_rotation gathers on-device."""
    cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=spyre_device)
    positions = torch.arange(32, device=spyre_device)
    out = cos_sin_cache.index_select(0, positions)
    expected = cos_sin_cache.cpu().index_select(0, positions.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_spyre_single_row_index_select(spyre_device):
    """A one-row index_select over the 4D RoPE rotation cache (single-token decode).

    Fixed by torch-spyre#3418; this now guards the on-device gather in
    SpyreRotaryEmbedding.gather_rotation."""
    cache = torch.randn(2048, 2, 2, 64, dtype=torch.float16, device=spyre_device)
    idx = torch.zeros(1, dtype=torch.int64, device=spyre_device)
    out = cache.index_select(0, idx)
    expected = cache.cpu().index_select(0, idx.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# Note: the embedding single-row probe lives in
# tests/test_vocab_parallel_embedding.py::test_single_token_embedding_on_device.
# It is intentionally not duplicated here.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Tensor.index_add_ / aten::index_add is unimplemented on Spyre "
        "(torch-spyre#3507). Upstream MeanPool uses index_add_ for segment "
        "sums; until this works we keep MEAN pooling on CPU. When this probe "
        "passes, add SpyreMeanPool (or drop MEAN from the unsupported list in "
        "configure_pooling_for_spyre) and keep the pooler on Spyre like CLS/LAST."
    ),
)
def test_spyre_index_add_for_mean_pooling(spyre_device):
    """Segment sum via index_add_ (MEAN pooling primitive).

    Shape mirrors a small pooled batch: values [T, H], segment ids [T] →
    out [B, H] with out.index_add_(0, ids, values).
    """
    num_tokens, hidden, num_seqs = 12, 64, 3
    values = torch.randn(num_tokens, hidden, dtype=torch.float16, device=spyre_device)
    # Three sequences of lengths 4, 3, 5 (ragged → flat with segment ids).
    segment_ids = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
        dtype=torch.int64,
        device=spyre_device,
    )
    out = torch.zeros(num_seqs, hidden, dtype=torch.float16, device=spyre_device)
    out.index_add_(0, segment_ids, values)

    expected = torch.zeros(num_seqs, hidden, dtype=torch.float16)
    expected.index_add_(0, segment_ids.cpu(), values.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Fancy indexing (aten::index.Tensor) is unreliable on Spyre for the "
        "shapes pooling / logits selection need. Related gather bugs: "
        "torch-spyre#3499 (L3_ADDEARIMM overflow), #3502 (fused two gathers, "
        "different indices), #3503 (fused two gathers, shared index). "
        "spyre-inference works around this with host-built indices + "
        "index_select (CLS/LAST) and CPU D2H before hidden_states[logits_indices]. "
        "When this probe passes, revisit those workarounds."
    ),
)
def test_spyre_fancy_index_tensor(spyre_device):
    """Row gather via advanced indexing ``hs[idx]`` (aten::index.Tensor).

    Upstream CLSPool / logits selection use this form; we use index_select
    instead. Probe uses a flat [T, H] activation and 1-D int64 row indices.
    """
    hidden_states = torch.randn(32, 128, dtype=torch.float16, device=spyre_device)
    # CLS-style first-token indices for a few sequences (not a simple arange).
    row_indices = torch.tensor([0, 7, 15, 24], dtype=torch.int64, device=spyre_device)
    out = hidden_states[row_indices]
    expected = hidden_states.cpu()[row_indices.cpu()]
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. Symbolic-offset in-place write
# ---------------------------------------------------------------------------


# The per-token KV-cache write in SpyreAttentionImpl is a narrow().copy_() into a
# page at a slot offset. Eager narrow().copy_() at a constant offset works
# on-device ("eager" mode); only *compiling* it with a data-dependent (SymInt)
# offset fails to lower ("compile" mode, xfail). That is why the loop stays eager
# and copies slot offsets to host int constants rather than indexing pages
# on-device. Same compiled gap (torch-spyre#3508) keeps encoder Q/K/V pack on CPU.


@pytest.mark.parametrize(
    "mode",
    [
        "eager",
        pytest.param(
            "compile",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Compiled narrow().copy_() at a data-dependent (SymInt) offset "
                    "fails to lower ('shape error in scatter op, can not broadcast "
                    "[.,1,.] to [.,u,.]') — torch-spyre#3508. Only compilation is "
                    "blocked; the eager path works, so slot_mapping is copied to "
                    "host int constants before KV writes. Same gap keeps encoder "
                    "Q/K/V pack on CPU; once this and test_spyre_strided_scatter_source "
                    "pass, move encoder packing back onto Spyre."
                ),
            ),
        ),
    ],
)
def test_spyre_narrow_copy_row_write(spyre_device, mode):
    """Per-token narrow().copy_() row write (KV-cache reshape_and_cache loop).

    Eager works at a constant offset; compiling with a symbolic offset does not.
    """
    page = torch.zeros(2, 256, 64, dtype=torch.float16, device=spyre_device)
    tok = torch.randn(2, 1, 64, dtype=torch.float16, device=spyre_device)

    if mode == "eager":
        page.narrow(1, 37, 1).copy_(tok)
    else:
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


# ---------------------------------------------------------------------------
# 5. Attention-result reshape + on-device scatter into output (issue #400)
# ---------------------------------------------------------------------------
#
# These two probes guard the on-device path in
# SpyreAttentionImpl._online_softmax_attention: the attention kernel returns
# [num_kv_heads, num_queries_per_kv, aligned_q, D] and must become
# [query_len, num_heads, D] written into the caller's output buffer. The
# head-axis transpose+contiguous and the per-seq scatter both run on-device;
# these probes catch a regression if a torch-spyre bump breaks either.


@pytest.mark.parametrize(
    ("head_size", "query_len", "aligned_q"),
    [
        (128, 1, 32),  # single-token decode, Granite 3.3 head_size
        (128, 17, 32),  # prefill chunk shorter than the aligned length
        (64, 8, 32),  # stick-boundary head_size
    ],
)
def test_spyre_attn_result_reshape_head_transpose(spyre_device, head_size, query_len, aligned_q):
    """Head-axis transpose+contiguous+slice of the attention result on device.

    Guards the on-device reshape in SpyreAttentionImpl._online_softmax_attention.

    Mirrors spyre_attn.py:1035-1038:
      [num_kv_heads, num_queries_per_kv, aligned_q, D]
        -> reshape [1, num_heads, aligned_q, D]
        -> transpose(1, 2).contiguous()
        -> [0, :query_len]  == [query_len, num_heads, D]
    """
    num_kv_heads, num_queries_per_kv = 8, 4
    num_heads = num_kv_heads * num_queries_per_kv
    result = torch.randn(
        num_kv_heads,
        num_queries_per_kv,
        aligned_q,
        head_size,
        dtype=torch.float16,
        device=spyre_device,
    )

    def reshape(r):
        r = r.reshape(1, num_heads, aligned_q, head_size)
        r = r.transpose(1, 2).contiguous()
        return r[0, :query_len, :, :]

    out = reshape(result)
    expected = reshape(result.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


def test_spyre_ondevice_scatter_into_output_at_offset(spyre_device):
    """Device->device slice-assign into output rows at a non-zero constant offset.

    q_start is a Python int per trace (spyre_attn.py:938), so the offset is a
    concrete constant. Guards the on-device scatter in
    SpyreAttentionImpl._online_softmax_attention (a non-zero dim-0 offset can
    silently write to row 0 if a torch-spyre bump regresses it)."""
    num_tokens, num_heads, head_size = 48, 32, 128
    q_start, query_len = 16, 17
    output = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float16, device=spyre_device)
    src = torch.randn(query_len, num_heads, head_size, dtype=torch.float16, device=spyre_device)

    output[q_start : q_start + query_len] = src

    expected = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float16)
    expected[q_start : q_start + query_len] = src.cpu()
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
