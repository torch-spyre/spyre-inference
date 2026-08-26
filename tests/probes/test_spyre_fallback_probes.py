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
skip silently (the same pattern used by attention/test_spyre_attn.py).
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


@pytest.mark.parametrize("mode", ["compile", "eager"])
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


# ---------------------------------------------------------------------------
# 2. Matmul output-dimension limitations
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre batchmatmul cannot restickify a size-1 output dimension: "
        "`x[T, in] @ w[in, 1]` fails to lower with 'cannot restickify any input "
        "layout of x to carry x_var=d1' (out=1 case; out>=2 works, so this is "
        "distinct from the 64*(k*32) work-division limit in torch-spyre#1918). "
        "Fails in both eager and compile. "
        "When supported, please adapt "
        "tests/custom_ops/test_mlp.py::test_replicated_matches_reference"
    ),
)
def test_spyre_matmul_output_dim_1(spyre_device, mode):
    """Mirrors spyre_linear_t: out = matmul(x[T, in], weight_t[in, out]) with out=1."""
    x = torch.randn(7, 128, dtype=torch.float16, device=spyre_device)
    weight_t = torch.randn(128, 1, dtype=torch.float16, device=spyre_device)

    def fn(a, b):
        return torch.matmul(a, b)

    if mode == "compile":
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    out = fn(x, weight_t)
    expected = torch.matmul(x.cpu().float(), weight_t.cpu().float())
    torch.testing.assert_close(out.cpu().float(), expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 3. Scatter / index_select / embedding
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre cannot use a non-contiguous (strided) tensor as the source of "
        "this eager 5D advanced-index scatter (torch-spyre#3508); the compiled "
        "index_copy_ takes one (test_spyre_slot_major_scatter_strided_source), "
        "which is how the paged KV cache write stays on device. Historically "
        "the eager gap forced SpyreQKVParallelLinear to D2H before return, and "
        "later to un-fuse QKV after load. Encoder-only attention sidesteps "
        "scatter with host indices + index_select (spyre_encoder_attn.py)."
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

    torch-spyre has a multi-row index_select kernel. The single-row case now works
    too (torch-spyre#3418; see test_spyre_single_row_index_select), so the RoPE
    per-token rotation gather runs on-device in the compile graph."""
    cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=spyre_device)
    positions = torch.arange(32, device=spyre_device)
    out = cos_sin_cache.index_select(0, positions)
    expected = cos_sin_cache.cpu().index_select(0, positions.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_spyre_single_row_index_select(spyre_device):
    """A one-row index_select over the 4D RoPE rotation cache (single-token decode).

    Fixed by torch-spyre#3418; this now guards the on-device RoPE rotation-cache
    gather run inside the compile graph."""
    cache = torch.randn(2048, 2, 2, 64, dtype=torch.float16, device=spyre_device)
    idx = torch.zeros(1, dtype=torch.int64, device=spyre_device)
    out = cache.index_select(0, idx)
    expected = cache.cpu().index_select(0, idx.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# Note: the embedding single-row probe lives in
# tests/custom_ops/test_vocab_parallel_embedding.py::test_single_token_embedding_on_device.
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
# 4. Indirect tensor access in matmul (attention page gathering)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A ZERO-DIM scalar device index silently produces wrong results: "
        "k_pages[torch.tensor(2)] fed through transpose into torch.matmul "
        "diverges from CPU. A ONE-ELEMENT index tensor works and is what the "
        "attention backend uses -- see "
        "test_spyre_indirect_page_gather_one_element_index below. Only this "
        "0-dim form remains broken."
    ),
)
def test_spyre_indirect_matmul_tensor_index(spyre_device):
    """Index a dense tensor by a 0-dim device index before matmul.

    Mirrors the page gather in _create_compilable_page_attn, but with a 0-dim
    index instead of the one-element index the kernel actually passes:
      k_page = k_pages[page_idx].unsqueeze(1).transpose(-2, -1)
      scores = torch.matmul(q, k_page)

    Pages here are head-major, so no permute: only the index form is under test.
    """
    num_kv_heads = 2
    block_size = 64
    head_size = 64
    num_blocks = 4
    query_len = 32

    q = torch.randn(1, num_kv_heads, query_len, head_size, dtype=torch.float16, device=spyre_device)
    k_pages = torch.randn(
        num_blocks,
        num_kv_heads,
        block_size,
        head_size,
        dtype=torch.float16,
        device=spyre_device,
    )
    page_idx = torch.tensor(2, dtype=torch.int32, device=spyre_device)

    @torch.compile(dynamic=False)
    def page_attn(q, k_pages, page_idx):
        k_page = k_pages[page_idx].unsqueeze(1).transpose(-2, -1)
        return torch.matmul(q, k_page)

    scores = page_attn(q, k_pages, page_idx)

    expected = torch.matmul(
        q.cpu(),
        k_pages.cpu()[2].unsqueeze(1).transpose(-2, -1),
    )
    torch.testing.assert_close(scores.cpu(), expected, atol=1e-1, rtol=5e-2)


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("head_size", [64, 128])
def test_spyre_indirect_page_gather_one_element_index(spyre_device, head_size, mode):
    """Guard the page gather used by SpyreAttentionImpl.

    The index must be a one-element tensor taken as a row slice of a stick-wide
    table (`table[b, 0:1]`), which is what SpyreAttentionMetadata.page_index_tables
    provides. Two nearby index forms do NOT work and are deliberately not used:
      - a 0-dim scalar index (see test_spyre_indirect_matmul_tensor_index), and
      - a slice of a plain 1-D index tensor, or of a shared table row, which
        fails to compile rather than returning wrong values.

    index_select works in both modes, so it guards the shape of the gather here.
    The subscript form the kernel uses when compiled is covered by
    test_spyre_indirect_page_gather_subscript_needs_compile.
    """
    num_kv_heads, block_size, num_blocks, query_len = 8, 64, 16, 32
    int32_elems_per_stick = 32
    page = 5

    q = torch.randn(num_kv_heads, 1, query_len, head_size, dtype=torch.float16, device=spyre_device)
    k_pages_cpu = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16)
    k_pages = k_pages_cpu.to(spyre_device)

    table_cpu = torch.zeros(num_blocks, int32_elems_per_stick, dtype=torch.int32)
    table_cpu[0, 0] = page
    table = table_cpu.to(spyre_device)

    def page_attn(q, k_pages, table):
        k_page = k_pages.index_select(0, table[0, 0:1]).squeeze(0).permute(1, 0, 2).unsqueeze(1)
        return torch.matmul(q, k_page.transpose(-2, -1))

    if mode == "compile":
        page_attn = torch.compile(page_attn, dynamic=False)

    scores = page_attn(q, k_pages, table)
    expected = torch.matmul(
        q.cpu(), k_pages_cpu[page].permute(1, 0, 2).unsqueeze(1).transpose(-2, -1)
    )
    torch.testing.assert_close(scores.cpu(), expected, atol=1e-1, rtol=5e-2)


@pytest.mark.parametrize(
    "mode",
    [
        "compile",
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Subscripting a dense Spyre tensor with an int32 index lowers "
                    "to aten.index, which upcasts to int64: eager fails with "
                    "'type conversion from torch.int32 to torch.int64'. Inductor "
                    "folds the conversion away, so the compiled path is fine."
                ),
            ),
        ),
    ],
)
def test_spyre_indirect_page_gather_subscript_needs_compile(spyre_device, mode):
    """`k_pages[idx]` for the page gather: works compiled, fails eager.

    This asymmetry is why _create_compilable_page_attn gathers with index_select,
    which works in both modes.
    """
    num_kv_heads, block_size, head_size, num_blocks, query_len = 8, 64, 128, 16, 32
    int32_elems_per_stick = 32
    page = 5

    q = torch.randn(num_kv_heads, 1, query_len, head_size, dtype=torch.float16, device=spyre_device)
    k_pages_cpu = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16)
    k_pages = k_pages_cpu.to(spyre_device)

    table_cpu = torch.zeros(num_blocks, int32_elems_per_stick, dtype=torch.int32)
    table_cpu[0, 0] = page
    table = table_cpu.to(spyre_device)

    def page_attn(q, k_pages, table):
        k_page = k_pages[table[0, 0:1]].squeeze(0).permute(1, 0, 2).unsqueeze(1)
        return torch.matmul(q, k_page.transpose(-2, -1))

    if mode == "compile":
        page_attn = torch.compile(page_attn, dynamic=False)

    scores = page_attn(q, k_pages, table)
    expected = torch.matmul(
        q.cpu(), k_pages_cpu[page].permute(1, 0, 2).unsqueeze(1).transpose(-2, -1)
    )
    torch.testing.assert_close(scores.cpu(), expected, atol=1e-1, rtol=5e-2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A core's share of a gather operand may span at most 256 MB, and the work "
        "divider splits this shape 4 ways along dim 0, so a 1 GB cache leaves a "
        "256 MB span and is rejected. This caps one layer's dense KV cache below "
        "1 GB (~7168 blocks here), which is why test_long_context_model_load is "
        "skipped. Lifted by chunking the cache or by multi-core indirect access "
        "(torch-spyre#2725, torch-spyre#3499)."
    ),
)
def test_spyre_dense_cache_gather_per_core_span(spyre_device):
    """Gather a page from a 1 GB dense KV cache — the long-context cache size.

    The allocation and the host-to-device transfer both succeed; only the gather
    is rejected, so the limit is on the operand of the page read, not on the
    cache itself.
    """
    num_blocks, block_size, num_kv_heads, head_size = 8192, 64, 8, 128

    k_pages = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16).to(
        spyre_device
    )
    table = torch.zeros(1, 32, dtype=torch.int32)
    table[0, 0] = 3
    table = table.to(spyre_device)

    k_page = k_pages.index_select(0, table[0, 0:1])
    assert k_page.cpu().shape == (1, block_size, num_kv_heads, head_size)


# ---------------------------------------------------------------------------
# 5. Symbolic-offset in-place write
# ---------------------------------------------------------------------------


# The per-token KV-cache write in SpyreAttentionImpl is a narrow().copy_() into a
# page at a slot offset. Eager narrow().copy_() at a constant offset works
# on-device ("eager" mode); only *compiling* it with a data-dependent (SymInt)
# offset fails to lower ("compile" mode, xfail). That is why the loop stays eager
# and copies slot offsets to host int constants rather than indexing pages
# on-device.


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
                    "host int constants before KV writes."
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
# 6. In-place mul on non-contiguous tensor (LogitsProcessor)
# ---------------------------------------------------------------------------


def test_spyre_inplace_mul_noncontiguous(spyre_device):
    """In-place mul on a transposed/logit-shaped non-contiguous Spyre tensor."""
    logits = torch.randn(32, 32000, dtype=torch.float16, device=spyre_device).t()[:32]
    assert not logits.is_contiguous()
    expected = logits.cpu().clone() * (1.0 / 6.0)
    logits *= 1.0 / 6.0
    torch.testing.assert_close(logits.cpu(), expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 7. Attention-result reshape + on-device scatter into output (issue #400)
# ---------------------------------------------------------------------------
#
# These probes guard the on-device path in
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


@pytest.mark.parametrize("source", ["clone", "view"])
def test_spyre_scatter_from_prefix_view_source(spyre_device, source):
    """Slice-assign whose source is a prefix view of a longer tensor.

    The attention kernel returns aligned_max_query_len rows and the write-back
    slices off the padding, so for any sequence shorter than the batch maximum
    the source is a prefix view rather than an exact-size tensor. Shapes mirror
    the batch that first exposed this: a 64-token sequence followed by a
    32-token one, so the short write starts at row 32 and the overrun runs to
    row 96 (q_start + aligned_q) instead of stopping at row 64.

    A single prefix-view slice write in a fresh process is always correct; the
    overrun only shows up once one at a *different* view length has already
    run. The warm-up below arms it, so the verdict does not depend on which
    other tests happened to run first in this process.

    Both sources land correctly as of torch-spyre#3826; before that the ``view``
    case overran, which forced a ``.clone()`` in the write-back.
    """
    num_heads, head_size = 32, 128
    aligned_q, query_len, q_start = 64, 32, 32
    num_tokens = 96

    warm_dst = torch.zeros(
        num_tokens, num_heads, head_size, dtype=torch.float16, device=spyre_device
    )
    warm_src = torch.randn(
        aligned_q, num_heads, head_size, dtype=torch.float16, device=spyre_device
    )
    warm_dst[q_start : q_start + 16] = warm_src[:16]

    output = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float16, device=spyre_device)
    result = torch.randn(aligned_q, num_heads, head_size, dtype=torch.float16, device=spyre_device)

    src = result[:query_len]
    if source == "clone":
        src = src.clone()
    output[q_start : q_start + query_len] = src

    expected = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float16)
    expected[q_start : q_start + query_len] = result.cpu()[:query_len]
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 8. storage_offset on compiled-graph inputs
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch-spyre#3770: a device view with storage_offset != 0 is read from offset 0 "
        "when passed into a compiled region; hence the per-sequence page_index_tables."
    ),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_spyre_compile_input_honors_storage_offset(spyre_device, dtype):
    """A compiled kernel must read a device input from its own storage offset.

    These views are is_contiguous(), so .contiguous() is a no-op; only a real copy works.
    """
    rows, width = 4, 64
    base_cpu = torch.stack([torch.full((rows, width), float(s)) for s in range(3)]).to(dtype)
    base = base_cpu.to(spyre_device)

    @torch.compile(dynamic=False)
    def fn(x):
        return x + x

    for s in range(3):
        view = base[s]
        assert view.is_contiguous() and view.storage_offset() == s * rows * width
        torch.testing.assert_close(fn(view).cpu(), (base_cpu[s] + base_cpu[s]), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 9. Slot-major KV cache: the indirect scatter write
# ---------------------------------------------------------------------------


def _slot_major_cache(num_slots, num_kv_heads, head_size, spyre_device):
    return torch.zeros(num_slots, num_kv_heads, head_size, dtype=torch.float16, device=spyre_device)


def test_spyre_slot_major_scatter_hits_exact_slots(spyre_device):
    """The reshape_and_cache scatter hits exactly slot_mapping.

    Was strict-xfail on torch-spyre#3705: index_copy_ on the default device
    layout silently wrote the wrong rows, which forced a host-allocated cache
    under a pinned slot-outermost layout.
    """
    num_blocks, block_size, num_kv_heads, head_size = 8, 64, 8, 128
    num_slots = num_blocks * block_size

    # Spanning pages and out of order, as a real slot_mapping can be.
    slots = torch.tensor([5, 70, 71, 300, 200, 201, 202, 511], dtype=torch.int32)
    kv_cpu = torch.randn(slots.numel(), num_kv_heads, head_size, dtype=torch.float16)

    def scatter(pages, index, src):
        pages.index_copy_(0, index, src)

    pages = _slot_major_cache(num_slots, num_kv_heads, head_size, spyre_device)
    torch.compile(scatter, dynamic=False)(pages, slots.to(spyre_device), kv_cpu.to(spyre_device))

    expected = torch.zeros(num_slots, num_kv_heads, head_size, dtype=torch.float16)
    expected[slots.long()] = kv_cpu
    got = pages.cpu()

    written = got.ne(0).any(-1).any(-1).nonzero().flatten().tolist()
    assert written == sorted(slots.tolist()), f"scatter hit the wrong rows: {written}"
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "mode",
    [
        "compile",
        pytest.param(
            "eager",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "index_copy_ has no eager on-device path: an int32 index is "
                    "rejected, an int64 one falls back to CPU. This is why "
                    "SpyreAttentionImpl compiles the write kernel unconditionally."
                ),
            ),
        ),
    ],
)
def test_spyre_slot_major_scatter_needs_compile(spyre_device, mode):
    """The write kernel must be compiled; eager either raises or leaves the device."""
    num_slots, num_kv_heads, head_size = 512, 8, 128
    slots = torch.arange(64, 96, dtype=torch.int32)
    kv_cpu = torch.randn(slots.numel(), num_kv_heads, head_size, dtype=torch.float16)

    def scatter(pages, index, src):
        pages.index_copy_(0, index, src)

    if mode == "compile":
        scatter = torch.compile(scatter, dynamic=False)

    pages = _slot_major_cache(num_slots, num_kv_heads, head_size, spyre_device)
    scatter(pages, slots.to(spyre_device), kv_cpu.to(spyre_device))

    assert pages.device.type == "spyre", "scatter left the device"
    expected = torch.zeros(num_slots, num_kv_heads, head_size, dtype=torch.float16)
    expected[slots.long()] = kv_cpu
    torch.testing.assert_close(pages.cpu(), expected, atol=1e-2, rtol=1e-2)


def test_spyre_slot_major_scatter_strided_source(spyre_device):
    """The compiled scatter takes k/v straight from the fused-QKV split; a
    regression here lands wrong data rather than raising."""
    num_tokens, num_heads, num_kv_heads, head_size = 8, 32, 8, 128
    q_size, kv_size = num_heads * head_size, num_kv_heads * head_size
    num_slots = 512
    slots = torch.tensor([5, 70, 71, 300, 200, 201, 202, 511], dtype=torch.int32)

    qkv_cpu = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=torch.float16)

    def kv_views(t):
        _, k, v = t.split([q_size, kv_size, kv_size], dim=-1)
        return (
            k.view(num_tokens, num_kv_heads, head_size),
            v.view(num_tokens, num_kv_heads, head_size),
        )

    pages = _slot_major_cache(num_slots, num_kv_heads, head_size, spyre_device)
    k_dev, _ = kv_views(qkv_cpu.to(spyre_device))
    assert not k_dev.is_contiguous() and k_dev.storage_offset() > 0

    def scatter(pages, index, src):
        pages.index_copy_(0, index, src)

    torch.compile(scatter, dynamic=False)(pages, slots.to(spyre_device), k_dev)

    k_ref, _ = kv_views(qkv_cpu)
    expected = torch.zeros(num_slots, num_kv_heads, head_size, dtype=torch.float16)
    expected[slots.long()] = k_ref
    got = pages.cpu()
    written = got.ne(0).any(-1).any(-1).nonzero().flatten().tolist()
    assert written == sorted(slots.tolist()), f"scatter hit the wrong rows: {written}"
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 9. Scalar pow
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch.pow(x, 3) returns |x| ** 4 on Spyre, so gelu_new degenerates to "
        "the identity for negative inputs. Exponents 2 and 4 are correct. When "
        "this passes, drop custom_ops/activation.py::SpyreNewGELU. Tracked by "
        "torch-spyre#4009."
    ),
)
def test_spyre_scalar_pow_cube(spyre_device):
    """torch.pow with exponent 3 on a device-produced tensor."""
    # x has to come from an on-device op: a host-copied tensor of unaligned width
    # is re-tiled and the comparison stops being meaningful.
    a = torch.randn(8, 256, dtype=torch.float16, device=spyre_device)
    b = torch.randn(256, 3072, dtype=torch.float16, device=spyre_device) / 32
    x = a @ b

    expected = x.cpu().float() ** 3
    torch.testing.assert_close(torch.pow(x, 3).cpu().float(), expected, atol=1e-1, rtol=5e-2)
