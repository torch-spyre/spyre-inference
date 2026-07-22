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
# Note: the QKV strided-scatter-source probe lives in tests/test_mlp.py as
# test_spyre_strided_scatter_source (xfail strict). It is intentionally not
# duplicated here.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre lacks a native index_select/embedding kernel. "
        "aten.embedding is a registered CPU fallback. This blocks on-device "
        "RoPE cos/sin gather and VocabParallelEmbedding."
    ),
)
def test_spyre_index_select_for_rope(spyre_device):
    """index_select rows from a cache (RoPE cos/sin gather primitive)."""
    cos_sin_cache = torch.randn(2048, 64, dtype=torch.float16, device=spyre_device)
    positions = torch.arange(32, device=spyre_device)
    out = cos_sin_cache.index_select(0, positions)
    expected = cos_sin_cache.cpu().index_select(0, positions.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_spyre_embedding_for_vocab(spyre_device):
    """torch.embedding on a Spyre weight (VocabParallelEmbedding primitive).

    aten.embedding is a registered CPU fallback in torch-spyre
    (torch-spyre#420), so the op moves the weight and indices to CPU,
    computes the embedding there, and returns the result to Spyre. The
    numerical path is correct and the result lands back on-device; only the
    performance cost remains. This test documents current behavior rather
    than xfail-ing it.
    """
    weight = torch.randn(4096, 4096, dtype=torch.float16, device=spyre_device)
    input_ids = torch.randint(0, 4096, (32,), device=spyre_device)
    out = torch.embedding(weight, input_ids)
    expected = torch.embedding(weight.cpu(), input_ids.cpu())
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)
    assert out.device.type == "spyre"


# ---------------------------------------------------------------------------
# 3. Symbolic-offset in-place write
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spyre does not support a symbolic-offset in-place write into a "
        "paged tensor (narrow().copy_() writes row 0 silently). "
        "This blocks eliminating the CPU staging buffer in attention output "
        "scatter and the deprecated torch.ops.spyre.overwrite path."
    ),
)
def test_spyre_symbolic_offset_page_write(spyre_device):
    """Symbolic-offset in-place write into a KV page."""
    page = torch.zeros(2, 256, 64, dtype=torch.float16, device=spyre_device)
    tok = torch.randn(2, 1, 64, dtype=torch.float16, device=spyre_device)
    offset = torch.tensor(37, device=spyre_device)
    page.narrow(1, int(offset.item()), 1).copy_(tok)

    expected_page = torch.zeros(2, 256, 64, dtype=torch.float16)
    expected_page[:, 37, :] = tok.cpu()[:, 0, :]
    torch.testing.assert_close(page.cpu(), expected_page, atol=0, rtol=0)


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
