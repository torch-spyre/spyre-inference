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

"""The csrc/ top-k / top-p filters behind V2's spyre_apply_top_k_top_p."""

import platform
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels
from spyre_inference.v1.worker.spyre.kernels import apply_top_k_top_p_kernels

# Skip, like vLLM's optional extensions, where the kernels are absent (a prebaked image
# without them, a dev checkout that never built them). CI's non-prebaked runs still fail
# on a missing extension: the build_kernels job and the post-build check in
# run-matrix-config enforce it, independent of these tests.
pytestmark = pytest.mark.skipif(
    not has_sampling_kernels(), reason="csrc/ sampling kernels are not built"
)


def _tied_logits(rows: int, vocab: int, ties: str) -> torch.Tensor:
    torch.manual_seed(rows + vocab)
    x = torch.randn(rows, vocab) * 3
    if ties == "fp16":  # Spyre logits are fp16: ties at the k-th value are common
        x = x.half().float()
    elif ties == "wide":  # one tie spans the whole top-k
        x[:, 5:305] = 20.0
    elif ties == "masked":  # a grammar mask leaves few finite tokens
        x[torch.rand(rows, vocab) < 0.99] = float("-inf")
    return x


def _assert_topp_close(
    x: torch.Tensor, actual: torch.Tensor, expected: torch.Tensor, p: torch.Tensor
) -> None:
    """Same bound as the top-p kernel's own claim: a few boundary tokens, mass >= p."""
    finite = torch.isfinite(x).any(dim=-1)
    if not bool(finite.any()):
        return
    x, actual, expected, p = x[finite], actual[finite], expected[finite], p[finite]
    kept = actual > float("-inf")
    assert torch.equal(actual[kept], x[kept])  # survivors keep their logits
    assert (kept.sum(dim=-1) > 0).all()
    diff = ((expected > float("-inf")).sum(dim=-1) - kept.sum(dim=-1)).abs()
    max_kept = int((expected > float("-inf")).sum(dim=-1).max())
    assert int(diff.max()) <= max(3, max_kept // 200)
    mass = torch.where(kept, x.softmax(-1).double(), 0.0).sum(-1)
    assert (mass >= p.double() - 1e-6).all()


def _ks(rows: int, vocab: int) -> torch.Tensor:
    # Per-row k, with the extremes and a disabled row (k == vocab).
    k = torch.randint(1, 129, (rows,))
    k[0], k[-1] = vocab, 1
    if rows > 2:
        k[1] = vocab - 1
    return k


# Vocabs off a SIMD-width multiple exercise the kernels' remainder handling.
@pytest.mark.parametrize("ties", ["none", "fp16", "wide", "masked"])
@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("vocab", [7, 32003, 151936])
def test_topk_filter_matches_pytorch(rows: int, vocab: int, ties: str) -> None:
    x = _tied_logits(rows, vocab, ties)
    # Serving passes InputBatch.top_k as int32. The other tests stay on int64.
    k = _ks(rows, vocab).clamp(max=vocab).to(torch.int32)
    expected = apply_top_k_top_p_pytorch(x.clone(), k, None, allow_cpu_sync=True)
    assert torch.equal(apply_top_k_top_p_kernels(x.clone(), k, None), expected)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("vocab", [32, 64])
@pytest.mark.parametrize("p", [None, 0.9999])
def test_topk_keeps_a_zero_tie_when_denormals_flush(rows: int, vocab: int, p: float | None) -> None:
    """The k-th value is 0. A vocab of 32 or more hits the 16-wide compare, and
    p < 1 also runs the top-k+top-p survivor scan. set_flush_denormal returns
    whether the mode is supported, not the previous mode, so the restore is off."""
    x = torch.full((rows, vocab), -5.0)
    x[:, :4] = torch.tensor([3.0, 2.0, 1.0, 0.0])
    k = torch.full((rows,), 4, dtype=torch.int32)
    p_t = None if p is None else torch.full((rows,), p)
    torch.set_flush_denormal(True)
    try:
        expected = apply_top_k_top_p_pytorch(x.clone(), k, p_t, allow_cpu_sync=True)
        actual = apply_top_k_top_p_kernels(x.clone(), k, p_t)
    finally:
        torch.set_flush_denormal(False)
    assert torch.equal(actual, expected)


# Top-p over the top-k survivors: the same kept values as the full sort. When a
# tie straddles the top-p cut, which tied token survives is arbitrary there.
@pytest.mark.parametrize("ties", ["none", "fp16", "wide", "masked"])
@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("vocab", [7, 32003, 151936])
def test_topk_topp_filter_matches_pytorch(rows: int, vocab: int, ties: str) -> None:
    x = _tied_logits(rows, vocab, ties)
    k = _ks(rows, vocab).clamp(max=vocab)
    k[0] = 1 if rows == 1 else vocab  # one row is top-p only
    p = torch.rand(rows) * 0.5 + 0.5
    p[-1] = 1.0
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_kernels(x.clone(), k, p)
    topk_rows = k < vocab
    if ties == "none":
        assert torch.equal(actual[topk_rows], expected[topk_rows])
    assert torch.equal(
        actual[topk_rows].sort(dim=-1).values, expected[topk_rows].sort(dim=-1).values
    )
    # k == vocab is the row cpu_topk_topp_sampling sends to the approximate top-p kernel.
    topp_only = ~topk_rows
    if bool(topp_only.any()):
        _assert_topp_close(x[topp_only], actual[topp_only], expected[topp_only], p[topp_only])


# Top-p alone is vLLM's kernel: a binary search over probabilities, so its cut
# may differ from the full sort by a few boundary tokens (vllm#62440's bound).
@pytest.mark.parametrize("ties", ["none", "fp16", "masked"])
@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("vocab", [1024, 32003, 151936])
def test_topp_filter_close_to_pytorch(rows: int, vocab: int, ties: str) -> None:
    x = _tied_logits(rows, vocab, ties)
    p = torch.rand(rows) * 0.9 + 0.1
    expected = apply_top_k_top_p_pytorch(x.clone(), None, p)
    actual = apply_top_k_top_p_kernels(x.clone(), None, p)
    _assert_topp_close(x, actual, expected, p)


def test_p_one_with_top_k_disabled_is_unchanged() -> None:
    """k == vocab and p == 1 is the no-op skip. A top-k row with p == 1 is exact."""
    vocab = 64
    x = torch.randn(4, vocab)
    k = torch.tensor([8, vocab, 4, vocab], dtype=torch.int32)
    p = torch.tensor([1.0, 1.0, 0.9, 0.5])
    actual = apply_top_k_top_p_kernels(x.clone(), k, p)
    assert torch.equal(actual[1], x[1])
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p, allow_cpu_sync=True)
    topk_p1 = (k < vocab) & (p >= 1)
    assert torch.equal(actual[topk_p1], expected[topk_p1])
    topp = apply_top_k_top_p_kernels(x.clone(), None, p)
    assert torch.equal(topp[p >= 1], x[p >= 1])


def test_filters_handle_degenerate_rows() -> None:
    vocab = 4099
    x = torch.randn(4, vocab)
    x[0] = float("-inf")  # fully masked: untouched
    x[1] = 1.5  # all tied: top-k keeps every token
    x[2, :] = float("-inf")
    x[2, [3, 70, 4000]] = torch.tensor([1.0, 2.0, 3.0])  # fewer finite than k
    k = torch.full((4,), 10)
    p = torch.full((4,), 0.9)
    expected = apply_top_k_top_p_pytorch(x.clone(), k, None, allow_cpu_sync=True)
    assert torch.equal(apply_top_k_top_p_kernels(x.clone(), k, None), expected)
    out = apply_top_k_top_p_kernels(x.clone(), k, p)
    assert (out[0] == float("-inf")).all()
    assert ((out[1:] > float("-inf")).sum(dim=-1) > 0).all()
    out = apply_top_k_top_p_kernels(x.clone(), None, p)
    assert (out[0] == float("-inf")).all()
    assert ((out[1:] > float("-inf")).sum(dim=-1) > 0).all()


def test_filters_take_strided_and_fp16_logits() -> None:
    x = torch.randn(8, 2 * 32000)[:, ::2]
    assert not x.is_contiguous()
    k = torch.full((8,), 40, dtype=torch.long)
    expected = apply_top_k_top_p_pytorch(x.clone(), k, None, allow_cpu_sync=True)
    assert torch.equal(apply_top_k_top_p_kernels(x, k, None), expected)
    half = x.half()
    out = apply_top_k_top_p_kernels(half, k, None)
    assert out.dtype == torch.float32
    assert torch.equal(out, apply_top_k_top_p_pytorch(half.float(), k, None, allow_cpu_sync=True))


def test_avx2_build_filters_match_pytorch() -> None:
    """CI runners have AVX512, so they never load _C_AVX2 unless a process asks."""
    import spyre_inference

    if platform.machine() != "x86_64":
        pytest.skip("the AVX2 extension is built on x86_64 only")
    root = Path(spyre_inference.__file__).resolve().parent
    if not any(root.glob("_C_AVX2*")):
        pytest.skip("the AVX2 extension is not installed")
    script = (
        "import sys\n"
        "import torch\n"
        "torch.cpu.get_capabilities = lambda: {\n"
        "    'avx2': True, 'fma3': True,\n"
        "    'avx512_f': False, 'avx512_vl': False,\n"
        "    'avx512_bw': False, 'avx512_dq': False,\n"
        "}\n"
        "import spyre_inference.v1.sample.sampling_kernels  # noqa: F401\n"
        "assert 'spyre_inference._C_AVX2' in sys.modules\n"
        "assert 'spyre_inference._C' not in sys.modules\n"
        "from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch\n"
        "from spyre_inference.v1.worker.spyre.kernels import apply_top_k_top_p_kernels\n"
        "def kept(t):\n"
        "    return t > float('-inf')\n"
        "x = torch.randn(4, 32003)\n"
        "k = torch.full((4,), 8)\n"
        "p = torch.full((4,), 0.9)\n"
        "for kk, pp in ((k, None), (k, p)):\n"
        "    actual = apply_top_k_top_p_kernels(x.clone(), kk, pp)\n"
        "    expected = apply_top_k_top_p_pytorch(x.clone(), kk, pp, allow_cpu_sync=True)\n"
        "    assert torch.equal(actual, expected)\n"
        "actual = apply_top_k_top_p_kernels(x.clone(), None, p)\n"
        "expected = apply_top_k_top_p_pytorch(x.clone(), None, p, allow_cpu_sync=True)\n"
        "diff = (kept(expected).sum(-1) - kept(actual).sum(-1)).abs()\n"
        "max_kept = int(kept(expected).sum(-1).max())\n"
        "assert int(diff.max()) <= max(3, max_kept // 200)\n"
        "assert (kept(actual).sum(-1) > 0).all()\n"
        "mass = torch.where(kept(actual), x.softmax(-1).double(), 0.0).sum(-1)\n"
        "assert (mass >= p.double() - 1e-6).all()\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
