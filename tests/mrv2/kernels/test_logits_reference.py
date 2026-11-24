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

"""The vocab-wide MRV2 host ops against a plain PyTorch reference.

test_parity.py checks them against triton-cpu, which exists on x86 only; this runs
wherever csrc/ builds, so it covers the s390x and ppc64le SIMD paths too. Logits sit on
a grid of eighths, so no reference op can round differently from the kernel's, and
every vocab size leaves a remainder for the scalar tail. 16421 spans two 16384-token
chunks.
"""

import math

import numpy as np
import pytest
import torch

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels

pytestmark = pytest.mark.skipif(
    not has_sampling_kernels(), reason="csrc/ sampling kernels are not built"
)

VOCABS = [7, 1029, 16421]
NEG_INF = float("-inf")


def _logits(rows: int, vocab: int, seed: int, specials: bool = True) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-160, 160, (rows, vocab), generator=g).float() / 8
    if specials and vocab > 6:
        x[0, [1, vocab // 2]] = float("nan")
        x[0, 2] = float("inf")
        x[0, 3] = NEG_INF
        x[-1, vocab - 1] = float("nan")
    return x


def _nan_as_neg_inf(x: torch.Tensor) -> torch.Tensor:
    """The kernels skip NaN logits; nan_to_num would also clamp +inf."""
    return x.masked_fill(x.isnan(), NEG_INF)


def _same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("vocab", VOCABS)
def test_temperature(vocab: int) -> None:
    logits = _logits(4, vocab, 0)
    temps = torch.tensor([0.0, 1.0, 0.7, 2.5])
    idx = torch.tensor([0, 1, 2, -1], dtype=torch.int32)
    expected = logits.clone()
    expected[2] /= temps[2]
    torch.ops._spyre_C.mrv2_temperature([4], logits, vocab, idx, temps, vocab, 1024)
    _same(logits, expected)


@pytest.mark.parametrize("vocab", VOCABS)
def test_greedy_takes_first_non_nan_max_per_block(vocab: int) -> None:
    logits = _logits(3, vocab, 1)
    logits[1] = float("nan")
    logits[2, :] = 0.0
    logits[2, vocab - 1] = -0.0
    block = 1024
    num_blocks = math.ceil(vocab / block)
    argmax = torch.full((3, num_blocks), -1, dtype=torch.int64)
    local_max = torch.zeros(3, num_blocks)
    zeros = torch.zeros(3, dtype=torch.int64)
    torch.ops._spyre_C.mrv2_gumbel_sample(
        [3, num_blocks], argmax, num_blocks, local_max, num_blocks,
        None, 0, 0, None, None, 0,
        logits, vocab, torch.arange(3, dtype=torch.int32), zeros, zeros, torch.zeros(3),
        vocab, block, False, True, False, False,
    )  # fmt: skip
    for b in range(num_blocks):
        lo = b * block
        part = _nan_as_neg_inf(logits[:, lo : lo + block])
        best = part.max(dim=-1).values
        at = torch.where(best == NEG_INF, 0, part.argmax(dim=-1)) + lo
        assert argmax[:, b].tolist() == at.tolist()
        _same(local_max[:, b], best)


@pytest.mark.parametrize("vocab", VOCABS)
def test_min_p(vocab: int) -> None:
    logits = _logits(3, vocab, 2)
    min_p = torch.tensor([0.0, 0.1, 1.0])
    expected = logits.clone()
    for r in (1, 2):
        threshold = _nan_as_neg_inf(expected[r]).max() + torch.log(min_p[r])
        expected[r][expected[r] < threshold] = NEG_INF
    torch.ops._spyre_C.mrv2_min_p(
        [3], logits, vocab, torch.arange(3, dtype=torch.int32), min_p, vocab, 1024
    )
    _same(logits, expected)


@pytest.mark.parametrize("vocab", VOCABS)
def test_penalties(vocab: int) -> None:
    # Rows 2-4 are one request's draft positions 0-2: earlier drafts count as output.
    g = torch.Generator().manual_seed(3)
    logits = _logits(5, vocab, 3)
    idx = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    local_pos = torch.tensor([0, 0, 0, 1, 2], dtype=torch.int32)
    token_ids = torch.randint(0, vocab, (5,), generator=g, dtype=torch.int32)
    rep = torch.tensor([1.0, 1.3, 0.8])
    freq = torch.tensor([0.0, 0.25, -0.5])
    pres = torch.tensor([0.5, 0.0, -0.75])
    words = math.ceil(vocab / 32)
    prompt = torch.rand(3, vocab, generator=g) < 0.2
    counts = torch.randint(0, 3, (3, vocab), generator=g, dtype=torch.int32)
    counts[torch.rand(3, vocab, generator=g) < 0.7] = 0
    bits = torch.zeros(3, words * 32, dtype=torch.int64)
    bits[:, :vocab] = prompt.long()
    weights = 1 << torch.arange(32, dtype=torch.int64)
    mask = (bits.view(3, words, 32) * weights).sum(-1)
    mask = torch.where(mask >= 1 << 31, mask - (1 << 32), mask).to(torch.int32)

    expected = logits.clone()
    for t in range(5):
        req = int(idx[t])
        count = counts[req].clone()
        p = int(local_pos[t])
        for prev in range(p):
            count[token_ids[t - p + prev + 1]] += 1
        x = expected[t]
        if rep[req] != 1.0:
            seen = prompt[req] | (count > 0)
            scale = torch.where(x > 0, torch.tensor(1.0) / rep[req], rep[req])
            x = torch.where(seen, x * scale, x)
        x = x - freq[req] * count.float()
        expected[t] = x - pres[req] * (count > 0).float()

    torch.ops._spyre_C.mrv2_penalties(
        [5], logits, vocab, idx, token_ids, local_pos, rep, freq, pres,
        mask, words, counts, vocab, vocab, 1024,
    )  # fmt: skip
    _same(logits, expected)


@pytest.mark.parametrize("vocab", VOCABS)
def test_topk_log_softmax(vocab: int) -> None:
    logits = _logits(3, vocab, 4, specials=False)
    topk_ids = torch.randint(0, vocab, (3, 4), generator=torch.Generator().manual_seed(4))
    out = torch.empty(3 * 4)
    torch.ops._spyre_C.mrv2_topk_log_softmax(
        [3], out, logits, vocab, topk_ids, 4, 4, vocab, 1024, 4
    )
    expected = logits.double().log_softmax(-1).gather(1, topk_ids).float().flatten()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("vocab", VOCABS)
def test_ranks(vocab: int) -> None:
    logits = _logits(3, vocab, 5)
    token_ids = torch.tensor([0, vocab // 3, vocab - 2])
    out = torch.empty(3, dtype=torch.int64)
    torch.ops._spyre_C.mrv2_ranks([3], out, logits, vocab, token_ids, vocab, 1024)
    target = logits.gather(1, token_ids[:, None])
    assert out.tolist() == (logits >= target).sum(-1).tolist()


@pytest.mark.parametrize("vocab", VOCABS)
def test_num_nans(vocab: int) -> None:
    logits = _logits(3, vocab, 6)
    logits[1] = float("nan")
    out = torch.empty(3, dtype=torch.int64)
    torch.ops._spyre_C.mrv2_num_nans([3], logits, vocab, out, vocab, 1024)
    assert out.tolist() == logits.isnan().sum(-1).tolist()


@pytest.mark.parametrize("vocab", VOCABS)
def test_grammar_bitmask(vocab: int) -> None:
    # Request 0 has two logits rows, request 1 one; the mask for request 1's second
    # slot is out of range and skipped. A bitmask narrower than the vocab disallows the rest.
    g = torch.Generator().manual_seed(7)
    logits = _logits(3, vocab, 7)
    stride = max(1, math.ceil(vocab / 32) - 1)
    allowed = torch.rand(4, stride * 32, generator=g) < 0.6
    allowed[0, :64] = True
    allowed[1, :64] = False
    weights = 1 << torch.arange(32, dtype=torch.int64)
    words = (allowed.long().view(4, stride, 32) * weights).sum(-1)
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
    mask_stride = 2
    mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32)  # req * MASK_STRIDE + slot
    cu = torch.tensor([0, 2, 3], dtype=torch.int32)

    expected = logits.clone()
    for m, row in ((0, 0), (1, 1), (2, 2)):
        keep = torch.zeros(vocab, dtype=torch.bool)
        width = min(vocab, stride * 32)
        keep[:width] = allowed[m, :width]
        expected[row][~keep] = NEG_INF

    torch.ops._spyre_C.mrv2_apply_grammar_bitmask(
        [4], logits, vocab, mapping, cu, words, stride, vocab, mask_stride, 1024
    )
    _same(logits, expected)


# ---------------------------------------------------------------- Gumbel noise
# NumPy models of the kernels' RNGs. The log/exp here and in the kernels (libm or
# SLEEF) may differ in the last bit, so values compare to 1e-5; the random logits
# never put two candidates that close, so the sampled tokens must match exactly.

M32 = 0xFFFFFFFF


def _rotl(v: np.ndarray, s: int) -> np.ndarray:
    return ((v << s) | (v >> (32 - s))) & M32


def _mix(h: np.ndarray, k: np.ndarray) -> np.ndarray:
    k = (k * 0xCC9E2D51) & M32
    k = (_rotl(k, 15) * 0x1B873593) & M32
    h = _rotl(h ^ k, 13)
    return (h * 5 + 0xE6546B64) & M32


def _fmix(h: np.ndarray) -> np.ndarray:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)


def _gumbel_noise(seed: int, pos: int, vocab: int) -> np.ndarray:
    """fp32 noise of tokens 0..vocab-1 at (seed, pos), as sample/gumbel.py draws it."""
    s = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    h = np.array([0], dtype=np.uint64)
    for word in (s & np.uint64(M32), s >> np.uint64(32), np.uint64(pos & M32)):
        h = _mix(h, np.array([word], dtype=np.uint64))
    bits = _fmix(_mix(h, np.arange(vocab, dtype=np.uint64)) ^ 16)
    hi16 = (bits >> 16).astype(np.float32)
    lo16 = (bits & 0xFFFF).astype(np.float32)
    u = hi16 * np.float32(1.52587890625e-05) + (lo16 + np.float32(0.5)) * np.float32(
        2.3283064365386963e-10
    )
    p = np.float32(1 / 8)
    for c in (1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0):
        p = np.float32(c) + u * p
    direct = np.log(np.maximum(np.float32(1) - u, np.float32(5.960464477539063e-08)))
    log1p_neg = np.where(u < np.float32(0.25), -u * p, direct)
    return -np.log(-log1p_neg)


def _first_max(values: torch.Tensor) -> tuple[int, float]:
    """The scalar loops' running strict-> max over a block: NaN never wins."""
    v = _nan_as_neg_inf(values)
    best = v.max().item()
    return (0 if best == NEG_INF else int(v.argmax())), best


@pytest.mark.parametrize("vocab", VOCABS)
def test_gumbel_sample_noisy(vocab: int) -> None:
    logits = _logits(4, vocab, 8)
    temps = torch.tensor([1.0, 0.7, 2.0, 0.0])
    seeds = torch.tensor([11, 12, -13, 14])
    pos = torch.tensor([3, 4, 5, 6])
    block = 1024
    num_blocks = math.ceil(vocab / block)
    argmax = torch.full((4, num_blocks), -1, dtype=torch.int64)
    local_max = torch.zeros(4, num_blocks)
    torch.ops._spyre_C.mrv2_gumbel_sample(
        [4, num_blocks], argmax, num_blocks, local_max, num_blocks,
        None, 0, 0, None, None, 0,
        logits, vocab, torch.arange(4, dtype=torch.int32), seeds, pos, temps,
        vocab, block, False, True, False, False,
    )  # fmt: skip
    for r in range(4):
        x = logits[r]
        if temps[r] != 0:
            noise = torch.from_numpy(_gumbel_noise(int(seeds[r]), int(pos[r]), vocab))
            x = x / temps[r] + noise
        for b in range(num_blocks):
            lo = b * block
            at, best = _first_max(x[lo : lo + block])
            assert int(argmax[r, b]) == lo + at
            torch.testing.assert_close(local_max[r, b].item(), best, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ Philox

PHILOX_M = (0xD2511F53, 0xCD9E8D57)
PHILOX_W = (0x9E3779B9, 0xBB67AE85)


def _philox(c: list[np.ndarray], k0: int, k1: int) -> list[np.ndarray]:
    c = [w.astype(np.uint64) for w in c]
    for rnd in range(10):
        p0 = np.uint64(PHILOX_M[0]) * c[0]
        p1 = np.uint64(PHILOX_M[1]) * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k0, p1 & M32, (p0 >> 32) ^ c[3] ^ k1, p0 & M32]
        if rnd != 9:
            k0, k1 = (k0 + PHILOX_W[0]) & M32, (k1 + PHILOX_W[1]) & M32
    return c


@pytest.mark.parametrize("vocab", VOCABS)
def test_philox_gumbel(vocab: int) -> None:
    rows, width, k0, k1 = 3, 5, 1234, 5678
    logits = _logits(rows, vocab, 9)
    contexts = torch.randint(0, vocab, (rows, width), generator=torch.Generator().manual_seed(9))
    temps = torch.tensor([1.0, 0.5, 0.0])
    argmax = torch.zeros(rows, 2, dtype=torch.int64)
    local_max = torch.zeros(rows, 2)
    torch.ops._spyre_C.mrv2_philox_gumbel(
        [rows, 2], argmax, 2, local_max, 2, logits, vocab, contexts.int(), width,
        None, None, None, None, temps, None, 0, 0, None, None, 0,
        k0, k1, vocab, width, 1024, False, False, False, True,
    )  # fmt: skip
    groups = math.ceil(vocab / 4)
    for r in range(rows):
        state = [np.array([w], dtype=np.uint64) for w in (0x574D4358, width, 0, 0)]
        for off in range(0, width, 4):
            for i in range(4):
                fill = int(contexts[r, off + i]) if off + i < width else M32 - i
                state[i] = state[i] ^ np.uint64(fill)
            state = _philox(state, k0 ^ off, k1)
        words = _philox(
            [np.arange(groups, dtype=np.uint64), *(np.repeat(s, groups) for s in state[:3])],
            int(k0 ^ int(state[3][0])),
            k1 ^ 0x574D544B,
        )
        bits = np.stack(words, axis=1).reshape(-1)[:vocab]
        scaled = ((bits >> 8) + 1).astype(np.float32) * np.float32(2.0**-24)
        u = (scaled.view(np.uint32) - 1).view(np.float32)
        temp = float(temps[r]) or 1.0
        x = _nan_as_neg_inf(logits[r] / temp) - torch.from_numpy(np.log(-np.log(u)))
        at, best = _first_max(x)
        assert argmax[r].tolist() == [at, at]
        torch.testing.assert_close(local_max[r, 0].item(), best, rtol=1e-5, atol=1e-5)
        assert local_max[r, 1] == NEG_INF


# ------------------------------------------------------- masks and the rest


@pytest.mark.parametrize("vocab", VOCABS)
def test_compact_sampling_mask(vocab: int) -> None:
    # Requests sample 2 and 1 rows; mostly -inf rows, as after top-k/top-p.
    g = torch.Generator().manual_seed(10)
    logits = torch.full((3, vocab), NEG_INF)
    logits[torch.rand(3, vocab, generator=g) < 0.05] = 1.0
    logits[0, : min(4, vocab)] = torch.tensor([float("nan"), float("inf"), -0.0, 2.0])[
        : min(4, vocab)
    ]
    rows, kept = 4, 5
    words = (vocab + 7) // 8
    token_ids = torch.full((rows, kept), -1, dtype=torch.int32)
    packed = torch.full((rows, words), 0xAA, dtype=torch.uint8)
    counts = torch.full((rows,), -1, dtype=torch.int32)
    torch.ops._spyre_C.mrv2_compact_sampling_mask(
        [rows], logits, vocab, 1, torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32), token_ids, kept, packed, words, counts,
        vocab, kept, 2, 1024,
    )  # fmt: skip
    for r, source in ((0, 0), (1, 1), (2, 2), (3, None)):
        if source is None:
            assert counts[r] == 0
            continue
        finite = logits[source].isfinite().nonzero().flatten()
        assert counts[r] == finite.numel()
        n = min(kept, finite.numel())
        assert token_ids[r, :n].tolist() == finite[:n].tolist()
        bits = torch.zeros(words * 8, dtype=torch.bool)
        bits[finite] = True
        weights = 1 << torch.arange(8)
        assert packed[r].tolist() == (bits.view(words, 8).long() * weights).sum(-1).tolist()


@pytest.mark.parametrize("vocab", VOCABS)
def test_logit_bias_restores_all_masked_row(vocab: int) -> None:
    # Before min_tokens the stop tokens are masked, unless that masks the whole row.
    logits = _logits(2, vocab, 11)
    stops = torch.tensor([[0, vocab - 1], [1, vocab // 2]], dtype=torch.int32)
    logits[0] = NEG_INF
    logits[0, stops[0].long()] = 1.5
    expected = logits.clone()
    expected[1, stops[1].long()] = NEG_INF
    zeros = torch.zeros(2, dtype=torch.int32)
    torch.ops._spyre_C.mrv2_logit_bias(
        [2], logits, vocab, vocab, torch.arange(2, dtype=torch.int32), zeros,
        torch.zeros(2, 1, dtype=torch.int32), 1, zeros, torch.zeros(2, 1, dtype=torch.int32), 1,
        torch.zeros(2, 1), 1, torch.zeros(2, dtype=torch.int64), torch.full((2,), 5),
        torch.full((2,), 2), torch.ones(2, dtype=torch.int32), stops, 2, 8, 1024, True,
    )  # fmt: skip
    _same(logits, expected)


# ------------------------------------------------------ rejection sampler


def _nan_first_max(values: torch.Tensor) -> int:
    """tl.max(return_indices=True): the first NaN, else the first max."""
    nan = values.isnan().nonzero()
    return int(nan[0]) if nan.numel() else int(values.argmax())


@pytest.mark.parametrize("vocab", VOCABS)
def test_local_logits_stats(vocab: int) -> None:
    # Requests of 2 drafts + bonus; request 0 is greedy. Bonus rows are left alone.
    g = torch.Generator().manual_seed(12)
    target = _logits(6, vocab, 12)
    draft = torch.randn(2, 3, vocab, generator=g) * 4
    req = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    temps = torch.tensor([0.0, 0.7])
    block = 1024
    nb = math.ceil(vocab / block)
    t_arg = torch.full((6, nb), -1, dtype=torch.int64)
    t_max, t_sum, d_max, d_sum = (torch.full((6, nb), 7.0) for _ in range(4))
    torch.ops._spyre_C.mrv2_compute_local_logits_stats(
        [6, nb], t_arg, nb, t_max, nb, t_sum, nb, d_max, nb, d_sum, nb, target, vocab,
        draft, 3 * vocab, vocab, req, local_pos, temps, vocab, 6, block, True,
    )  # fmt: skip
    for i in (0, 1, 3, 4):
        for b in range(nb):
            lo = b * block
            t = target[i, lo : lo + block]
            if i < 3:
                at = _nan_first_max(t)
                assert int(t_arg[i, b]) == lo + at
                _same(t_max[i, b], t[at])
                continue
            d = draft[1, int(local_pos[i]), lo : lo + block] / temps[1]
            for x, m_out, s_out in ((t, t_max, t_sum), (d, d_max, d_sum)):
                m = _nan_as_neg_inf(x).max()
                _same(m_out[i, b], m)
                s = (x - m).exp().nansum() if m > NEG_INF else torch.tensor(0.0)
                torch.testing.assert_close(s_out[i, b], s, rtol=1e-5, atol=0)
    assert (t_arg[[2, 5]] == -1).all() and (t_max[[2, 5]] == 7.0).all()


@pytest.mark.parametrize("vocab", VOCABS)
def test_resample(vocab: int) -> None:
    # Request 0 rejected draft 1 (residual of target over draft); request 1 took
    # every draft and samples its bonus token from the target.
    g = torch.Generator().manual_seed(13)
    target = torch.randn(6, vocab, generator=g) * 4
    draft = torch.randn(2, 3, vocab, generator=g) * 4
    req = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    rejected = torch.tensor([1, 2], dtype=torch.int32)
    draft_sampled = torch.randint(0, vocab, (6,), generator=g, dtype=torch.int32)
    temps, seeds, pos = torch.tensor([0.8, 1.2]), torch.tensor([21, 22]), torch.arange(6)
    t_lse = torch.stack([target[1].logsumexp(-1), target[5].logsumexp(-1)])
    d_lse = torch.stack([(draft[0, 1] / temps[0]).logsumexp(-1), torch.tensor(0.0)])
    block = 1024
    nb = math.ceil(vocab / block)
    out_arg = torch.full((2, nb), -1, dtype=torch.int64)
    out_max = torch.zeros(2, nb)
    torch.ops._spyre_C.mrv2_resample(
        [2, nb], out_arg, nb, out_max, nb, target, vocab, t_lse, draft, 3 * vocab, vocab,
        d_lse, rejected, cu, req, draft_sampled, temps, seeds, pos, None, None, 0, None,
        None, 0, 0, vocab, block, True, False, False, 0, False, False,
    )  # fmt: skip
    tl = target[1] - t_lse[0]
    ratio = ((draft[0, 1] / temps[0] - d_lse[0]) - tl).exp()
    residual = [torch.where(ratio < 1, tl + torch.log1p(-ratio), NEG_INF), target[5]]
    for r, token_idx in ((0, 1), (1, 5)):
        x = residual[r] + torch.from_numpy(_gumbel_noise(int(seeds[r]), token_idx, vocab))
        for b in range(nb):
            lo = b * block
            at = _nan_first_max(x[lo : lo + block])
            assert int(out_arg[r, b]) == lo + at
            torch.testing.assert_close(out_max[r, b], x[lo + at], rtol=1e-4, atol=1e-4)
