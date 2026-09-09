"""WIP reproducer -- ITS POSITIVE CONTROL DOES NOT PASS YET. Do not file it as-is.

Intended to isolate: a gather's per-core view saturates below the consumer's core count.

Standalone -- torch and torch_spyre only, nothing from spyre-inference -- so it can be
handed to torch-spyre as-is.

One step of a paged-attention inner loop, in the form where both gathered pages are
consumed exactly as gathered (no permute of either, so no restickify reads a page):

    page_k = table_k[index]            # [E, block, head]   E = entries gathered
    page_v = table_v[index]            # [E, block, head]
    scores = page_k @ q                # [E, block, query]  query is the stick axis
    probs  = exp(scores - amax)
    out    = probs.transpose @ page_v  # [E, query, head]

The gathered pages carry every splittable axis of both matmuls, and nothing is broadcast,
so both should be LX-pinnable at any core count. Measured: they pin while the consumer
runs on 8 cores, and stop pinning above that, with

    lx_pinning: <buf> (index) -> core div mismatch: broadcast read on '<consumer>':
        view covers <N> cores but op runs <M>

where N saturates well below M -- even when the entry axis alone is wide enough to supply
M, and even though the index is 2-D (`[E, 1]`), which is documented to remove the
32-entries-per-stick granularity.

Shapes mirror a granite-3.3-8b chunked-prefill step: the tables are a KV cache folded on
(page, kv_head), a row is block_size x head_size, and E rows are gathered per step.

    python scripts/probes/repro_gather_view_width.py

Env: ENTRIES, CORES, Q_LEN, BLOCK_SIZE, HEAD_SIZE, NUM_ROWS, OUT_DIR

STATUS 2026-09-09. Numerics are correct here (max abs diff 5e-03 to 9e-03, matching the
full kernel), and the refusal messages have the same form as the real kernel's. But the
E=8/cores=8 positive control reports 0/4 pinned, while the full kernel
(`lx_transposed_attn_probe.py`, MAX_CORES=8) pins all its K gathers at the *same* chosen
split `((0,4),(1,2))`. So this construction loses something the full kernel has, and until
the control reads PINS none of the other rows are evidence of anything.

The reproducer that does work today is `lx_transposed_attn_probe.py` on this branch: it
shows the pin at MAX_CORES=8 and the failure at 32 cores, deterministically. The right way
to finish this file is reduction *from* that probe -- strip it down until the pin
disappears -- rather than rebuilding the shape from scratch, which is what was tried here.

The E=8/cores=8 row is the positive control: it must PIN. If it does not, this harness is
not detecting residency and no other row means anything.
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="repro_gather_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))

import torch  # noqa: E402
import torch_spyre  # noqa: E402
from torch_spyre._C import (  # noqa: E402
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from torch_spyre._inductor import config as ts_config  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

DTYPE = torch.float16
D = int(os.environ.get("HEAD_SIZE", 128))
B = int(os.environ.get("BLOCK_SIZE", 128))
Q = int(os.environ.get("Q_LEN", 512))
NUM_ROWS = int(os.environ.get("NUM_ROWS", 256))
SCALE = D**-0.5


def rows_outermost_layout(num_rows, block_size, head_size, dtype):
    """The indexed axis at device position 0, so the gather moves only the rows it names."""
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_rows, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def attn_step(table_k, table_v, indices, q, masks, entries, block_size, head_size):
    """Online-softmax loop, as the real kernel runs it: the accumulation across blocks is
    part of what fixes the consumer's core division, so a single block does not reproduce."""
    acc_o = acc_s = acc_m = None
    for blk in range(len(indices)):
        index = indices[blk]
        # 2-D [E, 1] index keeps the entry variable off the index's own stick axis.
        page_k = table_k[index].reshape(entries, block_size, head_size)
        page_v = table_v[index].reshape(entries, block_size, head_size)
        scores = torch.matmul(page_k, q) * SCALE
        scores = scores + masks[blk]
        m = torch.amax(scores, dim=1, keepdim=True)
        probs = torch.exp(scores - m)
        o = torch.matmul(probs.transpose(1, 2), page_v)
        s = probs.sum(dim=1, keepdim=True).transpose(1, 2)
        mq = m.transpose(1, 2)
        if acc_o is None:
            acc_o, acc_s, acc_m = o, s, mq
        else:
            new_max = torch.maximum(acc_m, mq)
            r_old = torch.exp(acc_m - new_max)
            r_new = torch.exp(mq - new_max)
            acc_o = acc_o * r_old + o * r_new
            acc_s = acc_s * r_old + s * r_new
            acc_m = new_max
    return acc_o / acc_s


def run(entries, cores):
    log_before = PLANNER_LOG.stat().st_size if PLANNER_LOG.is_file() else 0
    layout = rows_outermost_layout(NUM_ROWS, B, D, DTYPE)

    k_host = torch.randn(NUM_ROWS, B, D, dtype=DTYPE)
    v_host = torch.randn(NUM_ROWS, B, D, dtype=DTYPE)
    table_k = k_host.to("spyre", device_layout=layout)
    table_v = v_host.to("spyre", device_layout=layout)
    nblocks = int(os.environ.get("NBLOCKS", 2))
    idx_host = [
        (torch.arange(entries, dtype=torch.int32) + blk * entries).reshape(entries, 1)
        for blk in range(nblocks)
    ]
    q_host = torch.randn(entries, D, Q, dtype=DTYPE)
    mask_host = [torch.zeros(entries, B, Q, dtype=DTYPE) for _ in range(nblocks)]

    prev = ts_config.sencores
    ts_config.sencores = cores
    try:
        got = torch.compile(attn_step, dynamic=False)(
            table_k,
            table_v,
            [t.to("spyre") for t in idx_host],
            q_host.to("spyre"),
            [m.to("spyre") for m in mask_host],
            entries,
            B,
            D,
        )
    finally:
        ts_config.sencores = prev

    # Reference: softmax over the concatenated token axis of all blocks.
    k_cat = torch.cat([k_host[i.reshape(-1).long()] for i in idx_host], dim=1).float()
    v_cat = torch.cat([v_host[i.reshape(-1).long()] for i in idx_host], dim=1).float()
    sc = torch.matmul(k_cat, q_host.float()) * SCALE
    p = torch.softmax(sc, dim=1)
    want = torch.matmul(p.transpose(1, 2), v_cat)
    rel = (got.cpu().float() - want).abs().max().item()

    text = PLANNER_LOG.read_text(errors="replace")[log_before:]
    verdicts = re.findall(r"lx_pinning: (\S+) \(index\)\s*.\s*([^\n]+)", text)
    pinned = sum(1 for _b, why in verdicts if why.strip() == "lx")
    reasons = sorted({why.strip()[:94] for _b, why in verdicts if why.strip() != "lx"})
    splits = sorted(set(re.findall(r"work_slice_dims=(\(\([^)]*\)(?:, \([^)]*\))*\))", text)))
    covers = sorted(set(re.findall(r"view covers (\d+) cores but op runs (\d+)", text)))

    print(f"\n--- ENTRIES={entries} CORES={cores}   (numerics max abs diff {rel:.2e})")
    print(f"    gather verdicts: {len(verdicts)}, pinned lx: {pinned}")
    for why in reasons:
        print(f"    refused: {why}")
    print(f"    consumer splits: {splits[:5]}")
    if covers:
        print(f"    view-vs-op cores: {covers}")
    return entries, cores, pinned, len(verdicts), rel


sweep = os.environ.get("ENTRIES")
cases = (
    [(int(sweep), int(os.environ.get("CORES", sweep)))]
    if sweep
    else [(8, 8), (8, 32), (32, 32), (64, 32)]
)
rows = [run(e, c) for e, c in cases]

print("\n=== SUMMARY")
for entries, cores, pinned, total, rel in rows:
    verdict = "PINS" if pinned and pinned == total else ("partial" if pinned else "HBM")
    print(
        f"  E={entries:3d} cores={cores:3d}  pinned {pinned}/{total}  {verdict}  absdiff={rel:.1e}"
    )
print(
    "\nE=8/cores=8 is the positive control and must read PINS. E=8/cores=32 isolates the\n"
    "core count with the shape held fixed; E=32/64 widen the entry axis to supply 32."
)
print(f"artifacts: {OUT}")
