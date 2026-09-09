"""K/V page LX residency of the folded LX attention kernel at prefill query lengths.

Drives PR #783's real `_lx_page_attn_kernel` over a (page, kv_head)-folded cache at a
configurable query length, so the same probe covers decode (Q_LEN=1) and chunked
prefill (Q_LEN=512). Verifies numerics against SDPA over the same pages, then reports
the layout planner's per-gather LX verdict and reason.

    Q_LEN=512 SEQ_LEN=2048 python scripts/probes/lx_kv_residency_prefill.py

Env: Q_LEN, SEQ_LEN, BLOCK_SIZE, KV_HEADS, QPK, HEAD_SIZE, NUM_PAGES,
     MAX_CORES (override the impl's core cap), OUT_DIR
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_prefill_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ["SPYRE_LX_KV_LAYOUT"] = "1"
if os.environ.get("MAX_CORES"):
    os.environ["SPYRE_ATTN_MAX_CORES"] = os.environ["MAX_CORES"]

import torch  # noqa: E402
import torch_spyre  # noqa: E402
import torch_spyre._inductor.pass_utils  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from spyre_inference.v1.attention.backends.spyre_attn import (  # noqa: E402
    _attn_max_cores,
    _capped_attn_cores,
    _folded_reshape_and_cache_kernel,
    _lx_page_attn_kernel,
    head_major_kv_layout,
)


def _int(name, default):
    return int(os.environ.get(name, default))


KV = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
D = _int("HEAD_SIZE", 128)
B = _int("BLOCK_SIZE", 128)
Q_LEN = _int("Q_LEN", 512)
SEQ_LEN = _int("SEQ_LEN", 2048)
NUM_HEADS = KV * QPK
NUM_BLOCKS = (SEQ_LEN + B - 1) // B
NUM_PAGES = _int("NUM_PAGES", max(NUM_BLOCKS, 8))
CTX = SEQ_LEN - Q_LEN
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min

print(
    f"config: Q_LEN={Q_LEN} SEQ_LEN={SEQ_LEN} CTX={CTX} NUM_BLOCKS={NUM_BLOCKS} "
    f"BLOCK_SIZE={B} KV={KV} QPK={QPK} D={D} NUM_PAGES={NUM_PAGES}"
)
print(f"impl core cap for output_units={KV * Q_LEN}: {_attn_max_cores(KV * Q_LEN) or 'uncapped'}")

fails = []


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


# Folded (page, kv_head) cache, materialised head-major so the indexed axis is at
# device position 0.
layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)
v_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)

# The folded store, as the runner writes this step's KV.
NUM_TOKENS = 8
slots = torch.arange(NUM_TOKENS, dtype=torch.int64) + 3 * B
pages = torch.div(slots, B, rounding_mode="floor")
offs = slots - pages * B
per_head = [(pages * KV + h) * B + offs for h in range(KV)]
heads = torch.arange(KV, dtype=torch.int64)
rows = ((pages.unsqueeze(1) * KV + heads) * B + offs.unsqueeze(1)).reshape(-1)
key = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
value = torch.randn(NUM_TOKENS, KV, D, dtype=torch.float16)
key_dev, value_dev = key.to("spyre"), value.to("spyre")
k_ref, v_ref = k_dev.cpu(), v_dev.cpu()
k_ref.view(-1, D).index_copy_(0, rows, key_dev.cpu().reshape(-1, D))
v_ref.view(-1, D).index_copy_(0, rows, value_dev.cpu().reshape(-1, D))
k_before = k_dev.cpu().view(-1, D)
torch.compile(_folded_reshape_and_cache_kernel, dynamic=False)(
    key_dev,
    value_dev,
    k_dev.view(-1, D),
    v_dev.view(-1, D),
    [t.to("spyre") for t in per_head],
    KV,
)
check("store K", k_dev.cpu(), k_ref, 0.0)
check("store V", v_dev.cpu(), v_ref, 0.0)
moved = (k_dev.cpu().view(-1, D).index_select(0, rows) - k_before.index_select(0, rows)).abs()
if moved.max() == 0:
    fails.append("store wrote nothing")

STAGING_ROWS = max(Q_LEN, 8)
query = torch.randn(STAGING_ROWS, NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32)
pages_used = torch.arange(NUM_BLOCKS, dtype=torch.int32)
head_ids = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
kv_tables = [(int(pages_used[i]) * KV + head_ids).contiguous() for i in range(NUM_BLOCKS)]
head_tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32) for g in range(QPK)
]

# Chunked-prefill causal mask: query row q sits at absolute position CTX + q and may
# attend to kv position p when p <= CTX + q and p < SEQ_LEN.
q_abs = CTX + torch.arange(Q_LEN).unsqueeze(1)
masks = []
for i in range(NUM_BLOCKS):
    p = i * B + torch.arange(B).unsqueeze(0)
    allow = (p <= q_abs) & (p < SEQ_LEN)
    tile = torch.zeros(Q_LEN, B, dtype=torch.float16)
    tile[~allow] = FP16_MIN
    masks.append(tile.contiguous())

dev_args = (
    query.to("spyre"),
    row_index.to("spyre"),
    k_dev,
    v_dev,
    [t.to("spyre") for t in kv_tables],
    [t.to("spyre") for t in head_tables],
    [m.to("spyre") for m in masks],
    SCALE,
    NUM_BLOCKS,
    Q_LEN,
    NUM_HEADS,
    KV,
    D,
    B,
)

with _capped_attn_cores(KV * Q_LEN):
    got = torch.compile(_lx_page_attn_kernel, dynamic=False)(*dev_args)
got = got.cpu()[:Q_LEN]

# Independent SDPA reference over the same pages.
k_flat = k_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
v_flat = v_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
k_ctx = torch.cat([k_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat([m.float() for m in masks], dim=-1)
q = query[:Q_LEN].float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
scores = torch.matmul(q, k_ctx.unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
probs = torch.softmax(scores, dim=-1)
want = torch.matmul(probs, v_ctx.unsqueeze(1)).reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)
check("attention vs SDPA", got, want, 2e-2)

text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
verdicts = re.findall(r"lx_pinning: (\S+) \(([^)]+)\)\s*.\s*([^\n]+)", text)
pinned = [b for b, kind, why in verdicts if why.strip() == "lx" and kind == "index"]
print(f"\ngathers pinned LX: {len(pinned)} / {2 * NUM_BLOCKS} expected (K+V per block)")
reasons: dict[str, int] = {}
for _b, kind, why in verdicts:
    if kind != "index" or why.strip() == "lx":
        continue
    reasons[why.strip()] = reasons.get(why.strip(), 0) + 1
for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  refused x{n}: {why}")
print(f"all lx_pinning verdicts (any kind): {len(verdicts)}")
print(f"restickify cross-frame barrier hits: {text.count('read by restickify')}")
print(f"mutation relayout copies: {text.count('mutation relayout copy')}")

pools = []
root = OUT / "inductor-cache" / "inductor-spyre"
for d in sorted(root.glob("*")) if root.is_dir() else []:
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text(errors="replace"))
        if m:
            pools.append((d.name[:56], int(m.group(1)) / 1024))
for name, kb in pools:
    print(f"  HBM pool {kb:9.1f} KB  {name}")

proof = "per_core_views_equal" in Path(torch_spyre._inductor.pass_utils.__file__).read_text(
    errors="replace"
)
print(f"torch-spyre #4153 restickify proof present: {proof}")
print(
    f"\nPINREPORT q_len={Q_LEN} seq_len={SEQ_LEN} blocks={NUM_BLOCKS} "
    f"pinned={len(pinned)}/{2 * NUM_BLOCKS} reasons={reasons}"
)
print("FAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
