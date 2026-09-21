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

"""K/V page LX residency of the head-major per-sequence attention kernel.

Runs the kernel over a head-major paged KV cache, checks it against SDPA over the same
pages, then reports the layout planner's residency verdict for every op in the gathered
page's chain (gather -> transpose -> matmul), so "LX-resident from the initial gather to
the point it is discarded" is read off the planner rather than inferred.

    Q_LEN=1 LAYOUT_SOLVER=greedy python scripts/probes/lx_head_major_residency.py

Needs torch-spyre#4153 for the K pages to pin; SPYRE_LX_PLANNER_RELAYOUT=0 shows what
their residency costs without it.

Env: Q_LEN, SEQ_LEN, BLOCK_SIZE, KV_HEADS, QPK, HEAD_SIZE, NUM_PAGES, MAX_CORES, OUT_DIR
"""

import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Import this checkout's spyre_inference, not whichever one the shared venv's editable
# install happens to point at.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_hm_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))

import torch  # noqa: E402
import torch_spyre  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from torch_spyre._inductor import config as ts_config  # noqa: E402

from spyre_inference.v1.attention.ops.layout import head_major_kv_layout  # noqa: E402
from spyre_inference.v1.attention.ops.page_attn_head_major_decode import (  # noqa: E402
    page_attn_head_major_decode_kernel,
)


def _int(name, default):
    return int(os.environ.get(name, default))


KV = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
D = _int("HEAD_SIZE", 128)
B = _int("BLOCK_SIZE", 128)
Q_LEN = _int("Q_LEN", 1)
SEQ_LEN = _int("SEQ_LEN", 512)
NUM_HEADS = KV * QPK
NUM_BLOCKS = (SEQ_LEN + B - 1) // B
NUM_PAGES = _int("NUM_PAGES", max(NUM_BLOCKS, 8))
CTX = SEQ_LEN - Q_LEN
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min
KERNEL = page_attn_head_major_decode_kernel
# Mirrors _lx_max_cores over the kernel's output units.
OUTPUT_UNITS = NUM_HEADS * Q_LEN
MAX_CORES = _int("MAX_CORES", 8 if OUTPUT_UNITS < 32 else 0)

print(
    f"config: Q_LEN={Q_LEN} SEQ_LEN={SEQ_LEN} CTX={CTX} "
    f"NUM_BLOCKS={NUM_BLOCKS} BLOCK_SIZE={B} KV={KV} QPK={QPK} D={D} NUM_PAGES={NUM_PAGES}"
)
print(
    f"solver={ts_config.layout_solver} lx_planning={ts_config.lx_planning} "
    f"co_opt={ts_config.co_optimizing_lx_planning} "
    f"relayout={getattr(ts_config, 'lx_planner_relayout', 'MISSING')} "
    f"max_cores={MAX_CORES or 'uncapped'} "
    f"kernel={KERNEL.__name__} output_units={OUTPUT_UNITS}"
)

fails = []


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


shape = (NUM_PAGES, KV, B, D)
layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_host = torch.randn(shape, dtype=torch.float16)
v_host = torch.randn(shape, dtype=torch.float16)
k_dev = k_host.to("spyre", device_layout=layout)
v_dev = v_host.to("spyre", device_layout=layout)
# Not exact: a host->device fp16 roundtrip quantizes, whatever the layout. This only
# confirms the layout put the bytes where the logical shape says they are.
check("cache roundtrip K", k_dev.cpu(), k_host, 8e-3)
check("cache roundtrip V", v_dev.cpu(), v_host, 8e-3)

STAGING_ROWS = max(Q_LEN, 8)
query = torch.randn(STAGING_ROWS, NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32)
pages_used = torch.arange(NUM_BLOCKS, dtype=torch.int32)

# Causal mask: query row q sits at absolute position CTX + q.
q_abs = CTX + torch.arange(Q_LEN).unsqueeze(1)
masks = []
for i in range(NUM_BLOCKS):
    p = i * B + torch.arange(B).unsqueeze(0)
    allow = (p <= q_abs) & (p < SEQ_LEN)
    tile = torch.zeros(Q_LEN, B, dtype=torch.float16)
    tile[~allow] = FP16_MIN
    masks.append(tile.contiguous())

head_ids = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
kv_tables = [(int(pages_used[i]) * KV + head_ids).contiguous() for i in range(NUM_BLOCKS)]
args = (
    query.to("spyre"),
    row_index.to("spyre"),
    k_dev.view(NUM_PAGES * KV, B, D),
    v_dev.view(NUM_PAGES * KV, B, D),
    [t.to("spyre") for t in kv_tables],
    [m.to("spyre") for m in masks],
    SCALE,
    NUM_BLOCKS,
    Q_LEN,
    NUM_HEADS,
    KV,
    D,
    B,
)

prev_cores = ts_config.sencores
if MAX_CORES:
    ts_config.sencores = MAX_CORES
try:
    got = torch.compile(KERNEL, dynamic=False)(*args).cpu()[:Q_LEN]
finally:
    ts_config.sencores = prev_cores

# Independent SDPA reference over the same pages.
k_ctx = torch.cat([k_host[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_host[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat([m.float() for m in masks], dim=-1)
q = query[:Q_LEN].float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
scores = torch.matmul(q, k_ctx.float().unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
probs = torch.softmax(scores, dim=-1)
want = torch.matmul(probs, v_ctx.float().unsqueeze(1)).reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)
check("attention vs SDPA", got, want, 2e-2)

text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
verdicts = re.findall(r"lx_pinning: (\S+) \(([^)]+)\) . ([^\n]+)", text)

# The K/V page gathers are 2 per block; the rest is the query-row gather, tiny and not
# what this probe is about.
gathers = [(op, kind, why.strip()) for op, kind, why in verdicts if kind == "index"]
pinned = [op for op, _, why in gathers if why == "lx"]
page_gathers = 2 * NUM_BLOCKS
query_gathers = 1
print(f"\ngathers: {len(gathers)} ops, {len(pinned)} pinned LX")
print(f"  K/V page gathers expected: {page_gathers} (K and V per block)")
print(f"  query-side gathers expected: {query_gathers} (not page residency)")
refused = Counter(why for _, _, why in gathers if why != "lx")
for why, n in refused.most_common():
    print(f"  refused x{n}: {why}")
if len(pinned) < page_gathers:
    fails.append(f"only {len(pinned)} gathers pinned LX, below the {page_gathers} K/V pages")

by_kind: dict[str, Counter] = {}
for op, kind, why in verdicts:
    by_kind.setdefault(kind, Counter())[why.strip()] += 1
print("\nresidency verdict by op kind (the whole graph):")
for kind in sorted(by_kind):
    inner = ", ".join(f"{why}: {n}" for why, n in by_kind[kind].most_common())
    print(f"  {kind:24s} {inner}")

spilled = [(op, kind, why) for op, kind, why in verdicts if why.strip() != "lx"]
print(f"\ntotal ops: {len(verdicts)}, spilled to HBM: {len(spilled)}")
print(f"restickify cross-frame barrier hits: {text.count('read by restickify')}")
print(f"mutation relayout copies: {text.count('mutation relayout copy')}")

# A page broadcast over the query-group axis is materialised as a clone; the folded
# form has no such axis, so any clone here is a page copy that should not exist.
clones = sum(n for kind, c in by_kind.items() if kind == "clone" for n in c.values())
print(f"clone ops (page materialisation): {clones}")

# The decisive check. One gathered page is KV * block_size * head_size fp16 values. If
# any page were spilled, the bundle's HBM pool would have to be at least that big, so a
# pool below one page proves no page ever lands in HBM between gather and discard.
page_kb = KV * B * D * 2 / 1024
pools = []
root = OUT / "inductor-cache" / "inductor-spyre"
for d in sorted(root.glob("*")) if root.is_dir() else []:
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text(errors="replace"))
        if m:
            pools.append((d.name[:56], int(m.group(1)) / 1024))
print(f"\none gathered K or V page: {page_kb:.1f} KB")
for name, kb in pools:
    verdict = "< one page: no page in HBM" if kb < page_kb else ">= one page: a page may be in HBM"
    print(f"  HBM pool {kb:9.1f} KB  ({verdict})  {name}")
if pools and max(kb for _, kb in pools) >= page_kb:
    fails.append("HBM pool is at least one page; K/V pages may be round-tripping")

print(
    f"\nPINREPORT q_len={Q_LEN} blocks={NUM_BLOCKS} "
    f"gathers_pinned={len(pinned)}/{len(gathers)} spilled_ops={len(spilled)} "
    f"refused={dict(refused)}"
)
print("FAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
