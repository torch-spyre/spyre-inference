"""Does folding G blocks into one gather spread it across 32 cores and keep K/V in LX?

PR #783's kernel gathers `(page, kv_head)` -- 8 entries -- so the gather can spread over
at most 8 cores. At prefill query lengths the consumer needs 32, so it splits the query
axis instead, which V does not have, and the page becomes a barred broadcast read.

Folding G blocks into the same gather makes the entry axis G * num_kv_heads. At G=4 that
is 32 entries, and both matmuls gain a (block, kv) pair of output axes that K and V both
carry, so a 32-core split needs no query split at all. The online softmax stays exact:
the G partials are combined with a max/rescale reduction over the block axis.

    G=4 Q_LEN=512 SEQ_LEN=512 python scripts/probes/lx_fold_blocks_probe.py

Env: G, Q_LEN, SEQ_LEN, BLOCK_SIZE, KV_HEADS, QPK, HEAD_SIZE, NUM_PAGES, MAX_CORES,
     LAYOUT_SOLVER, OUT_DIR
"""

import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_fold_"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "inductor-cache")
os.environ.setdefault("TORCH_LOGS", "+torch_spyre.inductor")
os.environ.setdefault("SPYRE_INDUCTOR_LOG", "1")
os.environ.setdefault("SPYRE_INDUCTOR_LOG_LEVEL", "DEBUG")
PLANNER_LOG = OUT / "planner.log"
os.environ.setdefault("SPYRE_LOG_FILE", str(PLANNER_LOG))
os.environ["SPYRE_LX_KV_LAYOUT"] = "1"

import torch  # noqa: E402
import torch_spyre  # noqa: E402

torch_spyre._autoload()
torch.spyre.set_device(0)
torch.zeros(1, dtype=torch.float16).to("spyre")

from spyre_inference.v1.attention.backends.spyre_attn import (  # noqa: E402
    head_major_kv_layout,
)


def _int(name, default):
    return int(os.environ.get(name, default))


KV = _int("KV_HEADS", 8)
QPK = _int("QPK", 4)
D = _int("HEAD_SIZE", 128)
B = _int("BLOCK_SIZE", 128)
Q_LEN = _int("Q_LEN", 512)
SEQ_LEN = _int("SEQ_LEN", 512)
G = _int("G", 4)
NUM_HEADS = KV * QPK
NUM_BLOCKS = (SEQ_LEN + B - 1) // B
NUM_PAGES = _int("NUM_PAGES", max(NUM_BLOCKS, 8))
CTX = SEQ_LEN - Q_LEN
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min

assert NUM_BLOCKS % G == 0, f"G={G} must divide NUM_BLOCKS={NUM_BLOCKS}"
NUM_GROUPS = NUM_BLOCKS // G

MAX_CORES = _int("MAX_CORES", 0)
print(
    f"config: G={G} entries_per_gather={G * KV} Q_LEN={Q_LEN} SEQ_LEN={SEQ_LEN} "
    f"NUM_BLOCKS={NUM_BLOCKS} groups={NUM_GROUPS} KV={KV} QPK={QPK} D={D} B={B}"
)
print(f"cores: {MAX_CORES or 32}")

fails = []


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


def block_folded_attn(
    query,
    query_row_index,
    k_pages,
    v_pages,
    kv_index_tables,
    head_index_tables,
    mask_tiles,
    scale,
    num_groups,
    g_blocks,
    padded_query_len,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    block_size,
):
    q_rows = query.index_select(0, query_row_index[:padded_query_len])
    q_groups = [
        q_rows.index_select(1, head_index_tables[g]).transpose(0, 1)
        for g in range(num_queries_per_kv)
    ]

    tile_max: list[torch.Tensor] = []
    tile_sum: list[torch.Tensor] = []
    tile_out: list[torch.Tensor] = []

    span = g_blocks * block_size
    for i in range(num_groups):
        rows = kv_index_tables[i]
        # Entries are ordered kv-major, block-minor, so (kv, G, B, D) collapses to
        # (kv, G*B, D) as a plain view: G blocks become one wider token tile. Identical
        # in shape to the unfolded kernel at block_size = G*block_size, but the gather's
        # entry axis is now kv * G, which is what lets it spread over 32 cores.
        k_f = k_pages[rows].reshape(num_kv_heads, span, head_size)
        v_f = v_pages[rows].reshape(num_kv_heads, span, head_size)
        k_t = k_f.permute(0, 2, 1)
        mask_tile = mask_tiles[i]

        for g in range(num_queries_per_kv):
            scores = torch.matmul(q_groups[g], k_t) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                probs = torch.exp(scores - scores_max)
                tile_max.append(scores_max)
                tile_out.append(torch.matmul(probs, v_f))
                tile_sum.append(probs.sum(dim=-1, keepdim=True))
            else:
                new_max = torch.maximum(tile_max[g], scores_max)
                rescale = torch.exp(tile_max[g] - new_max)
                tile_out[g] = tile_out[g] * rescale
                tile_sum[g] = tile_sum[g] * rescale
                probs = torch.exp(scores - new_max)
                tile_out[g] = tile_out[g] + torch.matmul(probs, v_f)
                tile_sum[g] = tile_sum[g] + probs.sum(dim=-1, keepdim=True)
                tile_max[g] = new_max

    groups = [tile_out[g] / tile_sum[g] for g in range(num_queries_per_kv)]
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(1, num_kv_heads * num_queries_per_kv, padded_query_len, head_size)
    attn = attn.transpose(1, 2)
    return attn.reshape(padded_query_len, num_kv_heads * num_queries_per_kv, head_size)


layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)
v_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)

STAGING_ROWS = max(Q_LEN, 8)
query = torch.randn(STAGING_ROWS, NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32)
pages_used = torch.arange(NUM_BLOCKS, dtype=torch.int32)
head_ids = torch.arange(KV, dtype=torch.int32)

# One [KV*G, 1] index per group, kv-major then block, so the gather output collapses
# to (kv, G*B, D) as a view.
kv_tables = []
for i in range(NUM_GROUPS):
    rows = torch.tensor(
        [int(pages_used[i * G + j]) * KV + kv for kv in range(KV) for j in range(G)],
        dtype=torch.int32,
    ).reshape(KV * G, 1)
    kv_tables.append(rows.contiguous())
head_tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32) for g in range(QPK)
]

# One [Q, G*B] tile per group: the G blocks are contiguous along the token axis.
q_abs = CTX + torch.arange(Q_LEN).unsqueeze(1)
mask_tiles = []
for i in range(NUM_GROUPS):
    p = i * G * B + torch.arange(G * B).unsqueeze(0)
    allow = (p <= q_abs) & (p < SEQ_LEN)
    tile = torch.zeros(Q_LEN, G * B, dtype=torch.float16)
    tile[~allow] = FP16_MIN
    mask_tiles.append(tile.contiguous())

dev_args = (
    query.to("spyre"),
    row_index.to("spyre"),
    k_dev,
    v_dev,
    [t.to("spyre") for t in kv_tables],
    [t.to("spyre") for t in head_tables],
    [m.to("spyre") for m in mask_tiles],
    SCALE,
    NUM_GROUPS,
    G,
    Q_LEN,
    KV,
    QPK,
    D,
    B,
)

import contextlib  # noqa: E402


@contextlib.contextmanager
def _cores(n):
    if not n:
        yield
        return
    from torch_spyre._inductor import config as ts_config

    prev = ts_config.sencores
    ts_config.sencores = n
    try:
        yield
    finally:
        ts_config.sencores = prev


with _cores(MAX_CORES):
    got = torch.compile(block_folded_attn, dynamic=False)(*dev_args)
got = got.cpu()[:Q_LEN]

k_flat = k_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
v_flat = v_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
k_ctx = torch.cat([k_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat([mask_tiles[i] for i in range(NUM_GROUPS)], dim=-1).float()
q = query[:Q_LEN].float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
scores = torch.matmul(q, k_ctx.unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
probs = torch.softmax(scores, dim=-1)
want = torch.matmul(probs, v_ctx.unsqueeze(1)).reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)
check("attention vs SDPA", got, want, 2e-2)

text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
verdicts = re.findall(r"lx_pinning: (\S+) \(([^)]+)\)\s*.\s*([^\n]+)", text)
pinned = [b for b, kind, why in verdicts if why.strip() == "lx" and kind == "index"]
want_pinned = 2 * NUM_GROUPS
print(f"\ngathers pinned LX: {len(pinned)} / {want_pinned} expected (one K + one V per group)")
reasons: dict[str, int] = {}
for _b, kind, why in verdicts:
    if kind != "index" or why.strip() == "lx":
        continue
    reasons[why.strip()[:110]] = reasons.get(why.strip()[:110], 0) + 1
for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  refused x{n}: {why}")
print(f"restickify cross-frame barrier hits: {text.count('read by restickify')}")
print(f"no room on scratchpad: {text.count('no room on scratchpad')}")

for d in sorted((OUT / "inductor-cache" / "inductor-spyre").glob("*")):
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text(errors="replace"))
        if m:
            print(f"  HBM pool {int(m.group(1)) / 1024:9.1f} KB  {d.name[:52]}")

print(
    f"\nFOLDREPORT G={G} q_len={Q_LEN} cores={MAX_CORES or 32} pinned={len(pinned)}/{want_pinned}"
)
print("FAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
