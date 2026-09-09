"""Can a query-as-stick ("transposed") attention kernel keep K/V in LX at 32 cores?

`work_division` never splits the last device axis (the stick): `coord_vars` is built from
`output_td.device_coords[:-1]`. In PR #783's kernel the second matmul writes
`[kv, Q, head_size]`, so head_size is the stick and the only splittable axes are kv (8)
and the query. V has no query axis, so its per-core view can only ever mirror the kv
factor -- measured as exactly `cores / 2` at every cap and every block size.

Transposing the kernel moves the query onto the stick axis:

    scoresT = k_page   @ q_gT   -> [kv, block_size, Q]   (K needs NO permute)
    outT    = v_pageT  @ probsT -> [kv, head_size,  Q]

Now the splittable axes are kv and block_size for the first matmul, kv and head_size for
the second -- and the gathered pages carry both in each case, so a 32-core split needs no
broadcast. The online softmax reduces along block_size (dim 1) instead of the last axis.

    Q_LEN=512 SEQ_LEN=1024 python scripts/probes/lx_transposed_attn_probe.py

`FLAT=1 G=4` widens the gather's entry axis to num_kv_heads * G = 32. That variant is
numerically correct but needs an upstream fix to compile at all:
`propagate_layouts._is_supported_layout` constructs a SpyreTensorLayout to *test* a
candidate dim_order and lets a RuntimeError escape, so an invalid candidate aborts the
compile instead of being rejected. Even with that fixed, work_division splits the 32
entries only 4 ways (`((0,4),(1,8))`, the other 8 on block_size) -- invariant to entry
width (8/32/64) and to block_size (128/64). `NO_COMBINE=1` drops the in-kernel G-reduction
and does get the planner to a single 32-way entry split (`((0,32),)`), but the gathers'
own views still cover only 16, so the pages remain in HBM. Best spill measured is the
unfolded transposed kernel: 6920 KB vs 22528 KB for #783 at the same shape.

Env: Q_LEN, SEQ_LEN, BLOCK_SIZE, KV_HEADS, QPK, HEAD_SIZE, NUM_PAGES, MAX_CORES,
     G, FLAT, LAYOUT_SOLVER, OUT_DIR
"""

import contextlib
import os
import re
import tempfile
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp(prefix="lx_tattn_"))
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
SEQ_LEN = _int("SEQ_LEN", 1024)
MAX_CORES = _int("MAX_CORES", 0)
G = _int("G", 1)
NUM_HEADS = KV * QPK
NUM_BLOCKS = (SEQ_LEN + B - 1) // B
NUM_PAGES = _int("NUM_PAGES", max(NUM_BLOCKS, 8))
CTX = SEQ_LEN - Q_LEN
SCALE = D**-0.5
FP16_MIN = torch.finfo(torch.float16).min

print(
    f"config: G={G} entries={KV * G} Q_LEN={Q_LEN} SEQ_LEN={SEQ_LEN} "
    f"NUM_BLOCKS={NUM_BLOCKS} B={B} KV={KV} QPK={QPK} D={D} cores={MAX_CORES or 32}"
)

fails = []


def check(name, got, want, tol):
    err = (got.float() - want.float()).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {err:.3e} (tol {tol:.0e})")
    if not ok:
        fails.append(name)


def flat_folded_attn(
    q_rep_groups,
    k_pages,
    v_pages,
    kv_index_tables,
    mask_tiles,
    entry_pick,
    scale,
    num_groups,
    g_blocks,
    padded_query_len,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    block_size,
    no_combine=False,
):
    """Folded entry axis kept FLAT at num_kv_heads * G, so every op stays 3-D.

    The gather's entry axis is the only axis it can split, so widening it to 32 is what
    lets it spread over 32 cores. Both pages are consumed exactly as gathered -- no
    permute of either -- and the query is replicated per entry instead, which costs one
    copy of q but keeps the pages off every broadcast and restickify path.
    """
    entries = num_kv_heads * g_blocks
    tile_max: list[torch.Tensor] = []
    tile_sum: list[torch.Tensor] = []
    tile_out: list[torch.Tensor] = []

    for i in range(num_groups):
        kv_rows = kv_index_tables[i]
        k_f = k_pages[kv_rows].reshape(entries, block_size, head_size)
        v_f = v_pages[kv_rows].reshape(entries, block_size, head_size)
        mask_tile = mask_tiles[i]

        for g in range(num_queries_per_kv):
            scores = torch.matmul(k_f, q_rep_groups[g]) * scale
            scores = scores + mask_tile
            m = torch.amax(scores, dim=1, keepdim=True)
            p = torch.exp(scores - m)
            o = torch.matmul(p.transpose(1, 2), v_f)
            # Everything stays rank 3: a 2-D [kv, Q] pointwise has no supported layout.
            s = p.sum(dim=1, keepdim=True).transpose(1, 2)
            mq = m.transpose(1, 2)

            if no_combine:
                # Isolates the pages' residency from the combine: the per-entry partials
                # leave the kernel and are reduced on the host instead.
                tile_out.append(o)
                tile_sum.append(s)
                tile_max.append(mq)
                continue

            # Collapse the G partials, one slice per folded block.
            for j in range(g_blocks):
                pick = entry_pick[j]
                o_j = o.index_select(0, pick)
                s_j = s.index_select(0, pick)
                m_j = mq.index_select(0, pick)
                if i == 0 and j == 0:
                    tile_max.append(m_j)
                    tile_out.append(o_j)
                    tile_sum.append(s_j)
                    continue
                new_max = torch.maximum(tile_max[g], m_j)
                r_old = torch.exp(tile_max[g] - new_max)
                r_new = torch.exp(m_j - new_max)
                tile_out[g] = tile_out[g] * r_old + o_j * r_new
                tile_sum[g] = tile_sum[g] * r_old + s_j * r_new
                tile_max[g] = new_max

    if no_combine:
        return tile_out + tile_sum + tile_max

    groups = [tile_out[g] / tile_sum[g] for g in range(num_queries_per_kv)]
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(num_kv_heads * num_queries_per_kv, padded_query_len, head_size)
    return attn.transpose(0, 1)


def folded_transposed_attn(
    q_groups,
    k_pages,
    v_pages,
    kv_index_tables,
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
    """Transposed, with G blocks folded into the gather's entry axis.

    Entries become num_kv_heads * G, so the gather can spread that wide, and the folded
    (kv, G) pair stays a device axis of every page operand. The query is the operand
    broadcast over G -- it is a graph input already in HBM, so nothing is lost by it
    being the broadcast one, whereas broadcasting a page is what bars the page from LX.
    """
    tile_max: list[torch.Tensor] = []
    tile_sum: list[torch.Tensor] = []
    tile_out: list[torch.Tensor] = []

    for i in range(num_groups):
        kv_rows = kv_index_tables[i]
        # Splitting the outermost (entry) axis into (kv, G) is a view.
        k_f = k_pages[kv_rows].reshape(num_kv_heads, g_blocks, block_size, head_size)
        v_f = v_pages[kv_rows].reshape(num_kv_heads, g_blocks, block_size, head_size)
        mask_tile = mask_tiles[i]

        for g in range(num_queries_per_kv):
            qg = q_groups[g].unsqueeze(1)
            # Both pages are consumed exactly as gathered; the only restickify is on
            # probs, an in-kernel buffer, so no page is ever a restickify's source.
            scores = torch.matmul(k_f, qg) * scale
            scores = scores + mask_tile
            m = torch.amax(scores, dim=2, keepdim=True)
            p = torch.exp(scores - m)
            o = torch.matmul(p.transpose(2, 3), v_f)
            s = p.sum(dim=2)
            mq = m.squeeze(2)

            # Collapse the G partials exactly.
            gm = torch.amax(mq, dim=1, keepdim=True)
            r = torch.exp(mq - gm)
            o = (o * r.unsqueeze(-1)).sum(dim=1)
            s = (s * r).sum(dim=1)
            gm = gm.squeeze(1)

            if i == 0:
                tile_max.append(gm)
                tile_out.append(o)
                tile_sum.append(s)
            else:
                new_max = torch.maximum(tile_max[g], gm)
                r_old = torch.exp(tile_max[g] - new_max)
                r_new = torch.exp(gm - new_max)
                tile_out[g] = tile_out[g] * r_old.unsqueeze(-1) + o * r_new.unsqueeze(-1)
                tile_sum[g] = tile_sum[g] * r_old + s * r_new
                tile_max[g] = new_max

    # [kv, Q, head_size] per group, as the unfolded kernel produces.
    groups = [tile_out[g] / tile_sum[g].unsqueeze(-1) for g in range(num_queries_per_kv)]
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(num_kv_heads * num_queries_per_kv, padded_query_len, head_size)
    return attn.transpose(0, 1)


def transposed_attn(
    q_groups,
    k_pages,
    v_pages,
    kv_index_tables,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    block_size,
):
    # q_groups arrive as [kv, head_size, Q], already transposed. Building them in-graph
    # needs a stick-changing permute on a doubly-gathered tensor, which lowers to a
    # 3-arg restickify and is rejected; in the real kernel they would come from the
    # staging buffer prepared outside the graph.
    tile_max: list[torch.Tensor] = []
    tile_sum: list[torch.Tensor] = []
    tile_out: list[torch.Tensor] = []

    for i in range(num_blocks):
        kv_rows = kv_index_tables[i]
        k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        v_page = v_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)
        # K is used as gathered; only V is permuted now.
        v_t = v_page.permute(0, 2, 1)
        mask_tile = mask_tiles[i]

        for g in range(num_queries_per_kv):
            scores = torch.matmul(k_page, q_groups[g]) * scale
            scores = scores + mask_tile
            scores_max = torch.amax(scores, dim=1, keepdim=True)

            if i == 0:
                probs = torch.exp(scores - scores_max)
                tile_max.append(scores_max)
                tile_out.append(torch.matmul(v_t, probs))
                tile_sum.append(probs.sum(dim=1, keepdim=True))
            else:
                new_max = torch.maximum(tile_max[g], scores_max)
                rescale = torch.exp(tile_max[g] - new_max)
                tile_out[g] = tile_out[g] * rescale
                tile_sum[g] = tile_sum[g] * rescale
                probs = torch.exp(scores - new_max)
                tile_out[g] = tile_out[g] + torch.matmul(v_t, probs)
                tile_sum[g] = tile_sum[g] + probs.sum(dim=1, keepdim=True)
                tile_max[g] = new_max

    # [KV, D, Q] per group -> [Q, NUM_HEADS, D]
    groups = [tile_out[g] / tile_sum[g] for g in range(num_queries_per_kv)]
    attn = torch.stack(groups, dim=1)
    attn = attn.reshape(num_kv_heads * num_queries_per_kv, head_size, padded_query_len)
    return attn.permute(2, 0, 1)


layout = head_major_kv_layout(NUM_PAGES * KV, B, D, torch.float16)
k_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)
v_dev = torch.randn(NUM_PAGES * KV, B, D, dtype=torch.float16).to("spyre", device_layout=layout)

STAGING_ROWS = max(Q_LEN, 8)
query = torch.randn(STAGING_ROWS, NUM_HEADS, D, dtype=torch.float16)
row_index = torch.arange(Q_LEN, dtype=torch.int32)
pages_used = torch.arange(NUM_BLOCKS, dtype=torch.int32)
head_ids = torch.arange(KV, dtype=torch.int32).reshape(KV, 1)
assert NUM_BLOCKS % G == 0, f"G={G} must divide NUM_BLOCKS={NUM_BLOCKS}"
NUM_GROUPS = NUM_BLOCKS // G
if G == 1:
    kv_tables = [(int(pages_used[i]) * KV + head_ids).contiguous() for i in range(NUM_BLOCKS)]
else:
    # kv-major, block-minor: the entry axis splits into (kv, G) as a view.
    kv_tables = [
        torch.tensor(
            [int(pages_used[i * G + j]) * KV + kv for kv in range(KV) for j in range(G)],
            dtype=torch.int32,
        )
        .reshape(KV * G, 1)
        .contiguous()
        for i in range(NUM_GROUPS)
    ]
head_tables = [
    torch.tensor([kv * QPK + g for kv in range(KV)], dtype=torch.int32) for g in range(QPK)
]

# Mask is now [block_size, Q] and broadcasts over kv.
q_abs = CTX + torch.arange(Q_LEN).unsqueeze(0)
mask_flat = []
for i in range(NUM_BLOCKS):
    pos = i * B + torch.arange(B).unsqueeze(1)
    allow = (pos <= q_abs) & (pos < SEQ_LEN)
    tile = torch.zeros(B, Q_LEN, dtype=torch.float16)
    tile[~allow] = FP16_MIN
    mask_flat.append(tile.contiguous())
if G == 1:
    mask_tiles = mask_flat
else:
    mask_tiles = [
        torch.stack(mask_flat[i * G : (i + 1) * G]).unsqueeze(0).contiguous()
        for i in range(NUM_GROUPS)
    ]

q_groups_host = [
    query[:Q_LEN].index_select(1, head_tables[g].long()).permute(1, 2, 0).contiguous()
    for g in range(QPK)
]
FLAT = os.environ.get("FLAT") == "1"
if FLAT:
    # entry index = kv * G + j, so q is replicated G times per kv head.
    rep = torch.tensor([kv for kv in range(KV) for _ in range(G)], dtype=torch.int64)
    q_rep_host = [q_groups_host[g].index_select(0, rep).contiguous() for g in range(QPK)]
    entry_pick = [
        torch.tensor([kv * G + j for kv in range(KV)], dtype=torch.int32) for j in range(G)
    ]
    # Per-entry mask: entry (kv, j) uses block i*G+j.
    mask_flat_tiles = [
        torch.stack([mask_flat[i * G + j] for _kv in range(KV) for j in range(G)]).contiguous()
        for i in range(NUM_GROUPS)
    ]

if FLAT:
    dev_args = (
        [t.to("spyre") for t in q_rep_host],
        k_dev,
        v_dev,
        [t.to("spyre") for t in kv_tables],
        [m.to("spyre") for m in mask_flat_tiles],
        [t.to("spyre") for t in entry_pick],
        SCALE,
        NUM_GROUPS,
        G,
        Q_LEN,
        KV,
        QPK,
        D,
        B,
        os.environ.get("NO_COMBINE") == "1",
    )
    _kernel = flat_folded_attn
else:
    dev_args = (
        [t.to("spyre") for t in q_groups_host],
        k_dev,
        v_dev,
        [t.to("spyre") for t in kv_tables],
        [m.to("spyre") for m in mask_tiles],
        SCALE,
        *((NUM_BLOCKS,) if G == 1 else (NUM_GROUPS, G)),
        Q_LEN,
        KV,
        QPK,
        D,
        B,
    )
    _kernel = transposed_attn if G == 1 else folded_transposed_attn


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
    got = torch.compile(_kernel, dynamic=False)(*dev_args)
if os.environ.get("NO_COMBINE") == "1":
    n = len(got) // 3
    print(f"no-combine: returned {len(got)} partial tensors, shapes {tuple(got[0].shape)}")
    got = None
else:
    got = got.cpu()[:Q_LEN]

k_flat = k_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
v_flat = v_dev.cpu().float().reshape(NUM_PAGES, KV, B, D)
k_ctx = torch.cat([k_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
v_ctx = torch.cat([v_flat[int(pages_used[i])] for i in range(NUM_BLOCKS)], dim=1)
mask_ctx = torch.cat(mask_flat, dim=0).float().transpose(0, 1)
q = query[:Q_LEN].float().reshape(Q_LEN, KV, QPK, D).permute(1, 2, 0, 3)
scores = torch.matmul(q, k_ctx.unsqueeze(1).transpose(-2, -1)) * SCALE + mask_ctx
probs = torch.softmax(scores, dim=-1)
want = torch.matmul(probs, v_ctx.unsqueeze(1)).reshape(NUM_HEADS, Q_LEN, D).transpose(0, 1)
if got is not None:
    check("attention vs SDPA", got, want, 2e-2)
else:
    print("skip  attention vs SDPA: no-combine mode returns partials")

text = PLANNER_LOG.read_text(errors="replace") if PLANNER_LOG.is_file() else ""
verdicts = re.findall(r"lx_pinning: (\S+) \(([^)]+)\)\s*.\s*([^\n]+)", text)
pinned = [b for b, kind, why in verdicts if why.strip() == "lx" and kind == "index"]
print(f"\ngathers pinned LX: {len(pinned)} / {2 * NUM_BLOCKS} expected (K+V per block)")
reasons: dict[str, int] = {}
for _b, kind, why in verdicts:
    if kind != "index" or why.strip() == "lx":
        continue
    reasons[why.strip()[:110]] = reasons.get(why.strip()[:110], 0) + 1
for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"  refused x{n}: {why}")
print(f"restickify cross-frame barrier hits: {text.count('read by restickify')}")
print(f"no room on scratchpad: {text.count('no room on scratchpad')}")
splits = re.findall(r"work_slice_dims=(\(\(\d+, \d+\)(?:, \(\d+, \d+\))*\))", text)
top: dict[str, int] = {}
for s in splits:
    top[s] = top.get(s, 0) + 1
for s, n in sorted(top.items(), key=lambda kv: -kv[1])[:4]:
    print(f"  split x{n}: {s}")

for d in sorted((OUT / "inductor-cache" / "inductor-spyre").glob("*")):
    bundle = d / "bundle.mlir"
    if bundle.is_file():
        m = re.search(r"device_mem_allocate (\d+) bytes", bundle.read_text(errors="replace"))
        if m:
            print(f"  HBM pool {int(m.group(1)) / 1024:9.1f} KB  {d.name[:52]}")

print(f"\nTREPORT q_len={Q_LEN} cores={MAX_CORES or 32} pinned={len(pinned)}/{2 * NUM_BLOCKS}")
print("FAILURES: " + (", ".join(fails) if fails else "none"))
print(f"artifacts: {OUT}")
raise SystemExit(1 if fails else 0)
