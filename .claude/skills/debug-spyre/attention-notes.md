# Attention-specific debugging notes

Companion to `SKILL.md`. Read this when the failing surface is the Spyre attention backend specifically. The generic workflow (cluster → HTML log → hypothesis queue → cluster validation) in `SKILL.md` still applies; this file just adds attention-flavored content to the hypothesis queue and pipeline-bisection steps.

## Files that matter for attention

- `spyre_inference/v1/attention/backends/spyre_attn.py` — the decoder backend: `SpyreAttentionImpl` (head-major allocation, per-head KV-write index, kernels), `SpyreAttentionMetadataBuilder` (shared with the encoder backend), and the per-sequence / batched-decode dispatch.
- `spyre_inference/v1/attention/backends/spyre_encoder_attn.py` — encoder-only (no KV cache) attention.
- `spyre_inference/v1/attention/ops/` — the kernel source code (`page_attn*.py`, `batched_decode*.py`, `reshape_and_cache*.py`) and the cache device layouts (`layout.py`). Kernel docstrings and inline comments name the torch-spyre limitation each shape choice routes around.
- `spyre_inference/v1/attention/attn_layer.py` — the traced KV write and query staging that replace `Attention.forward`.
- `tests/attention/test_spyre_attn.py` — builds real metadata via `SpyreAttentionMetadataBuilder`, seeds the cache through the impl's own store, calls `SpyreAttentionImpl.forward` (always compiled, on Spyre), compares against a CPU reference (`ref_attn`) with `assert_close_outliers` (base `atol=0.3, rtol=0.2` when a query is 32+ rows, `atol=0.2, rtol=0.2` otherwise; up to 5 outliers at twice the resulting tolerances) — if mismatch ratios are near 100 % with differences > 1.0, suspect a **structural** bug, not fp16 noise.

## Attention-specific limitations

In addition to the generic table in `SKILL.md`:

| Limitation | Workaround |
|---|---|
| KV alignment to bucketed length | `SpyreAttnBucketer` kv buckets: powers of two from `block_size` to `max_model_len`, consumed as a padded block count, so the same compiled kernel is reused as KV grows |
| Query length must be bucketed | `SpyreAttnBucketer` query buckets: `[1] + multiples of min(512, max_num_batched_tokens)`; each sequence's query is padded up to its own bucket |
| `head_size` must be a multiple of 64 | `SpyreAttentionBackend.supports_head_size` enforces this (128-byte stick / 2 bytes for fp16) |
| The page gather is a 2-D subscript (`aten.index`), which fails eager on Spyre (torch-spyre#4334) | Attention always compiles, even under `--enforce-eager` |

MHA and MQA (`test_spyre_attn_head_configs`), GQA, sliding window and soft-capping all have committed parametrizations.

## Useful test selectors (orientation only — not for `-k`)

These appear in current parametrize IDs. They're for reading collected node IDs; `-k` will not parse the parens/equals/commas.

- `decode_single` / `decode_batch` / `prefill_single` / `prefill_batch` / `mixed_batch` / `kv_padded_prefill`
- `device_spyre-compiled`, or `device_spyre` with `compiled`

## Reading LX residency: read the allocation, not the `lx_pinning:` log

When a hypothesis is "this buffer spills to HBM", do not read the answer off the planner's
debug log. With `SPYRE_INDUCTOR_LOG=1 SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` (note:
`TORCH_LOGS='+spyre.inductor...'` crashes torch — it rejects the non-module name) the allocator
prints one `lx_pinning:` line per op, and `→ lx` *reads* like "resident" while being the
fallback for a missing dict entry. Symbols below, in
`torch_spyre/_inductor/scratchpad/`; behaviour verified at torch-spyre
`b9daa0c3aa7ac7624e3ff6897da5893bb1fa54f4`, the rev `pyproject.toml` pins:

- `_log_lx_pinning` (`allocator.py`) prints `reasons.get(op.name, "lx")` and only runs at
  DEBUG. Anything missing from the dict reads as `lx`.
- Missing entries are rare, not routine. `cpsat` (the default) and `simulated_annealing` both
  write `_SOLVER_CHOSE_SPILL` for every spilled buffer (`ilp_solver_ortools.py`,
  `sa_cooptimizer.py`), and the base `ScratchpadAllocator._get_spill_reasons` synthesizes
  `no room on scratchpad` for every buffer with `address is None`. The co-opt override
  (`CoOptimizingAllocator._get_spill_reasons`) drops `None` reasons precisely because its two
  native solvers leave none.
- The one solver that does leave gaps is `ExhaustiveSearchSolver`: it copies a placement-only
  solver's `spill_reasons` (`exhaustive_search.py`), and `record_exclusions`
  (`plan_solver.py`) records only declared and capacity exclusions, so a buffer that merely
  did not fit keeps `spill_reason = None` and prints `→ lx`. Reached only with
  `ALLOW_EXHAUSTIVE_SEARCH=1` (for `greedy`/`bestfit`/`firstfit`, or `cpsat` without ortools);
  without that flag `select_allocator` raises `ValueError` instead, and
  `_cpsat_solver_class`'s fallback to greedy logs a `warn_once` rather than degrading quietly.
- Two buffer kinds are never logged at all, on any path: graph-input clones are keyed by the
  input name and relayout buffers by a synthetic `__spyre_lx_relayout__:` name, and neither
  appears in `graph.operations`, which is what `_log_lx_pinning` walks. On the placement-only
  path `_build_bound_buffers` additionally skips buffers with no lifetime, so those ops have
  no verdict either; the co-opt builder (`_build_cd_bound_buffers`) has no such skip, so there
  every op reaches the solver.

So the log is reliable except under `ExhaustiveSearchSolver` and, on the placement-only path,
for ops with no lifetime — but it is silent about input clones and relayouts either way. Read
residency off the allocation instead, where `address is None` means HBM:

```python
from torch_spyre._inductor.scratchpad import allocator as alloc

orig = alloc.ScratchpadAllocator._push_allocation

def push(self, graph, buffers, accepted):
    for b in buffers:
        where = "LX " if b.address is not None else "HBM"
        cd = getattr(b, "chosen_division", None)
        per_core = b.per_core_size(cd) if cd is not None else b.size
        print(f"{where} {b.name:50} addr={b.address} {per_core / 1024:.1f} KiB/core")
    return orig(self, graph, buffers, accepted)

alloc.ScratchpadAllocator._push_allocation = push
```

`b.size` means different things per path, which is why the snippet does not print it raw: a
co-opt `CoreDivisionBuffer` is sized by its **total** device footprint, while the placement-only
path builds `LifetimeBoundBuffer` from `size_per_core` and is already per-core. On co-opt the
divisor is the chosen division's `output_partition * tiling.output_tile_count` (see
`CoreDivisionBuffer.min_footprint`) — *not* the core count, since a reduction split retains a
full output part. `per_core_size(chosen_division)` applies exactly that divisor;
`b.core_divisions[b.chosen_division].label` prints which division was picked. Address deltas
between two same-sized buffers confirm the figure.

Take the capacity from `_lx_planning_size()` rather than from the physical LX size or the
tracker constant: the frontend plans into `1 - DXP_LX_FRAC_AVAIL` of the tracker capacity and
leaves the rest to the backend, so at the default `DXP_LX_FRAC_AVAIL=0.2` the solver's limit is
four fifths of the tracker figure.

This mattered concretely: it is what made an experiment aimed at giving the batched decode
kernel LX-resident page gathers look like it had found the shipped kernel *not* resident, when
both paths were resident all along, and the whole premise was void.

## Attention-specific hypotheses to add to the queue

When the failing surface is attention, in addition to the generic starter set, also consider:

- **KV write went to the wrong rows.** The write is `index_copy_` through a row view of the pages (`kv_slot_views`), one index tensor per KV head (`kv_write_index`). The cache must carry the device layout its impl allocated (`head_major_kv_layout`): the default tiled layout silently scatters to the wrong rows. A fixture that builds its own cache tensor rather than calling `allocate_pages` is the usual culprit.
- **Write source not materialized.** A single-token fused-QKV view reports contiguous at a nonzero storage offset; the store relies on an elementwise pass (`key * 1.0`) to copy it. Short-token tests pass while long prefill is wrong when this, or the per-head index split, regresses.
- **Attention mask off-by-one / wrong padded shape / wrong causal boundary.** `_build_attention_mask` (no sliding window) and `_build_single_tile` (sliding window) build per-block tiles padded to `aligned_query_len × block_size`. Off-by-one in causal/padding masking produces near-100 % mismatches that look like "random output." Print the mask stack for a small case and verify masked positions are `finfo(fp16).min` (-65504), unmasked are `0`, and the causal boundary matches `context_lens + q_pos`.
- **Page index table wrong.** `page_index_tables_cpu[s][i, 0]` must be the physical page of sequence `s`'s `i`-th *active* block (a position within `active_block_indices` under a window, not an absolute block index). A wrong order leaves the mask tiles misaligned with the pages they mask — silent disaster.
- **Batched vs per-sequence path.** Even one decode sequence can take the batched kernel when it is supported (compiled and sufficient page capacity); set `SPYRE_BATCHED_DECODE=0` to force the per-sequence loop and see whether the failure follows the path. Up to 64 decode sequences run on the shared-slot kernels (32 for a layer group whose KV row is too wide for two pages per core, see `shared_block_slots`). The kernel call sees them as one sequence with 64 (or 32) block slots per chunk (`num_seqs=1`, `blocks_per_chunk=64`), and a `[slots, rows]` one-hot maps each block slot back to its sequence's output row. `SPYRE_ATTN_SINGLE_CHUNK_KERNEL=0` moves a batch that fits in one chunk from the static single-chunk kernel to the dynamic one, which tells you whether a failure follows the kernel.
- **Tiled vs Python walk.** CPU tests swap `for_each_tile` for a Python loop (`use_python_for_each_tile` in `spyre_testing_plugin.attn_helpers`) that runs the identical kernel bodies. A failure those CPU tests don't reproduce points at the `for_each_tile` lowering or the torch-spyre pin.
- **A shape that escapes the `SpyreAttnBucketer` kv/query buckets** dispatches to an unrecorded kernel (correctness, not just recompilation cost).

## Pipeline bisection for `forward()`

A layer's attention runs as: KV write (`do_kv_cache_update`, traced by `attn_layer`) → query staging → per-sequence kernels (`_run_page_attn`) or the batched decode kernel (`_run_batched_decode`) → write-back into the output. Rerun each stage against a reference and compare:

- After the write, read the K/V pages back to CPU and check the rows `slot_mapping` names hold the new tokens.
- For the kernel, call the kernel function from `v1/attention/ops/` directly (eager, on CPU) with the same metadata the builder produced; it is plain PyTorch, so it runs on CPU for a reference.

A diff at the write step points at the index/layout; a diff only after the kernel points at the matmul / softmax / mask path.

## Per-row magnitude dump for broadcast/dispatch bugs

When you see huge values (near fp16 max, `~60000+`) in the attention output, the bug is usually in a specific broadcast or dispatch axis of a Spyre kernel rather than the math. Pull the result of one kernel call to CPU (`[padded_query_len, num_heads, head_size]`) and print the per-head magnitude:

```python
out = result.to("cpu")                       # [q, num_heads, head_size]
per_head_max = out.abs().amax(dim=(0, 2))    # heads are kv-major: h = kv_h * q_per_kv + g
for h, m in enumerate(per_head_max.tolist()):
    kv_h, g = divmod(h, num_heads // num_kv_heads)
    mark = "  <<< OVERFLOW" if m > 100 else ""
    print(f"  kv_h={kv_h} group={g}  max|out|={m:.4f}{mark}")
```

Look for **periodic patterns**: if every Nth head is huge and the rest are sane (e.g. every odd query group), that's near-certain evidence of a broadcast bug along that axis in either the matmul or softmax kernel. A workaround is usually `.expand(...).contiguous()` on the singleton dim of `k`/`v`/`mask` before the call — diagnostic (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix.

## Attention-specific gotcha: which mode the impl compiles in

Decoder attention (`SpyreAttentionImpl`) always runs compiled, even under `--enforce-eager`, because its page gather fails in eager mode (torch-spyre#4334). Encoder attention (`SpyreEncoderAttentionImpl`) compiles only when `compilation_config.mode` is `STOCK_TORCH_COMPILE` at the time it is built. So to reproduce an encoder bug, use the same compilation mode as the failing test.

## CPU-vs-Spyre standalone repro (attention-flavored)

The decoder attention tests run on Spyre only; `SpyreAttentionImpl` takes no CPU KV cache. For a standalone repro under `logs/<slug>/repro_cpu.py`, build the metadata with `SpyreAttentionMetadataBuilder.build` as `_run_spyre_attn_test` does, fill plain CPU head-major page tensors by indexing (`k_pages[block, :, offset] = key[t]`, as `test_spyre_attn_scatter` builds its expected pages), and call the kernel from `v1/attention/ops/` directly on CPU tensors. (`allocate_pages` needs a Spyre device: its layouts come from `torch_spyre._C`.) If the CPU kernel passes, the bug is in torch-spyre's realization of the operation; if it still fails on CPU, the bug is in our own logic. A compiled attention kernel does not run on CPU (Inductor's C++ codegen rejects it), so compare against the eager kernel.
