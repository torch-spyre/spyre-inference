# Spyre attention micro-benchmark

Measures the Spyre paged-attention kernel with the torch profiler. Spyre has no
CUDA-graph equivalent for excluding host overhead, so device time attributed to a
`record_function` span is the signal.

> **Notation:** `bs=64` / `bs=128` mean **`block_size`**, not batch size. Batch size
> (`num_reqs`) is 1 in every shipped shape.

## Run

```bash
SPYRE_ATTN_PROFILING=1 .venv/bin/python3 scripts/microbench/spyre_attn_microbench.py \
    --config scripts/microbench/configs/granite33_8b_bs64.json
```

### Which scope to measure

```bash
--span online_softmax   # _online_softmax_attention (default)
--span forward          # the whole SpyreAttentionImpl.forward path
--span reshape_and_cache
```

`forward` encloses the other two and sums every kernel starting inside its window, not
the innermost-span rule used for leaf spans. Use it for total attention cost, but do not
read `forward - online_softmax` as a `reshape_and_cache` measurement at long shapes —
profile `--span reshape_and_cache` directly.

### Prerequisite: AIUPTI

Device events only appear if torch-spyre was built with `USE_SPYRE_PROFILER=1`. Verify:

```bash
ldd .venv/lib/python3.12/site-packages/torch_spyre/_C.so | grep libaiupti
```

The runner's startup guard aborts when the probe profile has no device events rather
than reporting plausible-looking zeros.

## Input modes

Both lower onto the same `(query_lens, seq_lens)` path.

**Request list** — explicit per-request lengths:

```json
"capture_batches": [
  {"name": "prefill_512", "query_lens": [512], "seq_lens": [512]},
  {"name": "decode_ctx512", "query_lens": [1], "seq_lens": [512]}
]
```

`query_lens_rle` / `seq_lens_rle` accept `[[value, count], ...]` for wide batches.

**Cartesian grid** — `batch_size × sequence_length × decode_share × prompt_pattern`:

```json
"grid": {
  "batch_sizes": [1, 4],
  "sequence_lengths": [512, 2048],
  "decode_shares": [0.0, 0.5, 1.0],
  "partial_prefill_share": 0.0,
  "prompt_patterns": [[1.0], [1.0, 0.6, 0.3]]
}
```

`block_sizes: [64, 128]` sweeps block size as an extra axis.

## Output

Tab-separated, written after every measurement (a crash keeps what completed) plus a
`_final.csv`.

`ms`/`min_ms`/`max_ms` are **device time, not wall clock** — median/min/max over
`--iterations` separate profiled windows, so they are not comparable to GPU wall-clock
numbers. Spyre-specific columns: `device_time_memory_us` and `memory_share_pct`
(memcpy/memset/restickify share), `cpu_time_ms`, `fallback_clean`, `num_outliers`,
`span`, `kv_layout`.

One forward per profile window: the AIUPTI backend has a fixed pool of trace buffers
(`docs/user_guide/kineto_profiling.md` §4.5) and stops capturing once full, so a single
long window truncates the timeline.

Normalize by `num_kv_blocks_iterated` before concluding anything about scaling — raw µs
can suggest a knee that vanishes once divided by pages iterated.

## Device memory

The KV cache is `num_blocks * block_size * num_kv_heads * head_size * 2 B` per tensor, so
holding `num_blocks` fixed while doubling `block_size` doubles the footprint and
allocations can fail with `RAS::FLEXALLOCATOR::OutOfMemory`. Device memory is not fully
returned between configs (`kineto_profiling.md` §4.3) despite the runner's `gc.collect()`,
so it accumulates across a sweep; affected rows get `error` set and empty `ms`.

- Pin `num_blocks` constant across runs you intend to compare.
- Order decode captures before prefill (or run them separately) to avoid the
  fragmentation cascade.
- A failed allocation strands memory for the rest of the process and degrades later rows'
  timings, so re-run affected shapes in a fresh process.

## Attribution

Two Kineto limitations, both confirmed on hardware, force interval-overlap attribution:

1. Device time is not propagated to `record_function` parents — every span reports
   `self_device_time_total == 0.0`.
2. AIUPTI populates no correlation ids, so there is no CPU↔device linkage.

Each kernel is therefore attributed to the innermost span containing the kernel's
**start** timestamp. Start-based rather than containment-based because dispatch is async:
a kernel can start inside a span and end after it closes. This matters because
`reshape_and_cache` dwarfs `online_softmax` at some shapes, so summing the whole trace
would be dominated by the KV write.

## Correctness gate

Every configuration is checked against a CPU reference before timing, with the same
semantics as `tests/attention/test_spyre_attn.py`: relative tolerance `atol + rtol*|expected|`
(0.3/0.2) and up to `max_outliers` (default 5) fp16 stragglers. `allclose_pass` and
`max_abs_diff` are CSV columns, so a fast-but-wrong config is visible rather than silently
plotted as a win. Failures do not stop the sweep unless `--stop-on-failure`.

`fallback_clean` records whether a `FallbackWarning` fired — torch-spyre silently routes
unsupported ops to CPU, which would otherwise be counted as a Spyre result.

## Notes

- `block_size` 128 is what you get in practice: vLLM CPU platform defaults to 128 which is
  compatible with %64 by `platform.py`
- Compiled and eager variants need separate runs — the compilation mode is fixed per
  process.
- The kernel specializes per `(num_blocks, aligned_max_query_len)`, so a sweep
  legitimately triggers many recompiles; the dynamo recompile limit is raised to 4096.
  Warmup runs before the profiled windows so no compile lands inside a measured window.
