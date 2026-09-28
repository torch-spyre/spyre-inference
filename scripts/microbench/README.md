# Spyre attention micro-benchmark

Measures the Spyre paged-attention kernel with the torch profiler. Spyre has no
CUDA-graph equivalent for excluding host overhead, so device time attributed to a
`record_function` span is the signal.

> **Notation:** `bs=64` / `bs=128` mean **`block_size`**, not batch size.

## Run

```bash
SPYRE_ATTN_PROFILING=1 .venv/bin/python3 scripts/microbench/spyre_attn_microbench.py \
    --config scripts/microbench/configs/granite33_8b_e2e_ref.json
```

Reports device time for the whole emulated attention layer. `--span` narrows that
to one scope; see below.

`granite33_8b_e2e_ref.json` reproduces the shapes of one specific end-to-end run,
as a reference point for checking the harness against it. It declares the shapes
that run's kernels actually got, not the request lengths it was given: the
`SPYRE_ATTN_KV_BUCKETS=2048` pin made every step a 16-page kernel, so its four
prefill chunks — kv 512/1024/1536/1984, query 512/512/512/448 — were all the one
`q=512, kv=2048` shape, and appear here as a single capture. The run it mirrors:

```bash
LAYOUT_SOLVER=greedy SPYRE_NUM_CPUS=8 SPYRE_ATTN_KV_BUCKETS=2048 \
uv run --no-sync vllm bench latency \
    --model ibm-granite/granite-3.3-8b-instruct \
    --input-len 1984 --output-len 64 --batch-size 4 \
    --num-iters-warmup 2 --num-iters 1 --max-model-len 2048 \
    --profile --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=<dir> \
    --profiler-config.torch_profiler_use_gzip=false \
    --profiler-config.torch_profiler_record_shapes=true
```

Add `SPYRE_ATTN_PROFILING=1` to label the `spyre_attn::*` spans in that trace;
`SPYRE_BATCHED_DECODE=1` is the default, so it already takes the batched decode path.

### Which scope to measure

```bash
--span layer              # default: KV write + attention, the whole emulated layer
--span forward            # SpyreAttentionImpl.forward
--span online_softmax     # _online_softmax_attention
--span reshape_and_cache  # the KV write alone; needs --kv-write
```

`layer` and `reshape_and_cache` are emitted by this harness. `forward` and
`online_softmax` come from `spyre_attn` itself.

`layer` is the default because it is the only span that closes after a device
synchronisation, so it provably contains every kernel's start. The leaf spans can
drop a kernel and under-report; see Attribution. Use them to apportion cost within
a layer, once `layer` has told you the total.

## What the harness emulates

Production replaces `Attention.forward` with `attn_layer._spyre_attention_forward`,
which per layer:

1. scatters K/V into the cache (`do_kv_cache_update`),
2. stages the query into the impl's constant-shaped buffer,
3. calls the attention op on those buffers,
4. copies the result back.

Steps 1, 2 and 4 are traced into the *block* graph, so only step 3 sits inside
`spyre_attn::forward`. The harness runs the same four steps in the same order and
stages the query itself, which keeps that split: passing a non-staging query would
make `pre_staged` False and move the staging copies inside the measured span.

Step 1 only runs under `--kv-write`. Without it, `reshape_and_cache` and `layer`
have no KV write to measure — `forward` has not performed the KV write since it
moved to `attn_layer`.

### The KV write has no production counterpart

Inductor fuses the KV write into the block graph's RMSNorm + QKV-projection +
RoPE kernel, so production has no standalone reshape_and_cache kernel. What
`--kv-write --span reshape_and_cache` reports is the cost of that `index_copy`
pair *in isolation*; it cannot be subtracted from, or compared against, the fused
kernel. Treat it as a bound, not as production's KV-write cost.

## Shapes are exact

A config declares `(query_lens, seq_lens)` and the kernel runs those lengths. The
bucket lattice is not a second knob: `derive_lattice` builds it from the shape list
so that every declared length is its own bucket and the real
`SpyreAttentionMetadataBuilder`'s round-up is the identity. `max_model_len` and
`max_num_seqs` are the tops of those ladders, so they are derived too.
`max_num_batched_tokens` is pinned to the platform's 512 cap instead: `staging_rows`
is it plus one, and those buffers are the kernel's query and output arguments, so
deriving it from a decode-only shape list would narrow the gather the kernel runs.
A config that sets any of the three, or the old `attn_*_buckets` keys, is rejected.

Padding has not gone away — it is now something you ask for. The kernel specialises
on `(num_blocks, padded_query_len)`, so **to measure what a deployment's coarser
ladder does to a length, declare the padded length**: a `q=512, kv=512` chunk under
`SPYRE_ATTN_KV_BUCKETS=2048` is a 16-page kernel, so declare `kv=2048`. The mask
differs (fewer valid positions) but the shape, and so the cost, is the one
production runs.

Either way the realized shape is recorded per row, read back from the built metadata
rather than assumed: `num_kv_blocks_iterated` and `padded_query_len` are what the
kernel got, and a disagreement with the declared shape sets `error` instead of
quietly retitling the measurement.

`block_size` reaches `cache_config`, which the builder asserts its `kv_cache_spec`
against, so a `block_sizes` sweep runs one config context per block size. The
lattice itself is in tokens and sequences, so it is shared across that sweep —
which it has to be, since `spyre_inference.envs` caches on first read.

### Prefill is always chunked

`check_and_update_config` caps `max_num_batched_tokens` at
`min(max_num_batched_tokens, 512)` for decoder models, so a query longer than
that is unschedulable at **any** `max_model_len`: production chunks its prefills.
A prefill capture is therefore `query_len <= 512` against a growing `seq_len`, not
`query_len == seq_len`. The cap is checked against the declared batch, since the
derived limits come from the shapes and so can never rule one out; a batch over it
is reported as a row with `error` set rather than an assertion mid-sweep.

### Batched decode

The batched decode kernel needs the `batched_decode_compiled` variant, which sets
`SPYRE_BATCHED_DECODE` for the process (so it cannot be mixed with a per-seq
variant in one run). It also needs `num_decode_seqs >= 4`, a compiled build, and a
resolvable sequence/blocks bucket pair. Under the default tiled walk the kernel is
reached on the head-major layout only, which is `--attn-kv-layout`'s default; pairing
the batched variant with `token_major` is refused up front rather than measured as a
silent per-seq fallback.

The batch axis is bucketed like the other two — one bucket per captured batch size,
so each of `granite33_8b_batched_decode.json`'s 4/8/16/32 captures dispatches to its
own kernel. `num_blocks` has to hold every sequence's pages at the largest of them.

When the gate declines, the impl silently falls back to the per-seq loop. The
`attn_path` column records which path actually ran, and a declined gate sets
`error`, so a per-seq measurement cannot be read as a batched one.

## Prerequisite: AIUPTI

Device events only appear if torch-spyre was built with `USE_SPYRE_PROFILER=1`
(`pyproject.toml` sets `"0"` by default). Verify:

```bash
ldd .venv/lib/python3.12/site-packages/torch_spyre/_C.so | grep libaiupti
```

The runner's startup guard aborts when the probe profile has no device events
rather than reporting plausible-looking zeros.

## Input modes

Both lower onto the same `(query_lens, seq_lens)` path.

**Request list** — explicit per-request lengths:

```json
"capture_batches": [
  {"name": "prefill_chunk", "query_lens": [512], "seq_lens": [1024]},
  {"name": "decode_4seq", "query_lens_rle": [[1, 4]], "seq_lens_rle": [[2048, 4]]}
]
```

`query_lens_rle` / `seq_lens_rle` are the same two fields run-length encoded, for
batches too wide to spell out: each `[value, count]` pair expands to `count`
repeats of `value`, so `"seq_lens_rle": [[2048, 32]]` is 32 sequences of 2048.
Give one form or the other, not both.

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

Tab-separated, written after every measurement (a crash keeps what completed) plus
a `_final.csv`.

`ms`/`min_ms`/`max_ms` are **device time, not wall clock** — median/min/max over
`--iterations` separate profiled windows, so they are not comparable to GPU
wall-clock numbers. Spyre-specific columns: `device_time_memory_us` and
`memory_share_pct` (memcpy/memset/restickify/d2d-copy share), `cpu_time_ms`,
`fallback_clean`, `num_outliers`, `span`, `kv_layout`, `kv_write`,
`attn_path`, `num_decode_seqs`, `padded_num_seqs`, `decode_uniformity`,
`blocks_per_chunk`, `kernels_attributed`, `kernels_expected`, `late_compile`,
`num_kv_blocks_iterated`, `padded_query_len`.

One forward per profile window: the AIUPTI backend has a fixed pool of trace
buffers (`docs/user_guide/kineto_profiling.md` §4.5) and stops capturing once
full, so a single long window truncates the timeline. An end-to-end
`vllm bench latency --profile` run hits this and logs
`Exceeded max AIU buffer count`; its device timeline is truncated, which is the
reason this harness exists.

Normalize by `num_kv_blocks_iterated` before concluding anything about scaling —
raw µs can suggest a knee that vanishes once divided by pages iterated. It is the
count the kernel iterated, not one derived from `max_seq_len`, so it stays right
when a row is padded.

## Device memory

The KV cache is `num_blocks * block_size * num_kv_heads * head_size * 2 B` per
tensor, so holding `num_blocks` fixed while doubling `block_size` doubles the
footprint and allocations can fail with `RAS::FLEXALLOCATOR::OutOfMemory`. Device
memory is not fully returned between configs (`kineto_profiling.md` §4.3) despite
the runner's `gc.collect()`, so it accumulates across a sweep; affected rows get
`error` set and empty `ms`.

- Pin `num_blocks` constant across runs you intend to compare.
- Block tables are as wide as vLLM's, i.e. `max_model_len / block_size`, not as
  wide as the shape needs: the builder pads each sequence's block count onto a
  bucket and the kernel gathers every padded page. So `num_blocks` has to cover
  `num_reqs * ceil(max_model_len / block_size)`. Shapes that do not fit are
  skipped with the required value in the `error` column.
- Order decode captures before prefill (or run them separately) to avoid the
  fragmentation cascade.
- A failed allocation strands memory for the rest of the process and degrades
  later rows' timings, so re-run affected shapes in a fresh process.

## Attribution

Two Kineto limitations, both confirmed on hardware, force interval-overlap
attribution:

1. Device time is not propagated to `record_function` parents — every span reports
   `self_device_time_total == 0.0`.
2. AIUPTI populates no correlation ids, so there is no CPU↔device linkage.

Each kernel is therefore attributed to the innermost span containing the kernel's
**start** timestamp. Start-based rather than containment-based because dispatch is
async: a kernel can start inside a span and end after it closes.

### Start-based attribution is fragile at the closing edge

The device queue lags the CPU by roughly one kernel, so the **last** kernel of a
multi-sequence loop starts within a few µs of the enclosing span closing — inside
or outside it depending on the run. A dropped kernel silently divides the reported
time, which is why a batch of *n* sequences could read as the cost of *n-1*.

Two mitigations:

- The `layer` span closes after a device synchronisation, so every kernel has
  started by then. Its total is exact; the leaf spans are not. Hence the default.
- `kernels_attributed` vs `kernels_expected` is checked on every row, and a
  mismatch sets `error`. A short attribution is visible rather than silent.

`forward` and `layer` enclose the leaf spans, so they credit every kernel starting
inside their window rather than using the innermost-span rule.

## Correctness gate

Every configuration is checked against a CPU reference before timing, with the
same semantics as `tests/attention/test_spyre_attn.py`: relative tolerance
`atol + rtol*|expected|` (0.3/0.2) and up to `max_outliers` (default 5) fp16
stragglers. `allclose_pass` and `max_abs_diff` are CSV columns, so a
fast-but-wrong config is visible rather than silently plotted as a win. Failures
do not stop the sweep unless `--stop-on-failure`.

`fallback_clean` records whether a `FallbackWarning` fired — torch-spyre silently
routes unsupported ops to CPU, which would otherwise be counted as a Spyre result.

`late_compile` records whether Dynamo's graph counter moved during the measured
windows, i.e. whether an Inductor compile landed inside a measurement.

## Notes

- `block_size` 128 is what you get in practice: vLLM CPU platform defaults to 128
  which is compatible with %64 by `platform.py`
- Compiled and eager variants need separate runs, as do batched and per-seq decode
  variants — both are fixed per process.
- The kernel specializes per `(num_blocks, aligned_max_query_len)`, so a sweep
  legitimately triggers many recompiles; the dynamo recompile limit is raised to
  4096. Warmup runs before the profiled windows so no compile lands inside a
  measured window.
