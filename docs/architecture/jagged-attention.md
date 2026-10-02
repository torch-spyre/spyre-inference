# Jagged paged attention

`SPYRE_JAGGED_ATTENTION=1` enables experimental decoder attention over packed
queries and paged KV. It is off by default. The target is 128K tokens per sequence
with 128-token pages. Set `DXP_LOOP_UNROLL=0` to preserve counted device loops.

The [companion torch-spyre changes](https://github.com/torch-spyre/torch-spyre/pull/5086)
provide scatter/carry lowering, consecutive host correction steps, loop-unroll
control, and one-sided clamps that preserve DL16's range. ALiBi is unsupported.
Pooling and encoder attention use their existing paths.

## vLLM contract

The input contract follows vLLM's
[Triton backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py).
Here `T` is the number of new tokens, `S` the request count, and `B` the page size.

| Input | Meaning |
| --- | --- |
| Query and output | Packed `[T, Hq, D]` tensors in request order |
| `query_start_loc` | `[S + 1]` cumulative query lengths |
| `seq_lens` | `[S]` KV lengths including the new tokens |
| `block_table` | Logical-page to physical-page mapping per request |
| K/V cache | `[pages, Hkv, B, D]`, or token-major `[pages, B, Hkv, D]` |
| `slot_mapping` | Destinations for the separate KV update |

Query row `r` within request `s` has position
`seq_lens[s] - query_length[s] + r`. Causality and sliding windows use logical
positions. Attention reads the cache after the existing KV insertion operation.

## Serving schedule

The host groups requests by query width. Requests with up to eight new tokens
use grouped decode. Longer requests use 64/128/256/512-row prefill tiles. Tiles
retain indices into the original packed Q and output buffers.

Prefill uses an outer `for_each_tile` over query tiles and an inner loop over KV
pages. Each query is gathered once and its result is stored after the page loop.
Decode groups queries and independent page slots into parallel entries. It
maintains separate softmax states for the slots and merges them per query.
`SPYRE_JAGGED_PARALLEL_ENTRIES` defaults to 64 and must be a power of two; the
backend bounds it by the resident cache's page count.

Both loop counts are compile-time constants. Runtime tables supply query rows,
physical pages, output rows and mask bounds. Power-of-two capacities allow tables
to change between steps without recompilation. Prefill tables have these shapes:

| Table | Shape |
| --- | --- |
| Query and output indices | `[query_tiles, query_width]` |
| Page indices | `[query_tiles, pages_per_tile, 32]` |
| Query bounds | `[query_tiles, pages_per_tile, 2, query_width]` |
| Key offsets | `[query_tiles, pages_per_tile, B]` |

Decode uses `[groups, chunks, entries, ...]` page tables and one query row per
entry. Index rows use whole int32 sticks and a row-outermost layout; inheriting
an indexed tensor's interleaved layout can skip parts of a row. Inactive entries
write only reserved sink rows. Unwritten K/V lanes are sanitized before matmul,
including cache tails containing NaNs.

Absolute positions stay integer on the host. Only clipped page-relative bounds
are converted to fp16, avoiding precision loss at large positions. Supported
page sizes are multiples of 64 through 1024, whose local bounds fit DL16 exactly.

## State and metadata lifetime

Maxima, denominators and weighted sums remain DL16. Denominator and numerator
use compensated additions to retain rounding lost across pages:

```text
scaled = total * rescale
delta = page_contribution - rounding_error * rescale
updated = scaled + delta
rounding_error = (updated - scaled) - delta
```

Both sums are scaled by `1 / B`, leaving their ratio unchanged. Corrections
rescale with the running maximum. They do not recover rounding inside QK/PV
matmuls or exponentials.

Builders write final tables into reusable host storage. A live plan owns its
storage; callers retaining individual tensors must retain the plan. Device
allocations are reused and tables uploaded once per attention group per step.
Layers share the metadata. Groups write into shared output staging, which
reserves an additional maximum-width query tile for sink rows.

The surrounding projections, normalization, RoPE and MLP already consume packed
tokens in model-size buckets. Their bucket padding is unchanged. Metadata savings
occur once per attention group per step; attention work repeats in every layer.

Startup recording enumerates independent query/page group shapes and registers
both serving kernels with the compile guard. `SPYRE_ATTN_RECORD=0` permits lazy
compilation or explicit shape warmup. Attention still compiles when the rest of
the model runs eagerly.

## Validation and remaining limits

Tests cover both cache layouts, GQA/MQA/MHA, partial pages, NaN tails, sliding
windows, soft capping, changed runtime tables, workspace ownership and 128K
attention inputs. The current validation does not establish full-model generation
at 128K or generation quality for arbitrary long prompts.

The numerical investigation retained two short mixed model-step comparisons that
exceed a provisional 2% raw-logit difference gate: `[1,511]` and `[1,65,3]` at 1K
context. Both backends also have strict elementwise failures against float32 on
model-captured inputs. These failures are not waived.

Controlled frozen inputs showed small differences between attention algorithms;
removing compensation made the tested long-prefill results identical. Compiler
work divisions also changed QKV rounding before attention. With a common compiled
surrounding model and restored KV history, attention swaps produced smaller logit
differences. The exact original failing compilations were not replayed. Compare
each path independently to a reference, and control compiled model math and
history when attributing differences to attention.

## Benchmark alternatives

The latency harness also retains flat page visits, fixed-width nested loops and
split decode for design comparisons. Split decode emits partial softmax states
in one kernel and merges them in another. These are explicit benchmark choices;
serving uses the grouped decode/prefill schedule described above.

See the [microbenchmark guide](https://github.com/torch-spyre/spyre-inference/blob/main/scripts/microbench/README.md#jagged-latency-comparison)
for commands, measurement boundaries and correctness gates.
