# RFC: Graph recorder for static `torch.compile` in spyre-inference

| Field | Value |
|---|---|
| Status | Implemented |
| Authors | [@R3hankhan123](https://github.com/R3hankhan123), [@bringlein](https://github.com/bringlein), [@rishikakedia](https://github.com/rishikakedia), [@bohnstingl](https://github.com/bohnstingl), [@tdoublep](https://github.com/tdoublep), [@dilipgb](https://github.com/dilipgb) |
| Created | 2026-09-10 |
| Updated | 2026-11-04 |
| Tracking | Closes [#174](https://github.com/torch-spyre/spyre-inference/issues/174); shipped in [#480](https://github.com/torch-spyre/spyre-inference/pull/480) and [#638](https://github.com/torch-spyre/spyre-inference/pull/638) |
| Related | vLLM CUDA graphs; `compilation_config.compile_sizes`; `spyre_shape_bucketer.py`, `spyre_model_runner.py`, `platform.py` |

## 1. Motivation

Spyre's Inductor backend requires static shapes (`dynamic=False`). vLLM CUDA graphs are the GPU way
to pin those shapes, and Spyre has none. The graph recorder is that analogue for
`STOCK_TORCH_COMPILE`, for the **decoder body**.

[#174](https://github.com/torch-spyre/spyre-inference/issues/174) asked for the CUDA-graph analogue
on Spyre: compile complete forward passes for buckets of request lengths, in the model runner, while
paged attention stays a separate path.

Dynamo specializes a new graph on every unseen packed token count `T` and every lm_head row width.
Without a recorder, the first request at a new shape pays a full compile on the serving path and
often times out `execute_model`.

vLLM's CUDA-graph capture dummies each capture size at startup, then pads runtime batches onto a
captured graph. Spyre sets `use_cuda_graph = False`. The recorder keeps the same contract: compile
once per bucket at warmup, pad at runtime, never compile mid-request.

## 2. Goals and non-goals

Goals:

- Dummy each body token-count bucket at warmup.
- Pad every runtime batch up onto a warmed bucket through vLLM's `BatchDescriptor` /
  `_determine_batch_execution_and_padding` seam.
- Dummy lm_head row widths separately.
- Keep the default ladder sparse, so warmup stays affordable.
- Honor a user-supplied `compile_sizes` list.

Non-goals:

- Attention-kernel shape recording, which #174 explicitly excludes.
- Pooling and encoder body buckets, which have their own path (see
  [Encoder / pooling compile buckets](../../user_guide/configuration.md#encoder-pooling-compile-buckets)).
- Piecewise compilation or `VLLM_COMPILE`.

## 3. Background

### 3.1 Constraints

- **Compilation mode.** Only `NONE` (`--enforce-eager`) and `STOCK_TORCH_COMPILE` are supported.
  Selecting `VLLM_COMPILE` or `DYNAMO_TRACE_ONCE` logs a warning and falls back to
  `STOCK_TORCH_COMPILE`.
- **Packed `T` is the body shape.** With chunked prefill, one forward is `[prefill tokens] + [decode
  tokens]`. Linear and RMSNorm see one flattened `[T, H]`. There is no separate decode graph or
  prefill graph at the body.
- **Scheduler cap.** After buckets are chosen, `max_num_batched_tokens = max(compile_sizes)`, so the
  scheduler cannot emit a `T` with no bucket.

### 3.2 Terminology

- Call it the **graph recorder**, not CUDA graphs. The CUDA-graph vocabulary (`capture`, `replay`) is
  GPU-only; here the artifact is an Inductor graph keyed on `T`.
- **Bucket:** one padded size.
- **Ladder** or **`compile_sizes`:** the sorted list of buckets. It is the set of legal body shapes,
  the way capture sizes work on GPU, and runtime always pads *up* onto it.

## 4. Proposed architecture

There are two recorders. Each pads up to the smallest warmed size that fits, and sampling drops the
pad.

```text
                    scheduler batch
                           │
                           ▼
              ┌────────────────────────┐
              │  body T                │  SpyreShapeBucketer
              │  compile_sizes         │  BatchDescriptor pad
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  lm_head rows          │  logits_row_buckets
              │                        │  pad, then unpad
              └────────────────────────┘
```

## 5. Component design

### 5.1 Body recorder: `SpyreShapeBucketer`

`TorchSpyrePlatform.apply_config_platform_defaults` fills `compilation_config.compile_sizes` unless
the user already set them. The default ladder favors compile time:

- **Decode buckets:** powers of two up to `--max-num-seqs`, plus `--max-num-seqs` itself when it is
  not a power of two. Decode-only steps schedule one token per running sequence, so this ladder is
  exact for pure decode.
- **Prefill bucket:** the scheduler's `max_num_batched_tokens`, capped at the Spyre maximum (the same
  maximum as sendnn-inference). A lower scheduler limit gives a smaller bucket, and `--max-num-seqs`
  is clipped to it too. Full prefill chunks fill that budget.

Warmup dummies each body size largest-first, so later smaller shapes hit the Inductor cache, and then
calls `mark_warmed_up()`. Runtime padding overrides GPUModelRunner's
`_determine_batch_execution_and_padding` and returns `CUDAGraphMode.NONE` plus
`BatchDescriptor(num_tokens=padded)`. That is the CUDA-graph seam, so
`scheduler_output.total_num_scheduled_tokens` is not mutated. Pad tokens stay out of sampling through
`num_actual_tokens`.

`--enforce-eager` skips the bucketer.

### 5.2 lm_head row recorder

The lm_head sits outside every body graph and projects one row per *sampled* request, so a finishing
batch walks every width up to `--max-num-seqs`. Those widths pad onto `logits_row_buckets`: the body
sizes, clipped to the widest row count a step can produce, which includes draft rows under
speculative decoding. Warmup runs `_dummy_sampler_run` at each width, and pad rows are dropped before
sampling. A prefill bucket beyond that widest count is not a reachable row width, so it is not
recorded for the head.

### 5.3 Compile granularity

By default, `SPYRE_COMPILE_GRANULARITY=block` compiles each transformer block with `torch.compile` in
place (`dynamic=False`). Identical `forward` code objects share one Inductor artifact, though the
first layer still specializes on `residual is None`. A new `T` therefore recompiles a block, not a
whole model.

The alternative, `SPYRE_COMPILE_GRANULARITY=model`, compiles a single full-model graph. Its compile
cost grows with layer count, because nothing is shared between layers, so `block` is the path to use
unless one graph is specifically needed.

### 5.4 Mixed batches

The sparse ladder is cheap on the two common cases:

- **Decode-only:** `T` equals the number of running sequences and pads to the next power of two.
- **Full prefill chunks:** the scheduler fills the prefill budget and hits the prefill graph with no
  extra pad.

A mixed step has no bucket between `--max-num-seqs` and the prefill bucket, so a small prefill chunk
plus decode tokens pads all the way up to the prefill bucket. A denser user list is still honored:

```bash
vllm serve ... -cc '{"compile_sizes": [<size>, <size>, ...]}'
```

Do not split "decode graphs" and "prefill graphs" into two compilers. The body only sees `T`, and
separate ladders would still pad a mixed `T` onto the next available size.

### 5.5 Configuration

| Knob | Role |
|---|---|
| `--enforce-eager` | Disable compile and decoder recording |
| `-cc.compile_sizes` | Override body buckets; also caps `max_num_batched_tokens` |
| `SPYRE_COMPILE_GRANULARITY` | `block` (default) or `model` (whole-model graph, §5.3) |

The configuration docs should cover the default decoder ladder, how to override `-cc.compile_sizes`,
and that `max_num_batched_tokens` follows `max(compile_sizes)`. Existing vLLM users already know
capture sizes; the teaching point is that decode and prefill share one `T` axis under chunked prefill.

### 5.6 Trade-offs

- **Scheduler budget:** this is not a breaking API change, but it does cap `max_num_batched_tokens` to
  `max(compile_sizes)`. A user who expected a larger budget silently gets the largest bucket.
- **First boot:** startup is still blocked on warmup. Restarts are cheap only if an Inductor or kernel
  cache is configured.
- **Runtime pad:** mixed chunked-prefill and decode steps pad up to the prefill bucket. That is extra
  Linear and RMSNorm work on the common continuous-batching path.
- **Complexity:** the recorder adds a bucketer, a platform default, a
  `_determine_batch_execution_and_padding` override and a second row recorder for lm_head. Wrong
  `is_warmed_up` gating compiles during dummy runs and misses the warmed graph at runtime.

## 6. Implementation status

Shipped in [#480](https://github.com/torch-spyre/spyre-inference/pull/480). User-supplied
`compile_sizes` are respected and cap the scheduler
([#638](https://github.com/torch-spyre/spyre-inference/pull/638)).

The sparse decode and single-prefill ladder was chosen to cut warmup time after a dense ladder proved
too slow. Mixed-batch padding up to the prefill bucket is the accepted cost.

The recorder stays the body compile path, with the follow-ups in §8 and no exceptions.

## 7. Alternatives considered

| Approach | Why not |
|---|---|
| `VLLM_COMPILE` plus CUDA graphs | Spyre is not CUDA, and piecewise mode is unsupported. |
| Dense body ladder | Many body compiles. Fine padding, unacceptable startup. |
| Decode buckets only, no prefill bucket | Prefill and mixed `T` have no bucket, which means a mid-request compile or a scheduler capped to `--max-num-seqs`. |
| Mutate `scheduler_output.total_num_scheduled_tokens` | Fights upstream accounting. `BatchDescriptor` is the CUDA-graph seam. |
| Background or lazy compile | Startup UX win, but first-hit latency spikes. A follow-up, not a replacement. |
| Do nothing | Every new `T` compiles on the serving path. |

## 8. Open questions and follow-ups

- Should a production default persist the Inductor or kernel cache, so a restart is not a full
  re-record?
- Should the default ladder grow a few extra sizes in the mixed-batch gap without returning to a dense
  list?
- Should warmup run in the background, so first-boot serving does not block on the full record?

## 9. Acceptance criteria

- No mid-request Inductor compiles on a covered workload, which shows the ladder and scheduler cap are
  correct.
- No extra pad on the decode-only path when `T` equals the number of running sequences on a bucket.
- Warmup wall time tracked against the number of `compile_sizes` (startup cost).
- Mixed-batch body time tracked against a denser ladder (the padding cost of the gap).

Tests:

- `tests/runtime/test_platform.py`: default ladders, user `compile_sizes`, eager skip.
- `tests/runtime/test_shape_bucketer.py`: body dispatch, `logits_row_buckets`.
- `tests/e2e/test_compile.py`: compiled decoder cosine checks.

## 10. References

**spyre-inference:**

- `spyre_inference/v1/worker/spyre_shape_bucketer.py`: `SpyreShapeBucketer`, `logits_row_buckets`.
- `spyre_inference/v1/worker/spyre_model_runner.py`: warmup and runtime padding.
- `spyre_inference/platform.py`: default `compile_sizes` and the scheduler cap.
- [#174](https://github.com/torch-spyre/spyre-inference/issues/174),
  [#121](https://github.com/torch-spyre/spyre-inference/issues/121),
  [#5](https://github.com/torch-spyre/spyre-inference/issues/5),
  [#147](https://github.com/torch-spyre/spyre-inference/issues/147): the requests behind the
  comparison with CUDA graphs.

**Prior art:**

- vLLM CUDA graphs capture a graph per batch descriptor, pad runtime batches onto it and execute
  `CUDAGraphMode.FULL` or piecewise. The padding seam is the same
  (`_determine_batch_execution_and_padding`, `BatchDescriptor`); Spyre returns `CUDAGraphMode.NONE` and
  a padded token count instead of a graph.
- vLLM `compilation_config.compile_sizes` is already the knob for which shapes to compile. The recorder
  fills it and honors a user-supplied list.
- `torch.compile(dynamic=False)`: Dynamo's specialization model is why a finite bucket list is needed.
