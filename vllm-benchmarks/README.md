# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` holds a `defaults`
mapping and a `tests` list, and each test entry is merged over `defaults` so it
only spells out what differs; one entry per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

All three cover the same models at the same context lengths, so a regression can be read across the offline and online paths. `SPYRE_DEVICES` / `AIU_WORLD_SIZE` are derived from each entry's tensor-parallel size rather than spelled out per test.

Serve entries replay a recorded agentic trace: real router prompts, each request keeping the output length it actually produced, in recorded order. A trace rather than a fixed prompt shape is what makes prefill chunking, prefix reuse, and KV-block pressure visible. A serve entry's trailing suffix names the trace it replays.

Trace paths come from `SPYRE_AIOPS_DATASET` (`*_aiops`, run at 4k) and `SPYRE_CICS_DATASET` (`*_cics`, run at 8k). By default, `make perf-tests` resolves them from the cache and fetches missing or corrupt copies, preserving overrides that point to existing files. `FETCH_BENCH_DATA=0` bypasses this step. Variables left unset fall back to the Spyre benchmark hosts' paths; a selected entry whose file is absent fails the run.

Latency and throughput cannot replay those traces — `vllm bench latency` takes no dataset at all, and `vllm bench throughput` rejects the `custom` dataset the traces load through. They instead split each `max-model-len` evenly between prompt and output: `in2048_out2048` at 4k and `in4096_out4096` at 8k. Each model also runs `in1024_out1024` at 2k, which has no serve counterpart. `in<N>_out<N>` is part of the benchmark identity downstream, so changing a shape starts a new trend line rather than bending the old one. The `*_smoke` entries are the exception: a short shape on a small model, as a fast signal that needs no long compile.

## Running locally

Benchmarks run through the `perf-tests` Make target. Three optional filters,
which combine — a test runs only if it passes all of them:

- `MODELS` — comma-separated model names (matched case-insensitively). Empty =
  all models.
- `TPS` — comma-separated tensor-parallel sizes, e.g. `TPS=1,4`. Empty = all
  sizes.
- `BENCH_TYPES` — comma-separated subset of `latency,throughput,serve`. Empty =
  all types.

```bash
# Everything (all models, all bench types)
make perf-tests RESULTS_DIR=benchmark-results

# Just the serve benchmark for one model, at tensor-parallel 4
make perf-tests RESULTS_DIR=benchmark-results \
  MODELS=ibm-granite/granite-3.3-8b-instruct \
  TPS=4 \
  BENCH_TYPES=serve
```
