# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` is a YAML list of
test entries; one entry per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

Serve entries replay a recorded agentic trace: real router prompts, each request keeping the output length it actually produced, in recorded order. A trace rather than a fixed prompt shape is what makes prefill chunking, prefix reuse, and KV-block pressure visible. A serve entry's trailing suffix names the trace it replays.

Trace paths come from `SPYRE_AIOPS_DATASET` (`*_aiops`, run at 4k) and `SPYRE_CICS_DATASET` (`*_cics`, run at 8k), so each host can point at its own copy; unset, each falls back to its location on the Spyre benchmark hosts. A selected entry whose file is absent fails the run, so a serve-only job cannot go green without measuring anything.

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
