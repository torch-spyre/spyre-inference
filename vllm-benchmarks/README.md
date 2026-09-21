# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` is a YAML list of
test entries; one entry per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

Serve entries replay a recorded agentic trace: real router prompts, each request keeping the output length it actually produced, replayed in recorded order so prefix-cache behaviour is reproducible. The `*_4k` and `*_8k` suffixes give the context the trace needs, served with `max-model-len: 4096` and `8192` respectively. A trace rather than a fixed prompt shape is what makes prefill chunking, prefix reuse, and KV-block pressure visible.

Trace paths are environment variables, so each host can point them at its own copy: `SPYRE_AIOPS_DATASET` for the AIOps trace (`*_4k`) and `SPYRE_CICS_DATASET` for the CICS trace (`*_8k`). Unset, each falls back to its location on the Spyre benchmark hosts. Where a file is not present, the runner skips the entries using it with a warning instead of failing.

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
