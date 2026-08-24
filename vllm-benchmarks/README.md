# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` is a YAML list of
test entries; one entry per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

## Running locally

Benchmarks run through the `perf-tests` Make target. Two optional filters:

- `MODELS` — comma-separated model names (matched case-insensitively). Empty =
  all models.
- `BENCH_TYPES` — comma-separated subset of `latency,throughput,serve`. Empty =
  all types.

```bash
# Everything (all models, all bench types)
make perf-tests RESULTS_DIR=benchmark-results

# Just the serve benchmark for one model
make perf-tests RESULTS_DIR=benchmark-results \
  MODELS=ibm-granite/granite-3.3-8b-instruct \
  BENCH_TYPES=serve
```
