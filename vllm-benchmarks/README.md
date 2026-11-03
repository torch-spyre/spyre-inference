# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on Spyre hardware. There is one file per model and bench type, under `benchmarks/spyre/serve-tests/decoder/` (decoder models) and `benchmarks/spyre/serve-tests/encoder/` (pooling models). The file name is `<bench type>-<model>-runs.yaml`, where `<model>` is the Hugging Face model name without its org, and the prefix picks the command:

- `latency-*-runs.yaml` → `vllm bench latency`
- `throughput-*-runs.yaml` → `vllm bench throughput`
- `serve-*-runs.yaml` → `vllm bench serve` (starts a server, waits for health, then benchmarks against it)

Each file is a `defaults` mapping plus a `tests` list, where each test entry is merged over `defaults` so it only spells out what differs. `SPYRE_DEVICES` / `AIU_WORLD_SIZE` are derived from each entry's tensor-parallel size rather than spelled out per test. A `test_name` must be unique across all files, since it names the result file.

## Suites

Each entry lists the suites it belongs to in `labels`:

| Suite | Entries | Use |
|---|---|---|
| `minimal` | labelled `minimal` | Automated runs: the `vLLM Benchmark` workflow and the `perf` test type. Must stay short. |
| `base` | labelled `base` | The standard set for evaluating a change. |
| `reference` | labelled `reference` | The configs used for reporting and decisions. |
| `exhaustive` | every entry | The full sweep per model. An entry with no label runs only here. |

`BENCH_SUITE` picks the suite (default `minimal`), and both the runner and the CI matrix generator select from the same labels. A label other than these three fails the run.

Decoder serve entries share one setup per tensor-parallel size: TP4 runs at `max-model-len 32768` with `max-num-seqs 32`, TP1 at `max-model-len 8192` with `max-num-seqs 8`, since the TP4 KV cache does not fit on one card. Both use chunked prefill, prefix caching, 200 prompts, `temperature 0.1` and the trace's own output lengths. A decoder serve `test_name` reads `serve_<model>_tp<N>_conc<C>_<dataset>`.

Serve entries replay a recorded agentic trace: real router prompts, each request keeping the output length it actually produced, in recorded order. A trace rather than a fixed prompt shape is what makes prefill chunking, prefix reuse, and KV-block pressure visible. A serve entry's trailing suffix names the trace it replays.

Trace paths come from `SPYRE_AIOPS_DATASET` (`*_aiops`), `SPYRE_CICS_DATASET` (`*_cics`), `SPYRE_ALL_SEQUENCES_DATASET` (`*_all` at TP4), `SPYRE_ALL_SEQUENCES_TRUNCATED_8K_DATASET` (`*_all` at TP1), and `SPYRE_VISION_DATASET` (`*_vision`, with the `openai-chat` backend). By default, `make perf-tests` resolves them from the cache and fetches missing or corrupt copies, preserving overrides that point to existing files. `FETCH_BENCH_DATA=0` bypasses this step. Variables left unset default to their location under `SPYRE_BENCH_DATA_DIR`, except `SPYRE_VISION_DATASET`, which has no default and must be set explicitly; a selected entry whose file is absent fails the run.

The long-context latency and throughput configs cannot replay those traces — `vllm bench latency` takes no dataset at all, and `vllm bench throughput` rejects the `custom` dataset the traces load through. They instead split each `max-model-len` evenly between prompt and output: `in2048_out2048` at 4k and `in4096_out4096` at 8k. Each model also runs `in1024_out1024` at 2k, which has no serve counterpart. `in<N>_out<N>` is part of the benchmark identity downstream, so changing a shape starts a new trend line rather than bending the old one. The `in64_out64` entries are the exception: a short shape at `max-model-len 128`, as a fast signal that needs no long compile.

## Running locally

Benchmarks run through the `perf-tests` Make target. `BENCH_SUITE` picks the suite, and three optional filters narrow it further; a test runs only if it passes all of them:

- `MODELS` — comma-separated model names (matched case-insensitively). Empty =
  all models.
- `TPS` — comma-separated tensor-parallel sizes, e.g. `TPS=1,4`. Empty = all
  sizes.
- `BENCH_TYPES` — comma-separated subset of `latency,throughput,serve`. Empty =
  all types.

```bash
# The minimal suite (all models, all bench types)
make perf-tests RESULTS_DIR=benchmark-results

# The base serve entries of one model, at tensor-parallel 4
make perf-tests RESULTS_DIR=benchmark-results BENCH_SUITE=base \
  MODELS=ibm-granite/granite-4.1-8b \
  TPS=4 \
  BENCH_TYPES=serve

# Every config of one model
make perf-tests RESULTS_DIR=benchmark-results BENCH_SUITE=exhaustive \
  MODELS=google/gemma-4-26B-A4B

# Every entry of one file, whatever its labels
make perf-tests RESULTS_DIR=benchmark-results \
  BENCH_CONFIGS_DIR=vllm-benchmarks/benchmarks/spyre/serve-tests/decoder/serve-granite-4.1-8b-runs.yaml
```
