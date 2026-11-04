# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` is either a YAML list
of test entries or a `defaults` mapping plus a `tests` list, where each test
entry is merged over `defaults` so it only spells out what differs; one entry
per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

`latency-tests-dev.yaml`, `throughput-tests-dev.yaml` and `serve-tests-dev.yaml` are not run in CI. The offline ones run the offline benchmarks at the serve context lengths, and the serve one holds every `serve-tests.yaml` entry plus a 32k entry per model replaying the `all_sequences` trace. Each takes hours per model, so run them on your own pod with `make perf-tests BENCH_SUITE=dev`. `SPYRE_DEVICES` / `AIU_WORLD_SIZE` are derived from each entry's tensor-parallel size rather than spelled out per test.

Serve entries replay a recorded agentic trace: real router prompts, each request keeping the output length it actually produced, in recorded order. A trace rather than a fixed prompt shape is what makes prefill chunking, prefix reuse, and KV-block pressure visible. A serve entry's trailing suffix names the trace it replays.

Trace paths come from `SPYRE_AIOPS_DATASET` (`*_aiops`, run at 4k), `SPYRE_CICS_DATASET` (`*_cics`, run at 8k), `SPYRE_ALL_SEQUENCES_DATASET` (`*_all`, run at 32k), and `SPYRE_VISION_DATASET` (`*_vision`, run at 32k with `openai-chat` backend). By default, `make perf-tests` resolves them from the cache and fetches missing or corrupt copies, preserving overrides that point to existing files. `FETCH_BENCH_DATA=0` bypasses this step. Variables left unset default to their location under `SPYRE_BENCH_DATA_DIR`, except `SPYRE_VISION_DATASET`, which has no default and must be set explicitly; a selected entry whose file is absent fails the run.

The dev latency and throughput configs cannot replay those traces — `vllm bench latency` takes no dataset at all, and `vllm bench throughput` rejects the `custom` dataset the traces load through. They instead split each `max-model-len` evenly between prompt and output: `in2048_out2048` at 4k and `in4096_out4096` at 8k. Each model also runs `in1024_out1024` at 2k, which has no serve counterpart. `in<N>_out<N>` is part of the benchmark identity downstream, so changing a shape starts a new trend line rather than bending the old one. The `*_smoke` entries are the exception: a short shape on a small model, as a fast signal that needs no long compile.

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

# The long-context dev configs for one model
make perf-tests RESULTS_DIR=benchmark-results BENCH_SUITE=dev \
  MODELS=ibm-granite/granite-3.3-8b-instruct
```

## Run records and repeats

Every test also writes, under `<results-dir>/<test_name>/` (a subdirectory, so ingest's top-level `*.json` glob never reads it), records of the environment it ran in: `env.json` (host, CPU, cgroup limits, cards, knobs, `/home/senuser/spyre_artifact.json`), `phases.json` (monotonic phase spans: `run` for latency/throughput, `load` then `serve` for serve), and `host_resources.json` plus `host_series.jsonl` (about 1 Hz: CPU cores, RSS of the harness process tree, cgroup memory and throttling). Serve tests also write `outputs.json`, a SHA-256 of the greedy completion of four fixed prompts, taken after the timed run; `--no-output-hash` (or `BENCH_OUTPUT_HASH=0`) skips it. Device memory is not recorded: the benchmarked process is a subprocess.

Set `repeats: N` on a test to run it N times. The first run writes where ingest reads; run k writes under `<results-dir>/repeats/r<k>/` with the same layout, so repeats add samples without adding a series. For an A-A noise check, run the same commit twice (`repeats: 2`, or two runs into different `RESULTS_DIR`s) and compare the metric across the pair: the spread is the noise floor a change must exceed.

## Profiled runs

`--profile-level light|full` on `run_vllm_benchmarks.py` (`make perf-tests PROFILE_LEVEL=…`), or `profile_level:` on a test, runs that test under vLLM's torch profiler (`--profile` plus a `--profiler-config` the harness builds). `BENCH_SUITE=profile` runs `latency-tests-profile.yaml`, one small granite-3.3-8b TP1 latency test at `full`.

- `light`: CPU + PrivateUse1 activities, no stacks, shapes or memory. The `spyre_inference` plugin wraps each EngineCore step in a `ttft` (requests were waiting) or `itl_NNNN` region, the names spyre-perf-suite's analysers read, with a `spyre::<prefill|decode|mixed>:<num_tokens>` region inside it. It also writes the engine's resolved limits and phase markers (`weight_load`, `compile_warmup`, `warmup`, `prefill`, `decode`, …). TP>1 workers trace in their own processes, where vLLM's `execute_context_*` annotation marks the steps instead.
- `full`: `light` plus `record_shapes`, `profile_memory`, `SPYRE_ATTN_PROFILING=1`, torch-spyre's kernel-cost sidecar (`TORCH_SPYRE_KERNEL_COST_DIR`) and a device-memory sample from each worker every 0.1 s.

A profiled test writes nothing to `<results-dir>/*.json`, so it never reaches ingest or a trend line: its result, command, logs and analysis go to `<results-dir>/<test_name>/trace/`, the bundle torch-spyre's `vllm-trace` ingest reads (pass `--label <test_name>`):

| file | content |
|---|---|
| `manifest.json` | label, `profile_level`, artifact, env fingerprint, profiler config, traces, and `absent`: each input the run did not produce, with the reason |
| `e2e.json` | `args` (model, tp, batch_size, input_len, output_len, and the engine's resolved max_model_len, max_num_seqs, max_num_batched_tokens = budget), `e2e_s` of the profiled steps |
| `phases.jsonl` | every process's phase markers, wall clock |
| `res.json` | per traced phase (harness-only `startup`, `ready`, `profiled`, `post` are left out): wall, CPU, throttling, RSS and cgroup peaks, device memory peaks; `compile_summary` |
| `host.json` | per trace: device-row `audit`, per-step rows, prefill/decode/mixed summaries (wall, device busy, host-exposed time by host class, launches, copies, CPU fallbacks) |
| `roof.json` | per kernel and step kind: calls, median, and with the sidecar the compiler-model floor, ratio, bound and predicted time; per-step `timeline` |
| `profile_report.json` | the same analysis in spyre-perf-suite `spsprofile.profile_report()`'s shape: `schema`, `trace`, `audit`, `steps.{decode,prefill}`, `roofline`, `resources` |
| `raw/`, `kernel_cost/`, `markers/` | the vLLM traces, sidecar lines and per-process markers these came from |
| `<test_name>.json` | the bench result where one is written (throughput, serve), with `profile_level` added; `vllm bench latency --profile` writes none |

Profiled numbers are not comparable to unprofiled ones: the profiler, the regions and (at `full`) shapes and memory tracking all add host time.
