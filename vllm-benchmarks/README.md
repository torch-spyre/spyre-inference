# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on Spyre hardware. There is one file per model and bench type, under `benchmarks/decoder/` (decoder models) and `benchmarks/encoder/` (pooling models). The file name is `<bench type>-<model>-runs.yaml`, where `<model>` is the Hugging Face model name without its org, and the prefix picks the command:

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

The [config tables](#config-tables) at the end of this file list every entry and each suite. They are generated: after editing a config, run `python3 .github/scripts/generate_bench_config_tables.py` (the pre-commit hook does it for you). The `Benchmark config tables` workflow fails a change whose tables are stale or that reuses a `test_name`.

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
  BENCH_CONFIGS_DIR=vllm-benchmarks/benchmarks/decoder/serve-granite-4.1-8b-runs.yaml
```

<!-- BEGIN GENERATED: bench-config-tables -->

<!-- Generated by .github/scripts/generate_bench_config_tables.py; do not edit. -->

## Config tables

352 entries: 5 `minimal`, 28 `base`, 28 `reference`. Each cell gives the labels of one entry: `min` = minimal, `base` = base, `ref` = reference, `exh` = no label (exhaustive suite only). An empty cell has no entry.

### Decoder serve · `aiops`

| model · TP | conc4 | conc5 | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct · tp4 | exh | ref | exh | exh | exh |
| Llama-3.1-8B-Instruct · tp1 | exh | exh | exh |  |  |
| Ministral-3-14B-Instruct-2512-BF16 · tp4 | exh | ref | exh | exh | exh |
| Ministral-3-14B-Instruct-2512-BF16 · tp1 | exh | exh | exh |  |  |
| Mistral-Small-3.2-24B-Instruct-2506 · tp4 | exh | ref | exh | exh | exh |
| Mistral-Small-3.2-24B-Instruct-2506 · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B-it · tp4 | exh | ref | exh | exh | exh |
| gemma-4-26B-A4B-it · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B · tp4 | exh | min base ref | exh | exh | exh |
| gemma-4-26B-A4B · tp1 | exh | exh | exh |  |  |
| granite-3.3-8b-instruct-FP8 · tp4 | exh | base | exh | exh | exh |
| granite-3.3-8b-instruct-FP8 · tp1 | exh | exh | exh |  |  |
| granite-3.3-8b-instruct · tp4 | exh | exh | exh | exh | exh |
| granite-3.3-8b-instruct · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b-fp8 · tp4 | exh | base ref | exh | exh | exh |
| granite-4.1-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b · tp4 | exh | min base ref | exh | exh | exh |
| granite-4.1-8b · tp1 | exh | exh | exh |  |  |
| granite-4.2-8b-fp8 · tp4 | exh | ref | exh | exh | exh |
| granite-4.2-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.2-8b · tp4 | exh | base | exh | exh | exh |
| granite-4.2-8b · tp1 | exh | exh | exh |  |  |
| granite-vision-4.1-4b · tp4 | exh | exh | exh | exh | exh |
| granite-vision-4.1-4b · tp1 | exh | exh | exh |  |  |

### Decoder serve · `cics`

| model · TP | conc4 | conc5 | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct · tp4 | exh | exh | exh | exh | exh |
| Llama-3.1-8B-Instruct · tp1 | exh | exh | exh |  |  |
| Ministral-3-14B-Instruct-2512-BF16 · tp4 | exh | exh | exh | exh | exh |
| Ministral-3-14B-Instruct-2512-BF16 · tp1 | exh | exh | exh |  |  |
| Mistral-Small-3.2-24B-Instruct-2506 · tp4 | exh | exh | exh | exh | exh |
| Mistral-Small-3.2-24B-Instruct-2506 · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B-it · tp4 | exh | exh | exh | exh | exh |
| gemma-4-26B-A4B-it · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B · tp4 | exh | base | exh | exh | exh |
| gemma-4-26B-A4B · tp1 | exh | base | exh |  |  |
| granite-3.3-8b-instruct-FP8 · tp4 | exh | exh | exh | exh | exh |
| granite-3.3-8b-instruct-FP8 · tp1 | exh | exh | exh |  |  |
| granite-3.3-8b-instruct · tp4 | exh | exh | exh | exh | exh |
| granite-3.3-8b-instruct · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b-fp8 · tp4 | exh | exh | exh | exh | exh |
| granite-4.1-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b · tp4 | exh | base | exh | exh | exh |
| granite-4.1-8b · tp1 | exh | base | exh |  |  |
| granite-4.2-8b-fp8 · tp4 | exh | exh | exh | exh | exh |
| granite-4.2-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.2-8b · tp4 | exh | base | exh | exh | exh |
| granite-4.2-8b · tp1 | exh | exh | exh |  |  |
| granite-vision-4.1-4b · tp4 | exh | exh | exh | exh | exh |
| granite-vision-4.1-4b · tp1 | exh | exh | exh |  |  |

### Decoder serve · `all`

| model · TP | conc4 | conc5 | conc8 | conc16 | conc32 |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct · tp4 | exh | exh | exh | exh | exh |
| Llama-3.1-8B-Instruct · tp1 | exh | exh | exh |  |  |
| Ministral-3-14B-Instruct-2512-BF16 · tp4 | exh | exh | exh | exh | exh |
| Ministral-3-14B-Instruct-2512-BF16 · tp1 | exh | exh | exh |  |  |
| Mistral-Small-3.2-24B-Instruct-2506 · tp4 | exh | exh | exh | exh | exh |
| Mistral-Small-3.2-24B-Instruct-2506 · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B-it · tp4 | exh | exh | exh | exh | exh |
| gemma-4-26B-A4B-it · tp1 | exh | exh | exh |  |  |
| gemma-4-26B-A4B · tp4 | exh | base | exh | exh | exh |
| gemma-4-26B-A4B · tp1 | exh | exh | exh |  |  |
| granite-3.3-8b-instruct-FP8 · tp4 | exh | base | exh | exh | exh |
| granite-3.3-8b-instruct-FP8 · tp1 | exh | exh | exh |  |  |
| granite-3.3-8b-instruct · tp4 | exh | base | exh | exh | exh |
| granite-3.3-8b-instruct · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b-fp8 · tp4 | exh | base | exh | exh | exh |
| granite-4.1-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.1-8b · tp4 | exh | base | exh | exh | exh |
| granite-4.1-8b · tp1 | exh | exh | exh |  |  |
| granite-4.2-8b-fp8 · tp4 | exh | exh | exh | exh | exh |
| granite-4.2-8b-fp8 · tp1 | exh | exh | exh |  |  |
| granite-4.2-8b · tp4 | exh | base | exh | exh | exh |
| granite-4.2-8b · tp1 | exh | exh | exh |  |  |
| granite-vision-4.1-4b · tp4 | exh | exh | exh | exh | exh |
| granite-vision-4.1-4b · tp1 | exh | exh | exh |  |  |

### Decoder serve · vision and multimodal

| model · TP | conc4_mm_random | conc4_vision |
|---|---|---|
| Ministral-3-14B-Instruct-2512-BF16 · tp4 | base | ref |
| Mistral-Small-3.2-24B-Instruct-2506 · tp4 |  | ref |
| gemma-4-26B-A4B-it · tp4 | base | ref |
| gemma-4-26B-A4B · tp4 |  | ref |
| granite-vision-4.1-4b · tp4 |  | ref |

### Pooling serve

| model · TP | conc16 | conc32 | conc64 | conc64_ci |
|---|---|---|---|---|
| all-MiniLM-L6-v2 · tp1 | ref | ref | ref |  |
| all-roberta-large-v1 · tp1 |  |  |  | base |
| bert-base-NER · tp1 | ref | ref | ref |  |
| bge-reranker-large · tp1 |  |  |  | base |
| bge-reranker-v2-m3 · tp1 |  |  |  | base |
| granite-embedding-125m-english · tp1 |  |  |  | base |
| granite-embedding-278m-multilingual · tp1 | ref | ref | ref | min base |
| multilingual-e5-large · tp1 |  |  |  | base |
| roberta-large-mnli · tp1 | ref | ref | ref |  |
| roberta-large-ner-english · tp1 | ref | ref | ref |  |

### Latency / throughput

Each cell is one latency and one throughput entry; it shows both label sets only when they differ.

| model · TP | in64_out64 | in1024_out1024 | in2048_out2048 | in4096_out4096 |
|---|---|---|---|---|
| gemma-4-26B-A4B · tp4 |  | exh |  | exh |
| gemma-4-26B-A4B · tp1 |  | exh |  | exh |
| granite-3.3-8b-instruct-FP8 · tp4 |  | exh | exh |  |
| granite-3.3-8b-instruct · tp4 |  | exh |  | exh |
| granite-3.3-8b-instruct · tp1 | base | exh |  | exh |
| granite-4.1-8b-fp8 · tp4 |  | exh | exh |  |
| granite-4.1-8b · tp4 |  | exh |  | exh |
| granite-4.1-8b · tp1 |  | exh |  | exh |
| micro-g3.3-8b-instruct-1b · tp1 | min base |  |  |  |

### `minimal` (5 entries)

| test_name | model | TP | max-model-len | max-num-seqs | concurrency | dataset | prompts | labels | file |
|---|---|---|---|---|---|---|---|---|---|
| `latency_granite8B_tp1_in64_out64` | ibm-ai-platform/micro-g3.3-8b-instruct-1b | 1 | 128 | – | – | in64 / out64 | – | min base | [latency-micro-g3.3-8b-instruct-1b-runs.yaml](benchmarks/decoder/latency-micro-g3.3-8b-instruct-1b-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc5_aiops` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_granite41-8b_tp4_conc5_aiops` | ibm-granite/granite-4.1-8b | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `throughput_granite8B_tp1_in64_out64` | ibm-ai-platform/micro-g3.3-8b-instruct-1b | 1 | 128 | – | – | in64 / out64 | 16 | min base | [throughput-micro-g3.3-8b-instruct-1b-runs.yaml](benchmarks/decoder/throughput-micro-g3.3-8b-instruct-1b-runs.yaml) |
| `serve_granite-embedding-278m-multilingual_tp1_conc64_ci` | ibm-granite/granite-embedding-278m-multilingual | 1 | 512 | 4 | 64 | random 512 | 200 | min base | [serve-granite-embedding-278m-multilingual-runs.yaml](benchmarks/encoder/serve-granite-embedding-278m-multilingual-runs.yaml) |

### `base` (28 entries)

| test_name | model | TP | max-model-len | max-num-seqs | concurrency | dataset | prompts | labels | file |
|---|---|---|---|---|---|---|---|---|---|
| `latency_granite33-8b_tp1_in64_out64` | ibm-granite/granite-3.3-8b-instruct | 1 | 128 | – | – | in64 / out64 | – | base | [latency-granite-3.3-8b-instruct-runs.yaml](benchmarks/decoder/latency-granite-3.3-8b-instruct-runs.yaml) |
| `latency_granite8B_tp1_in64_out64` | ibm-ai-platform/micro-g3.3-8b-instruct-1b | 1 | 128 | – | – | in64 / out64 | – | min base | [latency-micro-g3.3-8b-instruct-1b-runs.yaml](benchmarks/decoder/latency-micro-g3.3-8b-instruct-1b-runs.yaml) |
| `serve_ministral3-14b_tp4_conc4_mm_random` | mistralai/Ministral-3-14B-Instruct-2512-BF16 | 4 | 32768 | 32 | 4 | random-mm | 200 | base | [serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml](benchmarks/decoder/serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml) |
| `serve_gemma4-26b-a4b-it_tp4_conc4_mm_random` | google/gemma-4-26B-A4B-it | 4 | 32768 | 32 | 4 | random-mm | 200 | base | [serve-gemma-4-26B-A4B-it-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-it-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc5_aiops` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc5_cics` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 5 | cics | 200 | base | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc5_all` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_gemma4-26b-a4b_tp1_conc5_cics` | google/gemma-4-26B-A4B | 1 | 8192 | 8 | 5 | cics | 200 | base | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_granite33-8b-fp8_tp4_conc5_aiops` | ibm-granite/granite-3.3-8b-instruct-FP8 | 4 | 32768 | 32 | 5 | aiops | 200 | base | [serve-granite-3.3-8b-instruct-FP8-runs.yaml](benchmarks/decoder/serve-granite-3.3-8b-instruct-FP8-runs.yaml) |
| `serve_granite33-8b-fp8_tp4_conc5_all` | ibm-granite/granite-3.3-8b-instruct-FP8 | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-granite-3.3-8b-instruct-FP8-runs.yaml](benchmarks/decoder/serve-granite-3.3-8b-instruct-FP8-runs.yaml) |
| `serve_granite33-8b_tp4_conc5_all` | ibm-granite/granite-3.3-8b-instruct | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-granite-3.3-8b-instruct-runs.yaml](benchmarks/decoder/serve-granite-3.3-8b-instruct-runs.yaml) |
| `serve_granite41-8b-fp8_tp4_conc5_aiops` | ibm-granite/granite-4.1-8b-fp8 | 4 | 32768 | 32 | 5 | aiops | 200 | base ref | [serve-granite-4.1-8b-fp8-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-fp8-runs.yaml) |
| `serve_granite41-8b-fp8_tp4_conc5_all` | ibm-granite/granite-4.1-8b-fp8 | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-granite-4.1-8b-fp8-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-fp8-runs.yaml) |
| `serve_granite41-8b_tp4_conc5_aiops` | ibm-granite/granite-4.1-8b | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `serve_granite41-8b_tp4_conc5_cics` | ibm-granite/granite-4.1-8b | 4 | 32768 | 32 | 5 | cics | 200 | base | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `serve_granite41-8b_tp4_conc5_all` | ibm-granite/granite-4.1-8b | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `serve_granite41-8b_tp1_conc5_cics` | ibm-granite/granite-4.1-8b | 1 | 8192 | 8 | 5 | cics | 200 | base | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `serve_granite42-8b_tp4_conc5_aiops` | ibm-granite/granite-4.2-8b | 4 | 32768 | 32 | 5 | aiops | 200 | base | [serve-granite-4.2-8b-runs.yaml](benchmarks/decoder/serve-granite-4.2-8b-runs.yaml) |
| `serve_granite42-8b_tp4_conc5_cics` | ibm-granite/granite-4.2-8b | 4 | 32768 | 32 | 5 | cics | 200 | base | [serve-granite-4.2-8b-runs.yaml](benchmarks/decoder/serve-granite-4.2-8b-runs.yaml) |
| `serve_granite42-8b_tp4_conc5_all` | ibm-granite/granite-4.2-8b | 4 | 32768 | 32 | 5 | all_sequences | 200 | base | [serve-granite-4.2-8b-runs.yaml](benchmarks/decoder/serve-granite-4.2-8b-runs.yaml) |
| `throughput_granite33-8b_tp1_in64_out64` | ibm-granite/granite-3.3-8b-instruct | 1 | 128 | – | – | in64 / out64 | 16 | base | [throughput-granite-3.3-8b-instruct-runs.yaml](benchmarks/decoder/throughput-granite-3.3-8b-instruct-runs.yaml) |
| `throughput_granite8B_tp1_in64_out64` | ibm-ai-platform/micro-g3.3-8b-instruct-1b | 1 | 128 | – | – | in64 / out64 | 16 | min base | [throughput-micro-g3.3-8b-instruct-1b-runs.yaml](benchmarks/decoder/throughput-micro-g3.3-8b-instruct-1b-runs.yaml) |
| `serve_all-roberta-large-v1_tp1_conc64_ci` | sentence-transformers/all-roberta-large-v1 | 1 | 256 | 4 | 64 | random 256 | 200 | base | [serve-all-roberta-large-v1-runs.yaml](benchmarks/encoder/serve-all-roberta-large-v1-runs.yaml) |
| `serve_bge-reranker-large_tp1_conc64_ci` | BAAI/bge-reranker-large | 1 | 512 | 4 | 64 | random-rerank 512 | 200 | base | [serve-bge-reranker-large-runs.yaml](benchmarks/encoder/serve-bge-reranker-large-runs.yaml) |
| `serve_bge-reranker-v2-m3_tp1_conc64_ci` | BAAI/bge-reranker-v2-m3 | 1 | 512 | 4 | 64 | random-rerank 512 | 200 | base | [serve-bge-reranker-v2-m3-runs.yaml](benchmarks/encoder/serve-bge-reranker-v2-m3-runs.yaml) |
| `serve_granite-embedding-125m-english_tp1_conc64_ci` | ibm-granite/granite-embedding-125m-english | 1 | 512 | 4 | 64 | random 512 | 200 | base | [serve-granite-embedding-125m-english-runs.yaml](benchmarks/encoder/serve-granite-embedding-125m-english-runs.yaml) |
| `serve_granite-embedding-278m-multilingual_tp1_conc64_ci` | ibm-granite/granite-embedding-278m-multilingual | 1 | 512 | 4 | 64 | random 512 | 200 | min base | [serve-granite-embedding-278m-multilingual-runs.yaml](benchmarks/encoder/serve-granite-embedding-278m-multilingual-runs.yaml) |
| `serve_multilingual-e5-large_tp1_conc64_ci` | intfloat/multilingual-e5-large | 1 | 512 | 4 | 64 | random 512 | 200 | base | [serve-multilingual-e5-large-runs.yaml](benchmarks/encoder/serve-multilingual-e5-large-runs.yaml) |

### `reference` (28 entries)

| test_name | model | TP | max-model-len | max-num-seqs | concurrency | dataset | prompts | labels | file |
|---|---|---|---|---|---|---|---|---|---|
| `serve_llama31-8b_tp4_conc5_aiops` | meta-llama/Llama-3.1-8B-Instruct | 4 | 32768 | 32 | 5 | aiops | 200 | ref | [serve-Llama-3.1-8B-Instruct-runs.yaml](benchmarks/decoder/serve-Llama-3.1-8B-Instruct-runs.yaml) |
| `serve_ministral3-14b_tp4_conc5_aiops` | mistralai/Ministral-3-14B-Instruct-2512-BF16 | 4 | 32768 | 32 | 5 | aiops | 200 | ref | [serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml](benchmarks/decoder/serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml) |
| `serve_ministral3-14b_tp4_conc4_vision` | mistralai/Ministral-3-14B-Instruct-2512-BF16 | 4 | 32768 | 32 | 4 | vision | 166 | ref | [serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml](benchmarks/decoder/serve-Ministral-3-14B-Instruct-2512-BF16-runs.yaml) |
| `serve_mistral-small32-24b_tp4_conc5_aiops` | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 4 | 32768 | 32 | 5 | aiops | 200 | ref | [serve-Mistral-Small-3.2-24B-Instruct-2506-runs.yaml](benchmarks/decoder/serve-Mistral-Small-3.2-24B-Instruct-2506-runs.yaml) |
| `serve_mistral-small32-24b_tp4_conc4_vision` | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 4 | 32768 | 32 | 4 | vision | 166 | ref | [serve-Mistral-Small-3.2-24B-Instruct-2506-runs.yaml](benchmarks/decoder/serve-Mistral-Small-3.2-24B-Instruct-2506-runs.yaml) |
| `serve_gemma4-26b-a4b-it_tp4_conc5_aiops` | google/gemma-4-26B-A4B-it | 4 | 32768 | 32 | 5 | aiops | 200 | ref | [serve-gemma-4-26B-A4B-it-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-it-runs.yaml) |
| `serve_gemma4-26b-a4b-it_tp4_conc4_vision` | google/gemma-4-26B-A4B-it | 4 | 32768 | 32 | 4 | vision | 166 | ref | [serve-gemma-4-26B-A4B-it-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-it-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc5_aiops` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_gemma4-26b-a4b_tp4_conc4_vision` | google/gemma-4-26B-A4B | 4 | 32768 | 32 | 4 | vision | 166 | ref | [serve-gemma-4-26B-A4B-runs.yaml](benchmarks/decoder/serve-gemma-4-26B-A4B-runs.yaml) |
| `serve_granite41-8b-fp8_tp4_conc5_aiops` | ibm-granite/granite-4.1-8b-fp8 | 4 | 32768 | 32 | 5 | aiops | 200 | base ref | [serve-granite-4.1-8b-fp8-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-fp8-runs.yaml) |
| `serve_granite41-8b_tp4_conc5_aiops` | ibm-granite/granite-4.1-8b | 4 | 32768 | 32 | 5 | aiops | 200 | min base ref | [serve-granite-4.1-8b-runs.yaml](benchmarks/decoder/serve-granite-4.1-8b-runs.yaml) |
| `serve_granite42-8b-fp8_tp4_conc5_aiops` | ibm-granite/granite-4.2-8b-fp8 | 4 | 32768 | 32 | 5 | aiops | 200 | ref | [serve-granite-4.2-8b-fp8-runs.yaml](benchmarks/decoder/serve-granite-4.2-8b-fp8-runs.yaml) |
| `serve_granite-vision41-4b_tp4_conc4_vision` | ibm-granite/granite-vision-4.1-4b | 4 | 32768 | 32 | 4 | vision | 166 | ref | [serve-granite-vision-4.1-4b-runs.yaml](benchmarks/decoder/serve-granite-vision-4.1-4b-runs.yaml) |
| `serve_all-MiniLM-L6-v2_tp1_conc16` | sentence-transformers/all-MiniLM-L6-v2 | 1 | 256 | 8 | 16 | random 196 | 10000 | ref | [serve-all-MiniLM-L6-v2-runs.yaml](benchmarks/encoder/serve-all-MiniLM-L6-v2-runs.yaml) |
| `serve_all-MiniLM-L6-v2_tp1_conc32` | sentence-transformers/all-MiniLM-L6-v2 | 1 | 256 | 8 | 32 | random 196 | 10000 | ref | [serve-all-MiniLM-L6-v2-runs.yaml](benchmarks/encoder/serve-all-MiniLM-L6-v2-runs.yaml) |
| `serve_all-MiniLM-L6-v2_tp1_conc64` | sentence-transformers/all-MiniLM-L6-v2 | 1 | 256 | 8 | 64 | random 196 | 10000 | ref | [serve-all-MiniLM-L6-v2-runs.yaml](benchmarks/encoder/serve-all-MiniLM-L6-v2-runs.yaml) |
| `serve_bert-base-NER_tp1_conc16` | dslim/bert-base-NER | 1 | 512 | 4 | 16 | random 512 | 10000 | ref | [serve-bert-base-NER-runs.yaml](benchmarks/encoder/serve-bert-base-NER-runs.yaml) |
| `serve_bert-base-NER_tp1_conc32` | dslim/bert-base-NER | 1 | 512 | 4 | 32 | random 512 | 10000 | ref | [serve-bert-base-NER-runs.yaml](benchmarks/encoder/serve-bert-base-NER-runs.yaml) |
| `serve_bert-base-NER_tp1_conc64` | dslim/bert-base-NER | 1 | 512 | 4 | 64 | random 512 | 10000 | ref | [serve-bert-base-NER-runs.yaml](benchmarks/encoder/serve-bert-base-NER-runs.yaml) |
| `serve_granite-embedding-278m-multilingual_tp1_conc16` | ibm-granite/granite-embedding-278m-multilingual | 1 | 512 | 4 | 16 | random 512 | 10000 | ref | [serve-granite-embedding-278m-multilingual-runs.yaml](benchmarks/encoder/serve-granite-embedding-278m-multilingual-runs.yaml) |
| `serve_granite-embedding-278m-multilingual_tp1_conc32` | ibm-granite/granite-embedding-278m-multilingual | 1 | 512 | 4 | 32 | random 512 | 10000 | ref | [serve-granite-embedding-278m-multilingual-runs.yaml](benchmarks/encoder/serve-granite-embedding-278m-multilingual-runs.yaml) |
| `serve_granite-embedding-278m-multilingual_tp1_conc64` | ibm-granite/granite-embedding-278m-multilingual | 1 | 512 | 4 | 64 | random 512 | 10000 | ref | [serve-granite-embedding-278m-multilingual-runs.yaml](benchmarks/encoder/serve-granite-embedding-278m-multilingual-runs.yaml) |
| `serve_roberta-large-mnli_tp1_conc16` | FacebookAI/roberta-large-mnli | 1 | 512 | 4 | 16 | random 512 | 10000 | ref | [serve-roberta-large-mnli-runs.yaml](benchmarks/encoder/serve-roberta-large-mnli-runs.yaml) |
| `serve_roberta-large-mnli_tp1_conc32` | FacebookAI/roberta-large-mnli | 1 | 512 | 4 | 32 | random 512 | 10000 | ref | [serve-roberta-large-mnli-runs.yaml](benchmarks/encoder/serve-roberta-large-mnli-runs.yaml) |
| `serve_roberta-large-mnli_tp1_conc64` | FacebookAI/roberta-large-mnli | 1 | 512 | 4 | 64 | random 512 | 10000 | ref | [serve-roberta-large-mnli-runs.yaml](benchmarks/encoder/serve-roberta-large-mnli-runs.yaml) |
| `serve_roberta-large-ner-english_tp1_conc16` | Jean-Baptiste/roberta-large-ner-english | 1 | 512 | 4 | 16 | random 512 | 10000 | ref | [serve-roberta-large-ner-english-runs.yaml](benchmarks/encoder/serve-roberta-large-ner-english-runs.yaml) |
| `serve_roberta-large-ner-english_tp1_conc32` | Jean-Baptiste/roberta-large-ner-english | 1 | 512 | 4 | 32 | random 512 | 10000 | ref | [serve-roberta-large-ner-english-runs.yaml](benchmarks/encoder/serve-roberta-large-ner-english-runs.yaml) |
| `serve_roberta-large-ner-english_tp1_conc64` | Jean-Baptiste/roberta-large-ner-english | 1 | 512 | 4 | 64 | random 512 | 10000 | ref | [serve-roberta-large-ner-english-runs.yaml](benchmarks/encoder/serve-roberta-large-ner-english-runs.yaml) |

<!-- END GENERATED: bench-config-tables -->
