# Benchmark definitions

Definitions for the `spyre-bench` engine (`bench/v1alpha1`), covering the `vllm bench` perf runs and the GSM8K
correctness gate. They are advisory: nothing here gates a check, and `make perf-tests` and the pytest gate still read
`vllm-benchmarks/` and `tests/`.

| File | Contents |
|---|---|
| `project.yaml` | Project, Suites (one per folder) and Selections (`smoke`, `perf-*`, `quality-*`) |
| `environments.yaml` | what a case needs: 1, 2 or 4 cards |
| `decoder/`, `encoder/`, `multimodal/` | perf Benchmarks; a variant name is a `test_name` from `vllm-benchmarks/`, lower-cased |
| `quality/` | the GSM8K gate models as Bench quality cases that record the score |
| `models.yaml`, `assets-*.yaml`, `stores.yaml`, `datasets.yaml` | weights, traces and where to fetch them |
| `schedule.yml` | nightly or weekly cadence per Selection |

```bash
make bench-validate                       # no-op with a notice unless spyre-bench is installed
make bench-validate BENCH_REQUIRED=1      # fail when the engine is missing
make bench-explain CASE=serve_granite33-8b_tp1_cics
```

The engine lives in `extensions/spyre-bench` of torch-spyre and is not published. Install it from a checkout, for example
`uv pip install "extensions/spyre-bench[jsonschema]"`. `bench-list`, `bench-plan`, `bench-run`, `bench-smoke` and
`bench-report` print a notice until the engine provides those verbs.

When a `vllm-benchmarks/` entry changes, change its variant here too: the two are not generated from each other yet.
