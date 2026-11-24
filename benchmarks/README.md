# Benchmark catalog

**This directory is generated. Do not edit it by hand.**

`vllm-benchmarks/benchmarks/spyre/*.yaml` is the single source of truth for the benchmarks. This
catalog is derived from it by `tools/generate_catalog.py`, and exists only to give the
[`spyre-bench`][engine] engine the shape it needs in order to add what those files cannot express:

- **what runs tonight** — a named `Selection` (`smoke`, `nightly-f1`, `perf-core`, `weekly-full`)
  instead of a command line full of `MODELS=`/`TPS=` filters;
- **which stored series a case belongs to** — an `identity` block reproducing the id the ClickHouse
  ingest already derives, so a trend line survives a refactor;
- **reuse, cost and verdicts** — a tier and a card count the planner can act on.

## Adding or changing a benchmark

Edit the config in `vllm-benchmarks/`, then regenerate:

```bash
make bench-generate     # rewrite this directory from the configs
make bench-validate     # schemas, references and cross-resource rules
make bench-check-ids    # identity blocks vs the ingest's own derivation
```

The only field a config needs beyond what it already had is `tier:` (`sanity`, `core` or
`extended`), which is what the Selections select on. Everything else is derived:

| Derived | From |
|---|---|
| model, precision | the hub id in the config, looked up against the engine's shared catalog; an FP8 repo is `fp8` |
| tensor parallelism | the config's own `tensor-parallel-size`, via `run_vllm_benchmarks.py._config_tp` |
| suite | the endpoint — completions → `decoder`, embeddings/rerank → `pooling`, chat → `multimodal` |
| endpoint, requests, concurrency, shapes | the config's `parameters` |
| dataset | the config's `dataset-path` variable, resolved through `.github/cache_config/artifactory_datasets.yaml` to the file a shared Asset declares |
| identity | the test name, parsed by `spyre_clickhouse_ingest.vllm`'s own code |

`make bench-check-catalog` fails if this directory is stale against the configs, and runs on every
PR — so the generated files cannot drift from their source.

## Nothing here runs a benchmark

`.github/scripts/run_vllm_benchmarks.py` is still the only executor. Every generated case invokes
it through one Make target:

```yaml
command: [make, bench-case, "CASE={case}", "BENCH_TYPES=serve", "RESULTS_DIR={raw_dir}"]
```

So the runner's path is written down in exactly one place (`mk/bench.mk`), not in twenty files.

## Case names are load-bearing

The ingest derives a stored benchmark's id from its **name**: `run_mode` is the first
`_`-separated token, and the `tp<N>_in<N>_out<N>` tokens are hashed as discriminators
(`spyre_clickhouse_ingest.vllm.parse_input_shapes`). Renaming a test in `vllm-benchmarks/` starts a
new trend line instead of continuing the old one. `make bench-check-ids` recomputes every id with
the ingest's own parser and fails on any difference, so this is checked rather than hoped about.

Names here are lowercase because the engine requires it, while two configs spell a model
`granite8B`. That is safe: the ingest lowercases before hashing (`identity.DerivedId.norm`), so both
spellings are the same stored benchmark, and `--tests` matches case-insensitively like `--models`.

## Known gap

`serve_ministral3-14b_tp2_mm_vision` is not generated. Its config sets `num-prompts: -1` ("replay
every record"), and no engine executor can express that — `closed` needs `requests` or `duration`,
`batch` needs `requests`. Give the config a request count and it will be picked up; the generator
reports the skip rather than omitting it silently.

[engine]: https://github.com/torch-spyre/torch-spyre/tree/main/extensions/spyre-bench
