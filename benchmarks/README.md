# Benchmark definitions

Definitions for the `spyre-bench` engine (`bench/v1alpha1`), covering the `vllm bench` perf runs and the
accuracy correctness gate. They are advisory: nothing here gates a check, and `make perf-tests` and the pytest gate still read
`vllm-benchmarks/` and `tests/`.

| File | Contents |
|---|---|
| `project.yaml` | Project, Suites (one per folder) and Selections (`smoke`, `perf-*`, `quality-*`) |
| `environments.yaml` | what a case needs: 1, 2 or 4 cards |
| `decoder/`, `encoder/`, `multimodal/` | perf Benchmarks; a variant name is a `test_name` from `vllm-benchmarks/`, lower-cased |
| `quality/` | the accuracy-gate models as Bench quality cases that record the score, and `profiles/<arch>.yml` (which cases an architecture runs) |
| `models.yaml`, `assets-models.yaml` | weights |
| `assets-datasets.yaml`, `assets-eval.yaml`, `datasets.yaml`, `stores.yaml` | every dataset a case reads: store path or public source, sha256, archive rule and the variable the configs read |
| `analysis/` | Analyses: `decode-per-token` derives `(latency@64 - latency@1) / 63` from the `out64` and `out1` pair |
| `schedule.yml` | nightly or weekly cadence per Selection |

```bash
make bench-validate                       # exits 2 with the install command when the engine is missing
make bench-validate BENCH_OPTIONAL=1      # a notice and exit 0 instead (the pre-commit hook does this)
make bench-explain CASE=serve_granite33-8b_tp1_cics
```

The engine lives in `extensions/spyre-bench` of torch-spyre and is not published. Install it from a checkout, for example
`uv pip install "extensions/spyre-bench[jsonschema]"`. `bench-list`, `bench-plan`, `bench-run`, `bench-smoke` and
`bench-report` need an engine that provides those verbs and exit 2 otherwise; `BENCH_REQUIRED=1` wins over `BENCH_OPTIONAL=1`.

Every entry the runner can reach (`*-tests.yaml` and `*-tests-<suite>.yaml`, `multimodal-serve-tests.yaml`) has one Benchmark here,
one file each, with an `identity` block, so `benchmark_id` is the id the existing ingest writes for the test name. Entries of the CI
configs are `core` or `sanity`; the dev-suite entries (`BENCH_SUITE=dev`: long contexts, the 32k `_all` traces) are `extended`.
When a `vllm-benchmarks/` entry is added or changes, change its Benchmark too: `test_bench_migrate.py` fails on a missing
definition, a changed discriminator, or a parameter that drifted. The config folders no option of the runner reaches
(`power_use_case/`, `serve-tests/pooling/`, `serve_tests/decoder/`) are not defined; nothing in this repo runs them.

Each command runs the existing runner through `.github/scripts/bench_perf_vllm.py run`, which leaves the native vLLM JSON where the
ingest reads it and also writes the `metrics.json` sidecar the engine turns into the repeat's samples (only metrics the
llm-serving pack registers; vLLM's combined `tokens_per_second` has no registered meaning and is not recorded).

## Quality

The accuracy gate (`make test-quality`, the upstream accuracy test per model) is unchanged: it keeps
asserting its own threshold. Bench quality only *records* the score it already measures, as one result per model, so the
history is kept. Nothing here judges or gates anything.

```bash
# the gate prints one results block per model with pytest -s; keep that output
make test-quality-shard QUALITY_SHARD_ID=0 2>&1 | tee results/bench/quality-gate.log
make bench-quality                       # or: make tests TEST_TYPE=quality
make bench-quality BENCH_INGEST=1        # also ingest (CLICKHOUSE_* set), test type quality
make bench-quality-cases                 # the cases this machine's profile enables
```

`TEST_TYPE=quality` means Bench quality. The pytest targets keep their names (`test-quality`, `test-quality-shard`) and are the
output gate.

| Piece | Role |
|---|---|
| `.github/scripts/bench_quality_gate.py record` | reads a model's results block from the gate log; writes `metrics.json` for the engine and a generic `benchmarks` file for the ingest |
| `quality/eval-slot-12-<model>.yaml` | the case: greedy (`temperature: 0`, no seed in identity), 64 questions, the question file's selection digest, identity profile `generic` |
| `quality/profiles/<arch>.yml` | which cases an architecture runs, as data. Only the file for the build machine is `enabled`; the other two are disabled stubs |

What is recorded per model: the score, its standard error (binomial, from the question count), the questions scored, the share
of invalid answers; and as run attributes the harness version, a digest of the protocol (few-shot count, question count, token
budget, temperature), the model's cached commit, and the digest of the questions scored. A missing pin is left out, not guessed,
so the engine can say the result is unpinned. The gate has no per-request error counter and prints no per-question outcomes, so
neither an error rate nor a per-question vector is recorded.

The `quality-sanity` and `quality-core` selections record from a gate run; used without one their commands fail by design
(no log). They are not part of `nightly`. The log must come from the run being recorded: a stale one records an old score.

Adding another architecture is a data change: set `enabled: true` and list its `cases` (and `skip_models` for models it cannot
load) in its profile. The recorder, the definitions and the Make targets do not change.

## Nightly route

Nightly and weekly runs of the `nightly` Selection are triggered from Jenkins presets, on the nightly-built image.
`.github/workflows/bench-on-demand.yml` is the manual route (workflow_dispatch only, no schedule): validate, plan, run,
ingest and verdict, one leg per model and tensor-parallel size on the labelled card runners. Both are advisory and
ingest into the dev instance unless told otherwise (the workflow's `target: prod` needs a dispatch from main).
`mode: smoke` runs a stub command on a hosted runner: no cards, no ingest.

| Make target | What |
|---|---|
| `bench-legs` | the legs as JSON (`MODELS=<csv>` narrows) |
| `bench-leg` | one leg: `BENCH_RUNNER=native` (`perf-tests` over `vllm-benchmarks/benchmarks/spyre/*-tests-nightly.yaml`), `engine` (`bench-plan`, then `spyre-bench run`, whose per-case `run.command` calls the same runner) or `stub` |
| `bench-dry-run`, `bench-verdict` | render the plan's commands without running them; the rolling-baseline verdict (words and counts only) of a run |
| `bench-conformance` | the definitions against the real ingest producer (`.github/scripts/test_bench_nightly.py`) |
| `bench-ingest` | write `BENCH_NATIVE_DIR` with `ingest_vllm_benchmarks.py`, the arguments the GitHub action passes (the Jenkins ingest) |

`bench-run` plans first and then runs the plan, so it needs an engine with both verbs. Both runners leave the vLLM JSON in `results/bench/native`, and one ingest step writes it with the existing
`ingest-vllm-benchmarks` action; the engine never writes ClickHouse rows and the benchmark process never sees the
ClickHouse secrets.

A Benchmark with an `identity` block (profile `vllm`) derives the same `benchmark_id` as `bench_entries` for the same
test name, so the route and `make tests TEST_TYPE=perf` write to one series. A name that ends in `_out1`, or a TP4
latency name, has no entry in the CI files: it is a new series. Keep the `*-tests-nightly.yaml` entries equal to the CI
entries of the same name; the tests check it.

### Side by side with the legacy rows

The workflow input `bench_mode` is `off` (the default: legacy rows only) or `side-by-side`; there is no `replace` yet. Side by
side needs `runner: engine` (only an engine run has Results to record) and `subject_id`, the tested artifact id. The extra step
is torch-spyre's `ingest-xml-to-clickhouse` action with `bench-dir: results/bench`; it records `ci` from `main` and `lab` from any
other ref, passes no `workflow` (the legacy v2 rows carry no source job, so the series follows the benchmark id), is advisory
(`continue-on-error`) and never touches the verdict. The legacy `ingest-vllm-benchmarks` step is unchanged and still writes the
v1 and v2 perf rows, so the HUD shows both until the projection is fixed. The CI `perf` job (`make tests TEST_TYPE=perf`) emits
no bundle: it runs `perf-tests` directly, so there are no engine Results to record. Jenkins passes the same results directory to
its shared library's `benchDir` with the orchestrator's artifact id as `BENCH_SUBJECT`.

### CPU pinning

Every Environment states `host.cpu_pinning`. The default is `none`, and `env.label.cpu_pinning` is a series label, so a
pinned and an unpinned run never share a baseline. The `-device-local` twins request `device_local` with
`require_verified`. The engine does not apply a policy: it samples the process tree and records `run.exec.cpu_pinning`
and the verification, and a violated or unobservable placement caps the result at `suspect`. Once the vLLM path pins
CPUs itself, dispatch the workflow with `runner: engine` and `cpu_pinning: device_local`; the native runner refuses
anything but `none`. Compare the two with `spyre-bench analyze` on `decode-per-token`. Until then nothing in the route
pins.

### What Jenkins provides

The `make bench-*` targets read no GitHub variable. A Jenkins leg on the nightly image provides:

| Need | How |
|---|---|
| Checkout of this repo at the tested ref | the job's workspace; `BENCH_BRANCH` and `BENCH_SHA` name it |
| Engine and ingest package | `pip install` of `spyre-bench` and `spyre-clickhouse-ingest` from torch-spyre `${BENCH_ENGINE_REF:-main}` (`#subdirectory=extensions/...`) into a venv apart from the product venv; `BENCH=<venv>/bin/spyre-bench`, `BENCH_HELPER_PY=<venv>/bin/python` |
| Product environment | the nightly image as built; `make` supplies the AIU setup (`BENCH_PREFIX`, `perf-tests`) |
| Request | `BENCH_PURPOSE=nightly`, `BENCH_REUSE=never`, `BENCH_SELECTION=nightly`, `BENCH_RUNNER=native` or `engine`, `BENCH_CPU_PINNING=none`, `BENCH_QUIET=1`, `BENCH_REQUIRED=1`, `BENCH_ENVIRONMENT=spyre-<N>card` |
| Legs | `make -s bench-legs` (JSON of model, TP, cards); per leg `BENCH_LEG_MODEL`, `BENCH_LEG_TP` |
| Run | `make bench-leg`; the vLLM JSON ends in `results/bench/native` |
| Ingest | `make bench-ingest BENCH_STORED_RUN_ID=<orchestrator run uuid> BENCH_BRANCH BENCH_SHA BENCH_LEG_KEY BENCH_REPOSITORY BENCH_STATE`, with `CLICKHOUSE_HOST/PORT/USER/PASS/DB/DB_V2` (credentials only for this command, never for `bench-leg`) |
| Verdict | `python .github/scripts/bench_nightly_verdict.py --bench $BENCH --run-id <the same uuid>`; exit 1 and `no verdict: <reason>` when nothing landed |

The stored run id of a Jenkins leg is the orchestrator's uuid, used verbatim by the ingest script (a bare build number
would be read as a GitHub run id and derive a different id). Legs of one run share it; `BENCH_LEG_KEY` keeps each leg's
rows apart and makes a retried leg a no-op. The test suite proves both derivations against the real ingest script.

### Dispatching `bench-on-demand.yml` (Jenkins and by hand)

The workflow is `workflow_dispatch` only, so a fork pull request never starts it and no secret reaches a fork.
The Jenkins orchestrator dispatches one workflow per x86 leg with these inputs:

| Input | Meaning | Validation |
|---|---|---|
| `prebaked_image` | runner-set label of the image the legs run inside (the nightly build's image set); empty uses `image_torch_spyre` and installs the pinned RPMs, a value runs the image as built: no RPM install, no rebuild | `^(image_[a-z0-9_]{1,60})?$` |
| `run_id` | the orchestrator's UUID for the leg; stored verbatim as the run id by the ingest (`--v2-run-id`) and read by the verdict step | lower-case UUID or empty (then derived from the workflow run) |
| `purpose` | `nightly`, `weekly` or `on-demand`; recorded on the rows as `run.purpose` so baseline eligibility can tell scheduled runs from ad-hoc ones | choice |
| `target` | `dev` (default) or `prod` (only from `main`) | choice |
| `mode`, `runner`, `cpu_pinning`, `models`, `engine_ref` | as in the workflow file | choice or `^[A-Za-z0-9._/,-]*$` |

With `prebaked_image` the legs run `make` inside the image's own tree (`/home/senuser/spyre-inference`), so the definitions and targets are those of the nightly commit; results go to the workspace. The ingest action installs the ingest package from `torch-spyre@main`, while the verdict uses the package at `engine_ref` (`main` or a full commit sha for `prod`). The verdict reads the whole run: with a shared `run_id` a leg that wrote nothing is covered by rows other legs landed.

Completion signal: the dispatched run's conclusion. It is `success` only when every leg ran, its rows were ingested and its
verdict was read; a leg whose ingest or verdict failed fails the run with `no verdict: <reason>` in the step summary. The
run id appears in the leg summary (`Run id`), as the `run_id` of the `benchmark_runs` rows, and in `run.purpose`. Jenkins
finds the run through the dispatch response or the `run_id` it passed.

### Datasets

The datasets (agentic traces, the vision archive) are defined once, as `Asset`, `Dataset` and `Store` resources, and
resolved by `.github/scripts/fetch_bench_datasets.py` (stdlib plus PyYAML) from the datasets namespace only. The Store
names the variables that configure it: `ARTIFACTORY_BASE_URL`, `ARTIFACTORY_DATASETS_PATH` (no fallback to another root)
and `ARTIFACTORY_TOKEN`, the CI credential for the datasets namespace. An Asset's annotations
carry the consumer variable (`consumer.env`, `consumer.media_env`) and, for a tarball, the extraction rule. A file is
fetched by digest into `SPYRE_BENCH_DATA_DIR`, verified, and published atomically; archive members that would leave the
extraction directory are refused. `make bench-datasets` (or `fetch_bench_datasets.py status`) shows each dataset's state
and the cases that read it; a selected case whose file is absent fails the run by name (`run_vllm_benchmarks.py`).
To add a dataset, add an Asset (path, sha256, `consumer.env`) and a Dataset; no table or secret changes.

`SPYRE_EE_DATASET` and `SPYRE_RAG_DATASET` (read by the `power_use_case` serve configs) are operator-set: they have no
Dataset, no store path and no cache default, so a host that lacks them fails those configs by name. Add an Asset and a
Dataset for them when their files are published to the datasets namespace.

Model Assets that pin no revision in `.github/cache_config/hf_models_and_datasets.yaml` carry an all-zero placeholder
revision and the annotation `revision.state: unpinned` (the engine schema cannot say "unpinned" with a pinned-revision
integrity). Treat their `model_id` as unpinned: the runner resolves the hub default.
