---
name: vllm-bench-serve
description: Benchmark decoder models with `vllm bench serve` on Spyre, comparing one or more targets (a PR, a commit, the working copy, or the same commit with different env vars) against a baseline (by default the merge base with torch-spyre/spyre-inference main), then write an HTML report with every metric and everything needed to reproduce it. Picks the configs (model, TP, context length, max-num-seqs, compile cache, ...) from what the diff changes, and asks for confirmation of the legs and the configs before running. Use when the user wants to know how a PR / commit / env var affects serving performance on Spyre. For Gemma-4 prefill (TTFT) and decode (ITL) latency measured apart, or a comparison with hf-adapters, use gemma4-benchmark-detailed instead.
user-invocable: true
argument-hint: "[<target>...] [--baseline <ref>] [--env KEY=v1,v2] [--extended]"
---

# Serving benchmark comparison on Spyre

Run the same set of `vllm bench serve` configs on each **leg**, then report them side by side. A leg is one `(spyre-inference ref, torch-spyre rev, env vars)` triple. There are one or more **target** legs and exactly one **baseline** leg.

This skill measures end-to-end serving under load. To isolate prefill time (TTFT) and decode time (ITL) with `vllm bench latency` at batch 1, and to compare spyre-inference with hf-adapters, use [gemma4-benchmark-detailed](../gemma4-benchmark-detailed/SKILL.md) instead.

## Hard constraints (do not violate)

- **Single accelerator, sequential only.** Never run two Spyre-backed commands at once. Legs run strictly one after another.
- **Targets first, baseline last.** If a target leg fails, still run the other targets. Run the baseline unless every target failed. Report each failure.
- **Identical configs and harness for every leg.** One `serve-tests.yaml` and one harness checkout serve every leg; only the installed package code (and the env vars the user asked to vary) changes between legs.
- **Two confirmation gates.** Confirm the legs, wait for the user's answer. Then confirm the configs, wait again. Do not start any run before both are confirmed.
- **Non-destructive.** Committed refs are benchmarked from throwaway `git worktree`s, synced into the harness `.venv`. Never `git checkout` / `git stash` in the user's working copy. `run_leg.sh` restores the install after each leg, even on failure, by syncing the harness checkout's lock, so run from a harness checkout on the branch whose install should survive. A hand-installed package (e.g. a local torch-spyre build) is not restored: `run_leg.sh` prints a `WARNING` with the diff of `uv pip freeze`; tell the user.
- **Always report, never invent.** Every number comes from a result JSON or log. A failed leg is reported with its error, not a number.

## Inputs

- **Targets** (positional, optional): any mix of PR numbers (`#1043` / `1043`), git refs/shas, or `.` (the working copy, including uncommitted changes). Default: `HEAD` of the current branch.
- **`--baseline <ref>`**: default the merge base of the first target with upstream main (step 1).
- **`--env KEY=v1,v2`**: compare env var values instead of (or in addition to) commits. With a single ref, each value is one leg; the first value listed is the target, the last is the baseline. E.g. `--env SPYRE_BATCHED_DECODE=1,0`.
- **`--extended`**: 1000 prompts instead of 200, and add configs where more than one is relevant (e.g. sweep max-num-seqs 4/8/16).
- The user can override any config axis in plain words ("at tp1", "with 32k context", ...).

## Steps

### 0. Environment

[scripts/run_leg.sh](scripts/run_leg.sh) sources `${SPYRE_LIBS:-~/spyre-libs}/env.sh` (when it exists) and `<spyre-inference>/.venv/bin/activate`, and sets `PYTHONSAFEPATH=1` so the harness checkout's `spyre_inference/` cannot shadow the leg's installed tree. Steps 1 to 3 need only git and `gh`.

`uv sync --frozen` installs torch-spyre into `.venv` from the git rev pinned in `pyproject.toml` (built once, then reused from uv's wheel cache keyed by that rev), so each leg's sync installs that leg's rev.

### 1. Resolve the legs

1. Fetch by URL, since the checkout may have no `upstream` remote: `git fetch https://github.com/torch-spyre/spyre-inference main` and keep `FETCH_HEAD`'s sha as `<main>`. For each PR number: `git fetch https://github.com/torch-spyre/spyre-inference pull/<N>/head` and use `FETCH_HEAD`'s sha.
2. Default target is `HEAD`. Look up its PR: `gh pr list --repo torch-spyre/spyre-inference --state all --search <sha> --json number,title,url,headRefOid`. If there is none, or more than one, ask the user. Refer to the PR number everywhere in the report.
3. Default baseline is `git merge-base <target> <main>`.
4. For each leg, read the torch-spyre rev it pins (`torch-spyre = { git = ..., rev = "<sha>" }` in that ref's `pyproject.toml`) and the vLLM rev. Also diff `spyre-rpms.lock` between legs.
   - **torch-spyre differs between legs**: each leg installs its own rev (step 4). The first sync of a rev not yet in uv's cache builds it. Note each rev in the report.
   - **vLLM rev or `spyre-rpms.lock` differs**: stop and ask the user. Rebuilding vLLM or the RPM tree (`scripts/install-pinned-rpms.sh --rebuild`) is out of scope for an unattended run.
5. **Gate 1**: show the legs in run order, one line each, then wait for the user:

```text
1. 2a11d3b  PR #1043 "<title>"   torch-spyre 8744437   (target)
2. 022255a  main (merge base)    torch-spyre 8744437   (baseline)
```

### 2. Pick the configs

Read the diff between baseline and target(s) (`git diff --stat`, then the relevant hunks). Choose the configs that exercise what the change touches. Defaults in **bold**:

| axis | choice |
|---|---|
| model | **`google/gemma-4-26B-A4B`**; another decoder from `vllm-benchmarks/benchmarks/spyre/serve-tests.yaml` when the change targets it: `ibm-granite/granite-3.3-8b-instruct`, `ibm-granite/granite-4.1-8b`, or an FP8 model (`ibm-granite/granite-3.3-8b-instruct-FP8`, `ibm-granite/granite-4.1-8b-fp8`) |
| tensor-parallel-size | **4**; 1 when the change targets single-card execution, or both if the change touches collectives |
| max-model-len + dataset | 4096 + `${SPYRE_AIOPS_DATASET}` for short-context changes; **8192 + `${SPYRE_CICS_DATASET}`**; 32768 + `${SPYRE_ALL_SEQUENCES_DATASET}` for long-context changes |
| max-num-seqs | **4**; another power of two (8, 16, 32) if the change targets batching. Keep `max-concurrency` equal to it |
| num-prompts | **200**; 1000 with `--extended` |
| compile cache | **`SPYRE_KERNEL_CACHE=1`**, default cache path; all caches off on every leg when the change can affect compile time (more graphs, new buckets, new compiled regions, torch-spyre codegen), so the report shows the warmup cost: `SPYRE_KERNEL_CACHE=0`, which makes `run_leg.sh` also set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` (inductor FX graph, AOTAutograd and autotune caches) and `VLLM_DISABLE_COMPILE_CACHE=1`. `SPYRE_KERNEL_CACHE=0` alone still lets the inductor and vLLM caches hit and shrink compile time |
| other | any env var or server flag the change introduces or reads, e.g. a new `SPYRE_*` knob |

The traces need a `max-model-len` that fits them: aiops at 4k, cics at 8k, all_sequences at 32k.

**Gate 2**: show one line per config, then wait for the user:

```text
serve_gemma4-26b-a4b_tp4_cics8k_bs4: gemma-4-26B-A4B, TP4, max-model-len 8192, max-num-seqs 4, concurrency 4, 200 prompts, cics, SPYRE_KERNEL_CACHE=1
```

### 3. Write the bench directory

Create `<spyre-inference>/../bench-<tag>/` (e.g. `bench-pr1043/`), outside the git tree so it survives worktree swaps. Write `serve-tests.yaml` there in the same `defaults`/`tests` format as `vllm-benchmarks/benchmarks/spyre/serve-tests.yaml`: copy its `defaults` block, then one entry per confirmed config. Give TP4 and 32k entries `server_health_timeout: 3600`, and double it when the caches are off. `test_name` must be unique per config; the leg is encoded in the results directory, not the name.

### 4. Run each leg (targets first)

Run [scripts/run_leg.sh](scripts/run_leg.sh) (`<skill>` is `<spyre-inference>/.claude/skills/vllm-bench-serve`) once per leg, strictly one after another, in the background (serve legs take from tens of minutes to hours), and wait for its completion instead of polling:

```bash
<skill>/scripts/run_leg.sh --bench <bench> --leg pr1043 --ref 2a11d3b --label "PR #1043" --role target [--env KEY=VAL]...
<skill>/scripts/run_leg.sh --bench <bench> --leg main --ref 022255a --label main --role baseline
```

`--ref .` benchmarks the working copy, uncommitted changes included. `SPYRE_KERNEL_CACHE=1` is set unless an `--env` gives it. With `SPYRE_KERNEL_CACHE=0`, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` and `VLLM_DISABLE_COMPILE_CACHE=1` are also set unless an `--env` gives them. For each leg the script:

1. Checks the ref out into a throwaway `git worktree` (`.`: the working copy itself) and runs `UV_PROJECT_ENVIRONMENT=<spyre-inference>/.venv uv sync --frozen --project <worktree>`.
2. Fails the leg unless `spyre_inference` imports from the worktree and the installed torch-spyre rev is the one that ref pins. This import check is the only guarantee that the leg runs its own code: the server inherits the same `.venv` and `PYTHONSAFEPATH=1`.
3. Exports the leg's env vars and writes `<bench>/<leg>/env.txt` (the `SPYRE|VLLM|TORCH|...` vars) and `<bench>/<leg>/leg.json`: label, role, sha, worktree path, torch-spyre, torch and vLLM versions, the RPMs and where they were read from, the `--env` overrides, host and date. The RPMs come from `components.txt` of the installed tree (`$SENTIENT_BASE_INSTALL_DIR`, or `$LIB_VERSION_FILE` when it lies in that tree); when the tree has none, from the versions `spyre-rpms.lock` requests for this arch. `env.txt` is the env before `make perf-tests`: the harness then re-sources `ibm-aiu-setup.sh` and applies each config's `environment_variables` (`SPYRE_DEVICES` and `AIU_WORLD_SIZE` included), so the per-config `env` in `summary.json` and the `.cmd` files are what the server ran with.
4. Runs `make perf-tests BENCH_TYPES=serve BENCH_CONFIGS_DIR=<bench> RESULTS_DIR=<bench>/<leg>/results` from the harness checkout, logging to `<bench>/<leg>/run.log`.
5. Gates the leg with `collect.py validate <bench> <leg>`: every config needs a result JSON with `completed == num_prompts` and `failed == 0` (`vllm bench serve` exits 0 and prints a complete-looking table even when most requests fail), and its `_server.log` must reach `Application startup complete`.
6. Always restores the install (`uv sync --frozen` in `<spyre-inference>`) and removes the worktree, even on failure or interrupt, and warns when `uv pip freeze` differs from before the leg. An interrupt (`INT` / `TERM`) stops the running config and its server.

Exit 0 means the leg is valid, 1 that the run or the gate failed, and 2 a usage or setup failure. If a target leg fails, keep its error for the report and go on to the next leg. If every target failed, skip the baseline.

### 5. Collect

```bash
python3 <skill>/scripts/collect.py summarize <bench> --legs pr1043,main
```

List the targets in run order and the baseline last. A leg that did not run is reported as pending. The script writes:

- `<bench>/summary.json`, with per leg and config:
    - every metric of the result JSON;
    - **server startup**: `server_startup_sec` of the result JSON, timed by the harness from the server spawn to the first healthy `/health`; from the `_server.log` timestamps only when the JSON has none;
    - **init and compile times** from `_server.log`, as the max across TP workers (each one logs its own, as vLLM does for `compilation`):
        - `init engine (...) took <s> s (compilation: <s> s)`, from vLLM `core.py`. Compilation is the wall time of the whole Spyre warmup (`SpyreWorker.compile_or_warm_up_model`), covering the two below;
        - `Warmup complete in <s>s for <n> buckets.`, from `SpyreModelRunner.warming_up_model`;
        - `Attention graph recording complete: <n> graphs in <s>s.`, from `SpyreModelRunner._record_attention_graphs`, with the per-layer `<p> per-seq + <d> batched-decode` split;
    - whether any compile cache was on (`compile_cache`: false only with `SPYRE_KERNEL_CACHE=0`, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` and `VLLM_DISABLE_COMPILE_CACHE=1`), the server env (`env`: `env.txt` overridden by the config's `environment_variables`), the validity errors, the ITL mean/median ratio (`compile_leak` when it is above 1.1, which points to compilation or a stall inside the measured window), and on failure the first `Error` / `Traceback` lines and the last 50 log lines;
    - the Δ of every row, target vs baseline, with its colour class.
- `<bench>/results-table.html`: the results table of step 6, with the rules applied: one column group per config, baseline then each target with its Δ, rows missing from every log left out, and timing rows marked `unreliable` when any compile cache is on.
- `<bench>/leg-validity.html`: the leg validity cards, plus an alert note per failed leg.

Read the verbatim `Serving Benchmark Result` block from `<bench>/<leg>/results/<test_name>_bench.log`, and the exact commands from `<test_name>_server.cmd` / `<test_name>_bench.cmd`.

### 6. Report

Follow [report-example.html](report-example.html): same CSS, same section order, same level of detail. Its PR and numbers are illustrative, never copy them. Save it as `bench-<tag>-<YYYY-MM-DD>.html` in the bench directory. Tell the user the path.

- **TL;DR** (header standfirst): 1-3 sentences. Say what is benchmarked and against what, and give the main observations, e.g. "PR #XXX optimizes decode by batching the per-sequence KV gathers into one kernel. Output throughput +30%, mean ITL −50%, TTFT flat, but server startup doubles (caches off)."
- **Headline tiles**: output throughput, mean ITL, mean TTFT and server startup, as the target-vs-baseline ratio or percentage, from `deltas` in `summary.json`. Use `tile flat` within ±2%, `tile warn` for 2–10% worse, `tile bad` for more than 10% worse.
- **What was compared**: one step per leg in run order: sha, label, what the leg is, env overrides, and its torch-spyre rev when it differs (all in `leg.json`).
- **Results table**: paste `results-table.html` as is, then the explanatory paragraph of the example. It holds every row of the vllm bench result block, plus server startup and the "Startup and compilation" band. Its Δ is `(target − baseline) / baseline`, coloured by whether it is better for that metric (lower is better for latencies and duration, higher for throughputs): `d-good` for better by ≥2%, no class within ±2%, `d-warn` for 2–10% worse, `d-bad` for more than 10% worse. Count rows get a Δ only when they differ. Unless all compile caches are off, server startup and the time rows are marked `unreliable` with the inline `caveat` "unreliable: compile cache on", since the cache hit rate is unknown.
- **Leg validity**: paste `leg-validity.html` (one card per leg and config, an alert note per failed leg, pending cards for legs that did not run), then the caveats note.
- **Environment**: date, host, spyre-inference and torch-spyre sha per leg, torch, vLLM, the RPMs and their source (all in `leg.json`), and the env vars (per leg and config where they differ), from `env` in `summary.json`, not `env.txt`. Show only the spyre-inference vars, i.e. the `SPYRE_*` keys of `environment_variables` in `spyre_inference/envs.py` (`SPYRE_KERNEL_CACHE` included), plus `TORCHINDUCTOR_FORCE_DISABLE_CACHES`, `VLLM_DISABLE_COMPILE_CACHE` and any `--env` var. Leave the other vars out of the report.
- **Reproduction**: the `serve-tests.yaml`, the exact server and bench commands from the `.cmd` files, and the `make perf-tests` line.
- Caveats only where they apply: n=1 per leg, different torch-spyre builds, a compile-leak flag, retried legs.
