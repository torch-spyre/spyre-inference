---
name: vllm-bench-serve-compare
description: Benchmark decoder models with `vllm bench serve` on Spyre, comparing one or more targets (a PR, a commit, the working copy, or the same commit with different env vars) against a baseline (by default the merge base with upstream/main), then write an HTML report with every metric and everything needed to reproduce it. Picks the configs (model, TP, context length, max-num-seqs, compile cache, ...) from what the diff changes, and asks for confirmation of the legs and the configs before running. Use when the user wants to know how a PR / commit / env var affects serving performance on Spyre. For Gemma-4 prefill (TTFT) and decode (ITL) latency measured apart, or a comparison with hf-adapters, use gemma4-benchmark-detailed instead.
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
- **Non-destructive.** Committed refs are benchmarked from throwaway `git worktree`s, synced into the harness `.venv`. Never `git checkout` / `git stash` in the user's working copy. Always restore the install at the end, even on failure.
- **Always report, never invent.** Every number comes from a result JSON or log. A failed leg is reported with its error, not a number.

## Inputs

- **Targets** (positional, optional): any mix of PR numbers (`#1043` / `1043`), git refs/shas, or `.` (the working copy, including uncommitted changes). Default: `HEAD` of the current branch.
- **`--baseline <ref>`**: default `git merge-base <first target> upstream/main` (after `git fetch upstream main`).
- **`--env KEY=v1,v2`**: compare env var values instead of (or in addition to) commits. With a single ref, each value is one leg; the first value listed is the target, the last is the baseline. E.g. `--env SPYRE_BATCHED_DECODE=1,0`.
- **`--extended`**: 1000 prompts instead of 200, and add configs where more than one is relevant (e.g. sweep max-num-seqs 4/8/16).
- The user can override any config axis in plain words ("at tp1", "with 32k context", ...).

## Steps

### 0. Environment

Run every command in one shell that sourced both, in order:

```bash
# Exists only if scripts/install-pinned-rpms.sh was run on this pod.
[ -f "$HOME/spyre-libs/env.sh" ] && source "$HOME/spyre-libs/env.sh"
source <spyre-inference>/.venv/bin/activate
# The server runs `python -m` from the harness checkout, which would put that
# checkout's spyre_inference/ ahead of the leg's installed tree on sys.path.
export PYTHONSAFEPATH=1
```

`uv sync --frozen` installs torch-spyre into `.venv` from the git rev pinned in `pyproject.toml` (built once, then reused from uv's wheel cache keyed by that rev), so each leg's sync installs that leg's rev.

### 1. Resolve the legs

1. `git fetch upstream main`. For each PR number: `git fetch upstream pull/<N>/head` and use `FETCH_HEAD`'s sha.
2. Default target is `HEAD`. Look up its PR: `gh pr list --repo torch-spyre/spyre-inference --state all --search <sha> --json number,title,url,headRefOid`. If there is none, or more than one, ask the user. Refer to the PR number everywhere in the report.
3. Default baseline is `git merge-base <target> upstream/main`.
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
| compile cache | **`SPYRE_KERNEL_CACHE=1`**, default cache path; `SPYRE_KERNEL_CACHE=0` on every leg when the change can affect compile time (more graphs, new buckets, new compiled regions, torch-spyre codegen), so the report shows the warmup cost |
| other | any env var or server flag the change introduces or reads, e.g. a new `SPYRE_*` knob |

The traces need a `max-model-len` that fits them: aiops at 4k, cics at 8k, all_sequences at 32k. `python3 .github/scripts/fetch_bench_datasets.py env` lists every available trace (truncated all_sequences variants included).

**Gate 2**: show one line per config, then wait for the user:

```text
serve_gemma4-26b-a4b_tp4_cics8k_bs4: gemma-4-26B-A4B, TP4, max-model-len 8192, max-num-seqs 4, concurrency 4, 200 prompts, cics, SPYRE_KERNEL_CACHE=1
```

### 3. Write the bench directory

Create `<spyre-inference>/../bench-<tag>/` (e.g. `bench-pr1043/`), outside the git tree so it survives worktree swaps. Write `serve-tests.yaml` there in the same `defaults`/`tests` format as `vllm-benchmarks/benchmarks/spyre/serve-tests.yaml`: copy its `defaults` block, then one entry per confirmed config. Give TP4 and 32k entries `server_health_timeout: 3600`, and double it when the kernel cache is off. `test_name` must be unique per config; the leg is encoded in the results directory, not the name.

### 4. Run each leg (targets first)

For each leg, in order:

1. **Code.** For `.`, use the working copy: `SRC=<spyre-inference>`. Otherwise: `SRC=$(mktemp -d); git worktree add --detach "$SRC" <sha>`. Then sync that tree into the harness `.venv`, which installs `spyre_inference` editable from `$SRC` and the torch-spyre rev that ref pins:

   ```bash
   UV_PROJECT_ENVIRONMENT=<spyre-inference>/.venv uv sync --frozen --project "$SRC"
   ```

   Verify, from `<spyre-inference>` with `PYTHONSAFEPATH=1` set, that `python -c "import spyre_inference; print(spyre_inference.__file__)"` prints a path under `$SRC`, and check the torch-spyre rev from `python -c "import importlib.metadata as m; print(m.distribution('torch-spyre').read_text('direct_url.json'))"`.
2. **Env.** Export the leg's env vars (the compile cache setting plus any `--env` value). Record the full env, for debugging: `env | grep -E '^(SPYRE|VLLM|TORCH|TORCHINDUCTOR|AIU|FLEX|DT|COLL|SENTIENT|OMP)_' | sort > <bench>/<leg>/env.txt`.
3. **Versions.** Record in `<bench>/<leg>/versions.txt`: the spyre-inference sha, the torch-spyre sha, `torch.__version__`, `vllm.__version__`, and the RPMs. If `~/spyre-libs` exists, use the `[packages]` block of the `spyre-rpms.lock` it was installed from. Otherwise use `/opt/ibm/spyre/components.txt`.
4. **Run**, from the harness checkout, in the background (serve legs take from tens of minutes to hours), and wait for completion instead of polling:

   ```bash
   cd <spyre-inference> && make perf-tests BENCH_TYPES=serve \
     BENCH_CONFIGS_DIR=<bench> RESULTS_DIR=<bench>/<leg>/results 2>&1 | tee <bench>/<leg>/run.log
   ```

   Do not set `MODELS` / `TPS`: the directory only holds the confirmed configs.
5. **Gate the leg.** The leg is valid only if every `spyre_inference` path in `<test_name>_server.log` (tracebacks, warnings) is under `$SRC`, and every config's result JSON has `completed == num-prompts` and `failed == 0`: `vllm bench serve` exits 0 and prints a complete-looking table even when most requests fail. If a target leg fails, keep its error for the report and go on to the next leg. If every target failed, skip the baseline.

### 5. Restore

Always, even on failure: `uv sync --frozen` in `<spyre-inference>` and `git worktree remove --force "$SRC"` for each worktree.

### 6. Collect

Per leg and config, from `<bench>/<leg>/results/`:

- `<test_name>.json`: every metric of the result table.
- `<test_name>_bench.log`: the printed `Serving Benchmark Result` block, copied verbatim into the report.
- `<test_name>_server.cmd` / `<test_name>_bench.cmd`: the exact server and bench commands.
- **Server startup**: the time from the first timestamp in `<test_name>_server.log` to the last timestamp before `Application startup complete`:

  ```bash
  python3 - <test_name>_server.log <<'EOF'
  import re, sys
  from datetime import datetime
  ts, first = None, None
  for line in open(sys.argv[1], errors="replace"):
      m = re.search(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b", line)
      if m:
          ts = datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
          first = first or ts
      if "Application startup complete" in line:
          print(f"{(ts - first).total_seconds():.0f} s"); break
  else:
      print("server never became ready")
  EOF
  ```

- **Init and compile times**, from `<test_name>_server.log`. Report each line that is present; leave out the lines that are missing (e.g. no attention line with `SPYRE_ATTN_RECORD=0` or `--enforce-eager`). Every TP worker logs its own times, so take the max across workers, as vLLM does for `compilation`. The counts are the same on every worker.
    - **Init engine**: `init engine (profile, create kv cache, warmup model) took <s> s`, from vLLM `core.py`.
    - **Total compilation**: `(compilation: <s> s)` on the same line. This is the wall time of the whole Spyre warmup (`SpyreWorker.compile_or_warm_up_model`), not just compile time. It covers the two lines below.
    - **Model graph warmup**: `Warmup complete in <s>s for <n> buckets.`, from `SpyreModelRunner.warming_up_model`. It covers the dummy runs for each token bucket, plus the sampler row widths.
    - **Attention graph recording**: `Attention graph recording complete: <n> graphs in <s>s.`, from `SpyreModelRunner._record_attention_graphs`. The per-layer split comes from `Recording <p> per-seq + <d> batched-decode attention variants for layer...`, which is logged once per layer: `<n> = (<p> + <d>) × layers`.

  ```bash
  python3 - <test_name>_server.log <<'EOF'
  import re, sys
  log = open(sys.argv[1], errors="replace").read()
  def max_of(pat, group):
      vals = [float(m.group(group)) for m in re.finditer(pat, log)]
      return max(vals) if vals else None
  init = r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s(?: \(compilation: ([\d.]+) s)?"
  warm = r"Warmup complete in ([\d.]+)s for (\d+) buckets"
  attn = r"Attention graph recording complete: (\d+) graphs in ([\d.]+)s"
  split = sorted(set(re.findall(r"Recording (\d+) per-seq \+ (\d+) batched-decode attention variants", log)))
  print("init_engine_s", max_of(init, 1))
  print("compilation_s", max_of(init, 2))
  print("model_warmup_s", max_of(warm, 1), "buckets", max_of(warm, 2))
  print("attn_record_s", max_of(attn, 2), "graphs", max_of(attn, 1), "per_seq+batched_decode per layer", split)
  EOF
  ```

- **Compile-leak check**: `mean_itl_ms / median_itl_ms > ~1.1` points to compilation or a stall inside the measured window. Flag it in the report rather than quoting the means as a clean result.
- On failure: the last ~50 lines of the relevant `_server.log` / `_bench.log`, and the first `Error` / `Traceback` lines.

### 7. Report

Follow [report-example.html](report-example.html): same CSS, same section order, same level of detail. Its PR and numbers are illustrative, never copy them. Save it as `bench-<tag>-<YYYY-MM-DD>.html` in the bench directory. Tell the user the path.

- **TL;DR** (header standfirst): 1-3 sentences. Say what is benchmarked and against what, and give the main observations, e.g. "PR #XXX optimizes decode by batching the per-sequence KV gathers into one kernel. Output throughput +30%, mean ITL −50%, TTFT flat, but server startup doubles (cache off)."
- **Headline tiles**: output throughput, mean ITL, mean TTFT and server startup, as the target-vs-baseline ratio or percentage. Use `tile flat` within ±2%, `tile warn` for 2–10% worse, `tile bad` for more than 10% worse.
- **What was compared**: one step per leg in run order: sha, label, what the leg is, env overrides, and its torch-spyre rev when it differs.
- **Results table**: every row of the vllm bench result block, plus server startup, with one column per leg and one column group per config, and a Δ column `(target − baseline) / baseline`. Colour each Δ by whether it is better for that metric (lower is better for latencies and duration, higher for throughputs): `d-good` for better by ≥2%, no class within ±2%, `d-warn` for 2–10% worse, `d-bad` for more than 10% worse. When `SPYRE_KERNEL_CACHE=1`, mark the server startup row `unreliable` with the inline `caveat` "unreliable: kernel cache on", since the cache hit rate is unknown. Below server startup, add a "Startup and compilation" band with the init and compile times from step 6: init engine, total compilation, model graph warmup (with its bucket count) and attention graph recording (with its graph count and the per-seq + batched-decode split per layer). Leave out rows whose line is not in the log. The time rows follow the same `unreliable` rule as server startup; the count rows do not, and they get a Δ only when they differ.
- **Leg validity**: one card per leg and config, giving completed/num-prompts and the ITL mean/median ratio. Give each failed leg an alert note with its error, and mark legs that were not run as pending.
- **Environment**: date, host, spyre-inference and torch-spyre sha per leg, torch, vLLM, the RPMs and their source, and the env vars (per leg where they differ). Show only the spyre-inference vars from `env.txt`, i.e. the `SPYRE_*` keys of `environment_variables` in `spyre_inference/envs.py` (`SPYRE_KERNEL_CACHE` included), plus any `--env` var. Leave the rest of `env.txt` out of the report.
- **Reproduction**: the `serve-tests.yaml`, the exact server and bench commands from the `.cmd` files, and the `make perf-tests` line.
- Caveats only where they apply: n=1 per leg, different torch-spyre builds, a compile-leak flag, retried legs.
