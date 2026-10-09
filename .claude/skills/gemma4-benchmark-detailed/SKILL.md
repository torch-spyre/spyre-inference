---
name: gemma4-benchmark
description: "Benchmark Gemma-4 26B-A4B on Spyre: hf-adapters (pinned PR #620, 9b075e4) against spyre-inference main and the spyre-inference branch under test, on the already installed environment. Two workloads on the same 1984-token prompt, each at whichever of TP1, TP2 and TP4 the user picks: the prefill TTFT of spyre-inference#1102 (the maintainer's ttft.py vs `vllm bench latency`) and decode ITL as the granite TP4 ITL report measured it ((lat64 - lat1) / 63). Every arm runs at the same TP (vLLM --tensor-parallel-size, hf-adapters under torchrun). Checks the environment read-only (pins, profiler-free build, consistent Spyre libraries, env knobs, a stale main, enough cards) and flags every inconsistency to the user instead of fixing it; never installs, syncs, rebuilds, fetches or checks anything out. Runs the arms one at a time and ends with a shareable Markdown report, TP scaling included. Use when asked to benchmark or reproduce Gemma-4 / gemma-4-26B-A4B TTFT, prefill or decode latency or ITL, at one or several tensor-parallel sizes, reproduce #1102, compare spyre-inference with hf-adapters, or measure a branch or an MoE change against main."
---

# Gemma-4 benchmark: hf-adapters vs spyre-inference main vs the branch under test

Two workloads on the same 1984-token prompt, batch 1, each at TP1, TP2 or TP4:

- **TTFT**: tdoublep's measurement from
  [spyre-inference#1102](https://github.com/torch-spyre/spyre-inference/issues/1102#issuecomment-6001115830),
  which ran it at TP1. When the issue was filed, vLLM `main` was about 1.5x slower than
  hf-adapters.
- **ITL** (`--itl`): decode latency the way the granite-3.3-8b TP4 ITL report (2026-10-02)
  measured it.

Three arms, alternating: **hf** (hf-adapters), **main** (spyre-inference at `--main-ref`),
**branch** (the spyre-inference checkout installed in the environment). At TP N every arm
runs on N cards, so every comparison is at equal card count. Every run ends with a
`report.md` meant to be shared as is.

## The one rule: never alter the environment

This skill measures the environment it is given, and changes nothing in it. Do not run
`uv sync`, `uv pip install`, `pip`, `git checkout`/`switch`/`fetch` in a checkout, a
rebuild, or an install script, not even into a fresh scratch venv to "match the recipe".
When the check flags something, report it to the user and stop. Changing the environment
is the user's call, made with their own tooling.

The main arm follows the same rule: `git archive` exports the main commit into a cache dir
(`$GEMMA4_BENCH_CACHE`, default `~/.cache/gemma4-benchmark/spyre-inference-<sha12>`), and
that export goes first on `PYTHONPATH`. The checkout, its refs and the environment stay as
they are; a gate proves `spyre_inference` imports from the export before anything runs. A
stale `--main-ref` is a WARN for the user to fetch, never a fetch.

## Quick start

```bash
SKILL=<this skill dir>
source <venv>/bin/activate                      # the environment to benchmark
$SKILL/scripts/run_suite.sh --check --ttft-tps "1 2" --itl-tps "1 2"    # read-only; exit 0 / 1 / 3
$SKILL/scripts/run_suite.sh --ttft-tps "1 2" --itl-tps "1 2" --repeat 2 --recompiles
```

`--ttft-tps` and `--itl-tps` each take any of `1 2 4`; `""` skips that workload; both
default to `1`. One workload at one size: `run_1102.sh [--itl] [--tp N]`, same options.
On dt-inductor pods: `--env-script /scratch/virtualenv/<env>/bin/activate`, or
`--python /scratch/virtualenv/<env>/bin/python`.

## Instructions

1. **Identify the environment.** Ask the user which venv or interpreter to benchmark if it
   is not obvious; never pick one by installing. Defaults: `$VIRTUAL_ENV`, then
   `$UV_PROJECT_ENVIRONMENT`. `--env-script` sources the host's runtime environment, which
   only sets variables in the script's own shell. `--hf-python` runs the hf arm in a
   different interpreter (the issue used a separate env for it). The branch under test is
   whatever spyre-inference checkout that environment has installed.
2. **Pick the combination.** Unless the user already said, ask which workloads to run at
   which TP sizes: TTFT and ITL each at any subset of TP1, TP2, TP4. TP N needs N cards.
   Tell them the run count: with all three arms, each TTFT size is 3 runs per `--repeat`,
   each ITL size 6.
3. **Check:** `run_suite.sh --check [flags]`. It checks every combination and prints every
   finding and the exact commands of each. Exit codes, worst over all combinations: 0 clean,
   1 ERROR, 3 WARN. Run it while the cards are idle: it imports torch_spyre.
4. **Flag the findings to the user, verbatim.**
   - **ERROR:** stop. Explain what is wrong and what the user would need to change, e.g.
     "torch-spyre links libaiupti: rebuild it with the profiler off", or "--tp 4 needs 4
     cards, this host has 2: drop TP4 from the list". Do not change it.
   - **WARN:** list each one and ask whether to benchmark anyway. Only after a yes, re-run
     with `--accept-warnings`. The accepted WARNs are printed in the report.
   - **INFO:** mention them in the report; no question needed.
5. **Run** (after step 4) in the background, polling its output. Nothing else may use the
   cards meanwhile. The suite re-checks every combination first and starts none if one
   would be refused.
   - A vLLM run takes two to three times as long as an hf run, about half of it engine
     init; much of an hf run is its first warmup call, which compiles.
   - `--repeat 3` for any decision; 2 shows the spread, 1 is indicative only.
6. **Gate, then share.** From `report.md` (the suite's, or one workload's):
   - a `FAILED` row: report the failure with the tail of that run's `run.log`, no number;
   - `LEAK` (max/min > 1.10): quote the p50 only and say the mean is compile-contaminated;
   - with `--recompiles`: a nonzero post-warmup recompile count means warmup missed a shape;
   - "lacks N commit(s) of main": main vs branch also measures those commits;
   - TP scaling compares sizes that ran one after another: fine for scaling, not for small
     differences.

   Hand the user the report's path; it already holds the headline, TP scaling, method,
   commands, raw results, environment and accepted findings.

## Workloads

| | TTFT (default) | ITL (`--itl`) |
|---|---|---|
| source | #1102, verbatim at TP1 | granite TP4 ITL report |
| output lengths | 1 | 1 and 64 (separate runs, alternating) |
| vLLM | `vllm bench latency ... --num-iters-warmup 2 --num-iters 3` | the same with `--num-iters 10`, under `OMP_WAIT_POLICY=PASSIVE SPYRE_NUM_CPUS=8` |
| hf | TP1: `ttft.py` (2 warmups, 5 timed calls); above: `scripts/hf_latency.py --output-len 1`, same counts | `scripts/hf_latency.py`, 2 warmups, 10 timed calls |
| metric | vLLM average latency vs hf's median | ITL = (lat64 - lat1) / 63 from averages (headline) and p50s; TTFT = lat1 |
| over replicates | median | median |

`hf_latency.py` is `ttft.py` generalized: the same timing and threads, `min_new_tokens` equal
to the output length (vLLM's bench ignores EOS), a seeded prompt (identical on every rank),
`tp_plan="auto"` under torchrun, and hf-adapters' own per-token timing as an in-run
cross-check. That timing only adds host timestamps to hf-adapters' generate loop, which
synchronizes every step anyway. hf sizes its KV cache to 2048 for any output length up to
64, and vLLM preallocates its own, so lat1 and lat64 run the same prefill graph on both
sides.

At batch 1 every decode step has one token. A branch that traces only stick-aligned MoE
batches (spyre-inference `SpyreMoERunner`) runs decode through the opaque op, like main.

## Tensor parallelism

`--tp N` (or one entry of a suite's TP list) sets the size of every arm of that workload:

- **vLLM** (main and branch): `--tensor-parallel-size N` on the command, omitted at TP1 as
  in the issue. spyre-inference supports TP ≥ 1 in fp16 with a native `all_reduce`.
- **hf-adapters**: `hf_latency.py` under `python -m torch.distributed.run --nproc-per-node N`,
  loading with `tp_plan="auto"` like hf-adapters' `scripts/run_multicard_smoke.py`. Gemma-4
  TP is hf-adapters #522, which the pinned `9b075e4` contains. Rank 0 writes the results.
- **Cards:** `$SPYRE_DEVICES` if set, else the numbered IOMMU-group nodes under `/dev/vfio`.
  TP above that is an ERROR at the check, and nothing runs. hf ranks get the first N pinned
  devices, else `SPYRE_DEVICES=0,…,N-1`.
- **Threads:** spyre-inference splits its CPU budget across its N workers:
  `ceil(SPYRE_NUM_CPUS / N)` each, or the physical cores / N when unset (the TTFT recipe).
  hf ranks pin 8 threads each. The report lists every vLLM run's threads × workers.
- **Scaling:** the suite report puts each arm's summary at every TP next to the same arm's
  TP1, with speedup and parallel efficiency. Include 1 in a list to get it.

## What the issue pins, and how the skill uses each pin

| issue recipe | in this skill |
|---|---|
| hf-adapters PR #620 at `9b075e46bc9689671fb7ba546ef12b3e509bbdc5` | checked; anything else is a WARN |
| hf-adapters' torch-spyre `e9d31328345f55ead94d1a65736380e6e07513bc` `[cpsat]` | checked; anything else is a WARN (it says whether the env's build contains `e9d3`) |
| `ttft.py` from the comment | [`scripts/ttft.py`](scripts/ttft.py): the comment's code (sha1 `58c92665e1258821878142ea8dcec1d803ec6b29`) plus this repository's license header and formatting; same syntax tree |
| `OMP_NUM_THREADS=8 uv run --no-sync python ttft.py --model <model> --input-len 1984` | the same command, with the env's interpreter in place of `uv run --no-sync python` |
| `uv run --no-sync vllm bench latency --model <model> --input-len 1984 --output-len 1 --batch-size 1 --num-iters-warmup 2 --num-iters 3 --max-model-len 2048 --max-num-seqs 1` | the same command via the env's `vllm` entry point, plus `--output-json`; the branch arm runs it from the checkout, main from its export |
| compare vLLM avg latency against `ttft.py`'s median TTFT | the TTFT report's headline ratio |

The issue's setup steps (`git clone`, `uv sync`, `uv pip install torch-spyre...`) describe
how the maintainer provisioned their host. They are **not** executed here.

## What `--check` looks at (read-only)

`scripts/check_env.py` imports only torch and torch_spyre. It locates every other package
without importing it, and reads checkouts with `git rev-parse` / `status` / `log` /
`merge-base --is-ancestor`, plus one `git ls-remote` of GitHub's main. It writes no bytecode
(`python -B`), and neither do the benchmark runs.

| finding | level |
|---|---|
| torch / torch_spyre do not import; `_C.so` has unresolved libraries | ERROR |
| `_C.so` links `libaiupti` (profiler build) | ERROR (WARN with `--allow-profiler`; numbers labelled inflated) |
| hf-adapters or vllm / spyre-inference not installed; model `config.json` unreadable | ERROR |
| main arm: no checkout, no remote pointing at `torch-spyre/spyre-inference`, or `--main-ref` does not resolve | ERROR |
| `--tp` above the host's cards (`$SPYRE_DEVICES`, else `/dev/vfio`) | ERROR |
| hf-adapters not at `9b075e4`; torch-spyre not `e9d3` | WARN |
| `--main-ref` differs from GitHub's main (stale); the branch lacks commits of main | WARN |
| `_C.so` built from another commit than the torch-spyre checkout's HEAD | WARN |
| modified tracked files in a checkout; unmet `transformers==5.15.0` / `torch~=2.13.0` / vllm requirement | WARN |
| Spyre libraries resolving from more than one install tree (e.g. in-tree `sentient/` plus `/opt/ibm/spyre`) | WARN |
| behaviour-changing env vars (`SPYRE_*`, `VLLM_*`, `TORCHINDUCTOR_*`, `TORCH_LOGS`, `SENCORES`, `FRONTEND_POOL_ALLOCATION`, `CO_OPTIMIZING*`, `DXP_*`) | WARN |
| cache dirs, `OMP_NUM_THREADS`, `VLLM_PLUGINS`, `SPYRE_DEVICES`; stale editable metadata; spyre-inference's own torch-spyre pin vs the env's; GitHub unreachable | INFO |
| model `model_type` not Gemma-4 | WARN |

The run itself refuses a busy card (`fuser /dev/vfio/vfio`). The `OMP_WAIT_POLICY` /
`SPYRE_NUM_CPUS` of an ITL run and the hf ranks' `SPYRE_DEVICES` are set on those commands
only, so they are not environment findings.

## Deviations from the issue

1. **One environment for all arms** by default, so the hf arm runs the env's torch-spyre
   rather than a dedicated `e9d3` build. The check flags this as a WARN, with the
   commit relation. Pass `--hf-python` to use a separate env the user built.
2. **The env's interpreter instead of `uv run --no-sync`.** Same execution, no uv involved.
3. **`ttft.py` runs from the skill directory**, not from inside the hf-adapters clone, so
   nothing is written into the checkout. With an installed `hf_adapters` the imports are
   identical.
4. **`--output-json` on `vllm bench latency`.** Output only.
5. **A main arm**, which the issue did not have; it shares the branch arm's environment.
6. **TP2 and TP4**, which the issue did not run. Above TP1 the hf arm is `hf_latency.py`,
   because `ttft.py` cannot shard; the measurement is `ttft.py`'s, on a seeded prompt.

## Pitfalls (measured)

- **Profiler builds.** torch-spyre's `setup.py` defaults to `USE_SPYRE_PROFILER=1`, and
  hf-adapters has no override, so the issue's literal hf-adapters install links
  `libaiupti`. spyre-inference's uv build pins the profiler off. On dt-inductor pods,
  `install-spyre-env.sh` needs `--no-profiler`. Tell the user; do not rebuild.
- **Threads.** The issue leaves `SPYRE_NUM_CPUS` unset. Without a cgroup CPU quota,
  spyre-inference then uses all physical cores, split across its TP workers, while every
  hf rank pins 8. Measured irrelevant for TTFT at TP1: `SPYRE_NUM_CPUS=8` moved main and
  the MoE stack by <0.5%. The ITL method sets it anyway, as its report did.
- **Traced MoE (spyre-inference #1154 and descendants).** It re-traces the block at the
  first real request on vLLM's lazy `moe_quant_config is None` guard, then recompiles
  prefill page attention. The timed iterations stay clean, and since #1109 there is no
  `compiled outside warmup` warning. Use `--recompiles`: the target is 0. At TP > 1 every
  worker logs its own recompiles; the report counts the busiest one.
- **torchrun failures (hf above TP1).** A failing rank makes torchrun stop the others and
  print `ChildFailedError`; the cause is above it in `run.log`. hf-adapters documents a
  `libsenlib-dd2.so` destructor abort at torchrun teardown: `hf_latency.py` exits through
  `os._exit` after a barrier to skip it.
- **Prefix caching.** `vllm bench latency` disables it by default, which is what makes the
  repeated prompt valid. Do not enable it.
- **Wrong runtime libraries.** `libsenlib-dd2.so` not found, or an undefined `flex::...`
  symbol from `libspyre_comms.so.1`, means an older in-tree Spyre build is mixed with
  `/opt/ibm/spyre`. On dt-inductor pods, `dt-inductor2/env.sh` is stale: use the venv's
  `bin/activate`.
- **Noise.** Medians agreed to <0.5% across two sessions on the reference pod, and drift
  within a session is of the same order. Alternate arms with `--repeat`; never run them in
  blocks.
- **Editing a running driver.** bash reads a running script incrementally: replace a skill
  script with `mv` of a new file, never in place, while a run is going.

## Reference results

From the dt-inductor2 pod, 2026-10-06/07, all arms on one profiler-free torch-spyre
(`2ab31d08`, which contains `e9d3`), TTFT at TP1. A ballpark, not a contract:

| arm | commit | median TTFT vs hf |
|---|---|---|
| hf-adapters #620 | `9b075e4` | 1.00x |
| spyre-inference main | `ad76a008` | 1.52x (1.54x issue metric) |
| MoE stack (#1058 + #1154) | `ef206a2a` | 1.14x |

ITL and TP2+ depend on the host and its cards: take them from your own runs.

## Output layout

`run_suite.sh` writes `<suite>/report.md` (all combinations in one shareable page, TP
scaling included) and `report.json`, plus one `run_1102.sh` directory per combination
(`ttft_tp1/`, `ttft_tp2/`, `itl_tp4/`, ...). The suite dir defaults to
`${GEMMA4_BENCH_OUT:-~/gemma4-benchmark}/suite_<timestamp>`. Each combination holds:

- `report.md` / `report.json`: headline, method with the exact commands, raw
  avg/p10/p50/p90/p99 per run, environment (commits, versions, cards, RPMs), findings.
- `<arm>_out<len>_<rep>/`: `run.log`, `meta.json`, and `latency.json` (vLLM's JSON, or
  `hf_latency.py`'s; `ttft.py`'s median is read from `run.log`).
- `env_<arm>.json`: everything the check found, with its findings.
- `workload.json`, `plan.txt` (the commands), `provenance.txt` (host, date, interpreters,
  CPU, cards, `ibm-*` RPMs), `args.txt` (the flags used).

## Installing this skill on another host

Copy or symlink this directory into a project's `.claude/skills/`, or into
`~/.claude/skills/`. Keep the directory name `gemma4-benchmark`.
