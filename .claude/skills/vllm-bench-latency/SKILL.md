---
name: vllm-bench-latency
description: Run `vllm bench latency` for a user-specified model on Spyre from within this checkout, then report avg latency + percentiles. If the user asks to compare, re-run the identical benchmark on `main` (via a throwaway git worktree, non-destructive to the working tree) and print a side-by-side comparison. Use when the user wants Spyre latency numbers for a model, or wants to see how the current branch's latency compares to main. Assumes you are already on the Spyre host with spyre-inference installed editable.
user-invocable: true
argument-hint: "<model> [--compare-main] [--input-len N] [--output-len N] [--batch-size N] [--max-model-len N] [--num-iters N] [--num-iters-warmup N] [--mode MODE]"
---

# Latency benchmark on Spyre

Run `vllm bench latency` for a model the user names against **this** `spyre-inference` checkout and report the numbers; optionally re-run the identical benchmark on `main` for a side-by-side comparison. Assumes you are already on the Spyre host, in the repo, with the package installed editable (`uv sync`) — like the other skills here. It does not provision hosts, sync code, or manage credentials.

## Hard constraints (do not violate)

- **Single accelerator, sequential only.** Spyre serves one process at a time — never run two Spyre-backed commands concurrently (no backgrounding, no `pytest -n`). In `--compare-main` mode the two runs are strictly sequential.
- **`uv run --no-sync`** is mandatory — a plain `uv run` / `uv sync` re-resolves the lockfile and clobbers the editable install.
- **Non-destructive.** The `main` comparison uses a throwaway `git worktree` + a temporary editable-install repoint — never `git checkout` / `git stash` in the working copy, and always restore (step 2).
- **Always report, never invent.** End every run with numbers from the JSON (step 3); if a run fails, report the failure with the log tail instead of a number.

## Inputs

- **`<model>`** (required): HF model id or locally-cached path. If omitted, default to `ibm-granite/granite-3.3-8b-instruct` and say so in the report.
- **`--compare-main`**: after the current-branch run, run the **identical** benchmark on local `main` and print a comparison. Only when the user asks to compare.
- Benchmark params (optional; defaults in parens): `--input-len` (64), `--output-len` (64), `--batch-size` (1), `--max-model-len` (128), `--num-iters` (10), `--num-iters-warmup` (2), `--mode` (`STOCK_TORCH_COMPILE`, passed as `-cc.mode`).

## Steps

### 0. Preflight

```bash
test -d .venv || { echo "no .venv — run 'uv sync' first"; exit 1; }
BRANCH=$(git rev-parse --abbrev-ref HEAD); SHA=$(git rev-parse --short HEAD)
echo "spyre-inference: $BRANCH ($SHA)"
```

### 1. Run the benchmark (current branch)

Fix the params in one block so a compare run is identical, then run and read back the JSON (source of truth):

```bash
MODEL="<model>"; INPUT_LEN=64; OUTPUT_LEN=64; BATCH_SIZE=1
MAX_LEN=128; ITERS=10; WARMUP=2; MODE=STOCK_TORCH_COMPILE
TAG=branch
OUT=.claude/skills/vllm-bench-latency/logs; mkdir -p "$OUT"   # artifacts live here, not the repo root

VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 SPYRE_FORCE_COMPILE_ATTN=1 \
uv run --no-sync vllm bench latency \
  --model "$MODEL" \
  --input-len $INPUT_LEN --output-len $OUTPUT_LEN --batch-size $BATCH_SIZE \
  --num-iters-warmup $WARMUP --num-iters $ITERS --max-model-len $MAX_LEN \
  -cc.mode $MODE \
  --output-json "$OUT/latency_${TAG}.json" 2>&1 | tee "$OUT/latency_${TAG}.log"
cat "$OUT/latency_${TAG}.json"
```

The JSON has `avg_latency` (s), `latencies` (per-iter), `percentiles` (keys 10/25/50/75/90/99). First run is slow (eager + cold compile — the `1800`s timeout is deliberately generous); `FallbackWarning` lines are normal. On error, report the tail of `$OUT/latency_${TAG}.log`, not a fake number.

### 2. Compare against `main` (only if `--compare-main`)

If the current branch already **is** `main`, skip and say so. Otherwise benchmark `main` via a throwaway worktree — repoint the editable install at it, run the **identical** command, then **always** restore (even on failure) so the env is left as found:

```bash
WT=$(mktemp -d); git worktree add --detach "$WT" main
MAIN_SHA=$(git rev-parse --short main)
uv pip install --no-deps -e "$WT"            # repoint editable install at main

TAG=main
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 SPYRE_FORCE_COMPILE_ATTN=1 \
uv run --no-sync vllm bench latency \
  --model "$MODEL" \
  --input-len $INPUT_LEN --output-len $OUTPUT_LEN --batch-size $BATCH_SIZE \
  --num-iters-warmup $WARMUP --num-iters $ITERS --max-model-len $MAX_LEN \
  -cc.mode $MODE \
  --output-json "$OUT/latency_${TAG}.json" 2>&1 | tee "$OUT/latency_${TAG}.log"
cat "$OUT/latency_${TAG}.json"

uv pip install --no-deps -e .                # restore editable install
git worktree remove --force "$WT"
```

The comparison is **current working copy (`$BRANCH` @ `$SHA`, incl. uncommitted changes) vs local `main` @ `$MAIN_SHA`**. For origin/main, the user should `git fetch` and re-run.

### 3. Report

Report the model + exact params, then numbers from the JSON:

- **Single run**: avg latency + p50/p90/p99 (s). Optionally decode throughput = `output-len × batch-size / avg_latency` tok/s (decode only — ignores prefill).
- **Compare mode**: a small table (branch vs main) of avg latency + key percentiles, the absolute and % change (`(branch − main) / main × 100`), and which is faster. Label each column with its branch/sha.

```text
model: <model>   params: in<N>/out<N>/bs<N>/max<N>/iters<N>/warmup<N>/<MODE>
                        main (<sha>)   <branch> (<sha>)   Δ
avg latency (s)         <a>            <b>                <b−a>  (<pct>%, <faster/slower>)
p50 (s)                 <a>            <b>                <b−a>
p90 (s)                 <a>            <b>                <b−a>
```

## Notes & pitfalls

- **First run is slow.** A cold cache forces a recompile even with a shared cache PVC, so the cold compile can take many minutes — *most of it during the first warmup iteration*. The `Fast Path Debug] SUCCESS` spam is per-decode-shape compilation and keeps going well after `Warming up...` prints, so reaching warmup is **not** "almost done"; don't promise an ETA while those lines still emit. In `--compare-main` mode the `main` leg often compiles cheaper (reuses the shapes the first leg warmed) — so compile cost doesn't explain latency deltas; those live in the timed iters.
- **`-cc.mode` isn't universally safe.** `STOCK_TORCH_COMPILE` is a good default (granite, llama), but some models crash *natively* at warmup under it (gemma-class here). If warmup dies natively rather than raising a Python error, retry with a different `-cc.mode` before calling it a regression.
- **Model not cached**: an uncached HF model downloads on first use (extra minutes); a locally-cached id skips that.
- **Metrics scope**: this reports *only* whole-generation latency (`avg_latency`, `latencies`, `percentiles`, all in seconds) — each iter is one opaque `llm.generate()` over the batch. There is **no** TTFT / ITL / TPOT / per-token breakdown and no throughput in the JSON (derive it as above). If the user wants TTFT or ITL/TPOT, that's `vllm bench serve` (needs a live `vllm serve` server) — say so rather than faking it from this run.
