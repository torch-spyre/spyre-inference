---
name: upgrade-torch-spyre
description: Bump the pinned `torch-spyre` git rev in `pyproject.toml`, binary-search to the latest commit that actually compiles against this host's `ibm-*` RPMs, clear the stale inductor cache, run the full test suite, and write a reviewer-ready PR description. Use whenever the user asks to "bump", "upgrade", "update", or "pull up" torch-spyre — typically after new `ibm-deeptools` / `ibm-flex` / `ibm-senlib` packages land that unblock previously-failing torch-spyre commits. Encodes that build failures are the expected signal that supporting libs need a matching bump, that the torchinductor cache must be wiped after the bump to avoid `TypeError: ...__init__() got an unexpected keyword argument ...` red herrings, and the curated PR-description shape (notable upstream PRs + bisect table + installed `ibm-*` RPM versions).
---

# Upgrade torch-spyre

`spyre-inference` pins `torch-spyre` to a specific commit SHA in `[tool.uv.sources]` of `pyproject.toml`. Bumping that pin is **not** mechanical: `torch-spyre` makes regular breaking changes that need matching updates in the host's `ibm-deeptools` / `ibm-flex` / `ibm-senlib` RPMs. If you naively bump to the tip of `main`, the C++ extension build will fail against whatever older RPMs are installed on this host. The job of this skill is to find the **latest commit that actually compiles here** — which is usually *not* the tip of `main`.

## When to use

Trigger phrases:

- "bump torch-spyre", "upgrade torch-spyre", "update torch-spyre"
- "torch-spyre is out of date", "new ibm libs landed"
- "we should pull up torch-spyre"

Do **not** invoke for unrelated debugging that happens to touch torch-spyre — use [[debug-spyre]] for that.

## The supporting-libs trap (why this isn't `uv lock && uv sync`)

When the user pins torch-spyre to a SHA, `uv sync --frozen` rebuilds the torch-spyre C++ extension from source (~50s) against headers shipped by `ibm-deeptools-devel` / `ibm-flex-devel` / `ibm-senlib-headers` on the build host. Every few weeks, torch-spyre lands commits that depend on a new symbol, struct field, or header layout that those RPMs haven't shipped yet. The build fails with `RuntimeError: Error compiling objects for extension` near the tail of `uv sync` — buried after thousands of warning lines.

**Treat compile failure as an expected outcome, not an error.** It's the signal that the supporting libs need a bump; until that happens, the right move is to find the newest torch-spyre commit that still builds.

## Workflow

### 1. Read current pin and latest main

```bash
grep -n "torch-spyre = " pyproject.toml
# → torch-spyre = { git = "...", rev = "<current-sha>" }

gh api repos/torch-spyre/torch-spyre/commits/main \
  --jq '{sha: .sha, message: .commit.message, date: .commit.author.date}'
```

If `current-sha == main`, there's nothing to do. Otherwise compare the range:

```bash
gh api 'repos/torch-spyre/torch-spyre/compare/<current-sha>...<main-sha>' \
  --jq '.commits | to_entries | .[] | "\(.key) \(.value.sha) \(.value.commit.message | split("\n")[0])"'
```

That gives you a numbered list (oldest → newest) of every candidate commit. Save the count — call it `N`. The bisect runs over indices `[-1, N-1]`, where `-1` is the known-good current pin and `N-1` is the candidate tip.

### 2. Curate notable changes for the reviewer

Skim the commit messages in the range. Flag commits that change *runtime* behavior, public APIs, or import-time side effects — those are what the reviewer needs to know about. Past examples worth flagging:

- New op / quant support (e.g. "Add FP8 quantization and dequantization support")
- Logging / observability rewrites ("Phase 1 new logging framework")
- Import-time side effects ("Apply the Spyre tensor monkey-patch at autoload time")
- Inductor / lowering pass changes that touch generated wrappers (any "Remove TensorArg.<field>" — see §4 below)
- Spec / format changes (SDSC, restickify, named-dim propagation)

Ignore CI/cicd-only commits, test-yaml shuffles, and xfail→pass moves unless they hint at a behavior change. Aim for 3–6 bullets in the final PR description, not a dump of all 50+ messages.

### 3. Try the tip; if it builds, you're done

Use the helper below. Even if you think the tip won't build, try it first — saves a bisect when you're lucky.

#### Bisect helper (`/tmp/spyre-bisect/try.sh`)

Write this once at the start of the session:

```bash
mkdir -p /tmp/spyre-bisect
cat > /tmp/spyre-bisect/try.sh <<'BASH'
#!/usr/bin/env bash
# Usage: try.sh <full-sha>
# Updates pyproject.toml, re-locks, attempts uv sync --frozen.
# Exit 0 = build succeeded, 1 = build failed, 2 = lock failed.
set -u
sha="${1:?need sha}"
log="/tmp/spyre-bisect/${sha:0:7}.log"
cd /home/senuser/spyre-inference

python3 - "$sha" <<'PY'
import re, sys, pathlib
sha = sys.argv[1]
p = pathlib.Path("pyproject.toml")
text = p.read_text()
new = re.sub(
    r'(torch-spyre = \{ git = "https://github\.com/torch-spyre/torch-spyre", rev = ")[0-9a-f]+(" \})',
    r'\g<1>' + sha + r'\g<2>',
    text,
    count=1,
)
assert new != text, "rev not updated"
p.write_text(new)
PY

uv lock >>"$log" 2>&1 || { echo "LOCK_FAILED"; exit 2; }

if uv sync --frozen >>"$log" 2>&1; then
    echo "BUILD_OK"
    exit 0
else
    echo "BUILD_FAILED"
    exit 1
fi
BASH
chmod +x /tmp/spyre-bisect/try.sh
```

Each invocation writes the full output to `/tmp/spyre-bisect/<short-sha>.log` so you can grep for the actual compile error (look for `error:` and `fatal`, not just `warning:`) without re-running the 50s build.

#### Bisect

If the tip fails, binary-search:

```text
low  = -1      # known good (current pin)
high = N - 1   # known bad (tip)
while high - low > 1:
    mid = (low + high) // 2
    sha = commits[mid]
    if try.sh sha succeeds:
        low = mid
    else:
        high = mid
target = commits[low]   # latest building commit
```

This takes ≤ `log2(N)` iterations, ~5–7 builds for typical ranges (50–100 commits). Each build is ~50s plus a few seconds of lock + small package syncs.

**State the bounds in chat after each iteration** ("Range [27, 55]. Next: index 41 = `a14b29e`.") so the user can interrupt early if they spot something off.

### 4. Clear the inductor cache — mandatory

> **CRITICAL.** After torch-spyre is rebuilt, the next pytest run will hit cached inductor wrappers from the *previous* install. These wrappers contain literal references to torch-spyre internals (kwargs, dataclass fields, attribute names). If the bump renamed or removed any of them, the cached `.py` files crash during model load with `TypeError: <Class>.__init__() got an unexpected keyword argument '<name>'` — even though the freshly-built `.venv` is consistent.

This is a confidently-recognizable signature. The actual fix is one line:

```bash
rm -rf /tmp/torchinductor_*
```

(Also clear `~/.cache/torch_inductor` if it exists.) Do this before the **first** test run after the rev change. Don't wait for the failure — preempt it.

If you forget, the symptom is a `TypeError` referencing a kwarg or attribute that you can grep for and find *only* in `/tmp/torchinductor_*/**/*.py`, never in `.venv/lib/python3.12/site-packages/torch_spyre/`. That's the confirmation.

### 5. Run the full test suite

Spyre is single-process — **no `-n`, no `xdist`, no backgrounding two pytest invocations**. Run sequentially:

```bash
uv run --no-sync pytest 2>&1 | tee /tmp/spyre-bisect/pytest-full.log
```

`--no-sync` is required when iterating after a manual rebuild (see CLAUDE.md "Iterating on a Local `torch-spyre` Checkout"); for the post-bump run from a clean `uv sync --frozen` you don't strictly need it, but it's harmless and saves a re-sync.

Expect ~18 minutes wall-clock. Triage failures:

- **`TypeError: ...__init__() got an unexpected keyword argument ...`** during model load → stale inductor cache (you forgot §4). Clear it and re-run just the failures.
- Numerical mismatches, fallback-warning storms, compile errors on `spyre` → real regressions introduced by the bump. Hand to [[debug-spyre]].
- A handful of `tests/test_distributed_tp2.py` / `tests/test_vllm_spyre_next.py` failures with the same root cause → probably one upstream change touched a path used by all of them. Read one traceback, fix once.

### 6. Capture installed `ibm-*` package versions

For the PR description, snapshot the RPMs that defined the build boundary:

```bash
rpm -qa 'ibm-*' 2>/dev/null | sort
```

This makes the build boundary reproducible — the next person bumping can tell at a glance whether their host has newer libs (and therefore should retry the commits this PR skipped).

### 7. Write the PR description

Write to `PR_torch_spyre_bump.md`. Follow `.github/pull_request_template.md`:

````markdown
## Description

Bumps `torch-spyre` from `<old-sha>` to `<new-sha>` — <K> of the <N> upstream commits since the previous pin. The remaining <N-K> commits (starting with <#FIRST-FAILING> "<first-failing-title>") need matching `ibm-*` updates and fail the torch-spyre C++ extension build against the currently-installed RPMs.

### Notable upstream changes in this range

<3–6 curated bullets from §2>

### Binary search for the latest building commit

| Iter | Index | Commit  | Result      |
|------|-------|---------|-------------|
| 1    | ...   | `<sha>` | build OK / failed |
...

→ Last compiling commit: **`<new-sha>`** (PR #<num>, "<title>").

### Installed `ibm-*` packages on the build host

```
<output of rpm -qa 'ibm-*' | sort>
```

## Test Plan

- [x] `uv lock` resolves cleanly to `<new-sha>`
- [x] `uv sync --frozen` builds the torch-spyre C++ extension successfully
- [x] `uv run --no-sync pytest` — **<P> passed, <S> skipped, <F> failed**
- [x] <note about inductor cache, if it bit you>

**Reviewer note:** when pulling this branch onto an existing checkout, `rm -rf /tmp/torchinductor_*` before running tests — the cache bakes in references to internals that were renamed/removed across the bump.

````

## Files touched (typical)

- `pyproject.toml` (the rev string)
- `uv.lock` (re-locked twice — once for the rev, plus uv may bump transitives)
- `PR_torch_spyre_bump.md` (scratch description for the user to paste)

Nothing in `spyre_inference/` should change during a pure bump. If you find yourself editing the package, you've crossed into "the bump exposed a regression" territory — stop, save state, and triage with [[debug-spyre]].

## Things that look like bugs but aren't

- **Hundreds of compiler warnings during `uv sync`.** Normal. The torch-spyre C++ extension intentionally compiles with `-Wall`. The actual failure (when present) is below the warnings, prefixed `error:` or `fatal`. Grep with `grep -iE "error:|fatal"`.
- **`Uninstalled 1 package … Installed 1 package …` near the top of pytest output** referring to torch-spyre. Means a previous `uv run` (without `--no-sync`) reverted your install. Re-run with `--no-sync`. See CLAUDE.md.
- **`FallbackWarning` lines in the pytest tail.** Not specific to bumps. These mean torch-spyre routed an op to CPU — pre-existing on most paths. Only act on these if the test *failed* and the warning is from a hot path.
- **A trailing `DeprecationWarning: builtin type swigvarlink has no __module__ attribute`** in the pytest output. Cosmetic interpreter teardown noise.

## Related skills

- [[debug-spyre]] — invoke when the bump exposes real numerical / compile regressions.
- [[upgrade-vllm]] — same shape of work for the vLLM pin; reuse the bisect helper pattern, but the cache pitfall is torch-spyre-specific.
