---
name: upgrade-torch-spyre
description: Bump the pinned `torch-spyre` git rev in `pyproject.toml` and update `spyre-rpms.lock` to match artifactory, binary-search to the latest commit that actually compiles against this host's `ibm-*` RPMs, clear the stale inductor cache, run a smoke test, and write a reviewer-ready PR description. Use whenever the user asks to "bump", "upgrade", "update", or "pull up" torch-spyre — typically after new `ibm-deeptools` / `ibm-flex` / `ibm-senlib` packages land that unblock previously-failing torch-spyre commits. Encodes that build failures are the expected signal that supporting libs need a matching bump, that the torchinductor cache must be wiped after the bump to avoid `TypeError: ...__init__() got an unexpected keyword argument ...` red herrings, and the curated PR-description shape (notable upstream PRs + bisect table + installed `ibm-*` RPM versions).
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

## Prerequisites

Three environment variables must be set for Artifactory access:

```bash
ARTIFACTORY_URL       # Base URL of the Artifactory instance
ARTIFACTORY_TOKEN     # Bearer token (rotate via Artifactory UI → Access Tokens)
ARTIFACTORY_RPM_REPO  # RPM repository name
```

Validate before starting (note: strip trailing slash from `ARTIFACTORY_URL` to avoid double-slash in API paths — some Artifactory endpoints silently 404 on `//artifactory/...`):

```bash
ARTIFACTORY_URL="${ARTIFACTORY_URL%/}"
curl -sf -H "Authorization: Bearer $ARTIFACTORY_TOKEN" \
  "$ARTIFACTORY_URL/artifactory/api/repositories/$ARTIFACTORY_RPM_REPO" \
  | jq .key
```

If any vars are missing or the request returns a 401 (revoked/expired token), **warn the user** that Artifactory is unavailable and that you'll fall back to scraping CI logs in §6. Continue with the rest of the workflow — Artifactory is only needed for the `spyre-rpms.lock` resolution step.

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

### 5. Run a smoke test

The full test suite takes ~18 minutes and is better left to CI (which parallelizes across runners). Instead, run a single quick test to verify the build is functional:

```bash
uv run --no-sync pytest tests/e2e/test_vllm_spyre_next.py::test_basic_model_load -m "not upstream" -x --timeout=120 -q 2>&1 | tail -20
```

This confirms torch-spyre loads and a model can be instantiated on the Spyre device — catching the most common bump failures (stale inductor cache, missing symbols, import errors) quickly.

**A green local build/smoke test does not guarantee CI links.** The build host has many `ibm-*` libs pre-installed system-wide, so a new link-time dependency the bump introduces (e.g. `-laiupti`) resolves locally but fails in CI, which installs only what's in `spyre-rpms.lock`. Watch the CI "Build spyre-inference" step for `/usr/bin/ld: cannot find -l<lib>`. The fix is to add the missing lib to the lock (§6) — the `next/` tree the lock resolves from publishes these link-time libs; the base `x86_64` tree does not.

If the smoke test passes, tell the user:

> Smoke test passed. The branch is ready for you to commit and push — CI will run the full suite.

Do **not** commit or push on behalf of the user. The human decides when to commit and what branch to push to.

If it fails, triage:

- **`TypeError: ...__init__() got an unexpected keyword argument ...`** during model load → stale inductor cache (you forgot §4). Clear it and re-run.
- **`ImportError` or `RuntimeError` referencing a missing symbol** → the bump pulled in a commit that needs newer RPMs than are installed. Bisect back.
- Numerical mismatches, fallback-warning storms, compile errors on `spyre` → real regressions introduced by the bump. Hand to [[debug-spyre]].

### 6. Update `spyre-rpms.lock`

The torch-spyre bump typically coincides with newer `ibm-*` RPMs on the build host. Lock entries are exact Artifactory filenames minus the `.x86_64.rpm` suffix; `download_rpms.sh` appends `.<arch>.rpm` and prepends the subdir to build the URL.

**The whole lock resolves from the `next/` subdir** (`<repo>/next/x86_64/`), not the base `<repo>/x86_64/` tree. `next/` is the only tree that publishes *every* lib we link against — including `ibm-libaiupti` (the torch-spyre profiler compiles under `-DHAS_AIUPTI -DUSE_KINETO` and links `-laiupti`), which the base tree doesn't carry at all. `download_rpms.sh --prefix` defaults to `next`, so entries are plain filenames with **no** prefix baked in.

**The trap:** `next/` has no monotonic `_NNN` build suffix. Instead it publishes *many rebuilds of the same commit*, distinguished by a 12-hex build hash (e.g. `…a54c45fb.0b76d1bf564c.a54c45f.el10` — dozens of variants for a single commit). There's no ordering in the filename, so the resolution rule is **latest by Artifactory publication time** (`lastModified`), which the `?list` endpoint returns.

#### Step 1: Read installed RPMs (to get package name + commit SHA)

```bash
rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 'ibm-*' \
  | grep -v '(none)' \
  | grep -E '^ibm-(deeptools|flex|senlib|spyre-comms|aiu-toolbox|libaiupti)' \
  | sort > /tmp/host-rpms.txt
```

Each line is `NAME  VERSION-RELEASE`; the commit SHA is the 7-hex after `+<count>.` (e.g. `…1+2142.a54c45f_0.el10` → `a54c45f`). We match `next/` builds at that commit.

#### Step 2: Resolve each package to its latest `next/` build (primary method)

```bash
ARTIFACTORY_URL="${ARTIFACTORY_URL%/}"

# Timestamped listing of the next/ tree (the ?list endpoint returns lastModified)
curl -sf -H "Authorization: Bearer $ARTIFACTORY_TOKEN" \
  "$ARTIFACTORY_URL/artifactory/api/storage/$ARTIFACTORY_RPM_REPO/next/x86_64?list&deep=0" \
  | jq -r '.files[] | "\(.lastModified)\t\(.uri)"' | sed 's|\t/|\t|' \
  > /tmp/next-timestamped.txt

# For each installed package, pick the newest next/ build at the host's commit.
: > /tmp/resolved-rpms.txt
while read -r name ver; do
  sha=$(echo "$ver" | grep -oP '\+[0-9]+\.\K[0-9a-f]{7}')
  fn=$(awk -F'\t' -v pat="^${name}-[0-9].*${sha}.*[.]el10[.]x86_64[.]rpm$" \
        '$2 ~ pat' /tmp/next-timestamped.txt | sort -r | head -1 | cut -f2)
  if [[ -n "$fn" ]]; then
    echo "${fn%.x86_64.rpm}" >> /tmp/resolved-rpms.txt
  else
    echo "NOT IN next/: $name @ $sha" >&2
  fi
done < /tmp/host-rpms.txt
cat /tmp/resolved-rpms.txt
```

`sort -r` orders by the leading ISO-8601 timestamp (lexical = chronological), so `head -1` is the most recently published build — a stable, automatable choice among the rebuild variants.

If a package prints `NOT IN next/`, it isn't published there at the host commit. Decide per package:

- **Not needed for the build/link/runtime** (e.g. `ibm-spyre-comms-test` — ships only unit-test binaries under `bin/`, no `.so`, not installed on the host) → drop it from the lock.
- **Actually needed** → stop and ask the user; do not fall back to the base tree silently (the goal is a single source).

#### Fallback: Artifactory unavailable

If the token is expired/missing and the user can't provide one immediately, fall back to scraping a recent `populate-rpm-cache` CI run's logs:

```bash
gh run list --repo torch-spyre/spyre-inference \
  --workflow populate-rpm-cache.yaml --limit 15 \
  --json databaseId,status,conclusion,createdAt

# Find a run that had a cache miss (actually listed RPMs):
gh run view <RUN_ID> --log 2>/dev/null | grep -c "ibm-deeptools"
# If > 0, extract filenames from its output. Note the CI listing has no
# timestamps, so pick the build matching the host commit by hand.
```

This is less reliable — cache-hit runs won't have the listing, and you may need to trigger a fresh run with a dummy lock-file change.

#### Step 3: Sanity-check — no accidental downgrades

Compare commit counts (the number after `+`) between old and new lock files. This works across `next/` naming schemes (`main.N+N`, `master.N+N`, `main.1+N`) because the count sits right after `+`:

```text
ibm-flex-2.0.0-0.main.456+456.33650c47.89b07ec13974.33650c4.el10
                         ^^^
                         commit count — monotonically increasing
```

```bash
parse_base() {
  echo "$1" | grep -oP '^.+?(?=-\d+\.\d+\.\d+)'
}
parse_commit_count() {
  echo "$1" | grep -oP '(?<=\+)\d+(?=\.)'
}

cp spyre-rpms.lock spyre-rpms.lock.old
echo "=== Downgrade check ==="
while IFS= read -r old_line; do
  base=$(parse_base "$old_line")
  old_num=$(parse_commit_count "$old_line")
  new_line=$(grep "^${base}-[0-9]" /tmp/resolved-rpms.txt | head -1)
  if [[ -z "$new_line" ]]; then
    echo "WARNING: $base not found in new RPMs"
    continue
  fi
  new_num=$(parse_commit_count "$new_line")
  if [[ "$new_num" -lt "$old_num" ]]; then
    echo "WARNING: $base downgraded: $old_num → $new_num"
  else
    echo "OK: $base $old_num → $new_num"
  fi
done < <(grep -v '^[[:space:]]*#' spyre-rpms.lock.old | grep -v '^[[:space:]]*$')
rm spyre-rpms.lock.old
```

Downgrade warnings are informational — the user may intentionally roll back — but confirm before proceeding.

#### Step 4: Write the new lock file

```bash
cat > spyre-rpms.lock <<'HEADER'
# Pinned Spyre RPM package names, one per line.
# Lines starting with # are ignored.
# Specify exact versions to pin (e.g. ibm-deeptools-1.2.3-1.el8).
# The cache key is derived from this file's hash. Changing any entry
# invalidates the GHA RPM cache and triggers a fresh download.
#
HEADER
cat /tmp/resolved-rpms.txt >> spyre-rpms.lock
```

Do **not** guess. If Artifactory (API or CI logs) can't confirm a filename, stop and ask the user. The `next/` build hashes are per-build and may be garbage-collected over time — always re-resolve to the latest build on each bump rather than assuming an old pin still exists.

Verify the download end-to-end before handing off (this is what CI's populate step runs):

```bash
ARTIFACTORY_TOKEN="$ARTIFACTORY_TOKEN" \
ARTIFACTORY_BASE_URL="${ARTIFACTORY_URL%/}" \
ARTIFACTORY_RPM_PATH="$ARTIFACTORY_RPM_REPO" \
RPM_NAMES_TXT=spyre-rpms.lock RPMS_DOWNLOAD_DIR=/tmp/rpm-dl-check \
  bash .github/scripts/download_rpms.sh
```

All entries should download (exit 0). Spot-check that the link-time lib is present:

```bash
rpm2cpio /tmp/rpm-dl-check/ibm-libaiupti-*.rpm | cpio -t 2>/dev/null | grep -i '\.so'
```

#### Cache population

The `populate-rpm-cache` workflow fires automatically via `pull_request_target` when `spyre-rpms.lock` changes, so no manual action is needed — opening the PR is sufficient.

### 7. Capture installed `ibm-*` package versions

For the PR description, snapshot the RPMs that defined the build boundary:

```bash
rpm -qa 'ibm-*' 2>/dev/null | sort
```

This makes the build boundary reproducible — the next person bumping can tell at a glance whether their host has newer libs (and therefore should retry the commits this PR skipped).

### 8. Write the PR description

Write to `PR_torch_spyre_bump.md`. Follow `.github/pull_request_template.md`:

Use one of the two templates below depending on whether a bisect was needed:

#### If the tip built (all N commits included)

````markdown
## Description

Bumps `torch-spyre` from `<old-sha>` to `<new-sha>` — all <N> upstream commits since the previous pin. The tip of `main` compiled cleanly against the currently-installed RPMs (no bisect needed).

### Notable upstream changes in this range

<3–6 curated bullets from §2>

### Installed `ibm-*` packages on the build host

```
<output of rpm -qa 'ibm-*' | sort>
```

## Test Plan

- [x] `uv lock` resolves cleanly to `<new-sha>`
- [x] `uv sync --frozen` builds the torch-spyre C++ extension successfully
- [x] Smoke test (`test_basic_model_load`) passes locally
- [ ] Full CI suite passes (pushed for CI validation)
- [x] `spyre-rpms.lock` updated — no downgrades (commit-count check passed)

**Reviewer note:** when pulling this branch onto an existing checkout, `rm -rf /tmp/torchinductor_*` before running tests — the cache bakes in references to internals that were renamed/removed across the bump.
````

#### If a bisect was needed (K < N commits included)

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
- [x] Smoke test (`test_basic_model_load`) passes locally
- [ ] Full CI suite passes (pushed for CI validation)
- [x] `spyre-rpms.lock` updated — no downgrades (commit-count check passed)

**Reviewer note:** when pulling this branch onto an existing checkout, `rm -rf /tmp/torchinductor_*` before running tests — the cache bakes in references to internals that were renamed/removed across the bump.
````

### 9. Stop — do not commit or push

The skill's job ends here. Present the user with a summary of what changed and the draft PR description. The user will:

1. Review the changes (`pyproject.toml`, `uv.lock`, `spyre-rpms.lock`)
2. Commit on their chosen branch
3. Push and open a PR (which triggers the full CI pipeline and `populate-rpm-cache`)

Do **not** run `git add`, `git commit`, `git push`, `gh pr create`, or any equivalent. The human owns that boundary.

## Files touched (typical)

- `pyproject.toml` (the rev string)
- `uv.lock` (re-locked twice — once for the rev, plus uv may bump transitives)
- `spyre-rpms.lock` (updated with correct artifactory build numbers)
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
