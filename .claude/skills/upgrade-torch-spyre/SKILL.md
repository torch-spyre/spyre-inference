---
name: upgrade-torch-spyre
description: Bump the pinned `torch-spyre` git rev in `pyproject.toml`, refresh the full `uv.lock` (all transitive deps except `vllm`, which is held by its git-tag pin), update `spyre-rpms.lock` to match artifactory, binary-search to the latest commit that actually compiles against this host's `ibm-*` RPMs, clear the stale inductor cache, run a smoke test, and write a reviewer-ready PR description. Use whenever the user asks to "bump", "upgrade", "update", or "pull up" torch-spyre — typically after new `ibm-deeptools` / `ibm-flex` / `ibm-senlib` packages land that unblock previously-failing torch-spyre commits. Encodes that build failures are the expected signal that supporting libs need a matching bump, that the full lockfile refresh happens once on the settled commit (not inside the bisect loop, which must isolate the build-viability signal), that the torchinductor cache must be wiped after the bump to avoid `TypeError: ...__init__() got an unexpected keyword argument ...` red herrings, and the curated PR-description shape (notable upstream PRs + bisect table + installed `ibm-*` RPM versions).
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

If any vars are missing or the request returns a 401 (revoked/expired token), **warn the user** that Artifactory is unavailable and that you'll fall back to scraping CI logs in §7. Continue with the rest of the workflow — Artifactory is only needed for the `spyre-rpms.lock` resolution step.

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
- Inductor / lowering pass changes that touch generated wrappers (any "Remove TensorArg.<field>" — see §5 below)
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

> The helper uses a plain `uv lock` (minimal re-resolve of the rev change), **not** `uv lock --upgrade`. Keep the bisect loop that way: a full transitive upgrade at each iteration would confound the build-viability signal — a failure could be a transitive bump rather than the torch-spyre commit under test. The full lockfile refresh happens exactly once, on the settled commit, in §4.

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

### 4. Refresh the full lockfile

Once the target commit is settled (the tip built, or the bisect landed on the latest building commit), do a **one-shot** refresh of the whole lockfile so transitive dependencies don't silently rot between bumps. The bisect helper only did a minimal `uv lock`; now upgrade everything:

```bash
uv lock --upgrade
```

**`vllm` is excluded automatically.** It's pinned to a git *tag* (`rev = "v0.28.0"`) in `[tool.uv.sources]`, and `--upgrade` only ignores pins in the *output* lockfile — not git-rev/tag pins in `pyproject.toml` sources. So `vllm` stays put (bump it with [[upgrade-vllm]], never here). `torch-spyre` likewise stays at the rev you just set. Confirm both held before continuing:

```bash
python3 - <<'PY'
import tomllib
with open("uv.lock", "rb") as f:
    lock = tomllib.load(f)
for pkg in lock["package"]:
    if pkg["name"] in ("vllm", "torch-spyre"):
        print(pkg["name"], pkg.get("version"), pkg.get("source", {}).get("git", "")[:70])
PY
```

`vllm` must still read `0.28.0+empty …rev=v0.28.0` and `torch-spyre` your new rev. If `--upgrade` moved `vllm`, stop — something changed the source pin and that's out of scope for this skill.

Skim the `Updated …` lines uv prints. A handful of transitives moving (pydantic, transformers, tiktoken, triton, …) is expected and desirable — that's the point of keeping the lock fresh. A **multi-version** package (e.g. `protobuf`, `scipy` resolved at two versions) can print what looks like a downgrade (`v7.x, v6.x -> v6.x`); that's a resolution consolidation, not a real rollback. Then re-sync against the fully-upgraded lock:

```bash
uv sync --frozen
```

This rebuilds torch-spyre *and* installs the new transitive versions, so the smoke test in §6 exercises the exact environment CI will lock to. List the notable transitive bumps in the PR description (§9).

### 5. Clear the inductor cache — mandatory

> **CRITICAL.** After torch-spyre is rebuilt, the next pytest run will hit cached inductor wrappers from the *previous* install. These wrappers contain literal references to torch-spyre internals (kwargs, dataclass fields, attribute names). If the bump renamed or removed any of them, the cached `.py` files crash during model load with `TypeError: <Class>.__init__() got an unexpected keyword argument '<name>'` — even though the freshly-built `.venv` is consistent.

This is a confidently-recognizable signature. The actual fix is one line:

```bash
rm -rf /tmp/torchinductor_*
```

(Also clear `~/.cache/torch_inductor` if it exists.) Do this before the **first** test run after the rev change. Don't wait for the failure — preempt it.

If you forget, the symptom is a `TypeError` referencing a kwarg or attribute that you can grep for and find *only* in `/tmp/torchinductor_*/**/*.py`, never in `.venv/lib/python3.12/site-packages/torch_spyre/`. That's the confirmation.

### 6. Run a smoke test

The full test suite takes ~18 minutes and is better left to CI (which parallelizes across runners). Instead, run a single quick test to verify the build is functional:

```bash
uv run --no-sync pytest tests/e2e/test_vllm_spyre_next.py::test_basic_model_load -m "not upstream" -x --timeout=120 -q 2>&1 | tail -20
```

This confirms torch-spyre loads and a model can be instantiated on the Spyre device — catching the most common bump failures (stale inductor cache, missing symbols, import errors) quickly.

**A green local build/smoke test does not guarantee CI links.** The build host has many `ibm-*` libs pre-installed system-wide, so a new link-time dependency the bump introduces (e.g. `-laiupti`) resolves locally but fails in CI, which installs only what's in `spyre-rpms.lock`. Watch the CI "Build spyre-inference" step for `/usr/bin/ld: cannot find -l<lib>`. The fix is to add the missing lib to the lock (§7). Every lib we depend on — including the profiler's `ibm-libaiupti` — is published in the **prod** (base) tree now, so a new dependency is just another `[packages]` entry — no need to reach into the `next` dev-preview tree.

If the smoke test passes, tell the user:

> Smoke test passed. The branch is ready for you to commit and push — CI will run the full suite.

Do **not** commit or push on behalf of the user. The human decides when to commit and what branch to push to.

If it fails, triage:

- **`TypeError: ...__init__() got an unexpected keyword argument ...`** during model load → stale inductor cache (you forgot §5). Clear it and re-run.
- **`ImportError` or `RuntimeError` referencing a missing symbol** → the bump pulled in a commit that needs newer RPMs than are installed. Bisect back.
- Numerical mismatches, fallback-warning storms, compile errors on `spyre` → real regressions introduced by the bump. Hand to [[debug-spyre]].

### 7. Update `spyre-rpms.lock`

The torch-spyre bump typically coincides with newer `ibm-*` RPMs on the build host. `spyre-rpms.lock` is **TOML**. All packages live in the **prod** (base) tree — `<repo>/<arch>/` — so `[defaults].tree = ""` and there are no per-package tree overrides (`ibm-libaiupti` now ships to prod, so the `next` tree — still available for dev-preview builds — is no longer needed for the baseline). `.github/scripts/resolve_rpms.py` turns the pins into per-arch filenames at CI time.

The lock encodes two kinds of pin:

- **`[packages]` — x86_64, EXACT build.** The trailing `_<buildnum>` is part of the pin. We have x86_64 CI, so we pin the exact build we tested; the build number matters — a newer build of the *same* commit can carry an ABI change that mismatches the compiled torch-spyre extension (we hit exactly this). `resolve_rpms.py` matches this filename verbatim; it does **not** resolve to a newer build.
- **`[overrides.ppc64le]` / `[overrides.s390x]` — P and Z, same commit, build wildcarded.** We have no CI to validate an exact build on P/Z, so we pin the same commit as x86_64 with a `_*` where the build number goes; `resolve_rpms.py` picks the newest matching build. Tighten these to exact builds once P/Z CI exists.

```toml
version = 2

[defaults]
tree = ""

[packages]                                                        # x86_64 — exact build
ibm-deeptools = { version = "2.0.0-0.main.1+2403.cc6e4df_322" }

[overrides.ppc64le]                                               # P — same commit, any build
ibm-deeptools = { version = "2.0.0-0.main.1+2403.cc6e4df_*" }

[overrides.s390x]                                                 # Z — same commit, any build
ibm-deeptools = { version = "2.0.0-0.main.1+2403.cc6e4df_*" }
```

The version-string anatomy:

```text
2.0.0-0.main.1+2403.cc6e4df_322.el10.x86_64.rpm
└──────── version ────────┘└build┘
             ^^^^ ^^^^^^^
            count  sha7   ← the commit
```

For x86_64 you keep the whole `version`, including `_<build>`. For the P/Z overrides you keep up to the sha and replace `_<build>` with `_*`.

#### Step 1: Read installed RPMs (name + full build)

```bash
rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 'ibm-*' \
  | grep -v '(none)' \
  | grep -E '^ibm-(deeptools|flex|senlib|spyre-comms|aiu-toolbox|libaiupti)' \
  | sort > /tmp/host-rpms.txt
```

Each line is `NAME  VERSION-RELEASE`, e.g. `ibm-deeptools 2.0.0-0.main.1+2403.cc6e4df_322.el10`. The x86_64 `version` you pin is everything before `.el10` (`…cc6e4df_322`, build included); the commit (for the P/Z overrides) is that string up to the 7-hex sha, with `_<build>` replaced by `_*`.

#### Step 2: Confirm the host build exists in prod on x86_64

CI installs from prod, not from the host, so the *exact* host build must be published there (a locally-built RPM won't be). Query Artifactory for each one. `resolve_rpms.py` reads the same env vars (with `ARTIFACTORY_URL`/`ARTIFACTORY_RPM_REPO` accepted as fallbacks for `ARTIFACTORY_BASE_URL`/`ARTIFACTORY_RPM_PATH`).

```bash
ARTIFACTORY_URL="${ARTIFACTORY_URL%/}"
aql() {  # POST an AQL query, return raw JSON
  curl -sf -H "Authorization: Bearer $ARTIFACTORY_TOKEN" \
    -H "Content-Type: text/plain" \
    -X POST "$ARTIFACTORY_URL/artifactory/api/search/aql" --data "$1"
}

while read -r name ver; do
  build="${ver%.el10}"        # strip trailing .el10 → …cc6e4df_322
  q='items.find({"repo":"'"$ARTIFACTORY_RPM_REPO"'","path":"x86_64",'
  q+='"name":{"$match":"'"$name"'-'"$build"'.el10.x86_64.rpm"}})'
  hit=$(aql "$q" | jq -r '.results[].name' | head -1)
  echo "${hit:-MISSING: $name @ $build}"
done < /tmp/host-rpms.txt
```

If a build prints `MISSING`, it wasn't published to prod (typically a local-only build). Pin the newest prod build of the **same commit** instead: query `"<name>-<…sha7>_*.el10.x86_64.rpm"`, `.sort({"$desc":["created"]})`, take the first, and note the substitution in the PR.

#### Step 3: Edit the lock

Set each `[packages]` entry's `version` to the exact x86_64 build from step 2, and set the matching `[overrides.ppc64le]` and `[overrides.s390x]` entries to the same commit with `_*`. Keep the `-core`/`-dd2`/`-headers`/`-devel` sub-packages of a lib on the **same** commit as their base package (they're built together). Leave `[defaults]` and `version = 2` untouched. Then confirm it parses:

```bash
python3 .github/scripts/resolve_rpms.py names   # lists package names, no network
```

#### Step 4: Validate the pin resolves on **all** arches — mandatory

A build/commit that's present on x86_64 but missing on P or Z must fail here, at upgrade time, not on a later per-arch run.

```bash
ARTIFACTORY_TOKEN="$ARTIFACTORY_TOKEN" \
ARTIFACTORY_BASE_URL="${ARTIFACTORY_URL}" \
ARTIFACTORY_RPM_PATH="$ARTIFACTORY_RPM_REPO" \
  python3 .github/scripts/resolve_rpms.py validate
```

Every line must print `OK`. A `FAIL … missing on: x86_64 (…_322)` means the exact build isn't in prod — fix per step 2. A `FAIL … missing on: ppc64le`/`s390x` means the pinned commit isn't published on that arch — pick a commit present on all arches, or ask the user. This is exactly the check CI's populate step runs before caching.

#### Step 5: Sanity-check — no accidental downgrades

Compare the commit count (the number right after `+`) of each new pin against the old lock. Both are TOML now, so read them with `tomllib`:

```bash
git show HEAD:spyre-rpms.lock > /tmp/spyre-rpms.lock.old
python3 - <<'PY'
import tomllib, re
def counts(path):
    with open(path, "rb") as f:
        pkgs = tomllib.load(f).get("packages", {})
    out = {}
    for name, spec in pkgs.items():
        m = re.search(r'\+(\d+)\.', spec["version"])
        out[name] = int(m.group(1)) if m else None
    return out
old, new = counts("/tmp/spyre-rpms.lock.old"), counts("spyre-rpms.lock")
for name in sorted(set(old) | set(new)):
    o, n = old.get(name), new.get(name)
    if o is None:   print(f"NEW:  {name} @ {n}")
    elif n is None: print(f"DROP: {name} (was {o})")
    elif n < o:     print(f"WARNING downgrade: {name} {o} → {n}")
    else:           print(f"OK:   {name} {o} → {n}")
PY
```

Downgrade warnings are informational — the user may intentionally roll back — but confirm before proceeding.

#### Step 6: Verify the download end-to-end

Resolve the per-arch filenames and download them, mirroring exactly what `populate-rpm-cache` does:

```bash
ARTIFACTORY_TOKEN="$ARTIFACTORY_TOKEN" \
ARTIFACTORY_BASE_URL="${ARTIFACTORY_URL}" \
ARTIFACTORY_RPM_PATH="$ARTIFACTORY_RPM_REPO" \
  python3 .github/scripts/resolve_rpms.py resolve --arch x86_64 > /tmp/relpaths.txt
cat /tmp/relpaths.txt   # one repo-relative <tree>/<arch>/<file> per package

BASE="${ARTIFACTORY_URL}/artifactory/${ARTIFACTORY_RPM_REPO}"
mkdir -p /tmp/rpm-dl-check
while read -r rel; do
  curl -fSL -H "Authorization: Bearer $ARTIFACTORY_TOKEN" \
    -o "/tmp/rpm-dl-check/$(basename "$rel")" "${BASE}/${rel}"
done < /tmp/relpaths.txt
```

All entries should download (exit 0). Spot-check that the link-time lib carries its `.so`:

```bash
rpm2cpio /tmp/rpm-dl-check/ibm-libaiupti-*.rpm | cpio -t 2>/dev/null | grep -i '\.so'
```

#### Fallback: Artifactory unavailable

If the token is expired/missing and the user can't provide one, you cannot resolve or validate — the lock's pins can't be checked against (or, for P/Z, resolved into) real filenames without querying Artifactory. Tell the user the lock can't be updated/verified this session and leave `spyre-rpms.lock` unchanged (the existing pins stay valid until a real bump). Do **not** hand-write version strings from CI logs into the TOML — a wrong commit passes `python3 … names` but fails `validate` on the runner.

#### Cache population

The `populate-rpm-cache` workflow fires automatically via `pull_request_target` when `spyre-rpms.lock` changes, so no manual action is needed — opening the PR is sufficient. It validates all arches, then downloads and caches the x86_64 build.

### 8. Capture installed `ibm-*` package versions

For the PR description, snapshot the RPMs that defined the build boundary:

```bash
rpm -qa 'ibm-*' 2>/dev/null | sort
```

This makes the build boundary reproducible — the next person bumping can tell at a glance whether their host has newer libs (and therefore should retry the commits this PR skipped).

### 9. Write the PR description

Write to `PR_torch_spyre_bump.md`. Follow `.github/pull_request_template.md`:

Use one of the two templates below depending on whether a bisect was needed:

#### If the tip built (all N commits included)

````markdown
## Description

Bumps `torch-spyre` from `<old-sha>` to `<new-sha>` — all <N> upstream commits since the previous pin. The tip of `main` compiled cleanly against the currently-installed RPMs (no bisect needed). Also refreshes the full `uv.lock` (`uv lock --upgrade`); `vllm` is held at its pinned git tag and intentionally **not** upgraded.

### Notable upstream changes in this range

<3–6 curated bullets from §2>

### Transitive dependency refresh

<count> transitive packages moved via `uv lock --upgrade`; notable bumps: <e.g. transformers X → Y, triton X → Y, …>. `vllm` and `torch-spyre` unchanged (pinned).

### Installed `ibm-*` packages on the build host

```
<output of rpm -qa 'ibm-*' | sort>
```

## Test Plan

- [x] `uv lock` resolves cleanly to `<new-sha>`
- [x] `uv lock --upgrade` refreshes all transitives (vllm held at its git tag)
- [x] `uv sync --frozen` builds the torch-spyre C++ extension successfully
- [x] Smoke test (`test_basic_model_load`) passes locally
- [ ] Full CI suite passes (pushed for CI validation)
- [x] `spyre-rpms.lock` updated — no downgrades (commit-count check passed)

**Reviewer note:** when pulling this branch onto an existing checkout, `rm -rf /tmp/torchinductor_*` before running tests — the cache bakes in references to internals that were renamed/removed across the bump.
````

#### If a bisect was needed (K < N commits included)

````markdown
## Description

Bumps `torch-spyre` from `<old-sha>` to `<new-sha>` — <K> of the <N> upstream commits since the previous pin. The remaining <N-K> commits (starting with <#FIRST-FAILING> "<first-failing-title>") need matching `ibm-*` updates and fail the torch-spyre C++ extension build against the currently-installed RPMs. Also refreshes the full `uv.lock` (`uv lock --upgrade`); `vllm` is held at its pinned git tag and intentionally **not** upgraded.

### Notable upstream changes in this range

<3–6 curated bullets from §2>

### Transitive dependency refresh

<count> transitive packages moved via `uv lock --upgrade`; notable bumps: <e.g. transformers X → Y, triton X → Y, …>. `vllm` and `torch-spyre` unchanged (pinned).

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
- [x] `uv lock --upgrade` refreshes all transitives (vllm held at its git tag)
- [x] `uv sync --frozen` builds the torch-spyre C++ extension successfully
- [x] Smoke test (`test_basic_model_load`) passes locally
- [ ] Full CI suite passes (pushed for CI validation)
- [x] `spyre-rpms.lock` updated — no downgrades (commit-count check passed)

**Reviewer note:** when pulling this branch onto an existing checkout, `rm -rf /tmp/torchinductor_*` before running tests — the cache bakes in references to internals that were renamed/removed across the bump.
````

### 10. Stop — do not commit or push

The skill's job ends here. Present the user with a summary of what changed and the draft PR description. The user will:

1. Review the changes (`pyproject.toml`, `uv.lock`, `spyre-rpms.lock`)
2. Commit on their chosen branch
3. Push and open a PR (which triggers the full CI pipeline and `populate-rpm-cache`)

Do **not** run `git add`, `git commit`, `git push`, `gh pr create`, or any equivalent. The human owns that boundary.

## Files touched (typical)

- `pyproject.toml` (the rev string)
- `uv.lock` (re-locked for the rev during bisect, then fully refreshed with `uv lock --upgrade` on the settled commit — expect many transitive bumps; `vllm` held at its git tag)
- `spyre-rpms.lock` (TOML — bumped x86_64 exact-build pins + P/Z commit overrides; resolved/validated via `.github/scripts/resolve_rpms.py`)
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
