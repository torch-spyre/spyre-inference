# vLLM Upgrade Procedure

This document describes the complete procedure for upgrading spyre-inference to a newly released vLLM version. It covers bumping the pin in pyproject.toml, re-syncing the upstream test plugin, and triaging the breakages that the bump exposes.

!!! note "AI Assistant Skills"
    This procedure is referenced by both Claude Code's `/upgrade-vllm` skill and IBM Bob's `upgrade-vllm` skill. When using these assistants, they will automatically follow this procedure and replace `{VERSION}` with the actual version number you specify.

## The OOT Platform Trap

Read this before doing anything else — this pitfall can appear in any test category.

Our platform sets `_enum = PlatformEnum.OOT`. This means `current_platform.is_cpu()` returns `False`, even though we inherit from `CpuPlatform` and set `device_type = "cpu"`. Every release, upstream adds new wrappers to `Worker` or `Platform` methods that call `is_cpu()` (or `is_cuda_alike()`, or similar predicates) to short-circuit to a no-op, and raise `RuntimeError` for anything else. We fall through to the raise.

**Don't** fix this by overriding `is_cpu()` to return `True`. `vllm/model_executor/custom_op.py` checks `is_cpu()` *before* `is_out_of_tree()` in its `forward_*` dispatch, and we rely on `forward_oot` for our custom-op overrides. Forcing `is_cpu()` reroutes every `CustomOp` to `forward_cpu`.

**Do** add a surgical override on `TorchSpyreWorker` (or `TorchSpyrePlatform` if the wrapper is on `Platform`). The fix is always the same shape: return whatever no-op the CPU branch would have returned (usually `nullcontext()` or `None`), and leave a comment explaining why you're not fixing `is_cpu()`. For example, the v0.23.0 bump introduced a memory-pool context wrapper in `Worker.load_model`:

```python
def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    # gpu_worker.Worker wraps weight loading in a memory-pool context that
    # short-circuits to nullcontext() when current_platform.is_cpu() is True.
    # Our platform reports OOT (so custom-op forward_oot dispatch works), so
    # the upstream check falls through and raises. Spyre weights live
    # on-device, not in a host-side cumem allocator, so a nullcontext is the
    # correct behaviour.
    return nullcontext()
```

The specific methods that need overrides change with every release — **read the upstream diff** rather than assuming the same set as last time. Search `gpu_worker.py` and `platform.py` in the new rev for `is_cpu()` / `is_cuda_alike()` / `is_out_of_tree()` guards that weren't there before.

## When to Use

User-facing triggers: "bump vllm", "upgrade vllm", "update to the latest vllm release", "pull up the lower bound", "track vllm <commit>", "rebase onto vllm main".

Don't use this for unpinning (`vllm = "*"`) or for widening the support window beyond a single release (e.g. `vllm>=0.22,<0.24`) — those need a real version policy discussion. The current policy is: support exactly one minor release at a time, and bump the lower bound + upper cap together with the pinned rev.

## Step 1: Pick the Target Rev

If the user named a rev, use it. Otherwise fetch the latest stable release:

```bash
gh api repos/vllm-project/vllm/releases --paginate \
  | python3 -c "import json,sys;[print(r['tag_name'],r['published_at']) for r in json.load(sys.stdin)[:10]]"
```

Tags look like `vMAJOR.MINOR.PATCH` (e.g. `v0.23.0`). Confirm "latest stable" with the user only if they were vague — otherwise just go.

Three places in `pyproject.toml` move together:

```toml
[project]
dependencies = [
    "torch-spyre",
    "vllm>=X.Y.Z,<X.(Y+1)",   # bump both halves
    "torch",
]

[tool.uv.sources]
vllm = [
  # On Linux, install from the prebuilt CPU wheel index (avoids ~30 min C++ compile in CI).
  { index = "vllm-cpu", marker = "sys_platform == 'linux'" },
  { git = "https://github.com/vllm-project/vllm", rev = "v{VERSION}", marker = "sys_platform != 'linux'" },
]

# ...further down...

[[tool.uv.index]]
name = "vllm-cpu"
url = "https://wheels.vllm.ai/<commit-sha>/cpu"  # update this SHA too
explicit = true
```

The `[[tool.uv.index]]` URL embeds the **commit SHA** (not the tag name) of the pinned rev. Get it with:

```bash
curl -s "https://api.github.com/repos/vllm-project/vllm/git/refs/tags/v{VERSION}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['object']['sha'])"
```

Then verify the wheels actually exist at that SHA before merging:

```bash
curl -s -o /dev/null -w "%{http_code}" "https://wheels.vllm.ai/<sha>/cpu/"
# Should return 200
```

Bump all three: the version constraint in `[project]`, the git rev in the non-linux source entry, and the SHA in the `vllm-cpu` index URL (also update the inline comment above it).

The runtime lower bound (`vllm>=X.Y.Z`) and the upper cap (`<X.(Y+1)`) exist because we only maintain compatibility with the latest release — any non-backwards-compatible change we land for the new rev will break earlier vLLMs, so they must be excluded from dependency resolution. The `<X.(Y+1)` cap excludes future minor releases that haven't been validated yet.

The version constraint in `[project.dependencies]` controls what downstream environments installing `spyre-inference` from a wheel will resolve to. It must agree with the pinned rev, or a wheel install will silently pull a different vLLM than we tested against.

## Step 2: Rebuild & Re-sync

On Linux, vLLM installs from a prebuilt CPU wheel (seconds). On macOS/Windows it is still built from source, which takes 4–5 minutes. Run sequentially:

```bash
uv sync --group dev
uv run --no-sync sync-upstream-test-deps
uv sync --group dev    # picks up changes the sync script wrote into tests/plugin/pyproject.toml
```

`sync-upstream-test-deps` rewrites `tests/plugin/pyproject.toml` with the dependency list pulled from `vllm/requirements/test.in` at the new rev (filters out forbidden packages like `terratorch`). Check the diff — large changes are normal across a minor-version bump (lm-eval, mistral-common, fastsafetensors versions move; arctic-inference / instanttensor pick up `; platform_machine == 'x86_64'` markers when upstream gates them).

If `uv sync` fails with a dependency conflict that wasn't there before, the override list in `pyproject.toml > [tool.uv] > override-dependencies` may need to grow. Don't add overrides speculatively — only if a real conflict appears.

If the lock file needs to be deleted and regenerated from scratch (e.g. after a messy merge conflict resolution), `uv lock` alone will fail because building `torch-spyre` metadata requires `SPYRE_COMMS_INSTALL_DIR`. Use `USE_SPYRE_CCL=0 uv lock` instead — that env var skips the multi-Spyre comm library requirement and lets uv resolve without a real Spyre build environment.

After the sync, smoke-test the import:

```bash
uv run --no-sync python -c "import spyre_inference; print('ok')"
```

If this fails with `ImportError` from inside vLLM, the upstream API moved out from under one of our imports — `grep -rn "<missing-symbol>" .venv/lib64/python3.12/site-packages/vllm/` to find where it went, then update our import.

## Step 3: Triage in This Order

Don't run the full `pytest` first — vLLM startup is ~30s per test file, and the failures cluster by category. Run the cheapest categories first so you can fix-and-rerun quickly.

**Always use `uv run --no-sync`** for every test invocation — without `--no-sync`, uv re-resolves dependencies on every call and reverts any local `torch-spyre` install you may have done.

### 3a. Custom Ops (No vLLM Engine)

```bash
uv run --no-sync pytest tests/test_platform.py tests/test_mlp.py tests/test_rms_norm.py \
  tests/test_silu_and_mul.py tests/test_parallel_lm_head.py \
  tests/test_vocab_parallel_embedding.py -m "not upstream" --no-header
```

These exercise the linear/RMSNorm/embedding/silu_and_mul ops against a CPU reference. Failures here usually mean a vLLM layer's constructor signature changed (new kwarg, renamed parameter) or a `vocab_parallel_embedding`-style helper was moved. Fix in `spyre_inference/custom_ops/`.

### 3b. Attention Backend

```bash
uv run --no-sync pytest tests/test_spyre_attn.py -m "not upstream" --no-header
```

~10 minutes. If this fails, likely culprits:

- `AttentionImpl.__init__` signature drifted — check `vllm/v1/attention/backends/*.py` for the new contract.
- `AttentionMetadata`/`CommonAttentionMetadata` field rename.
- `bind_kv_cache` layout flipped (see 3e for the upstream KV cache layout note — same change, exposed twice).

### 3c. End-to-End vLLM (Single-Process)

```bash
uv run --no-sync pytest tests/test_vllm_spyre_next.py -m "not upstream" --no-header
```

This is the most common place to hit OOT platform traps (see above). Fix those with surgical `TorchSpyreWorker` overrides as described.

Not all failures here are OOT traps — also watch for constructor or method signature changes on classes we instantiate or inherit from, new abstract methods on base classes, and import path moves (symbol relocated to a different vLLM module).

### 3d. Distributed

```bash
uv run --no-sync pytest -m "distributed" --no-header
```

~5 minutes. Failures usually point to changes in `vllm.distributed.parallel_state` (group construction, device_communicator dispatch). The OOT branches in `_platform_device_type` and `parallel_state.py:~448` use `current_platform.device_name` — if those branches moved to a different predicate, our `device_name = "cpu"` may stop being routed correctly.

### 3e. Upstream Tests

```bash
uv run --no-sync pytest -m "upstream" --no-header
```

These run upstream vLLM test files against our backend, filtered by `tests/plugin/spyre_testing_plugin/upstream_tests.yaml`. The most common failure mode is **upstream test infra refactoring** — the test file changes shape, our `patch_backend_list` fixture (in `tests/plugin/spyre_testing_plugin/pytest_plugin.py`) no longer matches. Read the upstream test file — check `upstream_tests.yaml` for the canonical path, or find it under `~/.cache/vllm-upstream-tests/worktree-<rev>/` (the subpath like `tests/v1/attention/test_attention_backends.py` may have moved between releases) — and diff its assumptions against what `patch_backend_list` does. The recurring trap is the **KV-cache tensor layout**: upstream has flipped between `(2, num_blocks, …)` and `(num_blocks, 2, …)` layouts more than once. Our fixture has to slice this tensor into `(k_pages, v_pages)` lists for `SpyreAttentionImpl.forward`. Check the slicing dim against wherever the upstream test constructs the KV cache (e.g. a function like `create_and_prepopulate_kv_cache` — name may differ in the new rev).

Other upstream-side patterns to watch for:

- New batch-spec entries that don't fit our block_size=64 / max_model_len=1024 defaults — extend `params.allow` in `upstream_tests.yaml` or skip the new variant.
- The list of backends the upstream test iterates over (e.g. a module-level `BACKENDS_TO_TEST` constant — name may differ): we patch it to `[CUSTOM]` only; if the variable name or location moves, the patch silently no-ops and every backend runs.
- Signature changes on the inner correctness-checking function that `patch_backend_list` wraps (e.g. `_test_backend_correctness` — name may differ): our wrapper forwards `*args, **kwargs` but pins `block_size=64`; if a new positional arg lands before `block_size`, that pin lands in the wrong slot.

If a test legitimately doesn't make sense for Spyre (e.g. it asserts triton-kernel-specific behaviour), block-list it in `upstream_tests.yaml` rather than papering over with a tolerance bump.

## Step 4: Full Sweep + Format

Once each category is green individually, run the full suite to catch any cross-test interactions:

```bash
uv run --no-sync pytest --no-header
```

Roughly 16 minutes on a Spyre host. Then format:

```bash
bash format.sh
```

`format.sh` runs prek (ruff, ruff-format, ty, codespell, markdownlint, actionlint) and exits non-zero if anything needs to be added/committed. Do **not** auto-add `.hypothesis/` — that directory is a per-run cache, gitignored.

## What to Commit

Five files typically change for a vLLM bump:

1. `pyproject.toml` — three things: the `rev` in the non-linux `[tool.uv.sources]` entry, the `vllm>=X.Y.Z,<X.(Y+1)` constraint in `[project] dependencies`, and the commit SHA in the `[[tool.uv.index]] name = "vllm-cpu"` URL.
2. `uv.lock` — auto-regenerated by `uv sync`. Don't hand-edit.
3. `tests/plugin/pyproject.toml` — auto-regenerated by `sync-upstream-test-deps`. Review the diff but don't hand-edit.
4. `spyre_inference/v1/worker/spyre_worker.py` (sometimes) or `spyre_inference/platform.py` — surgical overrides for new upstream wrappers.
5. `tests/plugin/spyre_testing_plugin/pytest_plugin.py` (sometimes) — fixture updates for upstream test-infra churn.

If your bump touched custom ops, attention, or distributed code in `spyre_inference/`, that's a sign the upstream API changed — make sure the change is *necessary*, not just "looked easier than figuring out the new contract."

The commit message convention in this repo uses gitmoji prefixes: `:arrow_up: support vllm X.Y.Z` is the standard form.

## When the Bump is Too Big to Land in One PR

If the rev jump skips multiple minor versions and the failure list is long, it may be cleaner to land it in stages:

1. **Rev-only commit**: bump rev, sync test plugin, override the obvious `_enum = OOT` traps, get the e2e test green. Skip the full upstream sweep for now.
2. **Upstream-test-infra commit**: re-enable upstream tests by updating `patch_backend_list` and `upstream_tests.yaml`.
3. **Coverage growback**: any tests that were temporarily block-listed get reviewed and either fixed, kept block-listed with a justification, or deleted.

Don't split the rev itself across commits — `pyproject.toml` and `uv.lock` need to move together or `uv sync` is non-deterministic.

## Scope Guardrails

- Do not bump the **lower bound** during a vLLM upgrade without explicit discussion. Lower-bound bumps require deleting compat code and pruning the test matrix.
- Do not "improve" code outside the compat surface. A vLLM upgrade PR should be reviewable as one focused change.
- If a transitive dep moves in a surprising way (`uv sync` output shows e.g. transformers/torch shifting unexpectedly), surface it to the user before continuing — don't silently accept.
