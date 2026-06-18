---
name: upgrade-vllm
description: Bump the pinned vLLM version in `pyproject.toml`, re-sync the upstream test plugin, and triage the breakages that the bump exposes. Use whenever a user asks to upgrade/bump/update vLLM, pull up the lower bound, or chase a specific upstream commit. Most failures after a bump are not in our code — they are upstream API churn (worker `load_model` wrappers, platform `is_*()` predicates, KV-cache layout flips, new constructor kwargs) that our `_enum = PlatformEnum.OOT` platform doesn't get the CPU/GPU short-circuits for. Encodes the order tests must be run in (custom ops → attention → e2e → distributed → upstream) and the patterns to look for first.
---

# Upgrade vLLM

`spyre_inference` pins vLLM to a specific git rev in `[tool.uv.sources]`. Bumping that rev is mostly mechanical, but every release introduces upstream churn that bites us specifically because our platform reports `_enum = PlatformEnum.OOT` while inheriting from `CpuPlatform` and setting `device_type = "cpu"`. New `if current_platform.is_cpu(): return nullcontext()` guards in upstream code paths fall through for us.

This skill runs the bump, then debugs the failures in a fixed order so you don't waste a 50s vllm rebuild on a fix that turns out to be in test-infra.

## When to use

User-facing triggers: "bump vllm", "upgrade vllm", "update to the latest vllm release", "pull up the lower bound", "track vllm <commit>", "rebase onto vllm main".

Don't use this for unpinning (`vllm = "*"`) or for widening the support window beyond a single release (e.g. `vllm>=0.22,<0.24`) — those need a real version policy discussion. The current policy is: support exactly one minor release at a time, and bump the lower bound + upper cap together with the pinned rev. This skill is for "switch the pinned rev to X and slide the support window onto X."

## Step 1: pick the target rev

If the user named a rev, use it. Otherwise fetch the latest stable release:

```bash
gh api repos/vllm-project/vllm/releases --paginate \
  | python3 -c "import json,sys;[print(r['tag_name'],r['published_at']) for r in json.load(sys.stdin)[:10]]"
```

Tags look like `vMAJOR.MINOR.PATCH` (e.g. `v0.23.0`). Confirm "latest stable" with the user only if they were vague — otherwise just go.

Two places in `pyproject.toml` move together:

```toml
[project]
dependencies = [
    "torch-spyre",
    "vllm>=X.Y.Z,<X.(Y+1)",   # bump both halves
    "torch",
]

[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "vX.Y.Z" }
```

Bump both. The runtime lower bound (`vllm>=X.Y.Z`) and the upper cap (`<X.(Y+1)`) exist because we only maintain compatibility with the latest release — any non-backwards-compatible change we land for the new rev will break earlier vLLMs, so they must be excluded from dependency resolution. The `<X.(Y+1)` cap excludes future minor releases that haven't been validated yet.

The two halves serve different consumers: `[tool.uv.sources]` controls the local build (git checkout of that exact tag), `[project.dependencies]` controls what downstream environments installing `spyre-inference` from a wheel will resolve to. They must agree on the version, or a wheel install will silently pull a different vLLM than we tested against.

## Step 2: rebuild & re-sync

vLLM is built from source for the CPU backend, so the first sync after the rev bump takes 4–5 minutes. Run sequentially:

```bash
uv sync --group dev
uv run --no-sync sync-upstream-test-deps
uv sync --group dev    # picks up changes the sync script wrote into tests/plugin/pyproject.toml
```

`sync-upstream-test-deps` rewrites `tests/plugin/pyproject.toml` with the dependency list pulled from `vllm/requirements/test.in` at the new rev (filters out forbidden packages like `terratorch`). Check the diff — large changes are normal across a minor-version bump (lm-eval, mistral-common, fastsafetensors versions move; arctic-inference / instanttensor pick up `; platform_machine == 'x86_64'` markers when upstream gates them).

If `uv sync` fails with a dependency conflict that wasn't there before, the override list in `pyproject.toml > [tool.uv] > override-dependencies` may need to grow. Don't add overrides speculatively — only if a real conflict appears.

After the sync, smoke-test the import:

```bash
uv run --no-sync python -c "import spyre_inference; print('ok')"
```

If this fails with `ImportError` from inside vLLM, the upstream API moved out from under one of our imports — `grep -rn "<missing-symbol>" .venv/lib64/python3.12/site-packages/vllm/` to find where it went, then update our import.

## Step 3: triage in this order

Don't run the full `pytest` first — vLLM startup is ~30s per test file, and the failures cluster by category. Run the cheapest categories first so you can fix-and-rerun quickly.

**Always use `uv run --no-sync`** for every test invocation — without `--no-sync`, uv re-resolves dependencies on every call and reverts any local `torch-spyre` install you may have done.

### 3a. Custom ops (no vLLM engine)

```bash
uv run --no-sync pytest tests/test_platform.py tests/test_mlp.py tests/test_rms_norm.py \
  tests/test_silu_and_mul.py tests/test_parallel_lm_head.py \
  tests/test_vocab_parallel_embedding.py -m "not upstream" --no-header
```

These exercise the linear/RMSNorm/embedding/silu_and_mul ops against a CPU reference. Failures here usually mean a vLLM layer's constructor signature changed (new kwarg, renamed parameter) or a `vocab_parallel_embedding`-style helper was moved. Fix in `spyre_inference/custom_ops/`.

### 3b. Attention backend

```bash
uv run --no-sync pytest tests/test_spyre_attn.py -m "not upstream" --no-header
```

~10 minutes. If this fails, likely culprits:

- `AttentionImpl.__init__` signature drifted — check `vllm/v1/attention/backends/*.py` for the new contract.
- `AttentionMetadata`/`CommonAttentionMetadata` field rename.
- `bind_kv_cache` layout flipped (see 3e for the upstream KV cache layout note — same change, exposed twice).

### 3c. End-to-end vLLM (single-process)

```bash
uv run --no-sync pytest tests/test_vllm_spyre_next.py -m "not upstream" --no-header
```

This is where the **`_enum = PlatformEnum.OOT` traps** live. The pattern: `vllm.v1.worker.gpu_worker.Worker.<some method>` adds a new wrapper that calls `current_platform.is_cpu()` (or `is_cuda_alike()`, etc.) to short-circuit, raises a `RuntimeError` otherwise. Our platform reports `is_cpu() == False` because `_enum = OOT`, so we hit the raise.

**Don't** override `is_cpu()` to return True — `vllm/model_executor/custom_op.py` checks `is_cpu()` *before* `is_out_of_tree()` in its `forward_*` dispatch, and we rely on `forward_oot` for our custom-op overrides. Forcing `is_cpu()` reroutes every CustomOp to `forward_cpu`.

**Do** add a surgical override in `TorchSpyreWorker` (`spyre_inference/v1/worker/spyre_worker.py`). For example, when v0.23.0 added the sleep-mode allocator pool context to `Worker.load_model`:

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

Look for analogous wrappers added to `Worker.init_device`, `Worker.compile_or_warm_up_model`, `Worker.execute_model`, etc. The pattern is always: import `nullcontext` (or whatever no-op the CPU branch returns), override the method in `TorchSpyreWorker`, leave a comment that explains why we're not just fixing `is_cpu()`.

If the new wrapper is on `Platform` rather than `Worker`, override on `TorchSpyrePlatform` instead — same principle, same comment.

### 3d. Distributed

```bash
uv run --no-sync pytest -m "distributed" --no-header
```

~5 minutes. Failures usually point to changes in `vllm.distributed.parallel_state` (group construction, device_communicator dispatch). The OOT branches in `_platform_device_type` and `parallel_state.py:~448` use `current_platform.device_name` — if those branches moved to a different predicate, our `device_name = "cpu"` may stop being routed correctly.

### 3e. Upstream tests

```bash
uv run --no-sync pytest -m "upstream" --no-header
```

These run upstream vLLM test files against our backend, filtered by `tests/plugin/spyre_testing_plugin/upstream_tests.yaml`. The most common failure mode is **upstream test infra refactoring** — the test file changes shape, our `patch_backend_list` fixture (in `tests/plugin/spyre_testing_plugin/pytest_plugin.py`) no longer matches. Read the upstream test file at:

```text
~/.cache/vllm-upstream-tests/worktree-<rev>/tests/v1/attention/test_attention_backends.py
```

…and diff its assumptions against what `patch_backend_list` does. The recurring trap is the **KV-cache tensor layout**: upstream has flipped between `(2, num_blocks, …)` and `(num_blocks, 2, …)` layouts more than once. Our fixture has to slice this tensor into `(k_pages, v_pages)` lists for `SpyreAttentionImpl.forward`. Check the slicing dim against `create_and_prepopulate_kv_cache` in the upstream test file.

Other upstream-side patterns to watch for:

- New `BatchSpec` entries that don't fit our block_size=64 / max_model_len=1024 defaults — extend `params.allow` in `upstream_tests.yaml` or skip the new variant.
- New `BACKENDS_TO_TEST` defaults (we patch `BACKENDS_TO_TEST` to `[CUSTOM]` only; if the variable name moves, the patch silently no-ops and every backend runs).
- `_test_backend_correctness` signature changes — our wrapper forwards `*args, **kwargs` but pins `block_size=64`; if a new positional arg lands before `block_size`, that pin lands in the wrong slot.

If a test legitimately doesn't make sense for Spyre (e.g. it asserts triton-kernel-specific behaviour), block-list it in `upstream_tests.yaml` rather than papering over with a tolerance bump.

## Step 4: full sweep + format

Once each category is green individually, run the full suite to catch any cross-test interactions:

```bash
uv run --no-sync pytest --no-header
```

Roughly 16 minutes on a Spyre host. Then format:

```bash
bash format.sh
```

`format.sh` runs prek (ruff, ruff-format, ty, codespell, markdownlint, actionlint) and exits non-zero if anything needs to be added/committed. Do **not** auto-add `.hypothesis/` — that directory is a per-run cache, gitignored.

## What to commit

Five files typically change for a vLLM bump:

1. `pyproject.toml` — the `rev` line in `[tool.uv.sources]` AND the `vllm>=X.Y.Z,<X.(Y+1)` constraint in `[project] dependencies`.
2. `uv.lock` — auto-regenerated by `uv sync`. Don't hand-edit.
3. `tests/plugin/pyproject.toml` — auto-regenerated by `sync-upstream-test-deps`. Review the diff but don't hand-edit.
4. `spyre_inference/v1/worker/spyre_worker.py` (sometimes) or `spyre_inference/platform.py` — surgical overrides for new upstream wrappers.
5. `tests/plugin/spyre_testing_plugin/pytest_plugin.py` (sometimes) — fixture updates for upstream test-infra churn.

If your bump touched custom ops, attention, or distributed code in `spyre_inference/`, that's a sign the upstream API changed — make sure the change is *necessary*, not just "looked easier than figuring out the new contract."

The commit message convention in this repo uses gitmoji prefixes: `:arrow_up: support vllm X.Y.Z` is the standard form.

## When the bump is too big to land in one PR

If the rev jump skips multiple minor versions and the failure list is long, it may be cleaner to land it in stages:

1. **Rev-only commit**: bump rev, sync test plugin, override the obvious `_enum = OOT` traps, get the e2e test green. Skip the full upstream sweep for now.
2. **Upstream-test-infra commit**: re-enable upstream tests by updating `patch_backend_list` and `upstream_tests.yaml`.
3. **Coverage growback**: any tests that were temporarily block-listed get reviewed and either fixed, kept block-listed with a justification, or deleted.

Don't split the rev itself across commits — `pyproject.toml` and `uv.lock` need to move together or `uv sync` is non-deterministic.
