# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`spyre-inference` is a vLLM platform plugin that integrates with `torch-spyre` to leverage IBM's Spyre AI accelerator hardware. It provides PyTorch-native attention implementations and custom operations optimized for Spyre devices.

## Architecture

### Platform Integration

- Registers as `spyre_inference` vLLM platform plugin via entry points (`spyre_inference:register`)
- `TorchSpyrePlatform` extends `CpuPlatform`, overriding device execution for Spyre hardware
- Forces `torch.float16` dtype and eager execution mode (torch.compile incompatible with current CPU fallback ops)

### Key Components

**Core Modules:**

- `spyre_inference/platform.py` - Platform registration and configuration
- `spyre_inference/v1/worker/spyre_worker.py` - Worker class that executes model on Spyre device
- `spyre_inference/v1/worker/spyre_model_runner.py` - Model runner with `torch.device("spyre")`
- `spyre_inference/v1/attention/backends/spyre_attn.py` - PyTorch-native attention with paged KV cache

**Custom Operations (OOT - Out-of-Tree):**

- `spyre_inference/custom_ops/` - Device-specific layer implementations:
    - `linear.py` - `SpyreQKVParallelLinear`, `SpyreRowParallelLinear`
    - `rms_norm.py`, `rotary_embedding.py`, `silu_and_mul.py`, `vocab_parallel_embedding.py`, `parallel_lm_head.py`

**Attention Implementation:**

- Uses transposed matmul kernel (`_attn_transposed`) for Spyre execution
- KV cache alignment: 256 tokens (avoids per-step recompilation)
- Query chunking: 32 tokens per chunk for consistent tensor sizes
- Block-diagonal masking for grouped-query attention

## Development Commands

```bash
# Install dependencies (requires sendnn availability for torch-spyre compilation)
uv sync --frozen

# Development install with test dependencies
uv sync --group dev

# Run all tests (cached in ~/.cache/vllm-upstream-tests)
uv run pytest

# Run distributed tests (cached in ~/.cache/vllm-upstream-tests)
uv run pytest -m distributed

# Run upstream vLLM tests (cached in ~/.cache/vllm-upstream-tests)
uv run pytest -m upstream

# Run specific test file
uv run pytest tests/test_spyre_attn.py

# Run a single parametrized test (parametrize IDs contain (), =, , which break pytest -k —
# list node IDs first, then quote the full node ID)
uv run pytest tests/test_spyre_attn.py -m "not upstream" --collect-only -q
uv run pytest -m "not upstream" 'tests/test_spyre_attn.py::test_spyre_attn[<id>]'

# Format code (uses prek via uvx)
bash format.sh

# Type checking
uv run ty
```

### Test Markers

- `spyre` - Tests defined in this repo
- `upstream` - vLLM upstream compatibility tests

When running a subset, prefer `-m "not upstream"` unless you specifically want upstream tests — broad selectors can otherwise match tests pulled in by `spyre-testing-plugin`. Tests that need real Spyre hardware are gated by the `requires_spyre` fixture and skip silently on CPU-only hosts; "all green" on a non-Spyre machine does not mean the change works.

### Test Layout

- `tests/test_spyre_attn.py` — attention backend (`SpyreAttentionImpl`, `SpyreAttentionMetadataBuilder`) vs a CPU reference (`ref_attn`).
- `tests/test_mlp.py`, `tests/test_rms_norm.py`, `tests/test_silu_and_mul.py`, `tests/test_parallel_lm_head.py` — per-custom-op tests, each comparing a Spyre-device run against a CPU reference.
- `tests/test_vllm_spyre_next.py` — end-to-end vLLM path.

Test runs are slow (~3 min) because of vLLM startup; prefer single-test invocations during iteration.

### Upstream Test Configuration

The pytest plugin is packaged separately as `spyre-testing-plugin` in `tests/plugin/`.
This keeps test infrastructure out of the production package.

**Plugin package structure:**

- `tests/plugin/` - Separate pytest plugin package
    - `spyre_testing_plugin/pytest_plugin.py` - Main plugin with pytest hooks
    - `spyre_testing_plugin/models.py` - Data models for YAML config
    - `spyre_testing_plugin/upstream_tests.yaml` - Test filter configuration
    - `spyre_testing_plugin/sync_upstream_test_deps.py` - Script to sync vLLM test dependencies

**Environment variables:**

- `SKIP_UPSTREAM_TESTS=1` - Skip upstream tests
- `VLLM_COMMIT=<sha>` - Override vLLM commit
- `UPSTREAM_TESTS_PATHS=models/language/generation` - Paths to sync from vLLM

**Syncing upstream test dependencies:**

Whenever vLLM is updated, the dependencies of the test plugin need to be updated as well.

```bash
# From the workspace root
uv run sync-upstream-test-deps

# Or run as a module
python -m spyre_testing_plugin.sync_upstream_test_deps
```

## Build Configuration

**pyproject.toml key settings:**

- `extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "empty" } }` - Empty backend for vLLM (no C kernels)
- `tool.uv.sources` - Pulls vllm and torch-spyre from GitHub (not PyPI)
- `[[tool.uv.index]]` - PyTorch CPU index for torch/torchvision

## Iterating on a Local `torch-spyre` Checkout

When editing `./torch-spyre/` (or any sibling checkout) and reinstalling via `uv pip install --no-deps --force-reinstall ./torch-spyre`, the next plain `uv run pytest …` will silently revert the install. `uv run` re-syncs deps from `pyproject.toml` on every invocation, and `pyproject.toml` pins `torch-spyre` to an upstream git rev — so each `uv run` reinstalls the upstream commit on top of the local-source install. Symptom: the installed file has your changes, but the test fails as if nothing happened (look for `Uninstalled 1 package … Installed 1 package …` near the top of pytest output).

**Always use `uv run --no-sync …`** for any iteration cycle that depends on a hand-installed local dependency. The `torch-spyre` wheel build is C++-heavy and takes ~50s, so batch source edits before each rebuild.

## Spyre-Specific Constraints

- **Device alignment**: Head size must be multiple of 64 (128-byte stick size / 2 bytes for float16)
- **Tensor parallelism**: TP≥1 supported (custom linear layers handle weight sharding + `all_reduce` via `SpyreCommunicator` — see `docs/architecture/index.md` "Distributed (TP)"). **DP>1 is rejected** in `TorchSpyrePlatform.check_and_update_config`; the spyre-comms global rank space hasn't been validated for DP×TP.
- **dtype**: float16 only (model_config.dtype check in platform.py)
- **Compilation**: Platform-level compile is set to `CompilationMode.NONE` due to CPU fallback ops creating intermediates. **Caveat**: under the pytest `default_vllm_config` fixture, `cfg.mode` is Python `None` (not the `NONE=0` enum), so per-module gates like `_maybe_compile` in `spyre_attn.py` may still wrap kernels with `torch.compile(..., dynamic=False)`. Don't assume "eager in tests" when comparing pytest behavior to a plain script.
- **Single accelerator**: Spyre is contested by one process at a time. Never run two Spyre-backed commands concurrently — no `pytest -n`/`xdist`, no parallel `uv run pytest` invocations, no backgrounding one Spyre test while starting another. Parallel invocations hang, produce undefined device state, or corrupt the compile cache.

## Spyre Knowledgebase

  Before searching code for questions about Spyre architecture, PyTorch integration, vLLM stack, or hardware interfaces, query the spyre-knowledgebase MCP server first. It contains curated documentation on:

- Hardware microarchitecture and execution models
- Upstream components (PyTorch, vLLM, Triton)
- Software stack layers (torch-spyre, deeptools, torch-runtime, flex-runtime)
- Interface contracts and end-to-end flows
- Tracked repository knowledge

  **Usage:**

  *Search the knowledgebase*

  ```txt
  mcp__spyre-knowledgebase__search(query="your question", scope="wiki")
  ```

  *Read specific pages*

  ```txt
  mcp__spyre-knowledgebase__read(path="wiki/stack/torch-spyre.md")
  ```

  *Get the wiki schema/structure*

  ```txt
  mcp__spyre-knowledgebase__guide()
  ```

  **When to use:**

- Checking hardware constraints or format requirements
- Finding upstream component behavior

  **When to skip:**

- Debugging a specific code bug (go straight to code)
- Questions about this repo's test layout or development commands
- Looking for this repo's specific implementation details

  Treat the knowledgebase as the authoritative map; verify against live code before making changes.

## Debugging

When a test fails with numerical mismatch, a compile error on `spyre`, or a silent CPU fallback, invoke the `debug-spyre` skill (`.claude/skills/debug-spyre/SKILL.md`). It encodes the cluster-of-failures workflow, hypothesis-queue protocol, torch-spyre site-packages tracing, and escalation criteria. Attention-specific notes live in `.claude/skills/debug-spyre/attention-notes.md`.

The single most important signal: `FallbackWarning`. `torch-spyre` silently routes unsupported ops to CPU and emits this warning. A fallback changes the numerical path (CPU vs Spyre matmul kernels differ in accumulation order for fp16) and can mask the real bug. Turn it into an error with `-W "error::torch_spyre.ops.fallbacks.FallbackWarning"` to get a traceback to the triggering line.

Most "Spyre is broken" bugs are not in our code — they are torch-spyre op gaps, dtype/layout limitations, or shape-bucket misses. Read the relevant `.venv/lib/python3.12/site-packages/torch_spyre/ops/{eager,fallbacks}.py` files before assuming a local bug.

## Researching torch-spyre Issues and PRs

When exploring a new feature area (e.g., FP8 quantization, new operators, hardware support), **survey the torch-spyre repo for existing issues and PRs first**. This avoids duplicating work and surfaces known blockers.

**How to search:**

1. GitHub issue/PR search: `https://github.com/torch-spyre/torch-spyre/issues?q=<keyword>` and `/pulls?q=<keyword>`
2. Search terms: Use both short and long forms (e.g., `FP8`, `float8`, `scaled_mm`, `_scaled_mm`)
3. Check both open and closed/merged items — merged PRs show what's landed, closed issues may explain why something was deferred

## Code Style

- **Line length**: 100 characters (ruff)
- **Type hints**: Ignore `possibly-missing-attribute` (ty)
- **Excluded from ty**: `spyre_inference/__init__.py`
- **Codespell**: Ignore list includes `dout, te, indicies, subtile, ElementE`
- **Minimize comments**: Code should be self-explanatory; add comments only for global context or non-obvious reasoning
- **Avoid trivial helpers**: Don't create 1-2 LOC functions used only once
- **Match existing style**: Follow established code patterns and architectural conventions
- **Prefer simplicity**: When uncertain, choose the simpler, more concise implementation
- **Assume vLLM familiarity**: The reader may not be an expert on the specific code being read, but should have general experience with vLLM
