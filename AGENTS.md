# CLAUDE.md

## Project Overview

`spyre-inference` is a vLLM platform plugin that integrates with `torch-spyre` to leverage IBM's Spyre AI accelerator hardware. It provides PyTorch-native attention implementations and custom operations optimized for Spyre devices.

## Architecture

- Registers as the `spyre_inference` vLLM platform plugin via entry points (`spyre_inference:register`).
- `TorchSpyrePlatform` extends `CpuPlatform`, forcing `torch.float16`. `--enforce-eager` is the only compile switch: set → `CompilationMode.NONE`; default → `STOCK_TORCH_COMPILE` (whole model + attention).

**Key modules:**

- `spyre_inference/platform.py` — platform registration and config
- `spyre_inference/v1/worker/` — `spyre_worker.py` (device execution), `spyre_model_runner.py` (`torch.device("spyre")`)
- `spyre_inference/v1/attention/backends/spyre_attn.py` — PyTorch-native paged-KV attention
- `spyre_inference/custom_ops/` — device-specific layer implementations (linear, rms_norm, rotary_embedding, silu_and_mul, embeddings, lm_head)

**Attention notes:** transposed matmul kernel (`_attn_transposed`), KV cache aligned to 256 tokens (avoids per-step recompile), query chunked at 32 tokens, block-diagonal masking for GQA.

## Development Commands

```bash
uv sync --group dev            # dev install with test deps
uv run pytest                  # local tests only
uv run pytest -m upstream      # upstream vLLM compat tests (cached in ~/.cache/vllm-upstream-tests)
uv run pytest --upstream       # local + upstream
bash format.sh                 # format (prek via uvx)
uv run ty                      # type check
```

Runs are slow (~3 min) due to vLLM startup — prefer single-test invocations while iterating. Parametrize IDs contain `()`, `=`, `,`, which break `pytest -k`; list node IDs first, then quote the full ID:

```bash
uv run pytest 'tests/attention/test_spyre_attn.py::test_spyre_attn[<id>]' -m "not upstream"
```

Upstream vLLM tests are opt-in: they are only cloned and collected when the `-m` expression names the `upstream` marker (a negative mention like `-m "not upstream"` doesn't count) or `--upstream` is passed, so a bare `pytest` won't pull them into a broad selector. Tests needing real hardware guard themselves with `spyre_available()` and skip on CPU-only hosts, so "all green" on a non-Spyre machine does **not** mean the change works.

Upstream test config lives in `spyre-testing-plugin` (`tests/plugin/`), which is loaded by the `addopts` entry in `pyproject.toml` — not a global `pytest11` entry point — so it stays inert in sibling repos sharing the venv (e.g. a local `torch-spyre` checkout). Env vars: `SKIP_UPSTREAM_TESTS=1` (hard off-switch, overrides `--upstream`), `VLLM_COMMIT=<sha>`, `UPSTREAM_TESTS_PATHS=<comma-separated paths>`. After a vLLM bump, resync deps with `uv run sync-upstream-test-deps`.

## Spyre-Specific Constraints

- **Head size** must be a multiple of 64 (128-byte stick / 2 bytes for fp16).
- **dtype**: float16 only (enforced in `platform.py`).
- **Tensor parallelism**: TP≥1 supported; `all_reduce` is provided natively by `libspyre_comms` (no longer overridden in `SpyreCommunicator`). **DP>1 is rejected** in `TorchSpyrePlatform.check_and_update_config` — the spyre-comms global rank space isn't validated for DP×TP.
- **Compilation**: default is compiled (`STOCK_TORCH_COMPILE`), not eager. Attention compiles in its own domain: `SpyreAttentionImpl` reads `get_current_vllm_config().compilation_config.mode` at construction, so building it outside a `set_current_vllm_config` context raises. Tests force `enforce_eager=True` where they want eager — don't assume "eager in tests." `STOCK_TORCH_COMPILE` compiles one transformer block at a time; `SPYRE_COMPILE_GRANULARITY=model` restores the whole-model graph.
- **Single accelerator**: the device is contested by one process at a time. Never run two Spyre-backed commands concurrently — no `pytest -n`/xdist, no parallel `uv run pytest`, no backgrounding one Spyre test while starting another. Parallel runs hang, corrupt device state, or corrupt the compile cache.

## Iterating on a Local `torch-spyre` Checkout

`uv run` re-syncs deps from `pyproject.toml` (which pins `torch-spyre` to a git rev) on every invocation, silently reverting a hand-installed local build. **Always use `uv run --no-sync …`** when iterating on a locally installed dependency. Symptom of forgetting: `Uninstalled 1 package … Installed 1 package …` near the top of pytest output, and your changes appear to have no effect. The wheel build takes ~50s, so batch source edits before each rebuild.

To run `uv sync` itself (e.g. to pick up other dependency changes) without touching an already-installed local `torch-spyre`, add `--no-install-package torch-spyre --inexact`: the first flag skips resolving/rebuilding the pinned git rev, the second stops uv from uninstalling the now-unreferenced local package.

## Debugging

Invoke the `debug-spyre` skill for numerical mismatches, `spyre` compile errors, or silent CPU fallbacks.

The single most important signal is `FallbackWarning`: torch-spyre silently routes unsupported ops to CPU, which changes the numerical path (fp16 accumulation order differs) and can mask the real bug. Turn it into an error to get a traceback: `-W "error::torch_spyre.ops.fallbacks.FallbackWarning"`. Most "Spyre is broken" bugs are torch-spyre op gaps or dtype/layout limits, not local bugs — read `site-packages/torch_spyre/ops/{eager,fallbacks}.py` before assuming otherwise. When exploring a new feature area, search torch-spyre issues/PRs (short and long forms, e.g. `FP8`/`float8`/`_scaled_mm`) before starting.

## Spyre Knowledgebase

For questions about Spyre architecture, the PyTorch/vLLM stack, or hardware interfaces, query the `spyre-knowledgebase` MCP server before searching code — treat it as the authoritative map, then verify against live code. Skip it for local code bugs and questions about this repo's own tests or implementation.

## Code Style

- **Minimize comments** — add them only for global context or non-obvious reasoning.
- **Avoid trivial helpers** — don't wrap 1-2 LOC used once.
- **Prefer simplicity** and match existing patterns/architecture when uncertain.
- Assume general vLLM familiarity in the reader.

## Opening a PR

Always use the `prepare-pull-request` skill to clean up the current branch and open a pull request.

## DCO (Developer Certificate of Origin)

All commits must include a `Signed-off-by` line. Use the `-s` flag or append manually:

```bash
# With git commit
git commit -s -m "Your commit message"

# Or amend an existing commit
git commit --amend -s --no-edit
```

The sign-off email must match your GitHub email. If they do not match, DCO checks will fail.
For help configuring git with your GitHub identity, invoke the `github-commit` skill.
