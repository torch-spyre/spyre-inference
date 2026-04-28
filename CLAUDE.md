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
  - `linear.py` - `SpyreMergedColumnParallelLinear`, `SpyreQKVParallelLinear`, `SpyreRowParallelLinear`
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

# Run Spyre device tests
uv run pytest -m spyre

# Run upstream vLLM tests (cached in ~/.cache/vllm-upstream-tests)
uv run pytest -m upstream

# Run specific test file
uv run pytest tests/test_spyre_attn.py

# Run single test
uv run pytest tests/test_spyre_attn.py::test_spyre_attn_single_sequence

# Format code (uses prek via uvx)
bash format.sh

# Type checking
uv run ty
```

### Test Markers

- `spyre` - Tests defined in this repo
- `upstream` - vLLM upstream compatibility tests

### Upstream Test Configuration

Environment variables for `tests/conftest.py`:
- `SKIP_UPSTREAM_TESTS=1` - Skip upstream tests
- `VLLM_COMMIT=<sha>` - Override vLLM commit
- `UPSTREAM_TESTS_PATHS=models/language/generation` - Paths to sync from vLLM

## Build Configuration

**pyproject.toml key settings:**
- `build-constraint-dependencies = ["torch==2.10.0"]` - Ensures consistent torch version
- `extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "cpu" } }` - CPU backend for vLLM
- `tool.uv.sources` - Pulls vllm and torch-spyre from GitHub (not PyPI)
- `[[tool.uv.index]]` - PyTorch CPU index for torch/torchvision

## Spyre-Specific Constraints

- **Device alignment**: Head size must be multiple of 64 (128-byte stick size / 2 bytes for float16)
- **No tensor parallelism**: Custom linear layers assume TP=1
- **dtype**: float16 only (model_config.dtype check in platform.py)
- **Compilation**: Disabled (`CompilationMode.NONE`) due to CPU fallback ops creating intermediates

## Code Style

- **Line length**: 100 characters (ruff)
- **Type hints**: Ignore `possibly-missing-attribute` (ty)
- **Excluded from ty**: `spyre_inference/__init__.py`
- **Codespell**: Ignore list includes `dout, te, indicies, subtile, ElementE`
