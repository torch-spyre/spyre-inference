# Configuration

## Plugin Setup

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference,spyre_inference_ops,spyre_inference_hf_adaptor
```

`spyre_inference` activates the platform, `spyre_inference_ops` registers the OOT custom
ops, and `spyre_inference_hf_adaptor` swaps in the hf-adapters Transformers backend
(needed for `model_impl="transformers"`).

## Usage

You can then use vLLM as usual:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_model_len=128,
    max_num_seqs=2,
)
```

See the [Examples](../examples/offline_inference/torch_spyre_inference.md) page for more usage patterns.

## Host sampler (async noise + log-space Gumbel)

Spyre replaces vLLM's default host sampler with a Spyre-optimized path
ported from [sendnn-inference#1046](https://github.com/torch-spyre/sendnn-inference/pull/1046)
(Holtz), in three stages:

1. **Async noise ring buffer** — Exp(1) log-noise is filled on a background
   thread; the decode loop borrows zero-copy rows instead of calling
   `exponential_()` on the critical path.
2. **TP rank-0 sampling** — when tensor-parallel and logits are on CPU, only
   rank 0 samples and broadcasts token ids (and logprobs) to the other ranks.
3. **Log-space Gumbel** — sample as `argmax(logits - log_noise)` instead of
   `argmax(probs / noise)`, which removes softmax from the hot path while
   preserving token order / distribution.

If `vllm_config` lacks `max_num_seqs` or vocab size, the runner falls back to
vLLM's default `Sampler` (same as sendnn). Sampling still runs on the **CPU**.

### Configuration (`spyre_inference.envs`)

All host-sampler knobs live in `spyre_inference/envs.py` (vLLM-style lazy
module with defaults, docs, optional `enable_envs_cache()`, and `is_set()` for
override detection). Prefer `import spyre_inference.envs as envs` over
scattered `os.environ.get` calls.

| Variable | Default | Meaning |
|---|---|---|
| `SPYRE_USE_SPYRE_SAMPLER` | `1` | Set to `0` to force upstream `Sampler` |
| `SPYRE_ASYNC_NOISE_SCALE` | `4` | Ring depth = scale × `max_num_seqs` (must be ≥ 2) |

Do **not** inject sampler timing into the production path. Measure with the
[Kineto / Spyre profiler](kineto_profiling.md) instead.

```bash
SPYRE_ASYNC_NOISE_SCALE=8 \
  python examples/offline_inference/torch_spyre_inference.py
```

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.11.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "empty", CMAKE_ARGS = "--fresh" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.11.0)
- vLLM is built with the **empty** backend — no device-specific C kernels. This avoids
  the torch-version coupling of prebuilt CPU wheels and the dependency on `vllm._C`
  (whose CPU-optimized ops we don't need; Spyre provides its own)

### Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
hf-adapters-spyre = { git = "https://github.com/torch-spyre/hf-adapters.git", rev = "..." }
```

This ensures that torch-spyre, hf-adapters, and vllm are compiled/installed from source, instead of pulling pre-compiled wheels from PyPI.

### PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This ensures the CPU flavor of PyTorch is installed, as CUDA support is not required.
