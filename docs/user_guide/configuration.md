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

## Encoder / pooling compile buckets

Spyre compile is on by default (`STOCK_TORCH_COMPILE`, `dynamic=False`). Pass
`--enforce-eager` to disable it. Each new encoder shape is a new graph (~60s):
SDPA specializes `[B, H, L, D]`, and Linear / LN specialize the flat token
count `[T, …]`.

At runtime the plugin pads **each sequence** to the next length bucket `L` and
the **batch** to `B`, so `T = B × L`. Attention still masks to the real prompt
lengths; pooling gathers the real tokens before CLS / LAST / MEAN.

| Env | Default | Meaning |
|---|---|---|
| `SPYRE_ENCODER_BUCKET_LENS` | `64,128,256,512,1024,2048` | Prompt-length ladder (rounded up to a multiple of 64) |
| `SPYRE_ENCODER_BUCKET_BATCH_SIZES` | `1, 2, 4, …, max_num_seqs` | Batch ladder |

A 30-token, 3-seq request with `--max-num-seqs 4` is padded to `(B=4, L=64)`
(`T=256`) and reuses that warmed graph. If `(B, L)` would exceed
`--max-num-batched-tokens`, the batch is left unpadded (a new compile).

Compiled pooling warmup dummies each `(B, L)` three ways: full `B × L`, then
`L-2` and `L-1` padded onto `T = B × L`. (`--random-input-len L` subtracts
tokenizer specials, often 2.) Eager pooling (`--enforce-eager`) still uses one
16-token dummy.

Example:

```bash
SPYRE_ENCODER_BUCKET_LENS=64,256 \
SPYRE_ENCODER_BUCKET_BATCH_SIZES=1,4 \
vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 4
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
