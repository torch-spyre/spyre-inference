# Configuration

## Plugin Setup

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference
```

`spyre_inference` names both the platform plugin and the general plugin: the first
activates the platform, and the second registers the OOT custom ops and the Spyre
model adaptations, including the Spyre Transformers backend (used for
`model_impl="transformers"`).

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

## Gemma-4: text-only use of a vision checkpoint

Every Gemma-4 repository carries a `vision_config`, so `google/gemma-4-31B` and
`google/gemma-4-26B-A4B` load as `Gemma4ForConditionalGeneration` and build a vision
tower — weights to load and graphs to warm up that a text-only workload never runs.

To use one of those repositories for text only, pin its decoder architecture:

```python
llm = LLM(
    model="google/gemma-4-26B-A4B",
    hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
    tensor_parallel_size=2,
)
```

That is also the configuration the tensor-parallel and compile e2e tests run these
checkpoints under. A repository with no vision tower gets the override by default, so
this is only needed for the multimodal ones.

## Decoder compile buckets

The body pads the packed token count to the next `compile_sizes` bucket, and warmup
dummies every bucket. The lm_head sits outside every body graph and compiles its own, so
it needs the same treatment: it projects one row per *sampled* request, a width that
would otherwise take every value in `1..--max-num-seqs` as requests finish. Those rows
pad onto the same buckets clipped to `--max-num-seqs`, and warmup projects each width, so
no shape reaches the lm_head uncompiled. Pad rows are dropped before sampling.

## Encoder / pooling compile buckets

Spyre compile is on by default (`STOCK_TORCH_COMPILE`, `dynamic=False`). Pass
`--enforce-eager` to disable it.

Everything derives from one number, `R` — the token budget, which is the most rows a
step may run. It is `--max-num-batched-tokens`, capped at 2048 (the measured throughput
argmax across pooling models), floored at `--max-model-len` rounded up to a multiple of
64, and capped again at what `--max-num-seqs` sequences of that length could carry, then
floored to a whole multiple of that longest length. `--max-num-seqs` is then lowered to
`R / 64` if it was higher, since no batch wider than that fits.

- **Body** (Linear / LN): every multiple of 64 from one stick up to `R`. A step pads
  to the next entry that holds its real tokens. Four prompts totalling 1650 tokens
  run 1664 rows, not `R`.
- **Lengths**: every multiple of 64 from one stick up to `--max-model-len`, rounded up
  to a whole stick. Attention pads each sequence to that length on its own; the other
  sequences in the batch do not inherit it. `SPYRE_ATTN_QUERY_BUCKETS` overrides the
  ladder, rounded the same way.
- **Attention**: Q/K/V stay packed. Requests are split into *groups* — the requests
  sharing one padded length. Each group is one fused gather/attend/scatter, staged
  on a fixed buffer so the kernel does not recompile for every body size. Widths are
  powers of two up to the cap; a wider group is chunked into descending powers of two.

The choice is made once per step from the step's metadata, before the forward, and the
runner counts it in `spyre_encoder_stats` (`rect_steps` / `ragged_steps`). The
ragged path always runs behind the opaque attention op. When compiled and the head size
fills whole 64-element sticks, the rectangular path is traced into the block graph
instead, so each rectangle is its own block graph. In eager mode both paths stay
opaque, with no block graph; a compiled, unpadded sub-stick head size also keeps
both paths opaque, sharing one block graph. Pooling models such as granite-30m
pad their native 32-wide heads to 64 before construction.

Compiled pooling warmup runs one dummy per body bucket. The first dummy traces the
group kernels on the fixed staging buffer; the rest compile the GEMMs at that row
count. Eager pooling uses one short dummy and always takes the packed path.

Example:

```bash
vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 4 --max-model-len 512
```

## Tuning buckets for padding

Bucketing trades warmup time for per-request padding. A request is padded up to the next
bucket on each axis and the padding is masked out, so buckets far above your real shapes
waste compute, while buckets that hug your workload cut that waste but add graphs to
compile at warmup. Per-sequence attention pairs each query bucket only with KV buckets
that fit its smallest real query length. Batched decode uses KV-length/num-sequences
pairs; both recording paths also enforce the allocated page-capacity limit. Extra
buckets can still multiply the recorded variants, so keep those lists short.

**Decoder body (packed token count).** Override the defaults with `compile_sizes`; the
platform clamps `--max-num-batched-tokens` to the largest entry. A decode-heavy run at
`--max-num-seqs 8` rarely needs the full power-of-two ladder:

```python
from vllm import LLM

llm = LLM(
    model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    max_num_seqs=8,
    max_model_len=2048,
    compilation_config={"compile_sizes": [1, 8, 512]},
)
```

`1` and `8` cover decode steps (one token per running sequence, up to 8); `512` is the
prefill bucket.

**Attention (KV length × query length).** Set the buckets directly as comma-separated
lists. Each is clamped to its limit: entries above `--max-model-len` (KV) or
`--max-num-batched-tokens` (query) are dropped, and the limit is appended if missing, so
every schedulable length keeps a bucket.

```bash
export SPYRE_ATTN_KV_BUCKETS=256,1024,2048    # default: powers of two from block_size
export SPYRE_ATTN_QUERY_BUCKETS=1,512         # 1 = decode; 512 = prefill chunk
```

The default KV buckets are geometric (powers of two) because the recorded set grows
with both axes, even after filtering. If your context never exceeds 2048, dropping the
higher powers removes variants from warmup at no serving cost.

With the batched-decode kernel enabled (`SPYRE_BATCHED_DECODE=1`, the default; under the
default tiled walk it is reached on the head-major layout only, and a token-major run keeps
the per-sequence loop), warmup records eligible KV-length × num-sequences combinations.
`SPYRE_ATTN_NUM_SEQS_BUCKETS` (default: powers of two from 1 to `--max-num-seqs`) is the
extra lever there, and the same keep-it-short advice applies.

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv.extra-build-variables.vllm]
VLLM_TARGET_DEVICE = "empty"
CMAKE_ARGS = "--fresh"

[tool.uv.extra-build-dependencies]
torch-spyre = ["torch==2.13.0"]
```

These settings ensure:

- torch-spyre is built against the same PyTorch version (2.13.0) the runtime pins, so its
  C++ extension cannot drift in ABI
- vLLM is built with the **empty** backend — no device-specific C kernels. This avoids
  the torch-version coupling of prebuilt CPU wheels and the dependency on `vllm._C`
  (whose CPU-optimized ops we don't need; Spyre provides its own)

### Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
```

This ensures that torch-spyre and vllm are compiled/installed from source, instead of pulling pre-compiled wheels from PyPI.

### PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This ensures the CPU flavor of PyTorch is installed, as CUDA support is not required.
