# Configuration

## Plugin Setup

To load the plugin, set the `VLLM_PLUGINS` environment variable before running vLLM:

```bash
export VLLM_PLUGINS=spyre_inference,spyre_inference_ops
```

`spyre_inference` activates the platform, and `spyre_inference_ops` registers the OOT
custom ops plus the Spyre Transformers backend (used for `model_impl="transformers"`).

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

Everything derives from one number, `R` — the token budget. It is
`--max-num-batched-tokens`, capped at 2048 (the measured throughput argmax across
pooling models), floored at `--max-model-len` rounded up to a power-of-two multiple of
64, and capped again at what `--max-num-seqs` sequences of that length could carry, then
floored to a whole multiple of it so that every length divides `R`. `--max-num-seqs` is
then lowered to `R / 64` if it was higher, since no batch wider than that fits.

- **Body** (Linear / LN): one shape, `R` rows. Every pooling step pads to it.
  Fixing it is what keeps the attention kernels keyed on sequence shapes alone.
- **Lengths**: powers of two from 64 (one Spyre stick) up to `--max-model-len`, rounded
  up to a power-of-two multiple of 64. Every length is then `64 * 2^k`, so each divides
  `R` and each rectangle covers the body exactly. `SPYRE_ATTN_QUERY_BUCKETS` overrides the
  ladder, rounded the same way.
- **Attention, rectangular path**: for each length `L`, one rectangle `B = R / L`. The
  runner pads every sequence to `L` and the batch to `B`, so Q/K/V *are* the grid:
  one reshape, one `F.scaled_dot_product_attention`, one store, no data movement
  inside the layer. Taken whenever `num_seqs <= B`.
- **Attention, ragged path**: for a batch too wide for any rectangle, Q/K/V stay
  packed and requests are split into *groups* — the requests sharing one padded
  length. Each group is one fused gather/attend/scatter keyed on `(group width,
  extent)`, so a ragged step makes one kernel call per group rather than per
  request. Widths are powers of two up to `B`; a wider group is chunked into
  descending powers of two.

With `--max-model-len 512 --max-num-seqs 32 --max-num-batched-tokens 2048` that is
23 shapes: one body, four rectangles (`(64,32) (128,16) (256,8) (512,4)`, each
exactly 2048 rows), and 18 group pairs. At `--max-num-seqs 4` the group family is
empty — no batch that narrow can miss the rectangular path — leaving five shapes.

Both paths go through the same opaque attention op, so the block graph is identical
for either and the choice is made once per step from the step's metadata. The
runner counts them in `spyre_encoder_rect_steps` /
`spyre_encoder_ragged_steps`.

Compiled pooling warmup runs one dummy at the body shape; the first attention call
in it traces every declared rectangle and group pair, against that call's own
tensors (a Spyre tensor's device layout is part of its cache key). Eager pooling
uses one short dummy and always takes the packed path.

Example:

```bash
vllm serve ibm-granite/granite-embedding-125m-english \
  --runner pooling --max-num-seqs 4 --max-model-len 512
```

## Tuning buckets for padding

Bucketing trades warmup time for per-request padding. A request is padded up to the next
bucket on each axis and the padding is masked out, so buckets far above your real shapes
waste compute, while buckets that hug your workload cut that waste but add graphs to
compile at warmup. Attention is recorded as the **product** of its KV-length and
query-length buckets (and, when the batched-decode kernel is enabled, a second KV-length ×
num-sequences product), so extra attention buckets cost multiplicatively — keep those
lists short.

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

The default KV buckets are geometric (powers of two) precisely because the recorded set
is a product. If your context never exceeds 2048, dropping the higher powers removes
variants from warmup at no serving cost.

With the batched-decode kernel enabled (`SPYRE_BATCHED_DECODE=1`, the default; the
head-major layout has no batched kernel and ignores it), warmup also records it over the
KV-length × num-sequences grid. `SPYRE_ATTN_NUM_SEQS_BUCKETS`
(default: powers of two from 4 to `--max-num-seqs`) is the extra lever there, and the same
keep-it-short advice applies.

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.13.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "empty", CMAKE_ARGS = "--fresh" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.13.0)
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
