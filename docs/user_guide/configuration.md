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

Every Gemma-4 repository carries a `vision_config`, so `google/gemma-4-31B`,
`google/gemma-4-26B-A4B`, `google/gemma-4-E2B` and `google/gemma-4-E4B` load as
`Gemma4ForConditionalGeneration` and build a vision tower — weights to load and graphs to
warm up that a text-only workload never runs.

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

## Gemma-4: per-request image sizes

The vision tower is compiled per token budget at warmup, for the image size the engine runs
at: `max_soft_tokens` from `--mm-processor-kwargs`, default 280. A request can ask for a
larger size (560 or 1120) in its own `mm_processor_kwargs`: offline through `LLM.chat` or
`LLM.generate`, and online only when the server runs with `--trust-request-mm-kwargs`.

Each larger size needs its own budget, compiled at warmup, so they are opt-in through
`encoder_cudagraph_token_budgets`. To serve requests up to 1120:

```bash
vllm serve google/gemma-4-26B-A4B-it --tensor-parallel-size 4 \
  --trust-request-mm-kwargs \
  -cc '{"encoder_cudagraph_token_budgets": [64, 128, 256, 280, 576, 1152]}'
```

The list replaces the default budgets, so keep the small ones in it: 64 to 280 cover the
engine's own 280, 576 covers 560 and 1152 covers 1120. Each budget is rounded up to a whole
patch group. An image larger than the largest entry, as listed, is rejected with a 400
(`VLLMValidationError` offline) rather than compiled mid-request, and the encoder cache is
raised to hold one at that size. An entry above `max_num_batched_tokens` (512 unless
`--max-num-batched-tokens` sets it) needs chunked MM input, so `--disable-chunked-mm-input`
with it fails at startup.
Raising the engine-wide `--mm-processor-kwargs '{"max_soft_tokens": 560}'` instead makes
every image that size.

## Decoder compile buckets

The body pads the packed token count to the next `compile_sizes` bucket, and warmup
dummies every bucket. The lm_head sits outside every body graph and compiles its own, so
it needs the same treatment: it projects one row per *sampled* request, a width that
would otherwise take every value in `1..--max-num-seqs` as requests finish. Those rows
pad onto *row buckets*: the `compile_sizes` buckets capped at `--max-num-seqs` times one
plus the number of speculative tokens, for example `{1, 2, 4, 8}` at `--max-num-seqs 8`.
Warmup projects each width, so no shape reaches the lm_head uncompiled. Pad rows are dropped before sampling.

The sampled rows are gathered on Spyre, so only the logits are copied to the host.
Warmup compiles that gather, and the lm_head on its output, for every row bucket at
every body bucket it fits in. The lm_head then sees a different device layout than the
host path gives it, so logits can differ at fp16 rounding level. The host gather is used without
buckets (`--enforce-eager` or an empty `compile_sizes`), for prompt-logprob rows, for
auxiliary hidden-state outputs, and in the V2 model runner.

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
Those counts assume the rectangular path is opaque; traced into the block graph (see
below), each rectangle is a block graph of its own rather than an attention kernel.

The choice is made once per step from the step's metadata, before the forward, and the
runner counts it in `spyre_encoder_stats` (`rect_steps` / `ragged_steps`). The
ragged path always runs behind the opaque attention op. When compiled and the head size
fills whole 64-element sticks, the rectangular path is traced into the block graph
instead, so each rectangle is its own block graph. In eager mode both paths stay
opaque, with no block graph; a compiled, unpadded sub-stick head size also keeps
both paths opaque, sharing one block graph. Pooling models such as granite-30m
pad their native 32-wide heads to 64 before construction.

Compiled pooling warmup runs one dummy at the body shape. With the rectangular path
opaque, the first attention call in it traces every declared rectangle and group pair,
against that call's own tensors (a Spyre tensor's device layout is part of its cache
key). With it traced in, warmup adds one dummy per rectangle, plus one forced onto the
ragged path (when any group is reachable) that traces the group pairs. Eager pooling
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
compile at warmup. Full-attention layers have no KV-length buckets (see below); their query and
num-sequences buckets each add recorded variants, so keep those lists short.

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

The default prefill bucket is 512 tokens, but larger ones compile. For prompts of many
thousands of tokens, a 1024-token prefill chunk shortens time to first token, on dense
models such as granite-3.3-8b as well as on MoE models such as gemma-4, where each
prefill step reads every expert's weights however many tokens it carries. Pass a larger
`--max-num-batched-tokens` to get one; the default buckets then keep `512` too, so short
prefills don't pad up to the larger one:

```bash
vllm serve google/gemma-4-26B-A4B -tp 4 --max-model-len 32768 \
    --max-num-batched-tokens 1024 \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
```

Warmup takes longer, and for prompts under a few thousand tokens it gains little; decode
can get slightly slower. The example serves text only: a larger token budget also raises
the number of image-batch widths the vision tower is warmed at, which can run out of
device memory (#1302). At a long `--max-model-len`, a 2048-token bucket can push the
largest attention variants past torch-spyre's 60 s backend compile limit at warmup, so
they are left to compile on first use (torch-spyre#5250 makes that limit configurable).

**Attention (query length and batch size).** Set the buckets as comma-separated lists.
Each is clamped to its limit: entries above `--max-num-batched-tokens` (query) or
`--max-num-seqs` (num-sequences) are dropped, and the limit is appended if missing, so every
schedulable length keeps a bucket.

```bash
export SPYRE_ATTN_QUERY_BUCKETS=1,512         # 1 = decode; 512 = prefill chunk
export SPYRE_ATTN_NUM_SEQS_BUCKETS=1,4,8      # default: powers of two up to --max-num-seqs
```

Warmup records one per-sequence kernel per query bucket and one batched-decode kernel per
num-sequences bucket.

### One per-sequence kernel for every page count

The per-sequence kernels (`page_attn_prefill`, which also runs any decode that batched decode
does not take) do not bucket the KV length. Warmup compiles one kernel per query bucket, and that kernel runs any
sequence length up to `--max-model-len`.

Prefill sequences are padded to a multiple of 8 pages, so the prefill kernel can read 8 pages
per loop step at every sequence length. When `--max-model-len` is too short for that, prefill
keeps its own page count, but covers at least two pages. Decode sequences keep their own page
count.

The page tables and mask stacks are copied to the device with torch-spyre's
`tensor.to("spyre", dynamic=...)`. For this, the plugin sets torch-spyre's
`MIN_RESERVED_DIM_SIZE` to 1 and raises its `MAX_BUCKETS` cap to at least
`--max-model-len / block_size` rounded up to a multiple of 8, whether or not you export them.

A `--max-model-len` of one page has no page range: every sequence runs that one page.

Sliding-window layers have no page range. Each sequence pads to the power of two at or above
its page count, capped at the pages its window touches, with one kernel per (page count,
query) bucket pair.

The batched decode kernel takes its chunk count at run time in the same way, and under the
same conditions. The kernel reads the KV cache one *chunk* per loop step: the next
`32 / num_seqs` pages (at least one) of every sequence in the batch, so that each of the 32
cores reads one page. For example, a batch of 8 sequences reads 4 pages of each per chunk,
and if its longest sequence has 256 pages, the batch runs 64 chunks. Shorter sequences are
masked past their own length. The same kernel runs a batch that fits in one chunk, so warmup
compiles one batched kernel per num-sequences bucket. A num-sequences bucket whose
`--max-model-len` fits in one chunk, or any bucket under the conditions above, pads every
batch to the page count of `--max-model-len` instead. A sliding-window layer pads its
batches to their power-of-two page count, capped at the pages a one-token window touches.

#### One batched decode kernel for every batch size

Batched decode of up to 64 sequences (or up to `--max-num-seqs`, if that is lower) does not
compile kernels per num-sequences bucket.

Each chunk of these kernels reads 64 pages, two per core. Each of these reads is a *block
slot*. A layer group whose per-card KV heads times head size is too wide for two pages per
core (for example 16 KV heads of 128 at TP1) reads 32, one per core, and serves up to 32
sequences. The per-bucket kernels above give every sequence the same number of block slots, 32
divided by the bucket. These kernels share their block slots by length instead, at every step:
a long sequence gets many, a short one gets few. So a batch runs only as many chunks as all
its pages need together, not as many as its longest sequence needs.

For example, take one sequence of 235 pages and four of 5 pages. The kernel for the bucket of
8 sequences gives each sequence 4 block slots, so the batch runs 59 chunks, most of them
reading nothing but padding. Shared by length over 32 block slots, the long sequence gets 27
and each short one gets 1, and the batch runs 9 chunks; over 64, it runs 5.

Warmup compiles a single such kernel for all batch sizes up to 64, instead of one per
num-sequences bucket. It runs every chunk count, from a batch that fits in a single chunk up to
every sequence at `--max-model-len`.

Batches of more than 64 sequences keep their per-bucket kernels, and sliding-window layers keep
their KV buckets. These kernels need the same torch-spyre as the dynamic chunk count above.

## KV cache size

The KV cache is sized by vLLM's own GPU worker code: after the weights load, it reads the
card's free memory (`torch.accelerator.get_memory_info`) and gives the KV cache what is
left of `--gpu-memory-utilization` (default 0.92) × total. The cache keeps every block
that memory pays for, usually far more than `--max-num-seqs` requests at `--max-model-len`
need; with prefix caching on, the spare blocks keep earlier prompts' KV for reuse.

Spyre turns vLLM's hybrid KV cache manager off by default, so every layer allocates its own
page per block and the budget is converted into blocks at the device cost of all layers' pages.
`--no-disable-hybrid-kv-cache-manager` turns it on: a model that mixes
sliding-window and full-attention layers (Gemma-3, Gemma-4) then gets one KV cache group per
kind, sliding-window groups return blocks that fall out of the window, and the groups overlay
one another as upstream lays them out, each layer reading the shared pages in its own shape, so
a block costs what vLLM's pool counts. Either way the pages are
laid out layer-outer, so `VLLM_KV_CACHE_LAYOUT` accepts only `LBNHC` (the default) or `LBHNC`,
and a block-outer value fails at startup. Every K and V page allocation is one tensor, and the
Spyre allocator never splits a tensor across its 16 GiB memory regions, so a model with few
layers gets fewer blocks than its budget: they must still pack into the regions. When memory
runs out before a full batch fits, a warning names the knobs, and requests wait for pages
instead of all running at full length.

The profile run is skipped, as the first forward on Spyre would trace the model before
the KV cache exists: a forward's activations come out of the `1 - --gpu-memory-utilization`
headroom. Lower it if a forward fails to allocate. `--kv-cache-memory-bytes` skips the
measurement; `--num-gpu-blocks-override` replaces the block count it yields, so the
warning above checks the override instead. `VLLM_CPU_KVCACHE_SPACE`, which sets the
KV cache size on vLLM's CPU backend, is ignored, with a warning when it is set to a size;
setting it to a size next to `--kv-cache-memory-bytes` is an error.

## FP8 KV cache

An FP8 model (FP8 linear weights: the `fp8` method, or a compressed-tensors checkpoint
with 8-bit float weights) stores its KV cache in float8 (e4m3) by default, halving the
bytes each cached token takes. Any other model opts in with `--kv-cache-dtype fp8`. An FP8
model opts out with its checkpoint's 16-bit dtype, `float16` or `bfloat16`; either gives a
float16 cache, the only 16-bit cache Spyre has:

```bash
vllm serve ibm-granite/granite-4.1-8b-fp8                             # FP8 KV cache
vllm serve ibm-granite/granite-4.1-8b-fp8 --kv-cache-dtype bfloat16   # float16 cache
```

It needs a head size that is a multiple of 128; with a smaller one an FP8 model keeps a
model-dtype cache, and an explicit `--kv-cache-dtype fp8` is rejected. Other quantized cache dtypes (`fp8_e5m2`, `fp8_inc`, `nvfp4`,
...) are rejected at startup. Since the KV cache is sized from free device memory (see
[KV cache size](#kv-cache-size)), an fp8 cache fits twice the tokens of an fp16 one.
Attention still computes in the model dtype: pages are dequantized on read with the
checkpoint's per-layer KV scales, or 1.0 when it has none. The speedup grows with the number of cached tokens each decode step
reads, so it is small at short contexts; check accuracy against an fp16 cache before
relying on it.

The scales are compiled into the graphs as constants. A checkpoint whose layers carry
different KV scales therefore compiles the transformer block and each recorded attention
kernel once per distinct pair of scales, which lengthens warmup; one without scales
compiles them once, as an fp16 cache does.

## Model runner

The plugin runs vLLM's Model Runner V2 (MRV2), its only model runner: the V1 runner was
removed (#1245), and `SPYRE_MODEL_RUNNER=v1` is refused at startup. MRV2's Triton
kernels are replaced by host kernels built into the plugin (`SPYRE_MRV2_KERNELS=host`), so
no Triton install is needed on any architecture. The plugin pins
`VLLM_USE_V2_MODEL_RUNNER=1`, so don't set it yourself.

Differences from the removed V1 runner:

- Seeded sampling draws a different token sequence for the same seed, because MRV2
  seeds each position rather than one generator per request. Greedy output is unchanged.
- These are refused at startup: speculative decoding other than DFlash/XPress,
  batch-sharded sampling, `cudagraph_mm_encoder`, prefill context parallelism, and,
  untested on Spyre, HiSparse, watermarking and AuxOutput (e.g. returning routed experts).

## Offline engine in the calling process

At `tensor_parallel_size=1`, set `VLLM_ENABLE_V1_MULTIPROCESSING=0` to run the offline
`LLM` engine in the calling process instead of a spawned EngineCore, which skips the
spawn and the second round of imports at startup. `vllm serve` always spawns EngineCore.

The engine then changes the calling process: vLLM freezes the garbage collector's heap,
caches its environment variables, seeds the global RNGs and leaves `RANK`/`WORLD_SIZE`
set, the worker sets torch's thread count, and the card stays open until the process
exits. Build an engine with `tensor_parallel_size > 1` in a new process. Set
`OMP_WAIT_POLICY=PASSIVE` before Python starts: OpenMP reads it when it loads, so the
default the plugin sets only reaches spawned processes.

## pyproject.toml Reference

The `pyproject.toml` includes several key build configurations:

### Build Configuration

```toml
[tool.uv.extra-build-variables.vllm]
VLLM_TARGET_DEVICE = "empty"
CMAKE_ARGS = "--fresh"

[tool.uv.extra-build-dependencies]
torch-spyre = ["torch==2.14.1"]
```

These settings ensure:

- torch-spyre is built against the same PyTorch version (2.14.1) the runtime pins, so its
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
