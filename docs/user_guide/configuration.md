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
Those counts assume the rectangular path is opaque; traced into the block graph (see
below), each rectangle is a block graph of its own rather than an attention kernel.

The choice is made once per step from the step's metadata, before the forward, and the
runner counts it in `spyre_encoder_rect_steps` / `spyre_encoder_ragged_steps`. The
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
export SPYRE_ATTN_KV_BUCKETS=256,1024,2048    # default: powers of two unioned with an 8/5 series
export SPYRE_ATTN_QUERY_BUCKETS=1,512         # 1 = decode; 512 = prefill chunk
```

The default KV buckets union the powers of two with an 8/5 geometric series on a 64-token
grain. Keeping every power of two is what makes the union safe: a KV length can only round
down relative to a pure power-of-two ladder, never up. The ladder stays geometric rather
than uniform because the recorded set grows with both the KV and num-sequences axes, even
after filtering, so each extra entry costs warmup time. If your context never exceeds 2048,
dropping the higher entries removes variants from warmup at no serving cost.

### Choosing a KV ladder

`SPYRE_ATTN_KV_LADDER` picks how finely the default ladder subdivides, without having to
write the buckets out. Every preset keeps the powers of two, so a KV length's bucket can only
move down relative to `pow2`, never up.

| preset | spacing | buckets at 4096 / 16384 / 32768 | mean overpay vs the real KV length |
|---|---|---|---|
| `pow2` | powers of two only | 6 / 8 / 9 | 39% |
| `8_5` (default) | 8/5 ratio | 10 / 15 / 18 | 21% |
| `4_3` | 4/3 ratio | 14 / 20 / 24 | 15% |
| `9_8` | 9/8 ratio | 20 / 34 / 41 | 8% |
| `uniform` | evenly spaced above a knee | 13 / 15 / 16 | 14% |

`uniform`'s entries are placed as fractions of `--max-model-len`, so its count stays flat as the
context grows (8 buckets at 1024, 18 at 131072) where the ratio presets add them logarithmically.

Overpay is averaged over every KV length above `block_size`, so it describes a workload whose
lengths are spread out. Warmup cost tracks the number of *distinct recorded kernels*, which is
close to but not the same as the bucket count, and it rises faster than linearly as buckets are
added. The presets are not supersets of one another: each series lands on its own points, so a
denser preset can lack a bucket a coarser one has.

The first four presets add buckets at a constant *ratio*; `uniform` adds them at a constant
*number of tokens* above a knee. Since what a bucket saves is a token count, and a ratio ladder's
gaps widen in absolute terms as they climb, the two families spend the same warmup budget on
different parts of the range: `8_5` subdivides most finely at the bottom, where its ratio steps are
only tens of tokens apart, while `uniform` spaces its entries evenly and so subdivides above its
knee, where `8_5` leaves its widest gaps. Because the knee and step are fractions of
`--max-model-len`, raising the context moves `uniform`'s whole series up rather than lengthening
it. The mean overpay above does not order the presets by measured throughput — it weights every KV
length equally, whereas a real workload only visits a narrow band of them.

Three workload properties decide which preset pays off:

- **Context length.** The proportional overpay barely moves with `--max-model-len`, but the
  buckets do, so a longer context means more warmup for the same percentage. What grows is the
  benefit: the saving is a number of KV tokens per decode step, so the same percentage is worth
  far more at long context than at short.
- **Batch size.** A denser ladder shortens the attention part of a decode step, which is a small
  share of that step at one or two sequences and most of it at tens. Below about four sequences
  no preset is worth its warmup. A larger batch widens the gaps between presets rather than
  changing their order.
- **Batch uniformity.** A batched step is sized by the longest sequence in it, so a batch whose
  contexts differ widely pays for the longest one regardless of the ladder. No preset changes
  that. The two effects compound, so a ragged batch loses more tokens to bucket round-up than a
  uniform one and has more to gain from a denser preset.

Because batch size decides how much of the step a ladder can touch, it also decides whether a
preset's weakness shows up at all: a preset can look equivalent at a small batch and lose several
percent at a large one, on the same KV lengths.

#### How the presets compare to the default

Throughput relative to `8_5`, from serving runs on granite-3.3-8b across five workloads:

| KV lengths | batch | `pow2` | `4_3` | `9_8` | `uniform` |
|---|---|---|---|---|---|
| ~7k–10k, context 16384 | 4 | 0.87x | 1.03x | **1.05x** | **1.05x** |
| ~7k–10k, context 32768 | 4 | 0.89x | 1.03x | **1.06x** | 1.01x |
| ~2k–3k, context 4096 | 32 | 0.90x | 1.05x | **1.07x** | 1.07x |
| ~400–700, context 1024 | 4 | 1.00x | 1.01x | **1.02x** | 1.00x |
| ~400–700, context 4096 | 32 | 0.97x | 1.00x | **1.02x** | 1.00x |

`uniform`'s spacing was changed to a fraction of `--max-model-len` after the first of these runs,
and its rows were re-measured against the shipping form.

Two measurement caveats apply to every number on this page. **Repeated identical runs agree to
about 1%**, so differences below a few percent are not resolvable — the 1.00x entries above are
nulls, not narrow wins. And **the generated-token count is not stable between runs**: on one
32-request workload, 6 or 7 requests would nondeterministically either stop at the model's own
end-of-sequence or run on to the requested output length, moving the total by 5%. Tokens per
second divides by that total, so the figures here are taken from benchmark *duration* and
inter-token latency, which do not depend on it. The context-16384 row is the only one whose four
presets were built and benchmarked in a single chain; the others pair against a baseline measured
separately, which is reliable only when both arms report the same realized input and output token
counts.

Warmup for the same runs, also relative to `8_5`. Repeated identical builds agree to about 2% in
one chain and 3% across chains, so the 1.00x entries are nulls:

| KV lengths | batch | `pow2` | `4_3` | `9_8` | `uniform` |
|---|---|---|---|---|---|
| ~7k–10k, context 16384 | 4 | **0.63x** | 1.09x | 1.54x | 1.07x |
| ~7k–10k, context 32768 | 4 | **0.57x** | 1.13x | 1.68x | 1.00x |
| ~2k–3k, context 4096 | 32 | **0.65x** | 1.11x | 1.38x | 1.36x |
| ~400–700, context 1024 | 4 | **0.80x** | 0.95x | 1.00x | 1.25x |
| ~400–700, context 4096 | 32 | **0.69x** | 1.18x | 1.48x | 1.44x |

Read the two together, because they pull in opposite directions:

- **`9_8` is the fastest preset in every workload above, or tied fastest**, and the dearest in all
  but one. It is the choice when warmup time does not matter.
- **`pow2` costs the least and gives up the most** — around a tenth of the throughput once the KV
  lengths are long, and nothing at all when they are short, where its buckets already fit.
- **`4_3` sits between the default and `9_8` on both axes**, which is what its density predicts.
- **`uniform` is the only preset whose bucket placement depends on `--max-model-len`.** The other
  four put their buckets at fixed token counts, so a given KV length is padded the same amount
  whatever you set the context to. `uniform` derives its step and knee from `--max-model-len`,
  which keeps its bucket count flat as the context grows — 8 buckets at 1024 rising to only 18 at
  131072 — but ties its buckets to the configured context rather than to your sequences. Setting a
  context far above the lengths you actually run pushes its buckets above them: for KV lengths
  around 400–700 the mean padded width is 4% *below* the default's at `--max-model-len 1024` and
  28% above it at 8192 and beyond, where the default's is unchanged. That is arithmetic over the
  bucket lists, not a measured throughput penalty — at the one shape where it was measured
  directly the two presets came out level. Its measured throughput is level with the default
  everywhere except context 16384, where it matches the fastest preset, and its warmup is level at
  a long context and dearer at a short one. So its case rests on the bucket count staying flat as
  the context grows, rather than on being faster. Match `--max-model-len` to your real lengths
  before choosing it.

Per unit of extra warmup, the denser ratio presets are the *worst* buys: `9_8` converts added
compile time into throughput least efficiently of the four in three of the five workloads, even
though it wins outright on throughput. So `8_5` remains the default because it is the compromise,
not because it is the fastest.

If the KV lengths are known and clustered, `SPYRE_ATTN_KV_BUCKETS` beats every preset: put the
buckets where the lengths actually are. A workload that sits on one power of two gains nothing
from any preset.

#### A denser ladder for large batches only

A denser ladder pays off in proportion to the tokens it saves times the batch size, so its gain
concentrates at large batches. `SPYRE_ATTN_KV_LADDER_LARGE_BATCH` names a second preset that
batched decode uses once the batch's num-sequences bucket reaches `SPYRE_ATTN_LARGE_BATCH_MIN_SEQS`
(default 16, which with the default buckets means 9 or more sequences), while the per-sequence
path and smaller batches keep `SPYRE_ATTN_KV_LADDER`. Below the threshold the
batched kernel already pads to whole chunks, so at short contexts dense buckets there would
mostly compile duplicate kernels. The second ladder is unioned with the first, so no length rounds up further.
It is off by default and has no effect when `SPYRE_ATTN_KV_BUCKETS` is set.

On granite-3.3-8b at context 4096, `SPYRE_ATTN_KV_LADDER=8_5` with
`SPYRE_ATTN_KV_LADDER_LARGE_BATCH=9_8` matched or beat plain `9_8` at every batch size from 1 to
32, for about 18% less warmup than `9_8` and about 28% more than `8_5`. At context 16384 it
matched `9_8` from 16 sequences up for about 23% less warmup, and fell between `8_5` and `9_8`
at 4 and 8 sequences, where the second ladder does not apply.

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
