# Plugin Architecture

`spyre-inference` is a vLLM out-of-tree (OOT) platform plugin that enables inference on
IBM's Spyre AI accelerator. It integrates with vLLM's plugin system to replace key
compute layers with Spyre-optimized implementations while preserving the rest of the
vLLM execution pipeline.

## System Overview

The diagram below shows how `spyre-inference` fits into vLLM's process architecture.
Blue boxes are Spyre-specific classes provided by this plugin; dark boxes are vLLM base
classes; the gold box is the model loaded from vLLM's model registry with Spyre custom
ops injected via OOT registration.

<figure markdown="span">
  ![System Overview](system-overview.svg){: style="width: 140%; max-width: 1200px; margin-left: -20%;" }
  <figcaption>
    Process-level view of vLLM with the spyre-inference plugin. Dashed arrows (▷)
    indicate inheritance; solid arrows indicate composition or dependency.
  </figcaption>
</figure>

The plugin registers via two entry points:

| Entry Point | Target | Purpose |
|---|---|---|
| `vllm.platform_plugins` | `spyre_inference:register` | Registers `TorchSpyrePlatform` — sets dtype, worker class, attention backend, and distributed backend |
| `vllm.general_plugins` | `spyre_inference:register_ops` | Calls `register_all()` — importing the ops package triggers every `@register_oot()` layer swap, and `register_all()` additionally registers the `spyre_convert` custom op (RoPE registers no op — its rotation runs in-graph). Also overrides vLLM's `TransformersForCausalLM` with `SpyreTransformersForCausalLM` |

`vLLM` is built from source with `VLLM_TARGET_DEVICE=empty` (no device-specific C
kernels), so the platform overrides a few CPU-backend assumptions: `import_kernels()` is
a no-op (there is no `vllm._C`), and the model runner reimplements the slot-mapping
kernel in pure PyTorch.

## Component view of a Granite model

<figure markdown="span">
  ![Plugin Architecture](plugin-architecture.svg){: style="width: 140%; max-width: 1000px; margin-left: -20%" }
  <figcaption>
    Static architecture of the spyre-inference plugin showing how it integrates with
    vLLM and which model layers are replaced for Spyre execution.
  </figcaption>
</figure>

## Custom Op Replacement

Most layers that require Spyre-specific handling are replaced via vLLM's
`@ClassName.register_oot()` decorator (a few, like `SiluAndMul`, need no replacement and
run upstream in the compiled graph). Most replacements are pure class swaps that run
when the ops package is imported. `register_all()` additionally registers the `spyre_convert`
custom op — the `convert` helper keeps device transfers invisible to `torch.compile`.
RoPE registers no op — its rotation-cache gather and 2×2 rotation run directly in the
compiled graph (see below).

| vLLM Layer | Spyre Replacement | Device | Notes |
|---|---|---|---|
| `GemmaRMSNorm` | `SpyreGemmaRMSNorm` | Spyre | A `maybe_compile(force=True)` `forward_native`, so the fp32 promotion is kept. Plain `RMSNorm` needs no replacement — upstream's fp32 `forward_native` lowers eagerly — but Gemma's trailing fp32 `weight` multiply does not: a STANDARD `[hidden]` operand that torch-spyre can neither broadcast against a staggered-EA activation nor de-stagger, so the kernel has to be compiled even under `enforce_eager`. |
| `RotaryEmbedding`, `Llama3RotaryEmbedding` | `SpyreRotaryEmbedding`, `SpyreLlama3RotaryEmbedding` | Spyre | Fully on-device, no opaque op. A device-resident 4D rotation cache (`[max_pos, 2, 2, rotary_dim//2]`) is built from `cos_sin_cache` and **primed on-device in `_apply` before `torch.compile`**; `forward_oot` then gathers this pass's per-token slice with `index_select` and applies the 2×2 rotation-matrix formulation (`_rotate_neox_2x2`) — both traced directly into the full-model compile graph. Priming before compile is the requirement: building the cache lazily inside the traced forward segfaults libsenlib during warmup, whereas a cache already materialized on-device indexes cleanly. Only neox-style full rotary is supported — other configs raise `NotImplementedError` at construction. The 2×2 inner dim `rotary_dim//2` must also be stick-aligned; this is not re-checked but is guaranteed by head-dim padding (see below) |
| `VocabParallelEmbedding` | `SpyreVocabParallelEmbedding` | Spyre (TP tables built on CPU at load) | The weight moves to Spyre with the model and the embedding gather runs on-device (`aten.embedding` now has a Spyre kernel, torch-spyre#420). TP=1 gathers directly. When TP>1, the per-vocab reindex/keep tables are built once on CPU at load and registered as device buffers; `forward` derives `masked_input`/`keep` from them on-device (`index_select`/`F.embedding`), applies the keep mask, and `all_reduce`s — no per-step CPU round-trip |
| `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear`, `ReplicatedLinear` | `SpyreColumnParallelLinear`, `SpyreMergedColumnParallelLinear`, `SpyreQKVParallelLinear`, `SpyreRowParallelLinear`, `SpyreReplicatedLinear` | Spyre | All five swap in `SpyreUnquantizedLinearMethod` (the transposed-weight fast path below). `SpyreQKVParallelLinear` additionally asserts `gather_output=False`; `SpyreRowParallelLinear` (`o_proj`, `down_proj`) inherits upstream's `all_reduce` when `reduce_results=True` under TP>1 |
| `SiluAndMul` | — (not replaced) | Spyre | No OOT class: vLLM's own `SiluAndMul` is traced into the compiled graph, so `silu(gate)·up` runs on Spyre and slices the fused `[..., 2*d]` on-device. The Spyre-specific piece is `mlp_pad.py`, which zero-pads `intermediate_size` to the 64-element stick at load time so that slice lands at a lowerable offset (inert since `silu(0) = 0`) |
| `NewGELU` | — (not replaced) | Spyre | No OOT class: vLLM's own `gelu_new` is traced into the compiled graph, cube term included — torch-spyre decomposes its `torch.pow(x, 3.0)` into a chain of `mul` ops (torch-spyre#4479) |
| `ParallelLMHead` | `SpyreParallelLMHead` | Spyre | TP≥1 with vocab sharding; per-rank weight padded to a multiple of 64×32 and pre-transposed; `apply` runs `x @ Wᵀ` then the un-pad slice, on Spyre — eager, no CPU detour; logits stay on Spyre for the TP `all_gather` |
| `LogitsProcessor` | `SpyreLogitsProcessor` | Spyre → CPU | Moves logits to CPU so all downstream sampling runs on the host. `_apply_head` D2Hs on the single-card path; when TP>1 `_gather_logits` runs the `all_gather` on Spyre and then converts the result. Either way the sampler's `logits.to(torch.float32)` never runs on Spyre, where it would crash torch-spyre's `copy_from_d2d` |
| `GateLinear` | `SpyreGateLinear` | Spyre | Clears `out_dtype` so MoE router logits stay in the weight dtype because Spyre cannot restickify fp32 (`spyre::ReStickifyOpHBM` is unsupported for IEEE_FP32). The MoE backend promotes stick-aligned full-softmax reductions to fp32 and returns them to the transport dtype |
| FP8 `ColumnParallelLinear` / fused QKV / `MergedColumnParallelLinear` / `RowParallelLinear` | `SpyreFp8LinearKernel` | Spyre | Checkpoint FP8 is dequanted to CPU fp16 at load so `model.to("spyre")` is a legal H2D. First Spyre forward eager-quantizes each SuperDSC N-tile to `qfp8wt` and caches it; later forwards compile `quantscalepertokenfp8` + `qfp8ch` + `aten._scaled_mm`. LM head stays FP16 |

### Transposed linear weights

`F.linear(x, W)` computes `x @ Wᵀ` however `W` is laid out, and on Spyre that transposed
matmul is ~3.5× slower than a plain `x @ A`
([torch-spyre#3512](https://github.com/torch-spyre/torch-spyre/issues/3512)).
`SpyreTransposedWeightMethod` (`custom_ops/linear.py`) is the shared base that closes that gap
in two overrides:

- `process_weights_after_loading` replaces the loaded `[out, in]` weight with a contiguous
  `[in, out]` `Wᵀ` (optionally padding the output rows first), so the transpose is paid once at
  load time rather than every forward.
- `apply` runs `spyre_linear_t` — `torch.matmul(x, Wᵀ)` plus optional bias — instead of
  `F.linear`, dropping any trailing pad columns with an eager on-device un-pad slice.

`SpyreUnquantizedLinearMethod` uses the base defaults (transpose in place, no padding); the five
linear subclasses install it in `__init__`, but only when `quant_method` is an
`UnquantizedLinearMethod`; quantized layers keep their own method. This is the
pure-PyTorch equivalent of torch-spyre's `[1,0]` weight layout, which only fires
for `nn.Linear` and so misses every vLLM parallel-linear. FP8 checkpoints use
`SpyreFp8LinearKernel` (`custom_ops/fp8_linear_kernel.py`) rather than `F.linear`:
load dequants to fp16, the first Spyre forward caches `qfp8wt` per N-tile, and
the compiled GEMM is `qfp8ch` + `aten._scaled_mm`.
`SpyreUnquantizedLMHeadMethod` reuses the same base with `WEIGHT_T_ATTR="padded_weight_t"` and
`ROW_ALIGN=64*32`, so the fast path and the padding/un-pad logic are defined once.

Fused projections stay fused. `SpyreQKVParallelLinear` returns the whole `[..., q+k+v]`
tensor and the unmodified upstream idiom `q, k, v = qkv.split(...)` slices it, exactly as
`SpyreMergedColumnParallelLinear`'s `[..., 2*d]` output feeds the upstream `SiluAndMul`,
which slices gate/up on-device. Earlier revisions instead split the QKV weight on CPU at load
time into three per-part GEMMs — a `SplitQKV` container built by an `analyze_and_unfuse`
pass — so that no fused output ever had to be sliced; one fused GEMM is faster than three,
so that pass is gone. The remaining slicing constraint is narrower than it was and lives
in the attention backend, where offset > 0 views still corrupt on transfer (see
[Attention Backend](#attention-backend)).

## Model adaptations

Some models need more than a swapped-out layer: a different transport for an input, a buffer
that has to follow `.to("spyre")`, an expert dispatch Spyre can lower. Those live in
`spyre_inference/models/`, one module per architecture, as **subclasses of the upstream vLLM
class** rather than runtime monkey-patches. `models/__init__.py` holds `spyre_models()`
(architecture string → Spyre class, built from the `_ADAPTED_MODULES` and `_ADAPTED_ARCHS`
tables) and `register_models()`, which points vLLM's `ModelRegistry` at them; `_`-prefixed
modules hold machinery those subclasses share or delegate to and register no architecture of
their own. Registration is lazy — nothing is imported until vLLM resolves the architecture —
and `register_models()` first checks every key against vLLM's own registry, so an upstream
rename fails loudly instead of silently falling through to the unadapted class.

Where upstream hardcodes a class and offers no hook (the BERT wrappers hardcode
`embedding_class`), the already-built instance is **retyped** to its Spyre subclass — same
`__init__`, same parameters, same module tree, only `forward` differs. Prefer a documented
upstream extension point where one exists: `CustomOp.register_oot` /
`PluggableLayer.register_oot` for a layer, and — for a MoE — the quant-method seam the
unquantized oracle leaves open for an out-of-tree platform.

Two adaptations worth knowing:

- **BERT / RoBERTa** (`models/_token_type.py`) carry `token_type_ids` in a side buffer
  owned by the embedding instead of vLLM's bit-pack into the high bits of `input_ids`,
  which Spyre cannot unpack ([torch-spyre#3509](https://github.com/torch-spyre/torch-spyre/issues/3509)).
- **MoE** (`moe.py`) supplies the routed-expert backend vLLM's unquantized MoE oracle lacks
  for an out-of-tree platform (it selects `UnquantizedMoeBackend.OOT` — no kernel — and
  leaves `process_weights_after_loading` to the plugin). A `CustomOp.register_oot`
  replacement for `UnquantizedFusedMoEMethod` computes the experts in two Spyre forms —
  gathered, which reads only a token's routed experts but lowers one token at a time, and
  all-expert persistent. Dispatch looks only at the packed token count: up to
  `SPYRE_MOE_GATHERED_MAX_TOKENS` — a decode batch, or a prompt that buckets that short — the
  gathered form is driven once per token, and above it the all-expert form takes the whole
  batch in one call. A batch whose rows are not stick-addressable takes the all-expert form
  whatever its size, since each row's storage offset has to span whole sticks for the compiled
  loop to address it.
  In the post-load hook it also rebuilds each layer's `w13 [E,2M,H]` / `w2 [E,H,M]` stacks into
  the `[E,H,M]` / `[E,M,H]` layout those forms contract on, freeing each source stack as it goes,
  since the device cannot hold both layouts at once. Tensor parallelism needs nothing
  further: upstream shards each expert's intermediate dim, so the forms just see a
  narrower `M` — zero-widened to whole sticks where a shard lands mid-stick — and
  `MoERunner` all-reduces the per-rank partial sums. Each model's own adaptation module
  supplies its recipe and any model-owned scaling (`configure_gemma4_moe_layers` in
  `models/gemma4.py`). `Gemma4DecoderLayer.forward` and `MoERunner` are untouched: vLLM
  reaches the experts through `torch.ops.vllm.moe_forward`, an opaque custom op, so the
  dispatch runs eagerly *inside* the block's compiled graph — the same seam the attention
  backend uses — and can drive compiled regions of its own.

## Compilation Granularity

Under `CompilationMode.STOCK_TORCH_COMPILE`, `_compile_for_spyre` compiles each entry of
the model's block `ModuleList` in place via `block.compile(backend="inductor",
fullgraph=True, dynamic=False)`. In place matters: rebinding the list entry to the
`OptimizedModule` that `torch.compile` returns would re-parent the block under an
`_orig_mod` child and rename every parameter, breaking weight save/reload.

Every graph is compiled `dynamic=False` because torch-spyre's Inductor backend rejects
`SymInt` shapes: a compiled graph is specialized to one concrete input shape. This is the
root reason the plugin buckets shapes everywhere — variable request shapes are padded up
to a small fixed set of compiled shapes (the padding masked out), and warmup pre-compiles
every reachable bucket so no request pays an Inductor compile mid-serving. Two shape axes
are bucketed independently: the packed `num_tokens` for the block graph (below) and
`(num_blocks, query_len)` for attention (see [Attention Backend](#attention-backend)).

Blocks are found structurally — a `ModuleList` whose non-`PPMissingLayer` entries own an
`Attention` somewhere, and are not themselves `Attention` layers — so decoder stacks
(`model.layers`) and encoder stacks (`bert.encoder.layer`) are both covered, as are
hybrid Mamba+attention stacks that mix layer classes in one list. A `ModuleList` of bare
`Attention` layers (Zamba2's shared `dpa_list`) is skipped: it is not a block stack.
Models whose attention is not a vLLM `Attention` — MLA (DeepSeek, Kimi), vision-tower
attention — match nothing and fall back to a whole-model graph.

Blocks of one class share one `forward` code object, so Dynamo traces the first and the
rest reuse that entry; whatever it re-traces hits the Inductor FX graph cache. The
backend compile count is independent of depth, but it is not 1: layer 0 specializes
separately because `residual is None` there, so a Llama-shaped stack yields two
artifacts, and stacks that vary per layer yield more — Gemma 3 alternates sliding-window
and full attention, giving four. A fresh `num_tokens` bucket then costs one block recompile
rather than a whole-model one. Note that `num_tokens` is the block graph's *only* shape
dependence: kv-cache length and its block-count buckets live inside
`unified_attention_with_output`, which is opaque to this graph and compiles its own
kernels (see [Kineto profiling](../user_guide/kineto_profiling.md)).

Depth independence relies on vLLM hoisting the per-layer attention name out of the graph,
which needs torch >= 2.11 and `VLLM_USE_LAYERNAME=1`. Without it each block bakes in its
own layer name and compiles separately, which is worse than the whole-model graph; the
runner logs a warning when it detects this. Inductor freezing (enabled by `max_autotune`)
defeats sharing the same way, by folding each block's weights into its own graph.

Embeddings and the final norm sit outside the block list, so no enclosing block graph
covers them. Both stay eager when compilation is off, except a Gemma final norm:
`SpyreGemmaRMSNorm` passes `force=True`, so it compiles as its own one-op graph
in every mode, including `enforce_eager`, because its fp32 weight multiply has no working
eager form. `lm_head` was never in the compiled region; `compute_logits` is a separate
call on the wrapper.

`SPYRE_COMPILE_GRANULARITY=model` restores the whole-model fullgraph, whose compile cost
grows with layer count.

## Attention Backend

Decoder attention has two backends, which differ only in how a page is laid out; the
default is the head-major one described below. `SpyreAttentionBackend` is the token-major
layout (`SPYRE_ATTN_KV_LAYOUT=token_major`) and the structure both share, so it is
described first. It implements paged attention using pure PyTorch operations
(no custom CUDA kernels). The KV cache is one dense tensor per layer on Spyre,
`[num_blocks, block_size, num_kv_heads, head_size]` — the shape
`SpyreAttentionBackend.get_kv_cache_shape` advertises. It runs a FlashAttention-style
online softmax that iterates over pages without any compact-gather step, reading each
page by indexing the dense tensor with a one-element int32 device tensor (an indirect
access, so the compiled bundle carries a real index rather than a constant slice) and
permuting the token-major page to head-major on device before the matmuls. The cache is
allocated with the slot axis outermost in the device layout (`slot_major_kv_layout`) so
the write can scatter through a slot-major view of it:

| Step | Device | Operation |
|---|---|---|
| 1. Build metadata & masks | CPU → Spyre | The metadata builder reads `query_start_loc`/`seq_lens`, pads each `query_len` and KV block count onto their buckets, builds the per-sequence query-row index tables and the additive mask on CPU, then copies them to the device |
| 2. Write new K/V to cache | Spyre | Compiled `index_copy_` scatter through a slot-major view of the paged cache (one per tensor, fused); only the slot-index vector is computed host-side and copied over |
| 3. Per-sequence page attention | Spyre | A host-driven loop dispatches one compiled kernel per sequence — the query rows are gathered on-device (never copied to CPU): `Q @ Kᵀ · scale` → optional soft-cap → `+ tile_mask` → online softmax → `@ V` |
| 4. Write-back | Spyre | Each sequence's result is written into the Spyre output buffer with a device-to-device copy |

The compiled kernels themselves — the per-sequence page attention, the batched decode
path, the KV store, and the cache's device layout — live under
`spyre_inference/v1/attention/ops/`; the backend module holds the metadata builder and
the host-side orchestration that calls them.

Because attention kernels are `dynamic=False` too, they are pre-compiled during warmup
rather than lazily on first use: by default (`SPYRE_ATTN_RECORD=1`) warmup traces every
variant `SpyreAttnBucketer` can produce — the product of the KV-length and query-length
buckets below — so a served request always lands on an already-compiled kernel. When the
batched-decode kernel is enabled (`SPYRE_BATCHED_DECODE=1`, the default, which under the
default tiled walk means the head-major layout) warmup also records its variants, the
product of the KV-length (`num_blocks`) and num-sequences buckets. A single step can carry a mix of prefill and decode sequences; each sequence is
padded to its own query bucket (decodes use the length-1 bucket) before dispatch.
`SPYRE_ATTN_RECORD=0` restores lazy per-variant compilation.

Under `dynamic=False` the Python loop over a sequence's KV pages is unrolled at trace
time, so a graph holds one copy of the attention body per page and compile time grows
with KV length. `SPYRE_ATTN_FOR_EACH_TILE=1` walks that axis with torch-spyre's
`for_each_tile` instead, leaving one body plus a tile spec, and the same applies to the
batched-decode kernel's walk over block chunks. Both kernels carry the online softmax as
a `(tile_max, tile_sum, tile_output)` triple either way; `walk_tiles` picks the walk and
is the only place that reads the variable. `SPYRE_ATTN_FOR_EACH_TILE=1` is the default;
setting it to `0` runs the identical bodies under Python loops as a rollback path. The switch is
read once at import, because it decides the `fullgraph` setting the tiled walk needs —
setting it after `spyre_inference` is imported has no effect.

### Head-major KV cache

`SpyreHeadMajorAttentionBackend` is the default decoder backend
(`SPYRE_ATTN_KV_LAYOUT=head_major`). It stores a page as
`[num_blocks, num_kv_heads, block_size, head_size]` instead. The page then arrives in the
shape the matmuls want, so the per-page permute in step 3 disappears — that is the whole
point of the layout. It moves the transpose to the write: a token's KV heads are
`block_size` rows apart, so step 2 becomes one `index_copy_` per KV head (`kv_write_index`
publishes one index per head) over a source materialized contiguously first, rather than a
single store of one contiguous run per token.

Everything above the cache's memory — the metadata builder, the bucketer, the mask tiles,
warmup recording and dispatch — is shared with the token-major backend. What differs is
duplicated rather than parameterised: the advertised shape, the allocation
(`head_major_kv_layout`), and the three kernels that touch a page. The worker follows the
layer's impl (`allocate_pages`) rather than a hardcoded shape, so the two cannot disagree.

#### LX-resident pages

The head-major per-sequence kernel keeps a gathered page in the LX scratchpad from its
gather to its last use instead of round-tripping through HBM. Two shape choices get it
there. The page is gathered on (page, kv_head) with a `[num_kv_heads, 1]` index, so the
gather's split lands per KV head — an output axis of `probs @ V` the consumer can mirror;
behind a 1-D index the entry axis instead splits in whole 32-entry sticks. And the query
groups fold into the query's row axis — a reshape, since heads are KV-major — so each
matmul carries a single batch dim: the batched GQA form leaves the page with two batch dims
and Inductor clones it out to a query-group axis it does not have (torch-spyre#4123). The
cache fold is free too — `[num_blocks, KV, block_size, D]` reshapes to
`[num_blocks * KV, block_size, D]` — but the cache is allocated with that folded axis at
device dim 0, which is where an indexed axis has to sit for the gather to cost one page
rather than the whole tensor.

Three things follow from those choices. The gather is a 2-D subscript, which lowers to
`aten.index` and fails eager by upcasting its int32 index, so this backend always compiles
attention even under `--enforce-eager` — attention compiles in its own domain, so the rest
of the model still runs eager. Because the bmm's output axes
(`num_kv_heads * padded_query_len`) cannot fill 32 cores at decode, and filling them would
mean K-splitting a reduction a gather cannot mirror, the attention compile alone is capped
at 8 cores; `SPYRE_ATTN_MAX_CORES` overrides that. And the layout carries no ALiBi (which
needs a bias tile per query group), so an ALiBi model needs `token_major`. Batched decode
runs the other way round: its page index is uploaded one entry per stick here, which is
what the tiled walk needs, so the batched kernel is reached on this layout and declined on
token-major until that port lands.

Residency is a property of the layout plan, not of a result, so it is measured off the
planner's own verdicts. K's residency needs torch-spyre#4153: `q @ Kᵀ` lowers the
transpose to a restickify, whose cross-frame barrier bars an LX-resident input without
that PR's local-read proof. V is read directly by `probs @ V` and stays resident either
way.

Key constraints:

- **KV length bucketing**: padded block count on power-of-two buckets from `block_size`
  to `max_model_len` (avoids per-step recompilation on Spyre)
- **Query length bucketing**: `[1] + multiples of min(512, max_num_batched_tokens)`
  (consistent tensor shapes for compilation)
- **Num-sequences bucketing** (batched-decode kernel only, `SPYRE_BATCHED_DECODE=1`, the
  default; under the default tiled walk, the head-major layout only):
  powers of two from 4 to `max_num_seqs` (`SPYRE_ATTN_NUM_SEQS_BUCKETS`); the decode-batch
  kernel is recorded over the `(num_blocks, num_seqs)` grid
- **Head size**: Must be a multiple of 64 (128-byte Spyre stick ÷ 2-byte float16)
- **Block size**: Must be a multiple of 64. The default is 128, and a user-supplied
  `block_size` is rounded up to the next multiple of 64
- **GQA only**: MHA (`num_queries_per_kv = 1`) currently fails in the Spyre compiler's
  layout-propagation pass; only GQA configurations are exercised today
- **Supported**: sliding-window masking (per layer, so a hybrid stack's full-attention
  layers stay unwindowed) and logits soft-capping are both handled; ALiBi slopes are not

### Encoder-only attention

Encoder-only (embedding) models take a separate path. For `ENCODER`/`ENCODER_ONLY`
layers, `TorchSpyrePlatform.get_attn_backend_cls` selects `SpyreEncoderAttentionBackend`
→ `SpyreEncoderAttentionImpl` (both subclass the decoder backend/impl in
`spyre_encoder_attn.py`). This path has **no KV cache** — attention is bidirectional over
the full sequence — so it skips the paged-cache machinery. There is no online-softmax
loop either: a sequence's whole K/V fits one tensor, so nothing forces the block-wise walk
the decoder needs.

Everything hangs off one number, `R = encoder_budget_rows(...)`: the token budget, capped
at 2048, floored at `max_model_len` rounded up to a power-of-two multiple of 64, capped at
what `max_num_seqs` sequences of that length could carry, and finally floored to a whole
multiple of that longest length — which is what makes every declared length divide `R`. **The pooling body is always `R` rows.** Fixing it
is what reduces the attention kernels' cache keys to the sequence shapes alone — both
kernels take the body buffer as an argument, so a varying buffer size would multiply every
attention graph.

On top of that one buffer sit two paths over a single power-of-two length ladder
(`ENCODER_LEN_ALIGNMENT = 64` doubling up to `max_model_len`, rounded up to a power-of-two
multiple of 64 — the same rounding a request's own extent gets, so the ladder declares
exactly the extents a request can be assigned):

1. **Rectangular path** — one rectangle per length, `B = R / L`, so a rectangle is exactly the
   body buffer. The runner pads each sequence to `L` and the batch to `B` in `_preprocess`
   (host-side, integer tensors only), so Q/K/V *are* the grid: `_encoder_rect_kernel`
   reshapes, runs one `F.scaled_dot_product_attention`, and stores — no data movement
   inside the layer. `_unpad_encoder_hidden` compacts the grid back before the pooler, at a
   fixed row count so the gather does not specialise per token total.
2. **Ragged path** — for a batch too wide for any rectangle. Q/K/V stay packed and requests
   are grouped by their own padded extent; `_encoder_fused_kernel` does gather, attend and
   scatter for one group in a single graph, keyed on `(width, extent)`. Request boundaries
   ride in int32 row-index tables, so a card never does offset arithmetic on *shapes* —
   offsets are data. A group wider than the widest declared width is chunked into
   descending powers of two, so every dispatch lands on a warmed pair.

The runner picks between them once per step in `_build_attention_metadata` and records the
choice as the *type* of `attn_metadata.encoder_plan` (`EncoderRectPlan` versus a list of
`EncoderGroupPlan`). It has to run there rather than in `forward`: the builder does a D2H
read and an H2D convert, which inside a traced region become graph nodes. Because both
paths stay behind the opaque `unified_attention_with_output`, the enclosing block graph is
identical for either — one shape, shared — so path selection is never a branch inside a
compiled region nor a dynamo guard.

Three torch-spyre constraints shape the rest: a compile input's `storage_offset` is a
Dynamo guard (torch-spyre#4449, which closed #3770) and for int32 is still dropped
outright, so rows are gathered with `index_select` rather than sliced — a slice would
either recompile per offset or read the wrong rows; there is no on-device `arange` or `full`, so every index and mask
tensor is host-built and reaches the device in one `convert` per plan; and SDPA's
decomposition does `amax` then `exp(scores - max)`, which NaNs a fully masked row — hence
the `finfo.min / 2` mask fill and the single attendable key a batch-pad lane gets.

## Encoder / embedding models: compile shape axes

The body is compiled once, at `R` rows. Attention is shape-managed separately behind the
opaque custom-op boundary: one rectangle per declared length on the rectangular path, one
`(width, extent)` pair per group on the ragged one (a *group* being the requests that
share one padded extent, attended together in one call). With `max_model_len=512`,
`max_num_seqs=32` and a 2048-token budget that is 23 shapes — one body, four rectangles,
18 group pairs — and at `max_num_seqs=4` only five, since no batch that narrow can miss
the rectangular path.

<figure markdown="span">
  ![Encoder target state](encoder-ideal-state.svg){: style="width: 140%; max-width: 1400px; margin-left: -20%" }
  <figcaption>
    Encoder / embedding models under <code>STOCK_TORCH_COMPILE</code>, <strong>ragged path
    only</strong>: attention over the packed list, grouped by each request's padded
    extent. Predates the rectangular path and the single <code>R</code>-row body, so read the
    body bucketing and the warmup sweep as historical; the grouping and the row-index
    tables are still current. Regenerating it needs the <code>d2</code> toolchain.
  </figcaption>
</figure>

## Device Placement Strategy

`TorchSpyreModelRunner` inherits from vLLM's `GPUModelRunner` and treats Spyre as the
"GPU" in the `CpuGpuBuffer` pattern. Buffers are created via a `SpyreCpuGpuBuffer`
override:

- **Float dtypes**: `.cpu` on CPU (numpy staging for the scheduler), `.gpu` on Spyre as
  `float16`
- **Int / bool dtypes**: `.gpu` aliased to `.cpu` (Spyre doesn't natively support these)

`self.device` stays `cpu` so that scatter, indexing, and block-table ops run on CPU, but
float compute tensors land on Spyre via `self._spyre_device`. Because there is no
`vllm._C` under `VLLM_TARGET_DEVICE=empty`, the runner also swaps in a pure-PyTorch
`_compute_slot_mapping` implementation for the paged-cache slot mapping.

At load time, `load_model` moves every module except `Attention` scale buffers onto Spyre.
The weight transposes happen before that, in each layer's
`process_weights_after_loading`, while the weights are still on CPU.

`_SpyreModelWrapper` sits between the model runner and the model and converts at the
call boundary:

- **Input**: CPU `int32`/`int64` tensors → Spyre `int64` (for embedding lookup)
- **Output**: Spyre `float16` tensors → CPU (for logits indexing and sampling)
- **`compute_logits`**: moves the CPU-sliced `hidden_states[logits_indices]` back onto
  Spyre for the `SpyreParallelLMHead` matmul, which returns logits on Spyre

`SpyreVocabParallelEmbedding` inherits weight loading and shard arithmetic from upstream
and overrides `forward`. The weight moves to Spyre with the rest of the model, and the
embedding gather runs on-device now that `aten.embedding` has a Spyre kernel
([torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420)) — this
replaces the earlier silent D2H/H2D CPU fallback that copied the full `[vocab, hidden]`
weight on every decode step. When TP>1 the shard mask is applied **on-device**:
`get_masked_input_and_mask` runs once at load to build per-vocab reindex/keep lookup
tables (its int64 comparisons against Python constants cannot lower on Spyre), registered
as device buffers; `forward` then gathers through them with `index_select`/`F.embedding`,
applies the keep mask, and `all_reduce`s — all on Spyre, no per-step CPU round-trip.

Hidden states flow on Spyre between decoder layers, with CPU round-trips only for
work that stays host-side: logits indexing for sampling, and the attention metadata the
builder prepares on CPU (slot mapping, the per-sequence index tables and additive mask).
The per-sequence attention loop is host-driven control flow, but its query-row gather and
kernels run on Spyre; the KV-cache write and the write-back are device-to-device. RoPE's
rotation-cache gather and the embedding gather also run on-device.

## Transformers backend

When `model_impl="transformers"`, `register_ops` swaps vLLM's `TransformersForCausalLM`
for `SpyreTransformersForCausalLM` (`spyre_inference/transformers_backend.py`). vLLM's
stock Transformers backend still handles model creation, weight loading, attention
routing, the KV cache, and scheduling, and its fusers replace HF's linear/norm/GLU
modules with vLLM layers — which the OOT registrations above then pick up automatically.

The subclass covers what upstream leaves to HF's module code. There is no RoPE fuser, so
HF's `rotary_emb` survives and would derive cos/sin inside the forward from int64
`position_ids`, a cast torch-spyre cannot lower. It is replaced with a precomputed
`[max_model_len, 2, 2, head_dim/2]` rotation cache — built on the host and moved to the
device before compile, leaving only an `index_select` in the graph — plus a matmul-based
`apply_rotary_pos_emb`. Head padding is shared with the native path: the platform widens
`head_dim` and the weight passes in `head_pad.py` pad Q/K interleaved, so this backend
only has to rebuild the rotation cache at the pre-pad frequencies.

`RMSNorm` registers no OOT op: upstream `forward_native` lowers its fp16→fp32 upcast
into the compiled graph. `GemmaRMSNorm` still needs one, because its trailing fp32
`weight` multiply hits a torch-spyre gap — a STANDARD `[hidden]` operand that can
neither broadcast against nor de-stagger the staggered-EA activation. Because the
fusers key on class names, the OOT registry covers the fused Gemma norm
(`SpyreTPAwareGemmaRMSNorm`) too.

## Distributed (TP)

`TorchSpyrePlatform.get_device_communicator_cls` returns `SpyreCommunicator`, a
`DeviceCommunicatorBase` override in
`spyre_inference/distributed/spyre_communicator.py`. The installed `libspyre_comms.so`
now implements `barrier`, `broadcast`, `send`/`recv`, list-form `allgather`, `gather`,
and `allreduce`; only `reduce` remains a throw-stub, and torch-spyre's spyreccl
backend still stubs `_allgather_base` (so `dist.all_gather_into_tensor` doesn't work).

`SpyreCommunicator` therefore overrides:

- **`all_reduce`** — uses the functional `_c10d_functional.all_reduce`, which torch-spyre
  lowers to `spyre::all_reduce_async` inside a compiled graph and which runs eagerly via
  `libspyre_comms` outside one, so one code path serves both modes.
- **`all_gather`** — inside a compiled graph the functional collective lowers to
  `spyre::all_gather_async` and stays on device; eager keeps native list-form
  `dist.all_gather`, because the functional entry point (`allgather_into_tensor_coalesced`)
  is rejected outside a graph. CPU tensors route through the gloo half of the multi-backend
  `cpu:gloo,spyre:spyreccl` group.
- **`reduce_scatter`** — raises; it is not on the TP forward path.

`gather` is not overridden — it works natively via `libspyre_comms`. The
`tests/probes/test_spyre_comms_native_probes.py` xfail-strict suite is the canonical
signal: when a probe flips green, delete the corresponding override or workaround.

The worker (`TorchSpyreWorker`) inherits directly from vLLM's `Worker` (gpu_worker), not
`CPUWorker` — Spyre needs none of the CPU-specific init (NUMA binding, host-RAM
profiling). Data parallelism (`data_parallel_size > 1`) is rejected in
`check_and_update_config`.
