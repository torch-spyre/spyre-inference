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

The plugin registers via three entry points:

| Entry Point | Target | Purpose |
|---|---|---|
| `vllm.platform_plugins` | `spyre_inference:register` | Registers `TorchSpyrePlatform` — sets dtype, worker class, attention backend, and distributed backend |
| `vllm.general_plugins` | `spyre_inference:register_ops` | Calls `register_all()` — importing the ops package triggers every `@register_oot()` layer swap, and `register_all()` additionally registers the opaque `spyre_rotary_cpu` and `spyre_convert` custom ops |
| `vllm.general_plugins` | `spyre_inference:register_hf_adapters` | Overrides vLLM's `TransformersForCausalLM` with `HfAdaptersForCausalLM` so `model_impl="transformers"` uses hf-adapters (matmul-based RoPE) on Spyre |

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

Each layer that requires Spyre-specific handling is replaced via vLLM's
`@ClassName.register_oot()` decorator. Most replacements are pure class swaps that run
when the ops package is imported; two layers (rotary embedding, the `convert` helper)
also register an opaque custom op via `register_all()` so their device transfers stay
invisible to `torch.compile`.

| vLLM Layer | Spyre Replacement | Device | Notes |
|---|---|---|---|
| `RMSNorm` | `SpyreRMSNorm` | Spyre | `forward_oot` runs a `maybe_compile`d kernel directly on Spyre; no float32 promotion (torch-spyre limitation) |
| `RotaryEmbedding`, `Llama3RotaryEmbedding` | `SpyreRotaryEmbedding`, `SpyreLlama3RotaryEmbedding` | CPU | Wrapped in the opaque `spyre_rotary_cpu` custom op; the whole rotary (incl. `index_select`) runs eagerly on CPU so Inductor sees one fallback kernel |
| `VocabParallelEmbedding` | `SpyreVocabParallelEmbedding` | Spyre + CPU | TP shard mask computed on CPU (Spyre inductor rejects int64 constants); the `aten.embedding` gather itself is a **silent CPU fallback** in torch-spyre ([torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420)); weights and output live on Spyre; `all_reduce` when TP>1 |
| `QKVParallelLinear` | `SpyreQKVParallelLinear` | Spyre | Subclass only asserts `gather_output=False`; the fused weight is split at load by the un-fusing pass, and `forward` runs `q`/`k`/`v` as three `F.linear` calls on Spyre |
| `SiluAndMul` | `SpyreSiluAndMul` | Spyre | Consumes the pre-split `gate`/`up` parts (see un-fusing); the fused fallback path slices on CPU (Spyre slicing corrupts views) |
| `ParallelLMHead` | `SpyreParallelLMHead` | Spyre → CPU | TP≥1 with vocab sharding; per-rank weight padded to a multiple of 64×32; logits returned on CPU for the downstream TP `all_gather` |
| `LogitsProcessor` | `SpyreLogitsProcessor` | — | Makes logits contiguous — the downstream in-place `logits *= scale` otherwise trips a torch-spyre compile issue |

`RowParallelLinear` and `MergedColumnParallelLinear` are **not** subclassed:

- `RowParallelLinear` (`o_proj`, `down_proj`) runs the upstream class unchanged — `F.linear`
  dispatches to Spyre, and `all_reduce` (via `SpyreCommunicator`) fires when `reduce_results=True`
  under TP>1.
- `MergedColumnParallelLinear` (`gate_up_proj`) is un-fused at load into separate `gate`/`up`
  Parameters (see below).

### Weight un-fusing

`analyze_and_unfuse` (`custom_ops/unfuse.py`) runs once after the checkpoint is loaded,
while weights are still on CPU. Fused projections are a problem on Spyre: splitting a
fused weight's output on-device yields strided views that corrupt when transferred. So
the pass splits each fused weight (`QKVParallelLinear`, and the `MergedColumnParallelLinear`
feeding a `SpyreSiluAndMul`) into contiguous per-part Parameters and rebinds `forward` to
run one `F.linear` per part. The result is a `SplitQKV` / `SplitSiluAndMul` container that
the unmodified downstream idioms — `q, k, v = qkv.split(...)` and `gate, up = proj` — keep
consuming unchanged.

## Attention Backend

The `SpyreAttentionBackend` implements paged attention using pure PyTorch operations
(no custom CUDA kernels). The KV cache is a list of per-page tensors on Spyre — each
page is `[num_kv_heads, block_size, head_size]` — rather than a monolithic tensor. It
runs a FlashAttention-style online softmax that iterates over pages without any
compact-gather step:

| Step | Device | Operation |
|---|---|---|
| 1. q/k/v → CPU | CPU | Bring `q`, `k`, `v` to CPU once (Spyre slicing corrupts strided views) |
| 2. Reshape & cache | Spyre | Per-token overwrite of new K/V into the list-of-pages cache |
| 3. Per-sequence varlen loop | CPU | Iterate sequences via `query_start_loc`, pad `query_len` to 32 |
| 4. Online softmax over pages | Spyre | Compiled per `(num_blocks, padded_query_len)` kernel: `Q @ Kᵀ · scale` → optional soft-cap → `+ tile_mask` → online softmax → `@ V` |
| 5. Write-back | CPU → Spyre | Stage each sequence's result into a CPU buffer, then one bulk copy into the Spyre output (per-token `spyre.overwrite` scatter doesn't scale) |

Key constraints:

- **KV length alignment**: 256 tokens (avoids per-step recompilation on Spyre)
- **Query chunk size**: 32 tokens (consistent tensor shapes for compilation)
- **Head size**: Must be a multiple of 64 (128-byte Spyre stick ÷ 2-byte float16)
- **Block size**: Must be a multiple of 64; the platform rounds a user-supplied
  `block_size` up to the next multiple of 64 automatically
- **GQA only**: MHA (`num_queries_per_kv = 1`) currently fails in the Spyre compiler's
  layout-propagation pass; only GQA configurations are exercised today
- **Supported**: sliding-window masking and logits soft-capping are both handled;
  ALiBi slopes are not

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

At load time, `load_model` calls `analyze_and_unfuse(self.model)` (weight un-fusing) and
then moves every module except `Attention` scale buffers onto Spyre.

`_SpyreModelWrapper` sits between the model runner and the model and converts at the
call boundary:

- **Input**: CPU `int32`/`int64` tensors → Spyre `int64` (for embedding lookup)
- **Output**: Spyre `float16` tensors → CPU (for logits indexing and sampling)
- **`compute_logits`**: moves the CPU-sliced `hidden_states[logits_indices]` back onto
  Spyre for the `SpyreParallelLMHead` matmul, which then returns logits on CPU

`SpyreVocabParallelEmbedding` inherits weight loading and shard arithmetic from upstream
and overrides `forward` to compute the TP shard mask on CPU (the upstream helper does
int64 comparisons against Python int constants, which the Spyre inductor backend
rejects). Its weight and output tensors live on Spyre, but the `aten.embedding` gather
itself is a **silent CPU fallback** in torch-spyre
([torch-spyre#420](https://github.com/torch-spyre/torch-spyre/issues/420)) — the lookup
is shuttled D2H/H2D even though the tensors are on Spyre, so no `FallbackWarning`-free
"Spyre embedding" exists yet.

Hidden states flow on Spyre between decoder layers, with CPU round-trips only for
operations that Spyre doesn't yet support natively (the embedding gather, rotary
embeddings, q/k/v slicing, the per-sequence attention varlen loop, logits indexing).

## HF-adapters Transformers backend

When `model_impl="transformers"`, the `register_hf_adapters` general plugin swaps vLLM's
`TransformersForCausalLM` for `HfAdaptersForCausalLM` (`spyre_inference/hf_adapters.py`).
vLLM's stock Transformers backend still handles model creation, weight loading, attention
routing, the KV cache, and scheduling; the Spyre OOT layers above apply automatically at
instantiation. The adapter's main job is to replace HF's `RotaryEmbedding` with a
matmul-based RoPE (`apply_rope_matmul`), padding Q/K into a stick-aligned dimension for
the rotation when `head_dim/2` is not a multiple of the Spyre block size and contracting
back afterward.

## Distributed (TP)

`TorchSpyrePlatform.get_device_communicator_cls` returns `SpyreCommunicator`, a
`DeviceCommunicatorBase` override in
`spyre_inference/distributed/spyre_communicator.py`. The installed `libspyre_comms.so`
now implements `barrier`, `broadcast`, `send`/`recv`, list-form `allgather`, and
`gather`; only `allreduce` and `reduce` remain throw-stubs, and torch-spyre's spyreccl
backend still stubs `_allgather_base` (so `dist.all_gather_into_tensor` doesn't work).

`SpyreCommunicator` therefore supplies:

- **`all_reduce`** — a manual TP=2 reduce-to-root + broadcast built from `send`/`recv`
  (native allreduce is not available yet; TP>2 raises).
- **`all_gather`** — routes CPU tensors through the gloo half of the multi-backend
  `cpu:gloo,spyre:spyreccl` group, and uses native list-form `dist.all_gather` for Spyre
  tensors (the base class's `dist.all_gather_into_tensor` path is blocked by the
  `_allgather_base` stub).
- **`reduce_scatter`** — raises; it is not on the TP forward path.

`gather` is no longer overridden — it now works natively. Each remaining fallback is
tagged `REPLACE-WITH-NATIVE`; the `tests/test_spyre_comms_native_probes.py` xfail-strict
suite is the canonical signal: when a probe flips green, delete the corresponding
override.

The worker (`TorchSpyreWorker`) inherits directly from vLLM's `Worker` (gpu_worker), not
`CPUWorker` — Spyre needs none of the CPU-specific init (NUMA binding, host-RAM
profiling). Data parallelism (`data_parallel_size > 1`) is rejected in
`check_and_update_config`.
