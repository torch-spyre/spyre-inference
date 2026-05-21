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

| Entry Point | Purpose |
|---|---|
| `vllm.platform_plugins` | Registers `TorchSpyrePlatform` — sets dtype, worker class, and attention backend |
| `vllm.general_plugins` | Calls `register_all()` — registers all custom ops before model loading |

## Component view of a Granite model

<figure markdown="span">
  ![Plugin Architecture](plugin-architecture.svg){: style="width: 140%; max-width: 1000px; margin-left: -20%" }
  <figcaption>
    Static architecture of the spyre-inference plugin showing how it integrates with
    vLLM and which model layers are replaced for Spyre execution.
  </figcaption>
</figure>

## Custom Op Replacement

Each layer in the Granite model that requires Spyre-specific handling is replaced via
vLLM's `@ClassName.register_oot()` decorator. When vLLM instantiates a layer, the
decorator intercepts and returns the Spyre implementation instead.

| vLLM Layer | Spyre Replacement | Device | Notes |
|---|---|---|---|
| `RMSNorm` | `SpyreRMSNorm` | Spyre | Custom op boundary for `torch.compile` |
| `RotaryEmbedding` | `SpyreRotaryEmbedding` | CPU | Fallback — Spyre doesn't support strided views |
| `QKVParallelLinear` | `SpyreQKVParallelLinear` | Spyre | TP=1 only, `F.linear()` |
| `RowParallelLinear` | `SpyreRowParallelLinear` | Spyre | TP=1 only, `F.linear()` |
| `MergedColumnParallelLinear` | `SpyreMergedColumnParallelLinear` | Spyre | TP=1 only, `F.linear()` |
| `SiluAndMul` | `SpyreSiluAndMul` | Spyre | Slicing done on CPU (Spyre limitation) |
| `ParallelLMHead` | `SpyreParallelLMHead` | Spyre | Vocab size padded to 64×32 boundary |

## Attention Backend

The `SpyreAttentionBackend` implements paged attention using pure PyTorch operations
(no custom CUDA kernels). Most prep work runs on CPU; the hot inner kernel is a single
compiled 4D attention call on Spyre:

| Step | Device | Operation |
|---|---|---|
| 1. KV cache update | CPU | Scatter new keys/values via `slot_mapping` |
| 2. Gather compact KV | CPU | Gather pages into per-seq tensors, pad `seq_len` to 256 |
| 3. Reshape Q + build mask | CPU | Pad `query_len` to 32, build additive mask (`0.0` / `-65504.0`) |
| 4. 4D attention kernel | Spyre | One compiled call: `Q @ Kᵀ` → `+ mask` → `softmax` → `@ V` |
| 5. Output extract | CPU | Trim query/sequence padding back to `num_actual_tokens` |

The backend also exposes an `use_sdpa=True` fallback path that routes through
`torch.nn.functional.scaled_dot_product_attention` on CPU — currently used for cases that
the 4D kernel doesn't cover (no GQA, non-square attention).

Key constraints:

- **KV length alignment**: 256 tokens (avoids per-step recompilation on Spyre)
- **Query chunk size**: 32 tokens (consistent tensor shapes for compilation)
- **Head size**: Must be a multiple of 64 (128-byte Spyre stick ÷ 2-byte float16)

## Device Placement Strategy

`TorchSpyreModelRunner` inherits from vLLM's `GPUModelRunner` and treats Spyre as the
"GPU" in the `CpuGpuBuffer` pattern. Buffers are created via a `SpyreCpuGpuBuffer`
override:

- **Float dtypes**: `.cpu` on CPU (numpy staging for the scheduler), `.gpu` on Spyre as
  `float16`
- **Int / bool dtypes**: `.gpu` aliased to `.cpu` (Spyre doesn't natively support these)

`self.device` stays `cpu` so that scatter, indexing, and block-table ops run on CPU, but
float compute tensors land on Spyre via `self._spyre_device`.

`_SpyreModelWrapper` sits between the model runner and the model and converts at the
call boundary:

- **Input**: CPU `int32`/`int64` tensors → Spyre `int64` (for embedding lookup)
- **Output**: Spyre `float16` tensors → CPU (for logits indexing and sampling)

`VocabParallelEmbedding` is **not** OOT-replaced — it is the upstream vLLM class. Its
weights are moved onto Spyre by `model.to(spyre_device)` during `load_model`, and the
wrapper supplies an `int64` Spyre tensor at the input, so the embedding lookup runs on
Spyre without any custom op.

Hidden states flow entirely on Spyre between decoder layers, with CPU round-trips only
for operations that Spyre doesn't yet support natively (rotary embeddings, attention
KV scatter/gather, logits indexing).
