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
(no custom CUDA kernels). The forward pass is split between CPU and Spyre:

| Step | Device | Operation |
|---|---|---|
| 1. KV cache update | CPU | Scatter new keys/values via `slot_mapping` |
| 2. KV gather | CPU | Gather pages into compact tensors, pad to 256-token alignment |
| 3. Q @ K^T | Spyre | Batched matmul |
| 4. Mask + softmax | Spyre | Additive causal mask, then softmax |
| 5. Attn @ V | Spyre | Batched matmul |
| 6. Output extract | CPU | Trim alignment padding |

Key constraints:

- **KV length alignment**: 256 tokens (avoids per-step recompilation on Spyre)
- **Query chunk size**: 32 tokens (consistent tensor shapes for compilation)
- **Head size**: Must be a multiple of 64 (128-byte Spyre stick ÷ 2-byte float16)

## Device Placement Strategy

The model runner uses `device="cpu"` for buffer management (scatter, indexing, scheduling)
while placing float compute tensors on Spyre via `_spyre_device`. The `_SpyreModelWrapper`
handles conversion at the boundary:

- **Input**: CPU `int32`/`int64` tensors → Spyre `int64` (for embedding lookup)
- **Output**: Spyre `float16` tensors → CPU (for logits indexing and sampling)

This means hidden states flow entirely on Spyre between decoder layers, with CPU
round-trips only for operations that Spyre doesn't yet support natively (rotary
embeddings, KV cache indexing).
