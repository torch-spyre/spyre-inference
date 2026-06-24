# Attention-specific debugging notes

Companion to `SKILL.md`. Read this when the failing surface is the Spyre attention backend specifically. The generic workflow (cluster → HTML log → hypothesis queue → cluster validation) in `SKILL.md` still applies; this file just adds attention-flavored content to the hypothesis queue and pipeline-bisection steps.

## Files that matter for attention

- `spyre_inference/v1/attention/backends/spyre_attn.py` — the backend, `SpyreAttentionImpl`, `SpyreAttentionMetadataBuilder`. List-based paged attention (PR #206): K/V are stored as a `list[Tensor]`, each `[num_kv_heads, block_size, head_size]`, written by `_reshape_and_cache` and consumed by `_online_softmax_attention`. The module docstring enumerates every torch-spyre limitation the backend routes around.
- `tests/test_spyre_attn.py` — builds real metadata via `SpyreAttentionMetadataBuilder`, calls `SpyreAttentionImpl.forward` on a Spyre device, compares against a CPU reference (`ref_attn`). Tolerances are loose (`atol=0.3, rtol=5.0` for prefill; `atol=0.2, rtol=0.2` for decode) — if mismatch ratios are near 100 % with differences > 1.0, suspect a **structural** bug, not fp16 noise. Multi-sequence specs (`batch_decode(2seqs)`, `batch_prefill(2seqs)`, `mixed(decode+prefill)`) are exercised and pass.
- Upstream attention coverage (via `tests/plugin/spyre_testing_plugin/upstream_tests.yaml`): `test_causal_backend_correctness` runs `single_decode`, `single_prefill`, `small_decode` (2 decoding seqs), `small_prefill` (2 prefilling seqs), and `mixed_small` (4 mixed) across `tensor_parallel_size ∈ {1, 2, 4}` — all pass. Multi-sequence and TP-sharded head counts are confirmed working at the backend level.

## Attention-specific limitations

In addition to the generic table in `SKILL.md`:

| Limitation | Workaround / status |
|---|---|
| KV alignment to bucketed length | `KV_LENGTH_ALIGNMENT=256` so the same compiled kernel is reused as KV grows |
| Query length must be bucketed | `QUERY_CHUNK_SIZE=32`; queries are chunked into multiples of this |
| `head_size` must be a multiple of 64 | `SpyreAttentionBackend.supports_head_size` enforces this (128-byte stick / 2 bytes for fp16) |
| `block_size` (KV reduction dim) must be a multiple of 64 | `SpyreAttentionImpl.get_supported_kernel_block_sizes` advertises `MultipleOf(64)`; `TorchSpyrePlatform.check_and_update_config` forces the framework block_size to 64. With smaller values the matmul reduction dim can't fit on the 64-element device stick and the layout pass aborts with `cannot restickify ... y_var=d2`. |
| MHA / MQA params still `#`-commented in `test_spyre_attn.py` | The compile-time `y_var=d2` failure that originally motivated the skip was a block-size issue (now fixed via `MultipleOf(64)`). MHA itself runs end-to-end (opt-125m is MHA and the single-seq model test passes); the standalone `(12, 1, 32, 64) @ (12, 1, 64, 64)` matmul matches CPU within fp16 noise. The local skips in `test_spyre_attn.py:302-303` are now stale and worth re-enabling once someone has time to pick parametrizations that match real model shapes. |
| Multi-sequence path at the **model_runner** level produces structurally-wrong logits | `SpyreAttentionImpl.forward` itself handles `num_seqs > 1` correctly (verified by every `batch_*` and `mixed_*` spec). The bug is somewhere in `spyre_model_runner.py` / metadata-builder orchestration when `max_num_seqs > 1` — `_SpyreModelWrapper.__call__` int-conversion, `SpyreCpuGpuBuffer` reuse across decode steps, or per-step block-index/offset computation. The `constrain_vllm_runner_kv_cache` test fixture forces `max_num_seqs=1` until this is fixed. |

## Useful test selectors (orientation only — not for `-k`)

These appear in current parametrize IDs. They're for reading collected node IDs; `-k` will not parse the parens/equals/commas.

`tests/test_spyre_attn.py`:
- `decode(q=1,kv=256)` / `decode(q=1,kv=1024)`
- `prefill(q=32,kv=256)` / `prefill(q=64,kv=512)` / `prefill(q=100,kv=512)`
- `batch_decode(2seqs)` / `batch_prefill(2seqs)` / `mixed(decode+prefill)`
- `GQA` only (MHA and MQA still `#`-commented in the file — see the table above)
- `head_size(128)` only (256 `#`-commented)
- `block_size(128)` / `dtype(fp16)` / `num_blocks(256)`
- `compilation_NONE` / `compilation_STOCK` × `device_cpu` / `device_spyre`

Upstream `test_causal_backend_correctness` (allow-listed):
- `single_decode` / `single_prefill` / `small_decode` / `small_prefill` / `mixed_small`
- `tensor_parallel_size ∈ {1, 2, 4}` (model is `meta-llama/Meta-Llama-3-8B`, GQA)

## Attention-specific hypotheses to add to the queue

When the failing surface is attention, in addition to the generic starter set, also consider:

- **`block_size < 64` trips the matmul stick layout.** Spyre's matmul reduction dim must fit on the 64-element device stick. A vLLM caller that passes a smaller `block_size` (the upstream default is 16) trips `cannot restickify ... y_var=d2` at compile time. The platform check forces `block_size=64` for the framework; `SpyreAttentionImpl.get_supported_kernel_block_sizes() = [MultipleOf(64)]`. If a new caller surfaces this error again, check the framework block_size before anything else.
- **`_reshape_and_cache` writes new K/V into the wrong block/offset.** Pages live as a `list[Tensor]` of `[num_kv_heads, block_size, head_size]`; the per-token write is `_overwrite(k_tok, k_pages[block_indices[t]], dims=[1], offsets=[block_offsets[t]])`. If `slot_mapping` (`physical_block_index * block_size + block_offset`) decomposes wrong in `SpyreAttentionMetadataBuilder`, new K/V land in the wrong slot — historical reads then mix new and old values. Print `block_indices` / `block_offsets` for one prefill+decode cycle.
- **Attention mask off-by-one / wrong padded shape / wrong causal boundary.** `_build_attention_mask` (line 387) pads to `aligned_max_query_len` and `aligned_max_seq_len` (round-up to `KV_LENGTH_ALIGNMENT=256`) and emits `mask_tiles_all[seq_idx][block]`. Off-by-one in causal/padding masking produces near-100 % mismatches that look like "random output." Print one tile and verify masked positions are `-65504` (fp16 -inf proxy), unmasked are `0`, and the causal boundary matches `context_lens + q_pos`.
- **Per-seq output overwrite in `_online_softmax_attention` collides between seqs.** The forward loops `for seq_idx in range(num_seqs)` and writes `_overwrite(tok, output, [0], [q_start + i])` per query token. If `query_start_loc` ranges overlap (or are computed wrong upstream in the metadata builder) two seqs stomp each other's hidden states. Backend-level multi-seq tests pass, so any failure here usually traces back to wrong metadata, not the loop.
- **A shape that escapes the `KV_LENGTH_ALIGNMENT=256` / `QUERY_CHUNK_SIZE=32` buckets** dispatches to an untested kernel path (correctness, not just recompilation cost).

## Pipeline bisection for `forward()`

`SpyreAttentionImpl.forward` (line 652) has three clean stages:

1. **`_reshape_and_cache`** (line 703) — write new K/V tokens into `k_pages[block_indices[t]][:, block_offsets[t], :]`.
2. **`_online_softmax_attention`** (line 730) — per-seq loop over pages: build `q` reshape `[num_kv_heads, num_queries_per_kv, padded_q, head_size]`, then for each block call `specialized_paged_attn_kernel` (line 218) which does `_indirect_matmul_mock(q, k_pages, ...)` (Q @ K^T) → mask add → online softmax → `_indirect_matmul_mock(probs, v_pages, ...)` (attn @ V).
3. **Per-token output overwrite** — write each query token's result back into `output` at `q_start + i`.

To bisect, instrument by reading `k_pages[block_indices[0]].cpu()` after step 1 and comparing to a CPU reference that does the same scatter. If pages are correct, the bug is downstream in step 2's attention math; if not, it's upstream in metadata or `_overwrite`.

## Per-row magnitude dump for broadcast/dispatch bugs

When you see huge values (near fp16 max, `~60000+`) in the attention output, the bug is usually in a specific broadcast or dispatch axis of a Spyre kernel rather than the math. Inside `specialized_paged_attn_kernel` after the per-block `tile_output` accumulator update (or right before `output` is written in `_online_softmax_attention`), print:

```python
ts = tile_output.to("cpu")  # [num_kv_heads, num_queries_per_kv, padded_q, head_size]
per_row_max = ts[:, :, 0, :].reshape(-1, ts.shape[-1]).abs().max(dim=-1).values
# Row layout: [kv_head, q_per_kv] in row-major order (0→NUM_KV_HEADS*QPK-1)
for i, m in enumerate(per_row_max.tolist()):
    kv_h, qpk = divmod(i, ts.shape[1])
    mark = "  <<< OVERFLOW" if m > 100 else ""
    print(f"  kv_h={kv_h} qpk={qpk}  max|out|={m:.4f}{mark}")
```

Look for **periodic patterns**: if every Nth row is huge and the rest are sane (e.g. every odd `q_per_kv`), that's near-certain evidence of a broadcast bug along that axis in either the matmul or softmax kernel. A workaround is usually `.expand(-1, N, -1, -1).contiguous()` on the singleton dim of `q` / a `k_pages` slice / `mask_tile` before the call — diagnostic (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix.

## Attention-specific gotcha: `_maybe_compile` wraps the per-shape kernels under pytest

The platform docs/comments suggest "`torch.compile` is globally off in this repo," which is only true when `cfg.mode == CompilationMode.NONE` (i.e. `0`). In the pytest `default_vllm_config` fixture, `CompilationConfig(custom_ops=["all"])` leaves `cfg.mode = None` (Python `None`, not the `NONE=0` enum). `_maybe_compile`'s first `if` (`cfg.mode == CompilationMode.NONE or cfg.backend == "eager"`) therefore falls through, so the kernels returned by `_get_reshape_fn` (specialized `_reshape_and_cache`) and `_get_attn_fn` (specialized `specialized_paged_attn_kernel`) end up wrapped with `torch.compile(..., dynamic=False)`. If you're debugging a kernel-looking bug and trying to reproduce it in a plain script with no `torch.compile`, you may be comparing apples to oranges.

## CPU-vs-Spyre standalone repro (attention-flavored)

`SpyreAttentionImpl.forward` now derives the target device from `k_pages[0].device` (not from a `_target_device` field on `self`). To run the same forward on CPU instead of Spyre, just allocate the K/V page list on CPU:

```python
# Pseudocode under logs/<slug>/repro_cpu.py:
device = torch.device("cpu")  # or torch.device("spyre")
k_pages = [torch.zeros(num_kv_heads, block_size, head_size, dtype=torch.float16, device=device)
           for _ in range(num_blocks)]
v_pages = [torch.zeros_like(k_pages[0]) for _ in range(num_blocks)]
# ... build SpyreAttentionMetadata with the same `block_table` / `slot_*` / `mask_tiles` / etc.
impl.forward(layer, q, k, v, (k_pages, v_pages), metadata, output)
```

`tests/test_spyre_attn.py` already parametrizes `configure_device ∈ {"cpu", "spyre"}`, so for most coverage gaps the cleanest first step is to add a parametrize ID rather than write a standalone script. The pytest fixture `requires_spyre` will still skip device-spyre tests on CPU-only hosts, so on a CPU-only box you can rely on the device_cpu parametrizations alone. If the CPU variant passes and the Spyre variant fails, the bug is in torch-spyre's realization of the operation; if both fail, the bug is in our own logic.
