# Attention-specific debugging notes

Companion to `SKILL.md`. Read this when the failing surface is the Spyre attention backend specifically. The generic workflow (cluster → HTML log → hypothesis queue → cluster validation) in `SKILL.md` still applies; this file just adds attention-flavored content to the hypothesis queue and pipeline-bisection steps.

## Files that matter for attention

- `spyre_inference/v1/attention/backends/spyre_attn.py` — the token-major backend, `SpyreAttentionImpl`, `SpyreAttentionMetadataBuilder` (shared by every Spyre backend), and the per-sequence / batched-decode dispatch.
- `spyre_inference/v1/attention/backends/spyre_head_major_attn.py` — the default head-major backend (`SPYRE_ATTN_KV_LAYOUT=head_major`); overrides the allocation, the KV-write index and the kernels.
- `spyre_inference/v1/attention/backends/spyre_encoder_attn.py` — encoder-only (no KV cache) attention.
- `spyre_inference/v1/attention/ops/` — the compiled kernels (`page_attn*.py`, `batched_decode*.py`, `reshape_and_cache*.py`), the cache device layouts (`layout.py`) and the tiled-walk driver (`tile_loop.py`). Kernel docstrings and inline comments name the torch-spyre limitation each shape choice routes around.
- `spyre_inference/v1/attention/attn_layer.py` — the traced KV write and query staging that replace `Attention.forward`.
- `tests/attention/test_spyre_head_major_attn.py` — the same checks for `SpyreHeadMajorAttentionImpl`, plus its write index, store fusion and a head-major-vs-token-major comparison.
- `tests/attention/test_spyre_attn.py` — builds real metadata via `SpyreAttentionMetadataBuilder`, calls the token-major `SpyreAttentionImpl.forward` on a Spyre device, compares against a CPU reference (`ref_attn`) with `assert_close_outliers` (`atol=0.3, rtol=0.2` when a query is 32+ rows, `atol=0.2, rtol=0.2` otherwise, up to 5 outliers at twice that) — if mismatch ratios are near 100 % with differences > 1.0, suspect a **structural** bug, not fp16 noise.

## Attention-specific limitations

In addition to the generic table in `SKILL.md`:

| Limitation | Workaround |
|---|---|
| KV alignment to bucketed length | `SpyreAttnBucketer` kv buckets: powers of two from `block_size` to `max_model_len`, consumed as a padded block count, so the same compiled kernel is reused as KV grows |
| Query length must be bucketed | `SpyreAttnBucketer` query buckets: `[1] + multiples of min(512, max_num_batched_tokens)`; each sequence's query is padded up to its own bucket |
| `head_size` must be a multiple of 64 | `SpyreAttentionBackend.supports_head_size` enforces this (128-byte stick / 2 bytes for fp16) |
| Head-major page gather is a 2-D subscript (`aten.index`), which fails eager | The head-major backend always compiles attention, even under `--enforce-eager`; batched decode is declined whenever attention is eager |

MHA and MQA (`test_spyre_attn_mha` / `test_spyre_attn_mqa`, and `test_head_major_attn_head_configs`), GQA, sliding window, soft-capping and ALiBi (token-major only) all have committed parametrizations; check which layout a failure ran on before assuming a head configuration is the cause.

## Useful test selectors (orientation only — not for `-k`)

These appear in current parametrize IDs. They're for reading collected node IDs; `-k` will not parse the parens/equals/commas.

- `decode(q=1,kv=256)` / `decode(q=1,kv=512)`
- `prefill(q=32,kv=256)` / `prefill(q=33,kv=96)`
- `batch_decode(2seqs)` / `batch_prefill(2seqs)` / `mixed(decode+prefill)`
- `kv_padded_decode(q=1,kv=300)` / `kv_padded_prefill(q=32,kv=65)`
- `head_size(64)` / `head_size(128)`, `block_size(64)` / `block_size(128)` / `block_size(256)`
- `swa_4` / `swa_16`, `soft_cap(50)`
- `device_cpu` / `device_spyre`, `compilation_NONE` / `compilation_STOCK`

## Attention-specific hypotheses to add to the queue

When the failing surface is attention, in addition to the generic starter set, also consider:

- **KV write went to the wrong rows.** The write is `index_copy_` through a slot-major view of the pages (`kv_slot_views`), and on head-major one index tensor per KV head (`kv_write_index`). The cache must carry the device layout its impl allocated (`slot_major_kv_layout` / `head_major_kv_layout`): the default tiled layout silently scatters to the wrong rows. A fixture that builds its own cache tensor rather than calling `allocate_pages` is the usual culprit.
- **Head-major write source not materialized.** A single-token fused-QKV view reports contiguous at a nonzero storage offset; the store relies on an elementwise pass (`key * 1.0`) to copy it. Short-token tests pass while long prefill is wrong when this, or the per-head index split, regresses.
- **Attention mask off-by-one / wrong padded shape / wrong causal boundary.** `_build_attention_mask` (no sliding window) and `_build_single_tile` (sliding window) build per-block tiles padded to `aligned_query_len × block_size`. Off-by-one in causal/padding masking produces near-100 % mismatches that look like "random output." Print the mask stack for a small case and verify masked positions are `finfo(fp16).min` (-65504), unmasked are `0`, and the causal boundary matches `context_lens + q_pos`.
- **Page index table wrong.** `page_index_tables_cpu[s][i, 0]` must be the physical page of sequence `s`'s `i`-th *active* block (a position within `active_block_indices` under a window, not an absolute block index). A wrong order leaves the mask tiles misaligned with the pages they mask — silent disaster.
- **Batched vs per-sequence path.** Decode batches of 4+ sequences take the batched kernel when it is supported (compiled, no ALiBi, and under the default tiled walk only on head-major); set `SPYRE_BATCHED_DECODE=0` to force the per-sequence loop and see whether the failure follows the path.
- **Tiled vs Python walk.** `SPYRE_ATTN_FOR_EACH_TILE=0` runs the identical kernel bodies under a Python loop; a failure that disappears there is a `for_each_tile` / torch-spyre pin problem. The flag is read at import.
- **A shape that escapes the `SpyreAttnBucketer` kv/query buckets** dispatches to an unrecorded kernel (correctness, not just recompilation cost).

## Pipeline bisection for `forward()`

A layer's attention runs as: KV write (`do_kv_cache_update`, traced by `attn_layer`) → query staging → per-sequence kernels (`_run_page_attn`) or the batched decode kernel (`_run_batched_decode`) → write-back into the output. Rerun each stage against a reference and compare:

- After the write, read the K/V pages back to CPU and check the rows `slot_mapping` names hold the new tokens.
- For the kernel, call the kernel function from `v1/attention/ops/` directly (eager, on CPU) with the same metadata the builder produced; it is plain PyTorch, so it runs on CPU for a reference.

A diff at the write step points at the index/layout; a diff only after the kernel points at the matmul / softmax / mask path.

## Per-row magnitude dump for broadcast/dispatch bugs

When you see huge values (near fp16 max, `~60000+`) in the attention output, the bug is usually in a specific broadcast or dispatch axis of a Spyre kernel rather than the math. Pull the result of one kernel call to CPU (`[padded_query_len, num_heads, head_size]`) and print the per-head magnitude:

```python
out = result.to("cpu")                       # [q, num_heads, head_size]
per_head_max = out.abs().amax(dim=(0, 2))    # heads are kv-major: h = kv_h * q_per_kv + g
for h, m in enumerate(per_head_max.tolist()):
    kv_h, g = divmod(h, num_heads // num_kv_heads)
    mark = "  <<< OVERFLOW" if m > 100 else ""
    print(f"  kv_h={kv_h} group={g}  max|out|={m:.4f}{mark}")
```

Look for **periodic patterns**: if every Nth head is huge and the rest are sane (e.g. every odd query group), that's near-certain evidence of a broadcast bug along that axis in either the matmul or softmax kernel. A workaround is usually `.expand(...).contiguous()` on the singleton dim of `k`/`v`/`mask` before the call — diagnostic (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix.

## Attention-specific gotcha: which mode the impl compiles in

`SpyreAttentionImpl` samples `get_current_vllm_config().compilation_config.mode` at construction and compiles its per-sequence kernel only when it equals `STOCK_TORCH_COMPILE`. The `default_vllm_config` fixture resolves to `STOCK_TORCH_COMPILE` (the platform hook sets it), so an impl built under it compiles unless the test's `configure_compilation` fixture sets `NONE`. The head-major decode, prefill and batched kernels are compiled regardless. If you're reproducing a kernel-looking bug in a plain script, match the mode the failing test used, or you may be comparing apples to oranges.

## CPU-vs-Spyre standalone repro (attention-flavored)

The tests already parametrize `device_cpu` / `device_spyre`: run the failing node id with `device_cpu` swapped in first. For a standalone repro under `logs/<slug>/repro_cpu.py`, copy the setup `_run_spyre_attn_test` does — plain CPU page tensors, metadata from `SpyreAttentionMetadataBuilder.build`, an impl under a `set_current_vllm_config` context — and call the kernel from `v1/attention/ops/` directly. (`allocate_pages` needs a Spyre device: its layouts come from `torch_spyre._C`.) If the CPU variant passes, the bug is in torch-spyre's realization of the operation; if it still fails on CPU, the bug is in our own logic. Note that a compiled attention kernel does not run on CPU (Inductor's C++ codegen rejects it), so compare against the eager kernel.
