# Attention-specific debugging notes

Companion to `SKILL.md`. Read this when the failing surface is the Spyre attention backend specifically. The generic workflow (cluster → HTML log → hypothesis queue → cluster validation) in `SKILL.md` still applies; this file just adds attention-flavored content to the hypothesis queue and pipeline-bisection steps.

## Files that matter for attention

- `spyre_inference/v1/attention/backends/spyre_attn.py` — the backend, `SpyreAttentionImpl`, `SpyreAttentionMetadataBuilder`. The module docstring enumerates every torch-spyre limitation the backend routes around.
- `tests/test_spyre_attn.py` — builds real metadata via `SpyreAttentionMetadataBuilder`, calls `SpyreAttentionImpl.forward` on a Spyre device, compares against a CPU reference (`ref_attn`). Tolerances are loose (`atol=0.3, rtol=5.0` for prefill; `atol=0.2, rtol=0.2` for decode) — if mismatch ratios are near 100 % with differences > 1.0, suspect a **structural** bug, not fp16 noise.

## Attention-specific limitations

In addition to the generic table in `SKILL.md`:

| Limitation | Workaround |
|---|---|
| KV alignment to bucketed length | `KV_LENGTH_ALIGNMENT=256` so the same compiled kernel is reused as KV grows |
| Query length must be bucketed | `QUERY_CHUNK_SIZE=32`; queries are chunked into multiples of this |
| `head_size` must be a multiple of 64 | `SpyreAttentionBackend.supports_head_size` enforces this (128-byte stick / 2 bytes for fp16) |
| MHA / MQA head configs fail to compile | Two test parametrizations are `pytest.mark.skip`'d with `reason="Spyre compilation fails for MHA/MQA head config"`. Only GQA is exercised. |
| `num_seqs=1` only | The forward path assumes a single sequence at a time |

## Useful test selectors (orientation only — not for `-k`)

These appear in current parametrize IDs. They're for reading collected node IDs; `-k` will not parse the parens/equals/commas.

- `decode(q=1,kv=256)` / `decode(q=1,kv=1024)`
- `prefill(q=32,kv=256)` / `prefill(q=64,kv=512)` / `prefill(q=100,kv=512)`
- `GQA` (MHA and MQA are skipped)
- `head_size(128)` / `head_size(256)`
- `num_blocks(2048)` / `num_blocks(32768)`

## Attention-specific hypotheses to add to the queue

When the failing surface is attention, in addition to the generic starter set, also consider:

- **Device KV cache not seeded from the passed-in `kv_cache`** (historical context lost). `SpyreAttentionImpl._init_device_cache` creates a fresh **zeroed** on-device cache; it does NOT seed from the `kv_cache` argument. The test fixture writes historical KV into the CPU `kv_cache`. For `decode(q=1, kv=N)` cases, the only KV values the device ever sees are the new token scattered in — historical tokens live on CPU and are never transferred. This is the canonical structural-bug hypothesis to check first when decode tests fail with near-100% mismatch.
- **Attention mask off-by-one / wrong padded shape / wrong causal boundary.** `_build_attention_mask` pads to `padded_query_len` and `aligned_max_seq_len`. Off-by-one in causal/padding masking produces near-100 % mismatches that look like "random output." Print the mask for a small case and verify masked positions are `-65504` (fp16 -inf proxy), unmasked are `0`, and the causal boundary matches `context_lens + q_pos`.
- **Scatter selectors wrong** (row/col one-hot mismatched with `slot_mapping`). A diff > 0 at the scatter step means the one-hot selectors are wrong.
- **Gather selector picks physical blocks in wrong order** → compact KV order mismatch with mask. `_gather_from_device_cache` does `bmm(sel_mask, cache)` then reshapes to `[num_kv_heads, aligned_max_seq_len, head_size]`. If the physical block order in `sel_mask` is wrong, the gathered tokens are in the wrong order and the mask no longer matches — silent disaster.
- **A shape that escapes the `KV_LENGTH_ALIGNMENT=256` / `QUERY_CHUNK_SIZE=32` buckets** dispatches to an untested kernel path (correctness, not just recompilation cost).

## Pipeline bisection for `forward()`

`SpyreAttentionImpl.forward` has five clean stages: scatter → gather → reshape → attention → slice. Rerun each stage on CPU against a reference and compare:

```python
# Rough scaffold
impl._init_device_cache(num_blocks, block_size)
impl._scatter_to_device_cache(...); k_dev = impl._k_cache_dev
compact_k, compact_v = impl._gather_from_device_cache(...)
# Pull to CPU and diff vs expected layout
```

A diff > 0 at the scatter step means the one-hot selectors are wrong. A diff only after attention means the `_attn_4d` path (matmul / softmax / mask add) is the issue.

## Per-row magnitude dump for broadcast/dispatch bugs

When you see huge values (near fp16 max, `~60000+`) in the attention output, the bug is usually in a specific broadcast or dispatch axis of a Spyre kernel rather than the math. Right after `self._attn_4d(q_spyre, k, v, self.scale, mask)` in `_compute_attention`, print:

```python
osp = output_spyre.to("cpu")
per_row_max = osp[:, :, 0, :].reshape(-1, osp.shape[-1]).abs().max(dim=-1).values
# Row layout: [kv_head, q_per_kv] in row-major order (0→NUM_KV_HEADS*QPK-1)
for i, m in enumerate(per_row_max.tolist()):
    kv_h, qpk = divmod(i, output_spyre.shape[1])
    mark = "  <<< OVERFLOW" if m > 100 else ""
    print(f"  kv_h={kv_h} qpk={qpk}  max|out|={m:.4f}{mark}")
```

Look for **periodic patterns**: if every Nth row is huge and the rest are sane (e.g. every odd `q_per_kv`), that's near-certain evidence of a broadcast bug along that axis in either the matmul or softmax kernel. A workaround is usually `.expand(-1, N, -1, -1).contiguous()` on the singleton dim of `k`/`v`/`mask` before the call — diagnostic (if expanding makes the overflow go away, the broadcast path is broken) even when it isn't the final fix.

## Attention-specific gotcha: `_maybe_compile` does wrap `_attn_4d` under pytest

The platform docs/comments suggest "`torch.compile` is globally off in this repo," which is only true when `cfg.mode == CompilationMode.NONE` (i.e. `0`). In the pytest `default_vllm_config` fixture, `CompilationConfig(custom_ops=["all"])` leaves `cfg.mode = None` (Python `None`, not the `NONE=0` enum). `_maybe_compile`'s first `if` therefore falls through and `cfg.backend == "inductor"` (default) fails the second `if` too, so `_attn_4d` ends up wrapped with `torch.compile(..., dynamic=False)`. If you're debugging a kernel-looking bug and trying to reproduce it in a plain script with no `torch.compile`, you may be comparing apples to oranges.

## CPU-vs-Spyre standalone repro (attention-flavored)

For attention specifically, a useful standalone-repro recipe under `logs/<slug>/repro_cpu.py`:

```python
# Pseudocode — instantiate SpyreAttentionImpl, then:
impl._target_device = torch.device("cpu")
# also do the same in SpyreAttentionMetadataBuilder
# then run forward() on the same inputs the failing parametrization uses
```

The pytest fixture `requires_spyre` will still skip the real test on CPU-only hosts, so don't rely on editing `_target_device` and rerunning pytest — do it in the standalone script instead. If the CPU variant passes, the bug is in torch-spyre's realization of the operation; if it still fails on CPU, the bug is in our own logic.
