# XPress speculative decoding

XPress runs a DFlash backbone once, refines a block of draft tokens, and lets
the target model verify the proposals. The experimental Spyre integration loads
the published `UIUC-SSAIL/Qwen3-8B-XPress-b16` checkpoint directly with
`Qwen/Qwen3-8B`; checkpoint conversion is not required.

Supported configuration: compiled FP16, unquantized weights, TP=1, PP=1,
the V1 runner, up to four active requests, and prefix caching
disabled. The checkpoint fixes the block at one anchor plus 15 proposals.
The four-request cap reflects the batches validated on Spyre so far. Larger
batches require additional device validation before raising this limit.

## Run Qwen3-8B

```bash
export SPYRE_MODEL_RUNNER=v1
export SPYRE_ATTN_QUERY_BUCKETS=1,16,64
uv run --no-sync vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype float16 --tensor-parallel-size 1 \
  --max-num-seqs 1 --max-model-len 512 --max-num-batched-tokens 64 \
  --no-enable-prefix-caching \
  --compilation-config '{"compile_sizes": [1, 16, 64]}' \
  --speculative-config '{"method": "dflash", "model": "UIUC-SSAIL/Qwen3-8B-XPress-b16", "revision": "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5", "num_speculative_tokens": 15, "xpress_topc": 128}'
```

Requests can use greedy (`temperature=0`) or stochastic target sampling,
including temperature, top-k and top-p. The drafter remains greedy;
vLLM treats its proposals as point-mass distributions during CPU rejection
sampling. The same speculative configuration works with `vllm.LLM`.
The end-to-end test and CI's public-model cache list pin the same model and
checkpoint revisions as this example; update all three together. A full
verification round needs at least 31 budgeted tokens because vLLM reserves
additional drafting slots; the adapter rejects a smaller budget.

For four concurrent requests, set `--max-num-seqs 4`,
`--max-num-batched-tokens 128`, and
`--compilation-config '{"compile_sizes": [1, 16, 32, 64, 128]}'`.
Set `SPYRE_ATTN_QUERY_BUCKETS=1,16,64,128` for this token budget.
Set `SPYRE_MAX_NUM_PARTIAL_PREFILLS=4` to allow four partial prefills.
The draft body uses batches of 1, 2, or 4; three active requests pad to four.
Each request retains its own accepted context, page table, and output limit.

The batch-four test also covers default attention buckets, compilation sizes
and the default single partial prefill. The settings above reduce query padding
and allow more concurrent partial prefills.

| Maximum requests | Minimum token budget |
| --- | --- |
| 1 | 31 |
| 2 | 62 |
| 3 | 93 |
| 4 | 124 |

## Refinement and compatibility

The refiner follows [merged vLLM XPress](https://github.com/vllm-project/vllm/pull/54448):
a fixed anchor, causal Jacobi updates, a block-wide hidden summary, and candidate
sets fixed across passes. Target input embeddings and output-head weights are
shared separately. Accepted target states populate the draft KV cache; rejected
and padded rows write only the reserved null page.

`xpress_topc` overrides the checkpoint field. Spyre defaults to 128 when neither
supplies a value; explicit 512 retains the upstream default and 0 enables
full-vocabulary refinement. A smaller shortlist can change acceptance. The
checkpoint selects the pass count, with six as the default.

Replaying a seeded request with the same configuration is deterministic.
Changing between ordinary decoding, DFlash and XPress changes RNG consumption;
the same seed need not produce identical tokens across these modes.

Upstream serves XPress only on the V2 runner, so Spyre subclasses its XPress model
and drives refinement from the V1 DFlash proposer. By default, initial selection
and per-pass argmax run on CPU; readout and refinement run on Spyre.

Spyre's V2 runner still lacks model loading, compilation and warmup. After that
runner is operational, XPress needs its auxiliary-state, draft-context and
CPU-sampling hooks ported to V2's speculator interface. See
[the V2 migration issue](https://github.com/torch-spyre/spyre-inference/issues/906).

## Optional device feedback

`SPYRE_XPRESS_DEVICE_FEEDBACK=1` keeps per-pass selection and proposal feedback
on Spyre. It defaults to zero and requires a shortlist of 1 to 1024 candidates.
DL16 represents these shortlist positions and their tie-breaking sentinel
exactly. The block size must match the checkpoint's learned mixer, as on CPU.
The initial vocabulary top-k still runs on CPU. Each refinement pass selects
the next token embeddings directly from the shortlist; only the final shortlist
positions return to CPU to recover token IDs.

This caches an additional input embedding per candidate for the current draft
block. Selecting those embeddings avoids the device integer-layout conversions
needed to feed token IDs back into an embedding lookup. Full-vocabulary device
selection remains follow-up work.

## Validation

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
uv run --no-sync pytest tests/e2e/test_xpress_feedback.py -m model_quality -s
SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_speculative_decoding.py tests/e2e/test_xpress_batch.py -m model_quality -s
SPYRE_XPRESS_DEVICE_FEEDBACK=1 uv run --no-sync pytest \
  tests/e2e/test_xpress_batch.py -m model_quality -s
```

PR regression jobs prefetch both pinned models before running the quality gate.
On Spyre, missing checkpoints fail the tests instead of skipping coverage.

CPU recurrence, sampling and lifecycle tests pass. At C128, device validation
passes seven reference cases and 63 strict greedy token comparisons across
scripted rejection, full acceptance, DFlash and XPress. Stopping, cancellation
and cache reuse checks also pass, including exact controlled-cache comparisons.

A known numerical limitation remains: nearly tied FP16 target scores can select
different greedy tokens between ordinary decoding and block verification. The
counting prompt avoids the previously observed Spain/Germany tie; the underlying
numerical difference is not fixed. Repeat device validation with C512 when
qualifying that configuration.

Seeded chunked prefill preserves CPU RNG state when discarding intermediate
samples. At C128, stochastic device validation passes 24 generations across
ordinary decoding, DFlash and XPress: temperature, top-k, top-p, and combined
filtering near the context limit, each repeated with exact token and cache replay.

The 503-token case starts too close to the context limit to draft. A separate
495-token case verifies that a previously nonempty draft is cleared at the limit
and that the next target step has no scheduled draft tokens. Validation timings
are collected by the test worker; the serving loop has no timing counters.

The batch-four test enables `SPYRE_COMPILE_GUARD=error`, including for partial
prefill and shrinking batches. Warmup compiles the draft context KV stores at
each target row bucket using the reserved null page.
