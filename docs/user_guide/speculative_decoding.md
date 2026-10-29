# XPress speculative decoding

XPress runs a DFlash backbone once, refines a block of draft tokens, and lets
the target model verify the proposals. The experimental Spyre integration loads
the published `UIUC-SSAIL/Qwen3-8B-XPress-b16` checkpoint directly with
`Qwen/Qwen3-8B`; checkpoint conversion is not required.

Supported configuration: compiled FP16, unquantized weights, TP=1, PP=1,
the V1 runner, one active request, and prefix caching
disabled. The checkpoint fixes the block at one anchor plus 15 proposals.

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
The end-to-end test pins the same model and checkpoint revisions as this example;
update both together. A full verification round needs at least 31 budgeted tokens because
vLLM reserves additional drafting slots; the adapter rejects a smaller budget.

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

The pinned vLLM 0.32 lacks native XPress registration, so this adapter uses its V1
DFlash classes with a configuration bridge. The merged GPU speculator uses V2.
Initial selection and per-pass argmax run on CPU; readout and refinement run on
Spyre. Jagged attention and compiler extensions are optional follow-up work.

Spyre's V2 runner is still a scaffold: model loading, KV-cache setup, compilation
and warmup are unimplemented. After that runner is operational, XPress needs its
auxiliary-state, draft-context and CPU-sampling hooks ported to V2's speculator
interface. See [the V2 migration issue](https://github.com/torch-spyre/spyre-inference/issues/906).

## Validation

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_speculative_decoding.py -m model_quality -s
```

CPU recurrence and lifecycle tests pass. Device qualification remains open:
the strict target-token comparison exposes a shape-dependent target score tie
between ordinary decoding and block verification. The equality assertion remains
in the test. Repeat device validation with C512 when qualifying that configuration.

The stochastic hardware test currently exposes an upstream CUDA-generator offset
rewind during seeded chunked prefill. CPU generators need state restoration at
that boundary; shorter seeded reference runs pass. This remains a draft integration.

The 503-token case starts too close to the context limit to draft. A separate
495-token case verifies that a previously nonempty draft is cleared at the limit
and that the next target step has no scheduled draft tokens. Validation timings
are collected by the test worker; the serving loop has no timing counters.
