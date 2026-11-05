# The Integration Tier

The `integration` tier is the check a torch-spyre (or Spyre RPM) PR pays for on every push.
It answers one question: does this change break a Spyre-specific path in spyre-inference?
It is not where correctness, accuracy or model coverage is proven; `trunk` and the quality
gate do that, and they keep running every e2e test.

## What runs

| Job | Selection | Runner |
|-----|-----------|--------|
| Unit-level shards (`test-smoke-unit-shard-*`) | the smoke marker expression, `--ignore=tests/e2e` | 1 card |
| e2e smoke shards (`test-e2e-smoke-shard-*`) | `e2e_smoke and not distributed` | 1 card |
| e2e smoke TP=2 shards (`test-e2e-smoke-dist-shard-*`) | `e2e_smoke and distributed` | 2 cards |
| Attention smoke (`test-attention-smoke`) | `attention_smoke` | 1 card |

Selection is opt-in: `tests/e2e/` reaches the tier only through the `e2e_smoke` marker, so a
new e2e test stays out of it unless someone tags it. The tagged cases keep their other
markers, so they also run in the full smoke and distributed shards under the other tiers;
the opt-in set is a copy, not a move.

To see exactly what a torch-spyre PR pays for:

```bash
pytest -m e2e_smoke --collect-only -q
make tests TEST_TYPE=integration   # the whole tier, unsharded
```

## Admission criteria

Tag a row with `marks=pytest.mark.e2e_smoke` (or a whole function with
`@pytest.mark.e2e_smoke`) only if all of these hold:

- **Names a distinct path.** The entry covers a Spyre-specific path: a kernel, lowering,
  layout, cache, padding pass or parallel mode. No current `e2e_smoke` entry covers the same
  path. "Another model" is not a path; "the `head_size=32` opaque encoder-attention
  fallback" is.
- **Not already covered more cheaply.** If a unit-level or model-level test already in the
  tier (`tests/custom_ops/`, `tests/models/`, `tests/attention/` via `attention_smoke`)
  reaches that path on card, don't admit it. For example, the compiled Pixtral graphs are
  covered by `tests/models/test_pixtral_vision.py`, so no compiled Ministral e2e case is
  needed for them.
- **Cheapest representative.** Use the smallest model that reaches the path
  (micro-g3.3, granite-embedding-30m, or a truncated or dummy-weight config where one exists).
  Pick one mode (compiled, unless the path is eager-only) and one parametrize row, not the
  whole function. Keep `max_model_len` and the prompts short.
- **Small.** About 10 minutes or less in a cold CI run (a compile-cache miss, which is what
  a torch-spyre PR sees), not a warm local rerun. State the figure in the PR and where it
  came from (a CI run's JUnit, or `test-durations.json`).
- **Stable.** Not xfail, no open flake issue, and green on the last 10 `trunk` runs. The tier
  gates other repos' PRs, so a flaky entry costs their authors, not just us.
- **Fits the budget.** The `e2e_smoke` total stays around 90 minutes, cold. Check it with
  `pytest -m e2e_smoke --collect-only -q` against the latest `test-durations.json`. If the
  case would go over, the same PR removes or shrinks another entry (see below).

There is deliberately no CI guard: test timings swing widely with compile-cache warmth, so a
hard limit would either flake or need slack wide enough to be useless. Reviewers enforce the
policy. The same policy covers `attention_smoke`.

The compiled gemma-4 MoE case runs `joerunde/gemma-4-moe-nano`, the 26B-A4B layout at 6
layers, so the suite's only end-to-end MoE model fits the size limit.

These device paths ran in the integration tier before `e2e_smoke` and were left out on
purpose when the set was first audited. The set was already at its budget, so a torch-spyre
regression on them is caught in `trunk`, not here:

- whole-model compile granularity (`test_whole_model_granularity`); block discovery and
  the granularity switch are covered on host by `tests/runtime/test_compile_granularity.py`
- FP8 KV cache at TP1 (`test_fp8_kv_cache_tracks_the_fp16_cache`); the TP2 row is in
- logprobs independent of KV-bucket padding (`test_logprobs_do_not_depend_on_the_padded_blocks`)
- the compiled gemma-3-1b decoder (sliding-window attention); its slot went to the gemma-4 MoE case

Admitting one of these follows the swap rule below.

## The swap rule

The budget is fixed, so admitting over it means removing something. The PR author proposes
the swap, comparing the two entries' paths, cost and failure history, and the reviewer
decides. Raising the budget is a separate, explicit decision by the CI owners, never a side
effect of an admission PR.

The PR that adds the marker is the admission. Its description names the path covered, the
cold-run cost and any swap.

## Removal and review

Remove an entry when another entry or a cheaper unit-level test comes to cover its path,
when it has grown well past the size limit and can't be brought back down, or when it goes
xfail or flaky. Removing an entry only drops the marker; the test keeps running in `trunk`.

Review the whole set quarterly. Re-check that each entry's path is still distinct, and look
at which entries actually caught a torch-spyre regression. A path that has never failed is
not removed for that alone; it is still the cheap guard on its path.

## Shard counts

`SMOKE_UNIT_SHARDS`, `E2E_SMOKE_SHARDS` and `E2E_SMOKE_DIST_SHARDS` in the `Makefile` set
the counts, and `.github/test-matrix.yaml` declares one job per shard. Resize
them with the `rebalance-test-shards` skill; `tests/test_sharding.py` checks that the two
files agree.
