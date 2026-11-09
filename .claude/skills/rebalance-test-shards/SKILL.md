---
name: rebalance-test-shards
description: Recompute the CI shard count per test suite from recorded durations and apply it to the Makefile *_SHARDS defaults. Use when a test suite has grown or shrunk and a shard job now runs too long (or wastefully short), when someone asks to rebalance/resize the test shards, or after a large batch of tests is added or removed. The Makefile is the only copy of each count; GHA's planner and the non-GHA CI both read it (guarded by tests/test_sharding.py).
---

# Rebalance test shards

CI fans each sharded suite out across parallel jobs and balances them by measured
runtime; this skill resizes the shard **count** so the busiest shard's test time stays
under the wall-clock budget (default 10 min) without over-sharding. It touches CI config
only — no production code.

The count lives in one place: the suite's `*_SHARDS` default in the Makefile. GHA's plan
job reads it through `make print-shards` and schedules that many jobs; the P and Z CI,
which runs the Makefile directly, reads the same default.

## 1. Get the recommendation

```bash
python3 .github/scripts/rebalance_test_shards.py            # pulls newest passing main durations
# or, offline / to reuse a file:
python3 .github/scripts/rebalance_test_shards.py --durations test-durations.json
python3 .github/scripts/rebalance_test_shards.py --max-minutes 8   # tighter budget
```

The script runs `pytest --collect-only` per suite, so run it where collection imports
succeed — the CI image or a full local checkout. If it exits complaining collection
failed, the environment isn't ready; don't guess the numbers by hand.

Read the `REC` column. Suites where `REC` equals the current count need no change. A
`<- single test over budget` flag means one test alone exceeds the budget — sharding
can't fix that; leave the suite and tell the maintainer to split or speed up that test.
The `new` column is collected tests with no recorded duration yet (excluded from the
estimate); a large `new` count means the recommendation is provisional until they run
once.

## 2. Apply each changed suite

For every suite whose `REC` differs from its current count, set its Makefile default:

| suite | Makefile var |
|-------|--------------|
| smoke | `SMOKE_SHARDS` |
| smoke-unit | `SMOKE_UNIT_SHARDS` |
| e2e-smoke | `E2E_SMOKE_SHARDS` |
| e2e-smoke-dist | `E2E_SMOKE_DIST_SHARDS` |
| attn | `ATTN_SHARDS` |
| upstream | `UPSTREAM_SHARDS` |
| dist | `DIST_SHARDS` |
| probe | `PROBE_SHARDS` |
| quality | `QUALITY_SHARDS` |

Nothing else changes: `.github/test-matrix.yaml` only marks a suite `sharded: true`, and
the CI job names, `(shard k/N)` labels and per-shard targets are generated from the count.

The e2e-smoke suites are sized by the `e2e_smoke` admission budget
(`docs/contributing/integration-tier.md`), not just the per-shard ceiling: if their total
has grown past ~90 min, flag it to the maintainer instead of only adding shards.

## 3. Verify

```bash
uv run pytest tests/test_sharding.py -m "not upstream" -q
```

`test_every_sharded_matrix_suite_has_a_makefile_count` checks that every suite the matrix
marks as sharded has a count in `make print-shards`, and that its recipe shards by it.

## 4. Open the PR

Hand off to the `prepare-pull-request` skill. In the body, state the old→new counts per
suite and the durations run the numbers came from.
