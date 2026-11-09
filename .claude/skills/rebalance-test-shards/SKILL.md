---
name: rebalance-test-shards
description: Recompute the CI shard count per test suite from recorded durations and apply it to the `shards:` values in .github/test-matrix.yaml. Use when a test suite has grown or shrunk and a shard job now runs too long (or wastefully short), when someone asks to rebalance/resize the test shards, or after a large batch of tests is added or removed. CI and the Makefile both read that one value (guarded by tests/test_sharding.py).
---

# Rebalance test shards

CI fans each sharded suite out across parallel jobs and balances them by measured
runtime; this skill resizes the shard **count** so the busiest shard's test time stays
under the wall-clock budget (default 10 min) without over-sharding. It touches CI config
only — no production code.

The count lives in one place: the `shards:` value on the suite's entry in
`.github/test-matrix.yaml`. The plan job expands it into that many CI jobs, and the
Makefile's `*_SHARDS` defaults read the same value, so a local run splits the suite the
way CI does.

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

For every suite whose `REC` differs from its current count, set `shards: <REC>` on the
entry whose `test_target` is `test-<name>-shard` (`<name>` = smoke / smoke-unit /
e2e-smoke / e2e-smoke-dist / attention / upstream / distributed / probes / quality).
Nothing else changes: the job names, `(shard k/N)` labels and per-shard targets are
generated, and the pod-level retry rebuilds its matrix from the failed legs.

The e2e-smoke suites are sized by the `e2e_smoke` admission budget
(`docs/contributing/integration-tier.md`), not just the per-shard ceiling: if their total
has grown past ~90 min, flag it to the maintainer instead of only adding shards.

## 3. Verify

```bash
uv run pytest tests/test_sharding.py -m "not upstream" -q
```

`test_makefile_reads_every_shard_count_from_the_matrix` confirms the Makefile resolves
the new count for each suite's shard target.

## 4. Open the PR

Hand off to the `prepare-pull-request` skill. In the body, state the old→new counts per
suite and the durations run the numbers came from.
