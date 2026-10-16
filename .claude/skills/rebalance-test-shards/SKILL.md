---
name: rebalance-test-shards
description: Recompute the CI shard count per test suite from recorded durations and apply it to the Makefile and the CI matrix. Use when a test suite has grown or shrunk and a shard job now runs too long (or wastefully short), when someone asks to rebalance/resize the test shards, or after a large batch of tests is added or removed. Keeps the Makefile *_SHARDS counts and the _test_matrix.yaml job entries in lockstep (guarded by tests/test_sharding.py).
---

# Rebalance test shards

CI fans each suite out across `*_SHARDS` parallel jobs and balances them by measured
runtime; this skill resizes the shard **count** so the busiest shard's test time stays
under the wall-clock budget (default 10 min) without over-sharding. It touches CI config
only — no production code.

The count lives in **two** places that a cross-check test forces to agree: the Makefile
`*_SHARDS` default and the per-shard job blocks in `.github/workflows/_test_matrix.yaml`.
Change both together or the suite silently loses (or duplicates) shards.

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

## 2. Apply each changed suite to BOTH files

For every suite whose `REC` differs from its current count, set N = REC and edit both:

**Makefile** — bump the one default:

| suite | Makefile var |
|-------|--------------|
| smoke | `SMOKE_SHARDS` |
| attn | `ATTN_SHARDS` |
| upstream | `UPSTREAM_SHARDS` |
| dist | `DIST_SHARDS` |
| probe | `PROBE_SHARDS` |

**`.github/workflows/_test_matrix.yaml`** — under the `test:` job's `matrix.include:`, each
suite has N consecutive `- cfg:` blocks keyed by `test_target: test-<name>-shard-<i>`
(`<name>` = smoke / attention / upstream / distributed / probes):

- **Growing** (N larger): copy the block **directly above the insertion point** — already
  a block of this same suite — and change only two things in the copy: its `test_target`
  id (the next 0-based `...-shard-<i>`) and its `(shard k/N)` label. Do **not** hand-type
  the block or borrow one from a neighbouring suite. `test_types`, `runs_on`, and
  `image_label` differ between suites and must be identical within one; in particular
  `test_types` decides which meta-suites a shard joins — attention, probes, and upstream
  are deliberately **not** `integration`, while smoke and dist are — so a block cloned
  from the wrong suite silently files the new shard under the wrong coverage set (and the
  id cross-check in step 3 won't catch it; the uniformity guard will).
- **Shrinking** (N smaller): delete the surplus highest-id blocks.
- **Always** renumber the `(shard k/N)` text in every block of that suite to the new N
  (e.g. `Spyre attention tests (shard 3/9)`). The label is cosmetic; the `test_target` id
  is what runs.

Do **not** touch the `test_retry` job — its matrix is built dynamically from
`collect_failed_suites`, so it inherits the new shards automatically.

## 3. Verify the two files agree

```bash
uv run pytest tests/test_sharding.py -m "not upstream" -q
```

Two guards in that file must pass before you open the PR:

- `test_matrix_shard_entries_match_makefile_counts` — the matrix shard ids are exactly
  `0..N-1` for the Makefile's N. A mismatch means a shard's tests never run in CI.
- `test_matrix_shard_blocks_within_a_suite_are_uniform` — every block of a suite shares
  the same `test_types`, `runs_on`, and `image_label`. This catches a shard cloned from
  the wrong suite (e.g. an attention shard that accidentally carries `integration`).

## 4. Open the PR

Hand off to the `prepare-pull-request` skill. In the body, state the old→new counts per
suite and the durations run the numbers came from.
