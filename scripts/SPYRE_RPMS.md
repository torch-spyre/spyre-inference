# `spyre_rpms.py`: ABI-compatible RPM pinning

`spyre-rpms.toml` names only **which source** each component comes from.
`scripts/spyre_rpms.py` turns it into `spyre-rpms.lock`, which names the
exact RPM files, and guarantees they are ABI compatible with each other.

```bash
# resolve the input into a lock file
./scripts/spyre_rpms.py resolve -i spyre-rpms.toml -o spyre-rpms.lock

# resolve one arch only (default when the input names no `arches`)
./scripts/spyre_rpms.py resolve -i spyre-rpms.toml -o spyre-rpms.lock -a x86_64

# CI lint: fail if the committed lock is stale
./scripts/spyre_rpms.py check -i spyre-rpms.toml -o spyre-rpms.lock

# re-check a committed lock's ABI invariant, offline
./scripts/spyre_rpms.py verify -o spyre-rpms.lock

# inspect a build plan when a resolve fails
./scripts/spyre_rpms.py show-plan --orchestrator-build 7457 --arch x86_64
```

The script is a [PEP 723](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)
single file: `uv` reads its inline metadata and installs `requests`/`pydantic`
on the fly, so `./scripts/spyre_rpms.py ...` needs no venv.

Deployment details come only from the environment, never from the toml, argv or
this source tree, because they are CI secrets in this project:

| variable | meaning |
| --- | --- |
| `ARTIFACTORY_TOKEN` | bearer token |
| `ARTIFACTORY_BASE_URL` | Artifactory host |
| `ARTIFACTORY_RPM_PATH` | repo holding the RPMs |
| `ARTIFACTORY_PLAN_PATH` | repo holding the orchestrator build plans (new) |

All four are required by `resolve`, `check` and `show-plan`; none has a default,
and the script fails with a named variable rather than guessing. `verify` needs
none of them, which is what keeps the committed lock checkable in a public repo
and on a fork PR.
`verify` needs neither, so it also gates pull requests from forks, where
[secrets are unavailable](../CI.md).

## Why a git commit is not enough

A commit does **not** uniquely identify an RPM. The orchestrator rebuilds a
component whenever its dependencies move, so one flex source commit
(`5aaf4902`) has several published RPMs that are different binaries:

| file                                                      | built against     |
| --------------------------------------------------------- | ----------------- |
| `ibm-flex-...5aaf4902.5cefcccce4cf.5aaf490.el10.x86_64.rpm` | one deeptools     |
| `ibm-flex-...5aaf4902.2feedcd04810.5aaf490.el10.x86_64.rpm` | a different one   |

The 12-hex field is the component's **identity** (`id12`), derived from the
source *and* the dependency shas:

```text
id12 = first12(sha256(version, source, recipe_sha256, recipe_files, dep_ids))
```

So picking "the newest file matching this version" mixes incompatible builds,
and it does so silently because every candidate is a real, installable RPM. In
the example above the wrong file is the newer one.

`id12` *does* identify a binary: two RPMs sharing one were compiled against the
same dependency shas.

## How compatibility is guaranteed

Every Jenkins orchestrator run publishes a build plan recording, per component,
its source sha and the exact dep shas it compiled against:

```text
<ARTIFACTORY_BASE_URL>/artifactory/<ARTIFACTORY_PLAN_PATH>/next/_nodes/orchestrator-<N>/plan-<arch>.json
```

The resolver takes that plan as the oracle. It resolves all refs against **one**
plan and enforces, for every dependency edge:

```text
consumer's recorded dep sha for <producer>  ==  <producer>'s own source sha
```

A set satisfying that for every edge was built as one coherent whole. The edges
are written into the lock, so `verify` re-checks the invariant with no network.
`resolve` refuses to emit a lock that fails it, and refuses to substitute a
different build when an RPM is missing — a partially published run is normal
while a build is still in flight, and silently reaching for another build is
exactly how an incompatible set gets assembled.

## Notes for maintainers

- **Plan arch names differ from RPM arch names.** `plan-amd64.json` describes
  the `x86_64` RPMs. Conflating them gives a confident 404.
- **Most runs are single-arch.** Only a small minority of orchestrator runs
  publish plans for all three arches, so with no `orchestrator_build` pinned the
  resolver selects the newest run covering every requested arch, not the newest
  run overall. Asking for fewer arches therefore reaches a much newer run: at
  the time of writing, unpinned, `x86_64` alone resolved orchestrator-7684 while
  all three resolved 7457. `--arches`/`-a` overrides the input's `arches` for a
  single invocation, and defaults to `x86_64` when the input names none.
- **`next/cache/<arch>` mirrors `next/<arch>`.** Queries match the arch
  directory exactly; a `next/*` glob returns every file twice.
- **Tags are rejected.** A plan records commits and branches, so a tag cannot be
  confirmed against one. Resolve it to a commit and pin that.
- **`id12` is arch-independent.** One component has the same identity on all
  arches in a run, so the lock needs no per-arch build numbers.
- Tests live in `scripts/tests/` (not `tests/`, whose `conftest.py` imports
  `spyre_inference` and needs Spyre hardware) and run on any machine:

  ```bash
  uv run --isolated --no-project --with pydantic --with requests --with pytest \
      pytest scripts/tests/ -q
  ```
