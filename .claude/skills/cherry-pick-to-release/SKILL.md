---
name: cherry-pick-to-release
description: "Cherry-pick a commit (usually a squash-merged PR) from main into a release branch, without committing. Fetches, branches `<release>-<sha7>-cp` off the latest upstream release branch, runs `git cherry-pick --no-commit`, resolves conflicts by applying only the commit's own changes, stages everything with a ready commit message (the original PR message, a cherry-pick note and links), runs the tests the commit added or changed, and verifies the staged diff against the original commit so that nothing else slips in. Use when the user asks to cherry-pick, backport or port a commit or PR to a release branch, or gives a sha or PR number together with a release branch. Stops before committing."
argument-hint: "<commit-sha | PR number> <release-branch>"
---

<!-- Keep a "$" followed by a digit out of this file (awk fields, shell positional
     parameters): the skill loader replaces it with an invocation argument. -->

# Cherry-pick a commit into a release branch

Arguments: `$ARGUMENTS` (the commit, as a sha or PR number, then the release branch). If
this is empty, take both from the user's request.

Port one commit from `main` onto a release branch and stop with everything staged and
nothing committed. The user reviews and commits. The result:

- a local branch `<release>-<short>-cp`, based on the latest `<remote>/<release>`;
- the commit's change staged, and nothing else;
- the commit message in `MERGE_MSG`, ready for `git commit -s -F`;
- the results of the tests the commit brought, and a verification report.

## Rules

- **Never commit, push or open a PR.** `git cherry-pick --continue` commits too, so don't
  run it.
- **Stage the commit's own change and nothing else.** A conflict means the release branch
  lacks earlier `main` commits. The "theirs" side of a conflict is `main`'s whole version
  of those lines, earlier commits included, not this commit's change. Never resolve a
  file with `git checkout --theirs` or `--ours`.
- **Ask before bringing in a prerequisite.** If the commit needs code the release branch
  lacks (a helper, an import, an API from an earlier PR), stop and ask whether to add the
  minimal piece with a note, or to cherry-pick the prerequisite PR first.
- **Leave the environment alone.** No `uv sync`, no `uv run` without `--no-sync` (it
  re-syncs and reverts local builds), and no reinstall unless the user agrees.
- **One Spyre process at a time.** Run test files one after another, never in parallel.
- **Don't clean up picked code.** The picked lines must stay identical to `main`. If the
  PR is later opened with `prepare-pull-request`, its comment and docstring cleanup must
  skip them.

## Values

Bash state does not persist between tool calls, so write these values literally into
every command:

| Value | How to get it |
| --- | --- |
| `<remote>` | The remote for `torch-spyre/spyre-inference`. Its name differs per checkout (`remote`, `upstream`, `origin`): `git remote -v \| grep -m1 -E '[:/]torch-spyre/spyre-inference(\.git)? \(fetch\)' \| cut -f1` |
| `<release>` | The branch the user named. If they named none, list `git branch -r --list '<remote>/release-*'` and ask. |
| `<sha>`, `<short>` | The full sha from `git rev-parse --verify '<input>^{commit}'`, and its first 7 characters. For a PR number, `git log <remote>/main --format='%H %s' --grep='(#<N>)$'`; confirm the subject with the user. |
| `<pr>` | The `(#NNNN)` at the end of the commit's subject, if there is one. |
| `<tools>` | A temporary copy of this skill's `scripts/` directory (step 0). |

## 0. Preflight

1. The tracked tree must be clean: `git status --porcelain --untracked-files=no` prints
   nothing. Otherwise stop and ask; don't stash the user's work.
2. Copy the helper script out of the tree, because step 2 switches to a release branch
   that may not carry this skill. On such a branch the skill's files are gone and it drops
   out of the skill list; keep following this text and use the copy:

    ```bash
    T=$(mktemp -d /tmp/cherry-pick-tools.XXXX) && cp <this skill's base directory>/scripts/cherry_pick.py "$T"/ && echo "$T"   # -> <tools>
    ```

## 1. Fetch and inspect the commit

```bash
git fetch <remote> main <release>
git rev-parse --verify '<input>^{commit}'      # -> <sha>
git merge-base --is-ancestor <sha> <remote>/main && echo on-main
git rev-list --parents -n1 <sha>               # three or more shas: a merge commit
git show --stat --format='%H%n%an <%ae>%n%s' <sha>
git log --oneline <remote>/<release> --grep='<sha>' --grep='(#<pr>)'
```

- **Not found after the fetch:** the commit may belong to another repository, such as
  torch-spyre. Say so and stop.
- **Not on `<remote>/main`:** stop and ask. It may be a commit from a PR branch or a fork.
- **A merge commit:** ask before picking it with `-m 1`. PRs here land as squash commits.
- **Already on the release branch** (the last command prints a commit): show it and ask.
  Drop its `--grep='(#<pr>)'` when the subject carries no PR number.
- The `--stat` list is what gets picked. If the user expects a different number of files,
  look at the PR's history (`git fetch <remote> pull/<pr>/head`): a PR can grow in its
  last commits. Never drop files to match an expectation.

## 2. Create the branch

```bash
git rev-parse --verify --quiet refs/heads/<release>-<short>-cp && echo exists
git switch --no-track -c <release>-<short>-cp <remote>/<release>
```

If the branch already exists, stop and ask; don't reset it. Afterwards `git rev-parse HEAD`
must equal `git rev-parse <remote>/<release>`.

## 3. Cherry-pick without committing

```bash
git cherry-pick --no-commit <sha>
```

If it applies cleanly, go to step 4. Otherwise, for each file that
`git diff --name-only --diff-filter=U` lists:

1. Read the commit's own change: `git show <sha> -- <file>`.
2. Read the conflict: `git diff <file>`. "Ours" is the release branch; "theirs" is `main`,
   earlier commits included.
3. Keep the release side and apply only the lines the commit adds or removes:

    | Conflict | Resolution |
    | --- | --- |
    | `main` reworded a nearby comment or docstring in an earlier commit | Keep the release wording and apply the commit's edit. |
    | An import line on `main` carries names from an earlier PR | Keep the release names and add only the commit's new import. |
    | The commit edits a line that the release branch doesn't have | Drop that edit and note it. |
    | The commit needs a symbol, file or API that the release branch lacks | Stop and ask (see Rules). |
    | Modify/delete: the file doesn't exist on the release branch | Stop and ask. |

4. Write one note per file: `<path>: <what conflicted>; <what you kept and applied>`.
5. `git add <file>`.

A clean merge can still be wrong, in two ways:

- **Code that exists only on `main`.** A hunk can apply without conflict and call a
  helper the release branch lacks. For each helper the commit's added lines call, check
  that it exists on the release branch: `git grep -n -e 'def <name>' -e 'class <name>' HEAD`.
- **The wrong place.** Git applies a hunk wherever its context lines match, nearest to the
  original line number. In a file with repeated blocks, such as the per-model entries of
  `spyre_inference/config/model_configs.yaml`, YAML or TOML tables, or parametrize lists,
  the same context matches in many places. For such a file, read each hunk with wider
  context on both sides and confirm it sits in the same section (model, table, function)
  as on `main`: `git show -U15 --format= <sha> -- <file>` against
  `git diff --cached -U15 -- <file>`.

## 4. Stage with the commit message

`git cherry-pick --no-commit` staged the clean files and step 3 staged the resolved ones,
so `git diff --name-only` must print nothing. Then write the message:

```bash
python3 <tools>/cherry_pick.py message <sha> --into <release> --clean
python3 <tools>/cherry_pick.py message <sha> --into <release> --note '<note>' --note '<note>'
```

Use `--clean` only when the pick applied without conflicts; otherwise pass one `--note`
per note from step 3. The script writes the message to `MERGE_MSG`, keeps a backup next to
it in the git directory, and prints it:

```text
[<release>] <original subject (#<pr>)>

Cherry-pick of #<pr> (<short>) from main onto `<release>`.
Original PR: https://github.com/torch-spyre/spyre-inference/pull/<pr>
Original commit: https://github.com/torch-spyre/spyre-inference/commit/<sha>

<original message body>

Cherry-pick notes for <release>:      (or: Applied cleanly; ...)
- <note>

(cherry picked from commit <sha>)

<original trailers>
```

The title follows the release branch's convention: when the cherry-pick PR is squashed,
GitHub appends its number, giving `[<release>] <title> (#<pr>) (#<cp-pr>)`. If the
original body is only a list of the PR's commit subjects, ask whether to use the PR
description instead.

The user commits with `git commit -s -F "$(git rev-parse --git-path MERGE_MSG)"`. A plain
`git commit` opens the editor in strip mode and silently drops every line starting with
`#`, which includes the `## Description` headings of the PR template. `git reset`,
`git switch` and `git stash` delete `MERGE_MSG`; copy the backup back if that happens.

## 5. Run the tests the commit brought

1. Find them:

    ```bash
    git show --name-only --format= --diff-filter=AMR <sha> -- tests | grep -E '(^|/)test_[^/]*\.py$'
    ```

    If the commit brought no tests, say so and suggest the closest existing tests (those
    covering the modules it touches) instead of running anything unasked. If it changed
    test infrastructure (`conftest.py`, `tests/plugin/`, shared helpers), also check that
    collection still works:
    `uv run --active --no-sync pytest --collect-only -q -m "not upstream" tests`.

2. Check the interpreter the tests will use:

    ```bash
    uv run --active --no-sync python <tools>/cherry_pick.py env
    ```

    | `MISMATCH` row | Meaning and action |
    | --- | --- |
    | `import` | The venv imports `spyre_inference` from another checkout, so the tests would run the wrong code. Stop and tell the user. |
    | `tests package` | Another editable install (for example hf-adapters) shadows `tests`, so `from tests...` imports fail. Count those failures as environment noise. |
    | `entry points` | The installed metadata predates a `[project.entry-points]` change, so plugins load under the old names. Ask before refreshing it with `uv pip install --no-deps -e .`. |
    | `torch-spyre`, `vllm`, `torch` | The venv runs a different stack from the one the release branch locks, usually `main`'s. Results are only indicative. Tell the user and ask whether to run anyway. Setting up the release stack (`uv sync`, `scripts/install-pinned-rpms.sh --rebuild`, see CLAUDE.md) is theirs to do. |

3. Run each file on its own, one after another, as the Makefile does:

    ```bash
    uv run --active --no-sync pytest -m "not upstream" -ra <file>
    ```

    Each run starts vLLM, so use a long Bash timeout, or run it in the background and
    wait for it to finish. Never start a second Spyre run while one is still going.
    Don't add `-p no:...` or other flags to force skipped tests to run.

4. Report the results as they are. Tests guarded by `spyre_available()` skip on a host
   without Spyre, and a skipped test is not a verified one, so say which tests skipped and
   why. For each failure, work out whether the cherry-pick caused it:

    - **Environment:** an `env` mismatch from above; `Device or resource busy` (another
      process holds the card, or the test's own parent process seeded torch first); a 403
      from a gated Hugging Face repo whose weights are cached (rerun with
      `HF_HUB_OFFLINE=1`); `No module named 'tests....'` (a shadowed `tests` package).
    - **Pre-existing:** for a test file the commit modified rather than added, run the same
      node ID on the release branch without the pick. Ask first: like `vllm-bench-latency`,
      this repoints the editable install. Use a throwaway worktree at
      `<remote>/<release>`, run `uv pip install --no-deps -e <worktree>`, run the test,
      then always run `uv pip install --no-deps -e .` and `git worktree remove`.
    - **Missing prerequisite:** otherwise the pick probably relies on a `main` change that
      the release branch lacks. Report the failing assertion or traceback and stop. Don't
      edit code or tests to make the test pass without the user's agreement; that would
      change the pick.

## 6. Verify the staged change

```bash
python3 <tools>/cherry_pick.py verify <sha>
```

The script compares the staged diff (index against `HEAD`) with the commit's diff, file
by file. It compares the lines each diff adds and removes, ignoring their position and
the surrounding lines:

| Status | Meaning |
| --- | --- |
| `identical` | Same starting text, same result. |
| `same change` | The same lines added and removed; the release branch's file differs elsewhere. |
| `DIFFERS` | Lines only in the commit or only staged, a different mode or status, or the same lines in a different place. |
| `MISSING` | The commit changes the file, but nothing is staged for it. |
| `EXTRA` | The file is staged, but the commit doesn't touch it. |

Exit status 0 means the staged change is the commit's change and nothing more, 1 means
there are differences, and 2 means something blocks: unresolved conflicts, unstaged edits
to tracked files, or conflict markers.

Fix every `DIFFERS`, `MISSING` and `EXTRA` file, or explain it in a cherry-pick note and
rerun `message`. Typical justified differences:

- an edit dropped because the release branch lacks the line it changes;
- `commit only -` and `staged only -` lines, where the release branch had other text in
  the lines the commit replaces;
- an import the release branch needs because `main` got it from an earlier PR (for
  example `from pathlib import Path`).

The comparison catches a line in the wrong place only when the release branch's file is
the one the commit started from (`identical` would turn into `DIFFERS`). Otherwise it
reports `same change` either way. So read every conflict hunk once more in
`git diff --cached`, and repeat the step-3 placement check for every file with repeated
blocks that `verify` reports as `same change`.

Then lint the staged Python files without changing them, with the ruff version that
`.pre-commit-config.yaml` pins:

```bash
uvx ruff@<pinned version, without the v> check --no-fix <staged .py files>
uvx ruff@<pinned version, without the v> format --check <staged .py files>
```

Finally, `git status --short` must show only the commit's files, all staged. Leave
untracked files alone.

## 7. Report

Tell the user:

- the branch, its base (`<remote>/<release>` and its sha), and the picked commit with a
  link to its PR;
- how many files are staged, against the commit's count, and the `verify` result;
- every conflict and how you resolved it (the notes);
- the test results per file (passed, failed, skipped and why) and any environment caveats;
- the next steps: review `git diff --cached`, commit with
  `git commit -s -F "$(git rev-parse --git-path MERGE_MSG)"`, push to their fork, and open
  the PR against `<release>`, not `main`.

Then remove `<tools>`.
