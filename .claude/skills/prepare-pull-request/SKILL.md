---
name: prepare-pull-request
description: Strip verbose comments, docstrings and commit messages out of a finished change, then push and open a PR with a brief plain-English body. Use whenever the user asks to clean up code for a PR, tidy up before review, or open/raise a PR for work that is already written — and always before running `gh pr create`. Encodes hard limits (comments deleted by default, 1-2 lines each if kept, no absolute performance numbers anywhere) because the unguided default is far too verbose.
---

# Clean up and open a PR

The change already works. This skill deletes the writing around it, then ships it.

The bias is **delete**. Every comment, docstring line, commit-message bullet and PR
sentence starts out condemned and survives only by earning it under the rules below.
Assume the reviewer knows vLLM and can read Python.

Scope: only lines this branch added or touched. Leave pre-existing comments alone —
unrelated deletions make the diff harder to review.

## 1. Comments

List every comment line the branch adds, then justify them one at a time:

```bash
git diff origin/main...HEAD -U0 -- '*.py' |
  awk '/^\+\+\+ /{f=$2} /^\+[^+]/{ if ($0 ~ /^\+[[:space:]]*#/ || $0 ~ /"""/) print f": "$0 }'
```

A comment survives **only** if it carries information that is not in the code and that a
reviewer would otherwise get wrong:

- an external constraint (hardware alignment, stick size, dtype or layout limitation)
- an upstream bug number that explains a workaround (`torch-spyre#3770`)
- a unit or convention that no identifier in scope states
- why the ugly path is used instead of the obvious one

Delete everything else. Always delete:

- restatements of the next line (`# loop over the sequences`)
- section banners and step numbering (`# --- Step 2: build the mask ---`)
- history and diff narration (`# previously we used X`, `# changed to fix Y`)
- explanations of what a well-named function or variable already says
- speculative TODOs and notes-to-self
- multi-sentence essays justifying a design choice

What survives is **1-2 lines**, stating the fact with no build-up:

```python
# Head size must be a multiple of 64: 128-byte stick / 2 bytes per fp16 element.
```

Not this:

```python
# NOTE: We have to be careful here, because the Spyre hardware has a stick size of
# 128 bytes, and we are working in float16 which is 2 bytes per element, which means
# the head size has to be a multiple of 64 or the device cannot lay the tensor out
# correctly and we end up hitting a CPU fallback instead.
```

## 2. Docstrings

Same bias, plus:

- private helpers and self-evident functions get **none**
- everything else gets **one line**, and only if the name cannot carry it
- no `Args:` / `Returns:` / `Raises:` blocks unless the file already uses them
- never restate the signature, types or shapes that are already annotated
- no usage examples, no rationale, no "This function is responsible for..."

## 3. Leftovers

Delete debug prints, commented-out code, unused imports, scratch files, and probes that
only existed to prove the fix. Check what is actually in the diff:

```bash
git status --short
git diff origin/main...HEAD --stat
```

## 4. Format and verify

```bash
bash format.sh
uv run pytest -m "not upstream" <the test covering this change>
```

Deleting comments cannot break code, but the formatter reflows what is left and those
changes must be committed. Re-run the test so the PR body can honestly claim it.

## 5. Verify commit signoffs

Ensure all commits have DCO-correct signoffs. Every commit needs `Signed-off-by:` (`-s`) with the email matching GitHub or DCO fails.

## 6. Push

```bash
git push fork <branch-name>
```

Never force-push, never push to `main`. If history needs fixing, push a new branch. If
git identity is unset, run the `github-commit` skill first.

## 7. PR body

Write the body to `.pr_drafts/$(git branch --show-current).md` and use `--body-file`; inline `--body`
mangles multi-line text. The `.pr_drafts/` folder is gitignored, so drafts never land in
the diff.

Fill the repo template and keep the whole thing **under ~150 words**:

- **Description** — 1-3 plain-English sentences: what was wrong, what this does. Name the
  root cause if it is not obvious from the diff. No sub-headings, no bullet list restating
  the diff, no account of how the debugging went.
- **Related Issues** — `Closes #123`, or drop the section.
- **Test Plan** — the commands run and whether they passed, one line each.
- **Checklist** — tick honestly; leave CI unticked until it is green.

**No absolute performance data** — no tok/s, ms, GB or wall-clock figures, in the PR body,
the test plan or the commit message. Relative statements are fine ("one less copy per
step", "~15% fewer device transfers"). If the user wants numbers, they will ask.

```bash
mkdir -p .pr_drafts
# ... write the body to .pr_drafts/$(git branch --show-current).md ...
gh pr create --repo torch-spyre/spyre-inference --base main \
  --title "type(scope): what changed" --body-file ".pr_drafts/$(git branch --show-current).md" --draft
```

## Final gate

Reread the diff and the body once as the reviewer, then delete the sentence you are least
sure about. That instinct is right more often than not.
