---
name: github-commit
description: Configure git with your GitHub identity and commit signing key, create a feature branch from origin/main, commit changes with DCO sign-off and a GPG/SSH signature, and push to your fork. Use whenever the user wants help committing and pushing changes from a remote dev pod or any environment where git user.name/user.email or the signing key is unset. Stops before PR creation.
---

# GitHub Commit and Push

This skill helps commit and push changes in environments where git identity is not
pre-configured (e.g., a fresh remote dev pod). It uses the `gh` CLI to discover your
GitHub name and email, then walks through branch creation, DCO sign-off, and pushing
to a fork.

It intentionally does **not** create the PR — the human reviews and opens the PR.

## When to use

Trigger phrases:

- "commit these changes"
- "push my changes"
- "set up git config"
- "I need to commit from a dev pod"
- "what git user.name should I use?"

## Prerequisites

- `gh` CLI installed and authenticated (`gh auth login`).
- A fork of `torch-spyre/spyre-inference` if you are not a direct collaborator.

## 1. Configure git identity

Run the helper script to set `user.name` and `user.email` from your GitHub profile:

```bash
# Local config (recommended for repo-specific identity)
./.claude/skills/github-commit/setup-git-config.sh

# Or global config
./.claude/skills/github-commit/setup-git-config.sh --global
```

This fetches your `name` and `email` via `gh api user` and falls back to your GitHub
noreply email if no public email is set.

Manual equivalent:

```bash
git config user.name "$(gh api user --jq '.name // .login')"
EMAIL=$(gh api user --jq '.email // empty')
[[ -z "$EMAIL" ]] && EMAIL="$(gh api user --jq '.login')@users.noreply.github.com"
git config user.email "$EMAIL"
```

## 1b. Make sure commits get signed

`main` requires signed commits, so every commit needs a GPG or SSH signature (`-S`) as well as the DCO sign-off (`-s`). An unsigned commit blocks the PR from merging. Check that a signing key is configured and usable:

```bash
git config --get user.signingkey
git commit-tree -S -m probe HEAD^{tree} >/dev/null && echo "signing works"
```

If no key is configured, ask the user which one to use rather than generating one: a new key is not registered on their GitHub account, so its signatures will not verify. Then point git at it:

```bash
git config user.signingkey <key-id-or-pubkey-path>
git config gpg.format openpgp   # or ssh
```

## 2. Create a feature branch

Always branch from the latest `origin/main`:

```bash
git fetch origin main
git checkout origin/main
git checkout -b <branch-name>
```

If the local branch already exists:

```bash
git branch -D <branch-name>
git checkout -b <branch-name>
```

## 3. Stage, format, and commit

Run the formatter before committing:

```bash
bash format.sh
```

Stage your changes and commit with DCO sign-off and a signature:

```bash
git add <files>
git commit -s -S -m "type: description of change"
```

Stage files by name, not `git add -A` / `git add .`. Do not stage `uv.lock`,
`pyproject.toml` or `spyre-rpms.lock` unless the change is meant to alter the
environment — a plain `uv lock` / `uv sync` rewrites the lockfile as a side effect.
Revert accidental edits with `git checkout origin/main -- <file>`.

To amend the last commit with a sign-off and signature:

```bash
git commit --amend -s -S --no-edit
```

Before pushing, confirm every commit on the branch is signed (`G`) and signed off:

```bash
git log --format='%h %G? %s' origin/main..
git log --format='%h %(trailers:key=Signed-off-by,valueonly)' origin/main..
```

## 4. Push to your fork

Add your fork as a remote if needed:

```bash
FORK_URL=$(gh api user --jq '"https://github.com/\(.login)/spyre-inference.git"')
git remote add fork "$FORK_URL" 2>/dev/null || git remote set-url fork "$FORK_URL"
```

Push the branch:

```bash
git push fork <branch-name>
```

Stop here. The human opens the PR.

## Never force-push

Never run `git push --force` or `git push --force-with-lease`. Force-pushing rewrites
history and can permanently lose work or invalidate review comments. If a branch needs
cleanup, create a new branch instead and push that.

Add new commits on top and push normally:

```bash
bash format.sh
git add <files>
git commit -s -S -m "type: description of follow-up change"
git push fork <branch-name>
```
