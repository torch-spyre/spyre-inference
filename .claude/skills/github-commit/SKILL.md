---
name: github-commit
description: Configure git with your GitHub identity, create a feature branch from origin/main, commit changes with DCO sign-off, and push to your fork. Use whenever the user wants help committing and pushing changes from a remote dev pod or any environment where git user.name/user.email is unset. Stops before PR creation.
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

Stage your changes and commit with DCO sign-off:

```bash
git add <files>
git commit -s -m "type: description of change"
```

To amend the last commit with a sign-off:

```bash
git commit --amend -s --no-edit
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
git commit -m "type: description of follow-up change" -s
git push fork <branch-name>
```
