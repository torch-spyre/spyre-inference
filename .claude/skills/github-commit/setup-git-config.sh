#!/usr/bin/env bash
#
# Configure git user.name and user.email from your GitHub account.
# Requires: gh CLI authenticated with `gh auth login`
#
# Usage:
#   ./scripts/setup-git-config.sh [--global]
#
# Options:
#   --global    Set the config globally (default: local to this repo)
#

set -euo pipefail

GLOBAL=false
if [[ "${1:-}" == "--global" ]]; then
    GLOBAL=true
fi

# Check gh is installed and authenticated
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI not found. Install from https://cli.github.com/" >&2
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: gh CLI not authenticated. Run 'gh auth login' first." >&2
    exit 1
fi

# Fetch user info from GitHub
NAME=$(gh api user --jq '.name // .login')
EMAIL=$(gh api user --jq '.email // empty')

# Fall back to noreply email if public email is not set
if [[ -z "$EMAIL" ]]; then
    LOGIN=$(gh api user --jq '.login')
    EMAIL="${LOGIN}@users.noreply.github.com"
fi

# Configure git
SCOPE=""
if [[ "$GLOBAL" == true ]]; then
    SCOPE="--global"
fi

git config $SCOPE user.name "$NAME"
git config $SCOPE user.email "$EMAIL"

echo "Git config set:"
echo "  user.name  = $NAME"
echo "  user.email = $EMAIL"
if [[ "$GLOBAL" == true ]]; then
    echo "  (global)"
else
    echo "  (local to this repository)"
fi
