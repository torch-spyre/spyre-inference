# Hive agent files

Prompts and data for the hive agents that work on this repo, kept here so they are
reviewed like code. The hive reads them live from `main` through each agent's
`prompt_source`, so a merged change applies on the agent's next kick.

| File | Agent | Read as |
|------|-------|---------|
| `prompts/deps.md` | `deps` | kick prompt (`prompt_source`) |
| `deps-known-bad.md` | `deps` | data the prompt tells it to read |

Some settings stay in the hive's own config, because a linked file cannot set them:

- the schedule (`governor.modes`). v5 parses `spec.cadences` in an `AgentDefinition` but never applies it.
- the sandbox (image, Spyre resources, secrets) and `paused`/`enabled`
- the GitHub allowlist that lets the hive fetch this repo (`variables.security`, base config only)

Don't confuse this directory with `.hive/`. That one is the hive's per-run scratch
directory inside an agent's workspace, and it is gitignored.
