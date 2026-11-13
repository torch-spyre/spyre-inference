# torch-spyre commits known to break spyre-inference

This file is read by the hive `deps` agent (`prompts/deps.md`). An `open` row caps the
`torch-spyre` pin: the agent will not bump to that commit or anything after it. When
upstream fixes it, set the status to `fixed in <sha>`. The cap then lifts once a
candidate commit includes that fix. Delete the row after the pin has moved past the fix.

The agent adds a row (issue `pending`) when it traces a red bump PR to a commit. A
maintainer files the upstream issue and fills in the link.

| Commit | Breaks | Found by | Issue | Status |
|--------|--------|----------|-------|--------|
