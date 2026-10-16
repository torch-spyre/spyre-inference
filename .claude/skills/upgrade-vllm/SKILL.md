---
name: upgrade-vllm
description: Bump the pinned vLLM version in `pyproject.toml`, re-sync the upstream test plugin and the vendored CPU sampling code (`csrc/cpu/`, `SpyreTopKTopPSampler`), and triage the breakages that the bump exposes. Use whenever a user asks to upgrade/bump/update vLLM, pull up the lower bound, or chase a specific upstream commit. Most failures after a bump are not in our code — they are upstream API churn (worker `load_model` wrappers, platform `is_*()` predicates, KV-cache layout flips, new constructor kwargs) that our `_enum = PlatformEnum.OOT` platform doesn't get the CPU/GPU short-circuits for.
argument-hint: <target-vllm-version e.g. 0.23.0>
---

Upgrade spyre-inference to vLLM `$ARGUMENTS`.

If `$ARGUMENTS` is empty, ask the user which target version before doing anything else.

**Follow the complete procedure in `docs/contributing/vllm-upgrade-procedure.md`.**

When you see `{VERSION}` in that document, replace it with `$ARGUMENTS` (the version provided by the user or requested from them).
