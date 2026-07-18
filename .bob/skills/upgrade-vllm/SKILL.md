---
name: upgrade-vllm
description: Bump the pinned vLLM version in `pyproject.toml`, re-sync the upstream test plugin, and triage the breakages that the bump exposes. Use whenever a user asks to upgrade/bump/update vLLM, pull up the lower bound, or chase a specific upstream commit. Most failures after a bump are not in our code — they are upstream API churn (worker `load_model` wrappers, platform `is_*()` predicates, KV-cache layout flips, new constructor kwargs) that our `_enum = PlatformEnum.OOT` platform doesn't get the CPU/GPU short-circuits for.
---

Upgrade spyre-inference to the vLLM version specified by the user.

**IMPORTANT**: If the user hasn't specified a target version, ask them which version before proceeding.

**Follow the complete procedure in `docs/contributing/vllm-upgrade-procedure.md`.**

When you see `{VERSION}` in that document, replace it with the actual version number provided by the user (e.g., `0.23.0`).
