# PD Disaggregation Next Lab Report

Branch: `tdeshane/spyre-inference-pd-disagg-next-lab` (baseline `b81de20`,
frozen draft PR branch untouched).

## What was implemented

1. **Serve lifecycle verified by trace, not new code.** Against pinned
   vLLM v0.20.1: scheduler connector creation, worker transfer-group
   init, paged `register_kv_caches` (via `GPUModelRunner.initialize_kv_cache`
   receiving the `SpyrePagedKVCache` dict from `initialize_kv_cache_tensors`),
   and execute_model bind/start_load/wait_for_save/get_finished all flow
   through inheritance. No execute_model override is needed or wanted.
2. **Removed the dead worker bridge** (`spyre_kv_connector_bridge.py`) and
   the inert `VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE` env var. Wiring the
   bridge would have double-invoked the inherited connector hooks; deleting
   it is the correct serve-readiness fix and shrinks review surface.
3. **Decode LIST_REQUEST retry** in `_load_saved_requests_nixl`: re-sends
   every 5 s during the 60 s window, replacing the pod-side 3-second
   settle workaround for the first-run metadata race.
4. **NIXL-absent hardening**: `_init_nixl_agent` now disables NIXL and
   warns instead of calling `None`.
5. **Path clarity**: `get_active_kv_path()` returns paged/heap/staging;
   legacy registration logs an explicit fallback warning. Paged stays the
   main Spyre path; heap stays bounded to non-paged layouts.
6. **ty clean**: fixed all 13 connector type errors (Optional narrowing,
   shm buf narrowing, stats dataclass `data` field, backend factory
   issubclass narrow) — CI type-check job now passes for the package.

## Tests

`tests/test_kv_connector_lifecycle.py` (new): bridge removal, inherited
lifecycle contracts, retry presence, NIXL-absent disable, heap-fallback
bounds, producer/consumer construction (vLLM-gated). Full set:
41 passed, 4 skipped (need vLLM: registration, smoke roundtrip,
two role-construction tests). compileall/ruff/format clean;
`ty check spyre_inference` 0 errors.

## Unproven

Real `vllm serve` prefill/decode with a model, scheduler-driven save/load
under load, NIXL retry on hardware. See
`docs/pd-disagg-serve-validation-plan.md` for the exact pod plan.

## Review readiness

Ready for Codex review. Suggested future split: connector ty/typing
fixes; bridge removal; NIXL retry hardening; docs.
