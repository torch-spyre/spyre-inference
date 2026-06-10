# PD Disaggregation: vllm serve Validation Plan (Next IBM Pod Run)

Goal: first real `vllm serve` prefill/decode pair over `InMemorySpyreConnector`
with a real model. The connector-level NIXL data path is already green
(synthetic two-pod smoke, 8/8 fp16 pages byte-identical). This run validates
the inherited vLLM v0.20.1 serve lifecycle: scheduler connector creation,
worker transfer-group init, paged `register_kv_caches`, save on prefill,
load on decode.

## Image

`us.icr.io/wxpe-cicd-internal/amd64/vllm-spyre-dev:dev-next` — use the
`/opt/spyre-inference` venv (the default PATH venv crashes vllm). Userspace
UCX/NIXL install activated via the standard activation script in `$HOME`.

## Branch

`tdeshane/spyre-inference-paged-kv-connector` from `toddllm/spyre-inference`
plus this lab branch (`tdeshane/spyre-inference-pd-disagg-next-lab`).
Stage via `git archive | oc exec tar -x` to both pods and set
`PYTHONPATH=$PWD` as in the connector smoke.

## Model

Tiny local model — `/mnt/models/tiny-granite-3.3-8b` (used by older PD
flow), or any small HF model already cached. The connector path is
model-agnostic; small + fp16 is enough.

## Prefill (pod 1)

```bash
export VLLM_SPYRE_ENABLE_NIXL_TRANSFER=1
export VLLM_SPYRE_NIXL_BLOCKING_TRANSFER=0
export VLLM_SPYRE_KV_ROLE=kv_producer
vllm serve /mnt/models/tiny-granite-3.3-8b \
  --host 0.0.0.0 --port 8000 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"InMemorySpyreConnector","kv_role":"kv_producer"}'
```

## Decode (pod 2)

```bash
export VLLM_SPYRE_ENABLE_NIXL_TRANSFER=1
export VLLM_SPYRE_KV_ROLE=kv_consumer
export VLLM_SPYRE_NIXL_REMOTE_IP=<prefill-pod-ip>
vllm serve /mnt/models/tiny-granite-3.3-8b \
  --host 0.0.0.0 --port 8000 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"InMemorySpyreConnector","kv_role":"kv_consumer"}'
```

Do NOT pass `--no-disable-hybrid-kv-cache-manager` (connector lacks
SupportsHMA; default auto-disable is required).

## Expected success markers

- Both pods: `Creating v1 connector with name: InMemorySpyreConnector`
- Both pods: `[InMemorySpyreConnector] Paged KV cache registered:
  {'layout': 'list_of_pages', ...}` (proves paged path active, not heap)
- Prefill on first completed request: `save_kv_bulk req=...` then NIXL
  pending exposure
- Decode after prompt routed: `NIXL load complete: req_id=...,
  blocks_loaded=N` with no `load_errors`; completion returned

## First blocker to capture if it fails

The first stack trace or error log after the `Paged KV cache registered`
marker, with which side emitted it. Three likeliest classes:

1. Engine init before registration: an HMA/scheduler error before
   `register_kv_caches` — capture vLLM config dump.
2. Save path: `save_kv_bulk` missing or stats stay zero — capture
   scheduler `build_connector_meta` request count.
3. Load path: NIXL pull timeout — capture both connector logs around
   LIST/PULL.

Not yet proven: any of the above markers under serve; runtime claims are
unverified until this run executes.
