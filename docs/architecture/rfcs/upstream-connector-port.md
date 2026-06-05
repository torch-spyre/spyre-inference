# RFC: Port the upstream KV Connector experience to spyre-inference

| Field | Value |
|---|---|
| Status | Draft |
| Author | Chen Wang |
| Created | 2026-06-05 |
| Target | First milestone in `spyre_inference/v1/kv_offload/` |
| Related | vLLM `OffloadingConnector`, llm-d `SharedStorageOffloadingSpec`, prior internal Spyre PD-disaggregation prototype |

## 1. Motivation

The upstream vLLM `OffloadingConnector` framework gives every CUDA platform three things for free:

1. A pluggable scheduler-side `OffloadingManager` that tracks where each block lives (G/H/F tiers).
2. A worker-side `OffloadingHandler` registry keyed by `(src_type, dst_type)` that performs the actual transfer.
3. An `OffloadingSpec` factory that lets out-of-tree platforms drop in their own manager + handlers without touching upstream code.

The existing `spyre-inference` plugin has **none** of this wired up. `TorchSpyreWorker` extends `CPUWorker` and never calls `register_kv_caches`, the upstream `CPUOffloadingSpec` errors out on non-CUDA platforms (`current_platform.is_cuda_alike()` check at `vllm/v1/kv_offload/cpu/spec.py:89`), and the only KV-tier story we have today is "the whole cache is on-device, full-stop."

Meanwhile, an earlier internal Spyre PD-disaggregation prototype has already demonstrated end-to-end KV transfer between two Spyre instances over NIXL, using a Spyre-specific device↔host copy primitive. That prototype is not packaged for vLLM's connector contract — it sits in standalone scripts that drive the model directly via `fms` — so it cannot ride the upstream connector ecosystem (LMCache, llm-d shared-storage backend, prefix caching, PD disaggregation) without an adaptor.

This RFC proposes how to combine the two: take the prototype's data-copy primitive, wrap it as an upstream-conformant `OffloadingHandler`, and register a `SpyreOffloadingSpec` so that every connector that already targets `OffloadingConnector` also works on Spyre.

## 2. Goals and non-goals

### Goals (M1)

- A user runs vLLM on Spyre with `--kv-transfer-config '{"kv_connector":"OffloadingConnector", "kv_connector_extra_config":{"spec_name":"SpyreOffloadingSpec","cpu_bytes_to_use":"8000000000"}}'` and gets host-RAM offload that survives across requests.
- All upstream connectors that compose `OffloadingConnector` (`SharedStorageOffloadingSpec`, llm-d-kv-cache `llmd_fs_backend`, etc.) work on Spyre with **no plugin-side handler changes** beyond what M1 ships.
- The Spyre device↔host copy goes through one named, testable primitive (`SpyreKvDmaCopier`) that wraps the prototype's `DmaiQPush`/`DmaoQPush` path (or its successor `torch_spyre._C.spyre_from_blob` path).
- `pytest tests/v1/kv_offload/` runs the same matrix as upstream for the CPU spec, plus a Spyre-specific test that round-trips a known-pattern block device→host→device.

### Non-goals (M1)

- PD disaggregation. The prototype's NIXL adapter is reusable but lives in a follow-up RFC because it requires worker-to-worker NIXL agent setup that is orthogonal to the connector port.
- Multi-tier (host RAM + file/object). Once the handler exists, layering `SharedStorageOffloadingSpec` on top is a configuration change, not code.
- Replacing the flit-offset addressing scheme with a stable on-device API. The `flit_offset` map is read from `perfdsc` artifacts and is fragile; M1 keeps the same approach as the prototype and we file a separate issue against torch-spyre to expose a stable kv-region descriptor.

## 3. Background: what the upstream `OffloadingConnector` actually requires

Three abstraction points matter on the worker side. References are to vLLM `main` at the version this fork tracks.

### 3.1 `OffloadingConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:46`)

Constructed once per role (`SCHEDULER`/`WORKER`) and delegates to `OffloadingConnectorScheduler` or `OffloadingConnectorWorker`. The worker side calls `connector_worker.register_kv_caches(kv_caches)` with the `dict[str, torch.Tensor]` that the runner has already allocated. **This is the only ingestion point for the on-device KV cache** — everything downstream operates on tensors handed in here.

### 3.2 `OffloadingSpec` (`vllm/v1/kv_offload/base.py:319`)

The contract a platform implements:

- `get_manager() -> OffloadingManager` — scheduler-side bookkeeping (which blocks are where, eviction policy).
- `get_handlers(kv_caches) -> Iterator[(src_type, dst_type, OffloadingHandler)]` — worker-side transfer dispatch. The handler is asked to move a list of `(src_block_ids, dst_block_ids)` pairs and to expose a `get_finished()` poll for completion.

### 3.3 `CpuGpuOffloadingHandlers` (`vllm/v1/kv_offload/cpu/gpu_worker.py:375`)

The reference CUDA implementation. It is **not directly reusable on Spyre** because:

- It allocates `torch.cuda.Stream` per transfer (`spyre` has no public stream API; `_sync_device` in `TorchSpyreModelRunner` is a stub).
- It asserts `gpu_tensor.is_cuda` on every registered KV tensor.
- It calls `ops.swap_blocks_batch`, a custom CUDA op.
- Optional `cudaHostRegister` pinning via `cudart()`.

The factory layer above it (`OffloadingSpecFactory.register_spec`, `vllm/v1/kv_offload/factory.py:21`) **is** trivially reusable — it just imports a string-named class.

## 4. Background: what the prototype's primitive actually provides

The prototype exposes the device↔host path in two layers:

### 4.1 Low level: senlib DMA queues

```python
self.pf.DmaoQPush(self.pf.DmaoDataCB(addr + offset, buf, 0))   # device → host
self.pf.DmaoQFlush()                                           # synchronous flush
self.pf.DmaiQPush(self.pf.DmaiDataCB(buf, addr + offset, 0))   # host → device
self.pf.DmaiQFlush()
```

Where `addr = kv_tensor["flit_offset"] * 128` (flit = 128 B), one `flit_offset` per `(layer_idx, k|v)` from the perfdsc `metadata.json`, and `buf` is a host-side `DataArray`. Throughput numbers in the script comments report ~3 GB/s on the demo hardware. The primitive is **synchronous** — no streams, no events, no async fence. This is fine for M1 and matches the way `SpyreCommunicator` falls back today.

### 4.2 Higher level: a Python accessor class

```python
self._C = importlib.import_module("torch_spyre._C")
addr = self._C.get_dmpa(x_spyre)
buf  = self._C.spyre_from_blob(addr, size=..., stride=..., dtype=torch.float16)
buf.copy_(data)             # host → device
host = buf.to('cpu')        # device → host
```

`get_dmpa` returns the device physical address of an existing Spyre tensor; `spyre_from_blob` constructs a Spyre tensor view at an arbitrary device address. The prototype's accessor uses these to read/write at flit-offset granularity from Python without touching senlib. **This is the path we should prefer for the M1 handler** because it operates on `torch.Tensor` objects (so `OffloadingHandler` doesn't need to know about `DmaoDataCB`) and because it does not depend on a pre-built libsenlib being available in the deployment image.

### 4.3 What's NIXL-specific (out of scope for M1)

The prototype's NIXL connector and its `CpuBufferManager` are about getting two _hosts_ to exchange those CPU tensors over the network. They are independent of how the Spyre-side data got into a CPU tensor in the first place. M1 reuses none of this.

## 5. Proposed architecture

```text
┌─────────────────────────────── vllm ───────────────────────────────┐
│ OffloadingConnector  ──register_kv_caches──►  OffloadingConnectorWorker
│        │                                            │
│        ▼ get_handlers (via factory)                 ▼ run handlers
│ OffloadingSpecFactory.create_spec("SpyreOffloadingSpec")
└──────────────────────────────┬─────────────────────────────────────┘
                               │ resolves to
┌──────────────────────────────▼─────────────────────────────── spyre-inference ──┐
│ SpyreOffloadingSpec                                                              │
│   .get_manager() -> CPUOffloadingManager     # reused verbatim from upstream     │
│   .get_handlers() -> SpyreCpuOffloadingHandlers                                  │
│         ├─ device_to_host_handler (Spyre → host RAM block tensor)                │
│         └─ host_to_device_handler (host RAM block tensor → Spyre)                │
│                                                                                  │
│ SpyreKvDmaCopier  (the one and only place we touch Spyre device addresses)       │
│   .copy_d2h(spyre_tensor, host_tensor, *, slice)                                 │
│   .copy_h2d(host_tensor,  spyre_tensor, *, slice)                                │
│   ── implementation: torch_spyre._C.{get_dmpa, spyre_from_blob}                  │
│   ── fallback:       libsenlib DmaiQPush/DmaoQPush  (gated by env var)           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Key shape: **only `SpyreCpuOffloadingHandlers` and `SpyreKvDmaCopier` are new code on the Spyre side.** Everything above (manager, factory, scheduler-side connector, eviction policies, llm-d composition) is unchanged upstream code.

### 5.1 Why we don't subclass `CpuGpuOffloadingHandlers`

The upstream class is structured around `torch.cuda.Stream`/`torch.Event`. Even ignoring the `is_cuda` assert, half the methods (`get_finished`, `wait`, `shutdown`) call `event.query()` / `event.synchronize()` / `event.elapsed_time()`. There is no "swap CUDA for Spyre" override point. A clean implementation of the same interface (`OffloadingHandler` from `vllm/v1/kv_offload/worker/worker.py`) is shorter than working around the CUDA assumptions.

### 5.2 Why we reuse `CPUOffloadingManager` verbatim

The manager is pure bookkeeping. It is keyed by `LoadStoreSpec` types, not by tensor backends, and the upstream pluggable cache policy registry (`lru`, `arc`) handles eviction. Nothing in it is CUDA-specific.

## 6. Component design

### 6.1 `SpyreKvDmaCopier`

```python
# spyre_inference/v1/kv_offload/copier.py
class SpyreKvDmaCopier:
    """Single-purpose owner of every host↔Spyre KV byte transfer.

    Wraps two backends:
      - 'spyre_from_blob'  : torch_spyre._C.{get_dmpa, spyre_from_blob}.
                             Operates on torch.Tensor inputs; preferred.
      - 'senlib_dma'       : libsenlib DmaiQPush/DmaoQPush over flit_offset
                             addresses; matches the demonstrated PD-disagg
                             plugin. Used when 'spyre_from_blob' isn't
                             available (older runtime images).

    Backend selection is driven by SPYRE_KV_DMA_BACKEND env var with auto-
    detect default ('spyre_from_blob' if importable, else 'senlib_dma').
    """

    def copy_d2h(self, src_spyre: torch.Tensor, dst_host: torch.Tensor) -> None: ...
    def copy_h2d(self, src_host: torch.Tensor, dst_spyre: torch.Tensor) -> None: ...
```

Constraints:

- Both methods are synchronous. There is no `non_blocking` on Spyre yet (`TorchSpyreModelRunner._sync_device` is a no-op for the same reason).
- Neither method allocates. The handler caller owns allocation.
- A single instance is shared across both directions. Construction reads the env var once and locks in a backend.

### 6.2 `SpyreCpuOffloadingHandlers`

```python
# spyre_inference/v1/kv_offload/handlers.py
class SpyreCpuOffloadingHandlers:
    def __init__(self,
                 kv_caches: CanonicalKVCaches,
                 block_size_factor: int,
                 num_cpu_blocks: int,
                 copier: SpyreKvDmaCopier): ...

    @property
    def device_to_host_handler(self) -> OffloadingHandler: ...
    @property
    def host_to_device_handler(self) -> OffloadingHandler: ...
```

Each direction is a `_SingleDirectionSpyreHandler(OffloadingHandler)` that:

1. On `transfer(spec)`, walks the `(src_block_ids, dst_block_ids)` pairs and calls `copier.copy_{d2h,h2d}` for each pair.
2. Returns a synchronous `TransferResult` with a job id, byte count, and elapsed time measured by `time.perf_counter()` (no CUDA events).
3. `get_finished()` drains the in-flight queue (which is always already done because every transfer is sync).
4. `shutdown()` clears references to the registered tensors.

Block tensors on the host side are a single `torch.empty(num_cpu_blocks, page_size_bytes, dtype=torch.int8)` per attention group, allocated at `__init__` time. We do **not** pin via `cudaHostRegister` — there is no equivalent on Spyre.

### 6.3 `SpyreOffloadingSpec`

```python
# spyre_inference/v1/kv_offload/spec.py
class SpyreOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)
        cpu_bytes = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes:
            raise ValueError("cpu_bytes_to_use must be set ...")
        # ... compute self.num_blocks identically to CPUOffloadingSpec
        self._copier = SpyreKvDmaCopier()
        self._manager = None
        self._handlers = None

    def get_manager(self) -> OffloadingManager:
        # Identical to CPUOffloadingSpec.get_manager (reuse upstream class).

    def get_handlers(self, kv_caches):
        if not self._handlers:
            self._handlers = SpyreCpuOffloadingHandlers(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
                copier=self._copier,
            )
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.device_to_host_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.host_to_device_handler
```

`GPULoadStoreSpec` is the upstream "device-side" type — it is a tag, not CUDA-specific, so we use it for Spyre. (The upstream class is named `GPULoadStoreSpec` for historical reasons; the tag is platform-agnostic.)

### 6.4 Registration

In `spyre_inference/__init__.py`, after the existing platform plugin registration:

```python
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
OffloadingSpecFactory.register_spec(
    "SpyreOffloadingSpec",
    "spyre_inference.v1.kv_offload.spec",
    "SpyreOffloadingSpec",
)
```

This mirrors how the upstream CPU spec is registered. No changes to `TorchSpyrePlatform`, no changes to `TorchSpyreWorker` — the connector is selected by `kv-transfer-config` at engine init.

### 6.5 Worker-side glue

`OffloadingConnectorWorker.register_kv_caches` is invoked by the engine after `_allocate_kv_cache_tensors` returns. This already happens through the upstream `KVConnectorBase_V1` machinery — **no plugin change is needed** as long as the tensors `_allocate_kv_cache_tensors` returns are real `torch.Tensor` objects on `device("spyre")`. They are: see `spyre_model_runner.py:339–345` (`device="spyre"`).

The one thing we have to verify in implementation is that `OffloadingConnectorWorker` does not assert tensor device type before handing the `kv_caches` dict to our spec. If it does, we fix that in upstream vLLM with a one-liner.

## 7. File-by-file plan

New files in `spyre_inference/v1/kv_offload/`:

| File | Purpose | Approx LOC |
|---|---|---|
| `__init__.py` | empty | 0 |
| `copier.py` | `SpyreKvDmaCopier` (both backends + auto-detect) | ~120 |
| `handlers.py` | `SpyreCpuOffloadingHandlers`, `_SingleDirectionSpyreHandler` | ~180 |
| `spec.py` | `SpyreOffloadingSpec` | ~70 |

Modified files:

| File | Change |
|---|---|
| `spyre_inference/__init__.py` | Add `OffloadingSpecFactory.register_spec(...)` call. |
| `pyproject.toml` | None (we don't add new deps for M1; senlib fallback is gated on import). |
| `spyre_inference/envs.py` | Add `SPYRE_KV_DMA_BACKEND` to the env var registry. |

New tests in `tests/v1/kv_offload/`:

| File | Coverage |
|---|---|
| `test_copier_round_trip.py` | Allocate a Spyre tensor with a known fp16 pattern, copy d2h, mutate host copy, copy h2d, assert content. Skipped if `device("spyre")` not available (CI gating already exists for other Spyre tests). |
| `test_spec_registration.py` | Import `spyre_inference`, then `OffloadingSpecFactory.create_spec(...)` resolves. Pure-CPU test — no Spyre device required. |
| `test_handler_dispatch.py` | Exercise `device_to_host_handler` / `host_to_device_handler` against `(src, dst)` tuples and assert the correct content lands. |

## 8. Compatibility with existing connectors

Because every connector ecosystem in the vLLM/llm-d world ultimately ends up at one or both of these two seams:

1. `OffloadingSpec.get_handlers` for the device↔host hop, then
2. `(host_tensor, host_tensor)` plumbing for any further hop (file, S3, NIXL, …),

once M1 ships, the following work on Spyre **without further plugin code**:

- `SharedStorageOffloadingSpec` (llm-d) — composes our spec for the device↔host hop, then reads/writes its own files for the host↔storage hop. The `FileMapper` content-hash path scheme is platform-agnostic.
- `llmd_fs_backend` POSIX/OBJ — same composition.
- LMCache offloading — talks to the same handler API.

The only connector that does **not** drop in is anything that requires async copy semantics (e.g. CUDA-graph-capturable transfers). None of the M1-relevant connectors do.

## 9. Migration: from the prior PD prototype to upstream

For users currently running the prior standalone NIXL demo, the migration shape is:

| Today (prior prototype) | After this RFC |
|---|---|
| Standalone `demo.py --role prefill/decode` | `vllm serve --kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'` on each side |
| Prototype's accessor driven directly from script | `SpyreKvDmaCopier` driven by the handler |
| Custom NIXL connector module | Upstream `NixlConnector`/llm-d router does the cross-host hop after the device→host hop is in place |
| flit-offsets read from `perfdsc` JSON | Same — until torch-spyre exposes a stable descriptor (filed separately) |

PD disaggregation is not delivered by M1 — but every component PD needs **except** the cross-host transport is delivered by M1. The follow-up RFC for PD on Spyre is purely about how to wire a NIXL agent into the upstream PD producer/consumer connectors.

## 10. Open questions

1. **`flit_offset` stability.** The prior prototype computes `flit_offset` from `perfdsc/metadata.json` produced at compile time. If torch-spyre's `_allocate_kv_cache_tensors` path uses `device="spyre"` allocation directly (it does — `spyre_model_runner.py:339`), can we recover the same flit-offset map without parsing perfdsc? The `_C.get_dmpa(tensor)` path implies yes; we need to confirm `get_dmpa` returns a stable address that survives a forward pass.
2. **`OffloadingConnectorWorker` device assertions.** Does any code in the worker path call `.is_cuda` on the registered tensors? A quick grep at implementation time will tell us; if so, we land a one-liner upstream.
3. **TP > 1.** `SpyreCommunicator` currently only supports TP=2. The connector handler operates per-rank, so TP>1 should be transparent, but we should verify the `kv_caches` dict the worker hands us at TP=2 contains exactly the local-rank slice. (It does on CUDA; we expect the same on Spyre because both go through the same upstream allocator.)
4. **Block alignment.** Spyre's `_allocate_kv_cache_tensors` rounds `num_blocks` up to a multiple of 64 (`spyre_model_runner.py:336`). The upstream `block_size_factor` machinery assumes the GPU/device block count and the offloaded block count are integer-related, which holds, but the alignment slack means a few blocks at the end are unusable. We should document this in the spec and not try to "use" the alignment slack on the host side.

## 11. Out of scope (filed as follow-ups)

- **PD disaggregation on Spyre.** Standalone RFC, builds on M1.
- **Async DMA on Spyre.** Depends on torch-spyre exposing a stream/event API. Until then, the synchronous handler is fine for offload/prefetch but precludes overlap with compute.
- **Stable on-device KV descriptor.** Depends on torch-spyre. Filed separately so the M1 handler can swap from `flit_offset+perfdsc` to the descriptor without changing the spec.
- **Multi-tier (host RAM + storage).** Already works once M1 lands by composing `SharedStorageOffloadingSpec` over our spec; no plugin code.

## 12. Acceptance criteria

- [ ] `vllm serve` on Spyre with `OffloadingConnector` + `SpyreOffloadingSpec` runs a 1k-prompt sweep and reports a non-zero `kv_offload_blocks_evicted` metric.
- [ ] `SharedStorageOffloadingSpec` over `SpyreOffloadingSpec` round-trips a known prompt-prefix across two `vllm serve` instances on the same node and the second instance reports a prefix-cache hit.
- [ ] `pytest tests/v1/kv_offload/` (the three new files above) green on a Spyre runner.
- [ ] No changes required to `TorchSpyreWorker` or `TorchSpyrePlatform`. (If we have to change them, the RFC's premise is wrong — pause and revise.)

## 13. References

- Upstream `OffloadingConnector`: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Upstream `OffloadingSpec`: `vllm/v1/kv_offload/base.py:319`
- Upstream CPU spec (CUDA-only today): `vllm/v1/kv_offload/cpu/spec.py`
- Upstream factory: `vllm/v1/kv_offload/factory.py:21`
- Spyre KV allocation today: `spyre_inference/v1/worker/spyre_model_runner.py:322–368`
