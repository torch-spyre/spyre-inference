# RFC: Port the upstream KV Connector experience to spyre-inference

| Field | Value |
|---|---|
| Status | Draft |
| Authors | Chen Wang ([@wangchen615](https://github.com/wangchen615)), Yue Zhu ([@yuezhu1](https://github.com/yuezhu1)) |
| Created | 2026-06-05 |
| Updated | 2026-06-08 — rebased on the upstream multi-tier framework (`TieringOffloadingSpec`, `SecondaryTierManager`, `tiering/fs`, `tiering/obj`); incorporated review feedback from [@yuezhu1](https://github.com/yuezhu1) |
| Tracking | First design doc for [#76 — \[Epic\] Develop KVCacheConnector for Spyre](https://github.com/torch-spyre/spyre-inference/issues/76) |
| Related | vLLM `OffloadingConnector`, vLLM `TieringOffloadingSpec` (PR #40020), vLLM `tiering/fs` (PR #41735), vLLM `tiering/obj` (PR #41968), prior internal Spyre PD-disaggregation prototype |

## 1. Motivation

The upstream vLLM `OffloadingConnector` framework gives every CUDA platform three things for free:

1. A pluggable scheduler-side `OffloadingManager` that tracks where each block lives (G/H/F tiers).
2. A worker-side `OffloadingHandler` registry keyed by `(src_type, dst_type)` that performs the actual transfer.
3. An `OffloadingSpec` factory that lets out-of-tree platforms drop in their own manager + handlers without touching upstream code.

As of vLLM v0.22, this stack has grown a fourth layer — a first-class **multi-tier framework** (`TieringOffloadingSpec` + `SecondaryTierManager` ABC + `SecondaryTierFactory`, PR #40020). Filesystem (`tiering/fs`, PR #41735) and object-store (`tiering/obj`, PR #41968) secondary tiers ship in-tree. The previously-standalone `llmd-fs-connector` is end-of-life as of `v0.22`; its logic is now upstream as the `fs` secondary tier.

The existing `spyre-inference` plugin has **none** of this wired up. `TorchSpyreWorker` extends `CPUWorker` and never calls `register_kv_caches`. Both the single-tier `CPUOffloadingSpec` and the new `TieringOffloadingSpec` (which subclasses `CPUOffloadingSpec`) error out on non-CUDA platforms via the `current_platform.is_cuda_alike()` check at `vllm/v1/kv_offload/cpu/spec.py:89`. So the entire upstream offload + tiering stack is unreachable from Spyre today, and the only KV-tier story we have is "the whole cache is on-device, full-stop."

Meanwhile, an earlier internal Spyre PD-disaggregation prototype has already demonstrated end-to-end KV transfer between two Spyre instances over NIXL, using a Spyre-specific device↔host copy primitive. That prototype is not packaged for vLLM's connector contract — it sits in standalone scripts that drive the model directly via `fms` — so it cannot ride the upstream connector ecosystem (LMCache, llm-d shared-storage backend, prefix caching, PD disaggregation) without an adaptor.

This RFC proposes how to combine the two: take the prototype's data-copy primitive, wrap it as an upstream-conformant `OffloadingHandler`, and register a `SpyreOffloadingSpec` so that the upstream `OffloadingConnector` works on Spyre. A small follow-on (`SpyreTieringOffloadingSpec`) extends the same handler with the `SecondaryTierManager` hook, so every secondary tier registered with `SecondaryTierFactory` (`fs`, `obj`, anything added later) works on Spyre with no per-tier plugin code.

## 2. Goals and non-goals

### Goals (M1)

- A user runs vLLM on Spyre with `--kv-transfer-config '{"kv_connector":"OffloadingConnector", "kv_connector_extra_config":{"spec_name":"SpyreOffloadingSpec","cpu_bytes_to_use":"8000000000"}}'` and gets host-RAM offload that survives across requests.
- The Spyre device↔host copy goes through one named, testable primitive (`SpyreKvDmaCopier`) that wraps the prototype's `DmaiQPush`/`DmaoQPush` path (or its successor `torch_spyre._C.spyre_from_blob` path).
- `pytest tests/v1/kv_offload/` runs the same matrix as upstream for the CPU spec, plus a Spyre-specific test that round-trips a known-pattern block device→host→device.

### Goals (M1.5 — small follow-on, scoped here)

- A user runs vLLM on Spyre with `spec_name: "SpyreTieringOffloadingSpec"` plus a `secondary_tiers: [{type: "fs", root_dir: "/mnt/kvcache"}]` entry and gets host-RAM-plus-FS tiered offload, with content-hashed paths that two instances on a shared volume cross-share.
- Every `SecondaryTierManager` registered with `SecondaryTierFactory` (`fs`, `obj`, future tiers) works on Spyre with **no per-tier plugin code** beyond what M1.5 ships.
- M1.5 lands at most ~50 LOC of glue on top of M1; if it grows, the design is wrong — pause and revise.

### Non-goals (M1 + M1.5)

- PD disaggregation. The prototype's NIXL adapter is reusable but lives in a follow-up RFC because it requires worker-to-worker NIXL agent setup that is orthogonal to the connector port.
- Replacing the flit-offset addressing scheme with a stable on-device API. The `flit_offset` map is read from `perfdsc` artifacts and is fragile; M1 keeps the same approach as the prototype and we file a separate issue against torch-spyre to expose a stable kv-region descriptor.
- Replacing `llmd-fs-connector` integrations *outside* the upstream `tiering/fs` path. The standalone connector is end-of-life as of `llmd-fs-connector==0.22`; the supported path forward is the upstream `fs` secondary tier, which M1.5 makes available.

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

### 3.4 Dynamic spec loading (`vllm/v1/kv_offload/factory.py:21`)

`OffloadingSpecFactory.register_spec(name, module_path, class_name)` records a tuple but does **not** import the module at registration time. The actual import happens lazily in `create_spec(...)` when the user's `kv_connector_extra_config.spec_name` selects this spec. That matters for an out-of-tree platform plugin: we can register `SpyreOffloadingSpec` from `spyre_inference/__init__.py` without dragging in any Spyre-only module at vLLM import time, and CUDA-only deployments that load `spyre-inference` for unrelated reasons pay zero cost for our spec.

The same pattern applies to `SecondaryTierFactory.register_tier(...)` (`vllm/v1/kv_offload/tiering/factory.py`). Adding a new secondary tier from a third-party package — including ours, if M2 ever ships one — is a one-line registration call, not an upstream PR.

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

The prototype's NIXL connector and its `CpuBufferManager` are about getting two *hosts* to exchange those CPU tensors over the network. They are independent of how the Spyre-side data got into a CPU tensor in the first place. M1 reuses none of this.

## 4a. Data paths in scope

There are two viable shapes for tiered KV caching during the v0.22 transition window. Both are in scope; the deployment-friendly one (Path A) is what M1.5 acceptance tests against.

| Path | Milestone | Compose how | Notes |
|---|---|---|---|
| Spyre device ↔ host RAM (single tier) | **M1** | `OffloadingConnector` + `SpyreOffloadingSpec` | Single-tier offload; survives across requests. |
| Spyre device ↔ host RAM ↔ filesystem (Path A) | **M1.5** | `MultiConnector` of two `OffloadingConnector`s: one with `SpyreOffloadingSpec` (D↔H), one with `SharedStorageOffloadingSpec` from `llmd_fs_backend` (H↔FS) | This is the shape llm-d v0.8 deployments actually run today. `MultiConnector` save fans out to both; load returns from the first connector reporting a hit (`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:304` and `:395`). |
| Spyre device ↔ host RAM ↔ filesystem/object (Path B) | M1.5+ | `OffloadingConnector` + `SpyreTieringOffloadingSpec` + `tiering/fs` or `tiering/obj` | The v0.22+ canonical shape. Internal cascade (`TieringOffloadingManager`), not save-to-both. Supersedes the standalone `llmd-fs-connector` (final standalone release `v0.22`). |
| Direct Spyre device ↔ filesystem / object store | Out of scope | n/a | Would require a Spyre-side analogue of NVIDIA GDS so a secondary tier can DMA without a host bounce. Not provided by torch-spyre today, and the upstream `SecondaryTierManager` contract assumes the `primary_kv_view` is over CPU memory; supporting this would change both. Filed as a future-work item. |

**Why Path A and Path B are both M1.5 in scope.** Path A is what's deployed today across llm-d v0.8 clusters. Path B is the upstream-canonical direction. Yue's example config (in the M1.5 acceptance criterion below) is Path A verbatim. Both paths reuse the same `SpyreOffloadingSpec` from M1 — the only difference is what sits on the other side of host RAM, which is configured per-deployment, not coded in the plugin.

**A subtle but important property of Path A.** `MultiConnector.register_kv_caches` hands the same `kv_caches: dict[str, torch.Tensor]` to *every* child connector — it does not pre-process or stage tensors between them. So the second `OffloadingConnector` (`SharedStorageOffloadingSpec`) receives the raw *Spyre* device tensors, just like the first one does. Whether `SharedStorageOffloadingSpec`'s `StorageOffloadingHandlers` work against Spyre tensors as-is is an open question — see §10 — and resolving it is part of M1.5's work, not deferred.

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

### 6.3a `SpyreTieringOffloadingSpec` (M1.5)

The upstream `TieringOffloadingSpec` (`vllm/v1/kv_offload/tiering/spec.py`) layers a `TieringOffloadingManager` over the same primary-tier handlers we already build in M1. Its only Spyre-incompatible piece is its `CPUOffloadingSpec` parentage, which inherits the `is_cuda_alike()` gate.

M1.5 ships a sibling spec that mirrors the upstream tiering shape but uses our handlers:

```python
# spyre_inference/v1/kv_offload/tiering_spec.py
class SpyreTieringOffloadingSpec(SpyreOffloadingSpec):
    """Spyre primary tier + upstream secondary tiers (fs, obj, ...).

    Mirrors vllm.v1.kv_offload.tiering.spec.TieringOffloadingSpec, but
    inherits from SpyreOffloadingSpec instead of CPUOffloadingSpec so it
    skips the CUDA gate. Reuses upstream TieringOffloadingManager and
    SecondaryTierFactory verbatim.
    """

    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

    def get_manager(self) -> OffloadingManager:
        # Build a TieringOffloadingManager wrapping a CPUPrimaryTierOffloadingManager
        # over our SharedOffloadRegion, plus one SecondaryTierManager per
        # entry in self.secondary_tier_configs (resolved via SecondaryTierFactory).
        ...
```

The `SharedOffloadRegion` (`vllm/v1/kv_offload/cpu/shared_offload_region.py`) is an mmap-backed CPU buffer; the `primary_kv_view: memoryview` it produces is what every `SecondaryTierManager` reads/writes from. Nothing in `SharedOffloadRegion` is CUDA-specific — it's plain `mmap` plus `multiprocessing.shared_memory`. We reuse it as-is.

User invocation:

```bash
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_connector_extra_config": {
    "spec_name": "SpyreTieringOffloadingSpec",
    "cpu_bytes_to_use": "8000000000",
    "secondary_tiers": [
      {"type": "fs", "root_dir": "/mnt/kvcache", "n_read_threads": 16}
    ]
  }
}'
```

For cross-instance sharing on a shared `root_dir`, set `PYTHONHASHSEED` to the same value on every instance — the upstream `FileSystemTierManager` documents this requirement (its block filenames depend on the `NONE_HASH` chain seed, which is randomized per-process otherwise).

### 6.4 Registration

In `spyre_inference/__init__.py`, after the existing platform plugin registration:

```python
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

OffloadingSpecFactory.register_spec(
    "SpyreOffloadingSpec",
    "spyre_inference.v1.kv_offload.spec",
    "SpyreOffloadingSpec",
)

# Added in M1.5:
OffloadingSpecFactory.register_spec(
    "SpyreTieringOffloadingSpec",
    "spyre_inference.v1.kv_offload.tiering_spec",
    "SpyreTieringOffloadingSpec",
)
```

This mirrors how the upstream CPU spec is registered. No changes to `TorchSpyrePlatform`, no changes to `TorchSpyreWorker` — the connector is selected by `kv-transfer-config` at engine init.

### 6.5 Worker-side glue

`OffloadingConnectorWorker.register_kv_caches` is invoked by the engine after `_allocate_kv_cache_tensors` returns. This already happens through the upstream `KVConnectorBase_V1` machinery — **no plugin change is needed** as long as the tensors `_allocate_kv_cache_tensors` returns are real `torch.Tensor` objects on `device("spyre")`. They are: see `spyre_model_runner.py:339–345` (`device="spyre"`).

The one thing we have to verify in implementation is that `OffloadingConnectorWorker` does not assert tensor device type before handing the `kv_caches` dict to our spec. If it does, we fix that in upstream vLLM with a one-liner.

## 7. File-by-file plan

### M1 files

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
| `spyre_inference/__init__.py` | Add `OffloadingSpecFactory.register_spec(...)` call for `SpyreOffloadingSpec`. |
| `pyproject.toml` | None (we don't add new deps for M1; senlib fallback is gated on import). |
| `spyre_inference/envs.py` | Add `SPYRE_KV_DMA_BACKEND` to the env var registry. |

New tests in `tests/v1/kv_offload/`:

| File | Coverage |
|---|---|
| `test_copier_round_trip.py` | Allocate a Spyre tensor with a known fp16 pattern, copy d2h, mutate host copy, copy h2d, assert content. Skipped if `device("spyre")` not available (CI gating already exists for other Spyre tests). |
| `test_spec_registration.py` | Import `spyre_inference`, then `OffloadingSpecFactory.create_spec(...)` resolves. Pure-CPU test — no Spyre device required. |
| `test_handler_dispatch.py` | Exercise `device_to_host_handler` / `host_to_device_handler` against `(src, dst)` tuples and assert the correct content lands. |

### M1.5 files (incremental)

| File | Purpose | Approx LOC |
|---|---|---|
| `spyre_inference/v1/kv_offload/tiering_spec.py` | `SpyreTieringOffloadingSpec` (subclasses `SpyreOffloadingSpec`; reuses upstream `TieringOffloadingManager` + `SecondaryTierFactory`) | ~50 |
| `spyre_inference/__init__.py` | Add a second `OffloadingSpecFactory.register_spec(...)` call for `SpyreTieringOffloadingSpec`. | +5 |
| `tests/v1/kv_offload/test_tiering_spec.py` | Pure-CPU test: build a `SpyreTieringOffloadingSpec` with a `dummy` secondary tier, store/load through the cascade, assert promotion semantics match upstream. | ~120 |

## 8. Compatibility with existing connectors and tiers

The two seams that matter:

1. **Device↔host hop** — `OffloadingSpec.get_handlers`. M1 makes this work on Spyre by registering `SpyreCpuOffloadingHandlers`.
2. **Host↔secondary hop** — `SecondaryTierManager.submit_store`/`submit_load`. M1.5 makes upstream's `tiering/*` framework usable on Spyre by giving `SpyreTieringOffloadingSpec` the same shape as upstream's `TieringOffloadingSpec`.

After M1 + M1.5 ship, the following work on Spyre **without further plugin code**:

- **Single-tier host-RAM offload** (M1) — via `SpyreOffloadingSpec`. Same prefix-cache semantics as the upstream CPU spec on CUDA.
- **`tiering/fs` secondary tier** (`vllm/v1/kv_offload/tiering/fs/manager.py:FileSystemTierManager`) — pure-Python disk-backed tier with content-hashed paths via `FileMapper`. With matching `PYTHONHASHSEED`, two Spyre instances on a shared `root_dir` (e.g. RWX PVC) cross-share blocks. Replaces the standalone `llmd-fs-connector` (EOL as of v0.22).
- **`tiering/obj` secondary tier** (PR #41968) — object-store backend (S3-style).
- **`tiering/example` secondary tier** — in-memory tier shipped upstream as a reference implementation; we use it in tests.
- **Future secondary tiers** — anything someone registers with `SecondaryTierFactory.register_tier(name, module, class)` works on Spyre as soon as it works on CUDA, since the Spyre-specific code stops at the primary tier.
- **LMCache offloading** — talks to the same `OffloadingHandler` API at the device↔host seam; M1 alone is enough.

The only connector that does **not** drop in is anything that requires async copy semantics (e.g. CUDA-graph-capturable transfers). None of the M1/M1.5-relevant tiers do — the upstream `SecondaryTierManager` contract is explicitly async-via-job-poll, not async-via-CUDA-events.

## 9. Migration: from the prior PD prototype to upstream

For users currently running the prior standalone NIXL demo, the migration shape is:

| Today (prior prototype) | After this RFC |
|---|---|
| Standalone `demo.py --role prefill/decode` | `vllm serve --kv-transfer-config '{"kv_connector":"OffloadingConnector",...}'` on each side |
| Prototype's accessor driven directly from script | `SpyreKvDmaCopier` driven by the handler |
| Custom NIXL connector module | Upstream `NixlConnector` does the cross-host hop after the device→host hop is in place |
| Cross-instance sharing via custom router copies | Built-in via `tiering/fs` content-hashed paths over a shared `root_dir` (M1.5; needs matching `PYTHONHASHSEED`) |
| flit-offsets read from `perfdsc` JSON | Same — until torch-spyre exposes a stable descriptor (filed separately) |

PD disaggregation is not delivered by M1 — but every component PD needs **except** the cross-host transport is delivered by M1. The follow-up RFC for PD on Spyre is purely about how to wire a NIXL agent into the upstream PD producer/consumer connectors.

## 10. Open questions

1. **`flit_offset` stability.** The prior prototype computes `flit_offset` from `perfdsc/metadata.json` produced at compile time. If torch-spyre's `_allocate_kv_cache_tensors` path uses `device="spyre"` allocation directly (it does — `spyre_model_runner.py:339`), can we recover the same flit-offset map without parsing perfdsc? The `_C.get_dmpa(tensor)` path implies yes; we need to confirm `get_dmpa` returns a stable address that survives a forward pass.
2. **`OffloadingConnectorWorker` device assertions.** Does any code in the worker path call `.is_cuda` on the registered tensors? A quick grep at implementation time will tell us; if so, we land a one-liner upstream.
3. **TP > 1.** `SpyreCommunicator` currently only supports TP=2. The connector handler operates per-rank, so TP>1 should be transparent, but we should verify the `kv_caches` dict the worker hands us at TP=2 contains exactly the local-rank slice. (It does on CUDA; we expect the same on Spyre because both go through the same upstream allocator.)
4. **Block alignment.** Spyre's `_allocate_kv_cache_tensors` rounds `num_blocks` up to a multiple of 64 (`spyre_model_runner.py:336`). The upstream `block_size_factor` machinery assumes the GPU/device block count and the offloaded block count are integer-related, which holds, but the alignment slack means a few blocks at the end are unusable. We should document this in the spec and not try to "use" the alignment slack on the host side.
5. **`SpyreOffloadingSpec` parent class.** Two viable bases: subclass `OffloadingSpec` directly (clean, but we duplicate the ~30 lines of `__init__` math from `CPUOffloadingSpec` that compute `num_blocks` from `cpu_bytes_to_use`); or subclass `CPUOffloadingSpec` and override `get_handlers` to skip the `is_cuda_alike()` gate (less duplication, but inherits a parent that documents itself as CUDA-only). The implementation will pick one once we see how much of `CPUOffloadingSpec` is genuinely CUDA-coupled vs. just gated. M1.5's `SpyreTieringOffloadingSpec` then subclasses whichever we picked, so the choice cascades.
6. **Mmap region on Spyre.** `TieringOffloadingManager` requires a `SharedOffloadRegion` to hand a `memoryview` to each `SecondaryTierManager`. The region is mmap-backed and platform-agnostic, but in M1 we may build host-side block tensors with `torch.empty` instead of from a `SharedOffloadRegion`. M1.5 needs to switch to the `SharedOffloadRegion` allocation path. Cost is small (one-line allocator swap) but worth noting up front so M1's `SpyreCpuOffloadingHandlers` accepts an optional pre-built region.
7. **Does `SharedStorageOffloadingSpec` work as the second connector in `MultiConnector` on Spyre?** This is the load-bearing question for M1.5 Path A. The spec itself extends `OffloadingSpec` directly (no `is_cuda_alike()` gate), but its worker-side `StorageOffloadingHandlers` ([`llmd_fs_backend/worker.py`](https://github.com/llm-d/llm-d-kv-cache/blob/main/kv_connectors/llmd_fs_backend/llmd_fs_backend/worker.py)) accepts a generic `GPULoadStoreSpec` tag and reads/writes via thread-pooled file I/O against a CPU staging buffer. It does **not** appear to call CUDA-specific copy ops on the device tensor itself — the device→host hop is owned by the *first* connector in the `MultiConnector` (us). If that holds, Path A works on Spyre with **zero** changes to `llmd_fs_backend`. Verifying this against a real run is M1.5 acceptance criterion #1 below; if it fails, the fix is a small upstream PR to `llmd_fs_backend` (likely a tag check) rather than Spyre-side glue.

## 11. Out of scope (filed as follow-ups)

- **M2 — Public Spyre device↔host primitive for third-party connectors.** Promote `spyre_inference.v1.kv_offload.copier.SpyreKvDmaCopier` to a stable, documented import surface so out-of-tree connectors that today target CUDA's `swap_blocks_batch` / `cudaMemcpy` can swap their device↔host hop for Spyre by importing one symbol. M1 builds the primitive; M2 commits to its API and documents it. (Raised by [@yuezhu1](https://github.com/yuezhu1) on the M1 draft.)
- **Direct device ↔ filesystem / object store.** Would need a Spyre-side analogue of NVIDIA GDS so a secondary tier can read/write device memory without a host bounce. Requires both a torch-spyre primitive and a contract change to upstream's `SecondaryTierManager` (which today takes a `primary_kv_view: memoryview` over CPU memory). Tracked separately. (Raised by [@yuezhu1](https://github.com/yuezhu1).)
- **PD disaggregation on Spyre.** Standalone RFC, builds on M1.
- **Async DMA on Spyre.** Depends on torch-spyre exposing a stream/event API. Until then, the synchronous handler is fine for offload/prefetch but precludes overlap with compute.
- **Stable on-device KV descriptor.** Depends on torch-spyre. Filed separately so the M1 handler can swap from `flit_offset+perfdsc` to the descriptor without changing the spec.
- **Authoring a new secondary tier.** Anything that does not slot into an existing `SecondaryTierManager` (e.g. a Spyre-to-Spyre direct fabric tier) is a separate design, not a milestone of this RFC.

## 12. Acceptance criteria

Each milestone's acceptance is a literal `vllm serve` invocation a deployment engineer can run, plus the observable behavior that confirms it works.

### M1 acceptance

**A1.1 — single-tier host-RAM offload runs end-to-end.**

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SpyreOffloadingSpec",
    "cpu_bytes_to_use": 8000000000,
    "lazy_offload": true
  }
}'
```

- [ ] Server boots. `OffloadingConnectorWorker.register_kv_caches` is reached on the Spyre worker without raising.
- [ ] A 1k-prompt sweep with overlapping prefixes reports a non-zero `kv_offload_blocks_evicted` (or the equivalent prefix-cache-hit metric exposed by `OffloadingConnector` in this vLLM version).
- [ ] Generated outputs match a non-offloading baseline run on the same prompts (allow ≤1e-3 token-distribution drift; with `temperature=0` outputs should be identical).

**A1.2 — plugin-side test suite green.**

- [ ] `pytest spyre_inference/tests/v1/kv_offload/test_copier_round_trip.py` passes on a Spyre runner.
- [ ] `pytest spyre_inference/tests/v1/kv_offload/test_spec_registration.py` and `test_handler_dispatch.py` pass on CPU-only runners.

**A1.3 — no plugin-platform-side regressions.**

- [ ] No changes required to `TorchSpyreWorker` or `TorchSpyrePlatform`. (If we have to change them, the RFC's premise is wrong — pause and revise.)
- [ ] `bash format.sh` clean.

### M1.5 acceptance

**A1.5.1 — MultiConnector + `llmd_fs_backend` (Path A — the deployment shape).** This is the literal config llm-d v0.8 clusters use today, with our `SpyreOffloadingSpec` slotted into the first child connector:

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "spec_name": "SpyreOffloadingSpec",
          "cpu_bytes_to_use": 8000000000,
          "lazy_offload": true
        }
      },
      {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "spec_name": "SharedStorageOffloadingSpec",
          "spec_module_path": "llmd_fs_backend.spec",
          "shared_storage_path": "/mnt/files-storage/",
          "block_size": 256,
          "threads_per_gpu": 64
        }
      }
    ]
  }
}'
```

- [ ] Server boots; both child connectors register `kv_caches` against the Spyre worker without raising. (The second connector receives Spyre device tensors per `MultiConnector.register_kv_caches`. If `llmd_fs_backend` raises here, file an upstream issue and either contribute a tag-only fix or document a workaround in this RFC — see §10 Q7.)
- [ ] After a warmup run, block files appear under `/mnt/files-storage/` with content-hashed filenames (the `FileMapper` scheme).
- [ ] Restart the server with the same config and same model; re-issue prompts that were warmed up; observe storage hits in `OffloadingConnector` metrics for the second connector.
- [ ] On a second host mounting the same `/mnt/files-storage`, a fresh `vllm serve` with the same config picks up the warmed prefixes (cross-instance share).
- [ ] Generated outputs match the M1 (single-tier) baseline.

**A1.5.2 — Forward-looking: upstream `tiering/fs` (Path B).** Smaller scope; tests that the upstream-canonical path works once we ship `SpyreTieringOffloadingSpec`:

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SpyreTieringOffloadingSpec",
    "cpu_bytes_to_use": 8000000000,
    "secondary_tiers": [
      {"type": "fs", "root_dir": "/mnt/kvcache", "n_read_threads": 16}
    ]
  }
}'
```

- [ ] Server boots; the warmup-restart-replay test from A1.5.1 reports a prefix-cache hit on the second run (with `PYTHONHASHSEED` pinned identically on both invocations).
- [ ] Cross-instance share works on a shared `root_dir`.

**A1.5.3 — plugin-side test suite green.**

- [ ] `pytest spyre_inference/tests/v1/kv_offload/test_tiering_spec.py` passes on CPU-only runners using the upstream `example` secondary tier.

**A1.5.4 — Engineering budget held.**

- [ ] M1.5 plugin-side LOC ≤ ~50 LOC of glue on top of M1, excluding tests. If A1.5.1 surfaces issues that require a Spyre-side adapter (rather than an upstream `llmd_fs_backend` fix), reassess the budget and revise this RFC.

## 13. References

- Upstream `OffloadingConnector`: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Upstream `OffloadingSpec`: `vllm/v1/kv_offload/base.py:319`
- Upstream CPU spec (CUDA-only today): `vllm/v1/kv_offload/cpu/spec.py`
- Upstream factory: `vllm/v1/kv_offload/factory.py:21`
- Upstream tiering framework (PR #40020, merged 2026-05-13): `vllm/v1/kv_offload/tiering/{base,manager,spec,factory}.py`
- Upstream FS secondary tier (PR #41735, merged 2026-05-24): `vllm/v1/kv_offload/tiering/fs/manager.py`
- Upstream object-store secondary tier (PR #41968, merged 2026-06-05): `vllm/v1/kv_offload/tiering/obj/`
- Upstream `SharedOffloadRegion`: `vllm/v1/kv_offload/cpu/shared_offload_region.py`
- Upstream `FileMapper` (content-hashed paths): `vllm/v1/kv_offload/file_mapper.py`
- Spyre KV allocation today: `spyre_inference/v1/worker/spyre_model_runner.py:322–368`
