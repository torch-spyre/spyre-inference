# RFC: Port upstream NixlConnector for PD disaggregation to spyre-inference

| Field | Value |
|---|---|
| Status | Draft |
| Authors | Pravein Govindan Kannan ([@praveingk](https://github.com/praveingk)), Vijay Naik ([@vknaik](https://github.com/vknaik)), Todd Deshane ([@toddllm](https://github.com/toddllm)) |
| Created | 2026-06-12 |
| Tracking | Companion design doc to [#76 — \[Epic\] Develop KVCacheConnector for Spyre](https://github.com/torch-spyre/spyre-inference/issues/76) |
| Related | vLLM `NixlConnector`, vLLM `OffloadingConnector`, [PR #240](https://github.com/torch-spyre/spyre-inference/pull/240) (upstream-connector-port — the offload substrate), [PR #264](https://github.com/torch-spyre/spyre-inference/pull/264) (`InMemorySpyreConnector` — tactical NIXL demo path), [PR #266](https://github.com/torch-spyre/spyre-inference/pull/266) (`SpyrePagedKVCacheAccessor`), prior internal Spyre PD-disaggregation prototype |

## 1. Motivation

Spyre deployments need PD (prefill-decode) disaggregation: route long-prompt prefill to one pool of workers, decode the resulting requests on a separate pool. PD is the standard architecture for production LLM serving on heterogeneous accelerators — it lets each phase scale independently and gives long-context prompts somewhere to land without dominating a serving pod's compute budget.

vLLM ships the canonical PD mechanism upstream as `NixlConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`) which has been optimized for 1) Heterogenous Tensor-parallelism, 2) Hybrid Memory allocator for hybrid models, and 3) Fault-tolerance. 
The NIXL Connector uses UCX backend by default, and has option to configure other transport backends. Additionally, it supports transfer through a CPU buffer device if the device doesn't support GPU-Direct RDMA.

This RFC describes how to wire `spyre-inference` into upstream `NixlConnector` so that:

- A user runs `vllm serve` on Spyre, sets `kv_connector: SpyreNixlConnector`, and gets working PD via upstream `NixlConnector` (the Spyre name routes to a one-method subclass — see §5.7).
- All cross-host machinery — handshake, agent metadata exchange, transfer state polling, compatibility hash, per-request `kv_transfer_params` routing — is upstream lifecycle, unchanged.
- The plugin-side Spyre code is small (~150 LOC of glue, **zero upstream patches** — Spyre admits itself via the existing `Platform.get_nixl_supported_devices` classmethod hook) and **inherits upstream NixlConnector evolution**: bug fixes, heterogeneous-TP improvements, and ecosystem connectors land on Spyre as soon as they land on CUDA.
- Uses CPU buffer to transfer KV Cache as an intermediate solution. When Spyre direct communication is available, it would be a transport backend change, and no NIXLConnector change would be required.
- Composing PD with prefix-cache reuse and tiering on Spyre is a standard `MultiConnector` configuration, not a new design problem.

The maintenance argument is the load-bearing one. A Spyre-specific PD connector that forks NIXL's lifecycle costs thousands of LOC and requires tracking upstream's evolution forever. Reusing upstream `NixlConnector` and overriding a single method on its worker — the host-buffer allocation, since Spyre's KV layout is a page-list NamedTuple rather than a single Tensor — costs ~150 LOC of plugin-side subclass and tracks upstream automatically. Every additional NIXL feature (new transports, async polling improvements, observability) becomes available immediately for Spyre.

## 2. Goals and non-goals

### Prerequisites (M0)

- **`SpyreKvDmaCopier`** — the device↔host DMA primitive originally designed in [PR #240](https://github.com/torch-spyre/spyre-inference/pull/240) and intended for public stable export in PR #240's M2. This RFC depends on that primitive being importable from `spyre_inference.v1.kv_offload.copier` (or its eventual stable surface).
- **`SpyrePagedKVCacheAccessor`** — the page-list ↔ block-layout abstraction from [PR #266](https://github.com/torch-spyre/spyre-inference/pull/266). PR #266 is foundation work, dependency-light. We import it directly; no carve-out fallback needed.

These two primitives bound the work specific to this RFC: everything else is a thin worker subclass and registration plumbing.

### Goals (M1)

- A user runs `vllm serve` on Spyre with `--kv-transfer-config '{"kv_connector":"SpyreNixlConnector","kv_role":"kv_producer","kv_buffer_device":"cpu",...}'` (prefill role) and a matching `kv_consumer` config on a separate decode worker; PD inference works end-to-end across them.
- Intra-host PD also works (two Spyre processes on the same host); UCX auto-selects SHM transport for same-host peers.
- All cross-host machinery — handshake, agent metadata exchange, transfer state polling, compatibility hash — is upstream `NixlConnector` lifecycle, unchanged. Plugin-side Spyre code is restricted to: (1) a one-method override of `NixlBaseConnectorWorker.initialize_host_xfer_buffer` (subclassed via `NixlPullConnectorWorker`) for the page-list-aware host buffer, (2) a `CopyBlocksOp` factory using `SpyrePagedKVCacheAccessor` + `SpyreKvDmaCopier`, and (3) factory registration of the subclass under the name `"SpyreNixlConnector"`.
- Imports reuse existing modules: `SpyreKvDmaCopier` from `spyre_inference.v1.kv_offload.copier` (PR #240 M2 stable surface) and `SpyrePagedKVCacheAccessor` from `spyre_inference.distributed.kv_transfer.kv_connector.v1.spyre_paged_kv_accessor` (PR #266). No duplicate device↔host primitive, no duplicate page-list abstraction.
- `pytest tests/v1/kv_transfer/test_spyre_pd_*` exercises the override and adapter on CPU-only runners with mocked upstream worker.
- Two-process intra-host PD test passes on a Spyre runner.

### Goals (M1.5)

- Compose with `OffloadingConnector(SpyreOffloadingSpec)` from PR #240 via `MultiConnector`: prefix-cache reuse stays available alongside PD. A request that arrives with `do_remote_prefill=True` is handled by `SpyreNixlConnector`; a request that misses but hits a previously-cached prefix is served by `OffloadingConnector`. Save fans out to both.


### Non-goals (M1 + M1.5)

- Async or overlapping DMA. Until torch-spyre exposes streams/events, all transfers are synchronous (same constraint as PR #240's `SpyreKvDmaCopier`). PD throughput is bounded by sync-DMA serialization until that lands.
- Heterogeneous-TP between prefill and decode beyond what upstream `NixlConnector` already supports. We verify against Spyre's TP=2 ceiling but do not extend.

## 3. Background

### 3.1 Upstream `NixlConnector` (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`)

Upstream split the connector into multiple files in 2026 Q2. Verified against `vllm-project/vllm` `main`:

- `nixl/connector.py` — `NixlBaseConnector`, `NixlPullConnector` (pull/READ mode), `NixlPushConnector` (push/WRITE mode). The name `NixlConnector` is preserved as a backward-compat alias for `NixlPullConnector` (`connector.py:386`), so `kv_connector: NixlConnector` continues to resolve to the pull-mode connector.
- `nixl/base_worker.py` — `NixlBaseConnectorWorker` (~2,300 LOC). Holds `initialize_host_xfer_buffer` (line 645), `set_host_xfer_buffer_ops` (line 690), and the `register_kv_caches` flow (line 863-864) that calls `initialize_host_xfer_buffer` when `use_host_buffer=True`.
- `nixl/pull_worker.py` / `nixl/push_worker.py` — `NixlPullConnectorWorker` / `NixlPushConnectorWorker`, mode-specific subclasses of `NixlBaseConnectorWorker`.
- `nixl/metadata.py` — `compute_nixl_compatibility_hash` (line 79).
- `nixl/utils.py` — `_NIXL_SUPPORTED_DEVICE` (line 18) and the platform-extension hook described in §3.3.

Pieces relevant to this RFC:

- **Per-request routing** — `kv_transfer_params: dict[str, Any]` carries `do_remote_prefill: bool`, `remote_engine_id: str`, `remote_host: str`, `remote_port: int`, plus block IDs. Set by the orchestrator.
- **Side-channel handshake** — listener thread bound to `VLLM_NIXL_SIDE_CHANNEL_PORT`, per-host-pair, one-time at startup. Decode-side worker fetches prefill agent metadata + memory layout via ZMQ.
- **Compatibility hash** — `compute_nixl_compatibility_hash()` hashes vLLM version + model + dtype + backend. Both sides must match before handshake completes.
- **Backend selection** — `nixl_backends` config defaults to `["UCX"]`. UCX auto-selects transport: SHM (same host), TCP/RDMA (different hosts).

### 3.2 The host-buffer path

For accelerators NIXL doesn't talk to natively, `NixlBaseConnectorWorker` exposes a CPU-staging path. Three relevant entry points in `nixl/base_worker.py`:

```python
# line 645
def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """Initialize transfer buffer in CPU mem for accelerators NOT directly
    supported by NIXL (e.g., tpu)"""
    for layer_name, kv_cache in kv_caches.items():
        kv_shape = kv_cache.shape           # ← assumes kv_cache is a Tensor
        kv_dtype = kv_cache.dtype
        ...
        xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")

# line 690
def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp): ...

# line 863-864 (inside register_kv_caches)
if self.use_host_buffer:
    self.initialize_host_xfer_buffer(kv_caches=kv_caches)
```

The `CopyBlocksOp` signature is `Callable[[device_kv, host_buf, src_blocks, dst_blocks, "h2d" | "d2h"], None]`. Upstream `NixlConnector` then calls `register_memory()` on the host buffer (not the device tensor) and ships *that* over the wire.

This is the seam Spyre plugs into. The same path TPU uses today.

### 3.3 Out-of-tree platform admission via `Platform.get_nixl_supported_devices`

Earlier versions of this RFC proposed a one-line patch to `_NIXL_SUPPORTED_DEVICE` to admit `"spyre"`. **Upstream already provides the hook for this.** At `nixl/utils.py:18-31`:

```python
_NIXL_SUPPORTED_DEVICE = {
    "cuda": ("cuda", "cpu"),
    "tpu":  ("cpu",),
    "xpu":  ("cpu", "xpu"),
    "cpu":  ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())
```

`Platform.get_nixl_supported_devices()` is a classmethod on `vllm/platforms/interface.py:999` (default `{}`) **specifically intended for out-of-tree platforms** to declare their supported `(device_type, kv_buffer_device)` pairs. Spyre's `TorchSpyrePlatform` implements this classmethod returning `{"spyre": ("cpu",)}`, and upstream picks it up automatically at import time.

**No upstream patch is required for device admission.** This is purely a plugin-side classmethod implementation.

### 3.4 The substrate from PR #240

PR #240 ships `SpyreKvDmaCopier` at `spyre_inference/v1/kv_offload/copier.py`:

```python
class SpyreKvDmaCopier:
    def copy_d2h(self, src_spyre: torch.Tensor, dst_host: torch.Tensor) -> None: ...
    def copy_h2d(self, src_host: torch.Tensor, dst_spyre: torch.Tensor) -> None: ...
```

Sync. Non-allocating. Two backends (`spyre_from_blob` preferred, `senlib_dma` fallback). PR #240 §11 commits M2 to making this a public stable import surface.

### 3.5 Spyre's KV cache layout — and where to handle the mismatch

The Spyre V1 attention backend (`spyre_inference/v1/attention/backends/spyre_attn.py`) allocates each layer's KV cache as a `SpyrePagedKVCache(k_pages, v_pages)` NamedTuple — two equal-length lists of per-page tensors of shape `[num_kv_heads, block_size, head_size]`. That layout is the attention backend's choice (driven by Spyre device-tensor slicing constraints) and is a Spyre-internal concern.

Upstream `NixlConnector.initialize_host_xfer_buffer` reads `kv_cache.shape` / `kv_cache.dtype` to allocate a host buffer of matching shape. On CUDA the device-side cache happens to be a single Tensor and the host mirror has the same shape — convenient, but not a fundamental contract. The host buffer's *purpose* is just to hold the right number of bytes per layer, addressable by block IDs; it doesn't need to mirror the device's storage container.

**Two wrong directions, ruled out:**

1. *Patch upstream `initialize_host_xfer_buffer` to know about page lists.* Pushes Spyre-specific behavior into upstream `NixlConnector`. Won't be accepted upstream — and shouldn't be.
2. *Change Spyre's attention-backend layout (page list → contiguous Tensor) to satisfy a host-buffer allocation step.* Drives a device-side memory-layout choice from a host-side concern. Wrong direction architecturally.

**The right place to handle the mismatch is the connector layer, plugin-side.** A small `NixlConnectorWorker` subclass in `spyre_inference` overrides `initialize_host_xfer_buffer` to compute the host-mirror shape from Spyre's page geometry (via `SpyrePagedKVCacheAccessor`) and allocate a NIXL-registerable host buffer. Override is local to one method; lives in the plugin; upstream `NixlConnector` is not modified; Spyre's attention-backend layout is not modified.

The subclass is registered under the factory name `"SpyreNixlConnector"` (see §5.7 for why a Spyre-specific name rather than re-registering `"NixlConnector"`). Operators on Spyre type that name in `kv_transfer_config`; CUDA / TPU deployments are unaffected and continue to use `"NixlConnector"`.

### 3.6 PR #266's `SpyrePagedKVCacheAccessor` — device page layout vs canonical block layout

PR #266 introduces `SpyrePagedKVCacheAccessor` — a uniform abstraction over the Spyre paged-cache layout that exposes geometry (`num_pages`, `num_kv_heads`, `block_size`, `head_dim`, `dtype`) and per-page read/write.

Two layouts matter here and must not be conflated (this is precisely where the prior internal prototype found bugs, and it's what §5.2/§5.3 must be consistent about):

| Layout | Shape | Where it lives | Who produces / consumes it |
|---|---|---|---|
| **Device page layout** | `[num_kv_heads, block_size, head_size]` per page | Individual page tensors inside `SpyrePagedKVCache(k_pages, v_pages)` on the Spyre device. This is what `spyre_attn.py` allocates and what the attention kernel reads. | Spyre attention backend. Not surfaced to the connector directly. |
| **Canonical block layout** | `[block_size, num_kv_heads, head_size]` per block | The CPU-side representation the accessor exposes. `read_block(...)` returns this shape; `write_block(..., values=...)` expects this shape. | `SpyrePagedKVCacheAccessor.read_block` / `write_block`, and everything downstream on the CPU side (our host xfer buffer, NIXL registration, the peer's host buffer). |

The two differ by a `(num_kv_heads, block_size) → (block_size, num_kv_heads)` transpose on the first two dims. The accessor absorbs this permutation so the connector layer only ever deals with the canonical block layout.

**This RFC reuses the accessor directly.** The plugin-side `SpyrePdNixlWorker` subclass (§5.2; subclasses upstream `NixlPullConnectorWorker`) calls `SpyrePagedKVCacheAccessor.try_from_kv_caches(kv_caches)` to derive the host-buffer shape and uses `read_block` / `write_block` inside the `CopyBlocksOp` (§5.3). No duplication between this RFC and PR #266 — same accessor, two consumers.

### 3.7 The two-track relationship to PR #264

PR #264 (`InMemorySpyreConnector`) implements PD on Spyre by reusing the previous standalone prototype's design — its own connector class, its own metadata flow, its own NIXL transfer logic. ~2,420 LOC for the connector class alone, plus stores, helpers, smoke harness.

That work is independently valuable: it's a working PD path that ships on Spyre soon, validates the page-list transport path, and gives users with deadlines a path forward. This RFC's path is the parallel upstream-aligned alternative — minimal plugin code (~150 LOC of subclass + `CopyBlocksOp`), reusing upstream `NixlConnector`'s lifecycle wholesale. Users select between the two paths via `kv_connector` config:

| Config | Behavior |
|---|---|
| `kv_connector: SpyreNixlConnector` | This RFC's path. Upstream `NixlConnector` lifecycle, with a Spyre-side worker subclass that handles the page-list host-buffer allocation. |
| `kv_connector: InMemorySpyreConnector` | PR #264's path. Self-contained Spyre PD connector. |
| `kv_connector: MultiConnector` with both | Stack them; load picks first hit, save fans out. |

Migration from one to the other is a config change (§7).

## 4. Proposed architecture

```text
┌─────────────────────────────── vllm upstream (untouched) ──────────────────────┐
│                                                                                  │
│  vLLM scheduler ──KVConnectorBase_V1 lifecycle──► NixlConnector                 │
│                                                       (= NixlPullConnector)     │
│                                                          │                       │
│        ┌─────────────────────────────────────────────────┴──────────────────┐  │
│        │ NixlBaseConnectorScheduler   NixlBaseConnectorWorker               │  │
│        │  · kv_transfer_params         · ZMQ handshake                      │  │
│        │  · per-request routing        · register_kv_caches                 │  │
│        │                               · initialize_host_xfer_buffer        │  │
│        │                               · NIXL register_memory               │  │
│        │                               · transfer state polling             │  │
│        │                                                                     │  │
│        │   No Spyre-specific code here. Untouched.                           │  │
│        └─────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│   Upstream OOT-platform hook (already exists):                                   │
│     nixl/utils.py:31  _NIXL_SUPPORTED_DEVICE.update(                             │
│                          current_platform.get_nixl_supported_devices())         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                          ▲
                                                          │ Spyre platform supplies:
                                                          │   {"spyre": ("cpu",)}
                                                          │
┌──────────────────── spyre-inference plugin (NEW) ──────────────────────────────┐
│                                                                                  │
│  spyre_inference/platform.py                                                     │
│                                                                                  │
│    class TorchSpyrePlatform(CpuPlatform):                                        │
│      @classmethod                                                                │
│      def get_nixl_supported_devices(cls):                                        │
│        return {"spyre": ("cpu",)}                                                │
│                                                                                  │
│  spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_pd_nixl.py        │
│                                                                                  │
│    class SpyrePdNixlWorker(NixlPullConnectorWorker):                             │
│      def initialize_host_xfer_buffer(self, kv_caches):                           │
│        # Spyre's per-layer cache is a page-list NamedTuple, not a Tensor.        │
│        acc = SpyrePagedKVCacheAccessor.try_from_kv_caches(kv_caches)             │
│        self.host_xfer_buffers = {                                                │
│          layer: torch.empty(                                                     │
│            (2, acc.num_pages, acc.block_size, acc.num_kv_heads, acc.head_dim),   │
│            dtype=acc.dtype, device='cpu')                                        │
│          for layer in acc.layer_names                                            │
│        }                                                                         │
│        self._spyre_accessor = acc                                                │
│                                                                                  │
│    class SpyrePdNixlConnector(NixlPullConnector):                                │
│      # constructs SpyrePdNixlWorker on the WORKER role                           │
│                                                                                  │
│    Registered under the factory name "SpyreNixlConnector" — see §5.7.            │
│                                                                                  │
│  Imports:                                                                        │
│    SpyreKvDmaCopier            ← PR #240 M2 public surface                       │
│    SpyrePagedKVCacheAccessor   ← PR #266                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Key shape: **upstream `NixlConnector` is not modified, and zero upstream patches are required.** Spyre admits itself via the existing `Platform.get_nixl_supported_devices` classmethod; the page-list-aware host-buffer allocation lives in a small plugin-side worker subclass that overrides one method. Spyre's attention-backend KV layout is also not modified.

### 4.1 Why a plugin-side worker subclass

Three things determine where this fix should live:

- **Upstream cleanliness.** Spyre-specific layout knowledge does not belong inside upstream `NixlConnector`. Upstream maintainers will reasonably refuse patches that put it there.
- **Layering correctness.** Spyre's device-side memory layout is the attention backend's concern. The connector's host-buffer allocation is the connector's concern. Driving a device-layout choice from a host-buffer requirement crosses the layer boundary the wrong direction.
- **Minimum surface area.** The mismatch between page-list and Tensor only matters at the one method that allocates the host mirror. Override that one method; nothing else changes.

A small plugin-side subclass that overrides `initialize_host_xfer_buffer` satisfies all three. The override is ~50 LOC and reads its inputs through PR #266's existing `SpyrePagedKVCacheAccessor`, so there is no duplicated layout knowledge inside the subclass itself.

### 4.2 Why this still counts as "use upstream NixlConnector"

The subclass is `class SpyrePdNixlWorker(NixlPullConnectorWorker)` — a one-method override. Lifecycle, handshake, agent metadata exchange, NIXL `register_memory()`, transfer state polling, compatibility hash, per-request `kv_transfer_params` routing — all of it inherits from upstream and is reused unchanged. Upstream bug fixes in any of those areas flow to Spyre automatically; we don't fork the lifecycle.

We do not re-register `"NixlConnector"` in the factory (upstream's `KVConnectorFactory.register_connector` raises `ValueError` on duplicate names — see `factory.py:33-34`). The Spyre subclass is registered under a distinct name `"SpyreNixlConnector"` (§5.7); operators on Spyre type that name in `kv_transfer_config`. The user-facing UX cost is one extra config token; in exchange, zero upstream patches.

## 5. Component design

### 5.1 Platform admission via `Platform.get_nixl_supported_devices`

`TorchSpyrePlatform` declares Spyre's supported NIXL `(device, kv_buffer_device)` pair through the existing upstream classmethod hook:

```python
# spyre_inference/platform.py — additive
class TorchSpyrePlatform(CpuPlatform):
    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        return {"spyre": ("cpu",)}
```

Upstream `nixl/utils.py:31` already calls `_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())` at module-load time, so the entry is picked up automatically when `spyre_inference` is the active platform. **No upstream patch.**

(If torch-spyre later exposes a NIXL-registerable device-memory path, we'd extend this to `{"spyre": ("cpu", "spyre")}` — same hook, no upstream change.)

### 5.2 Plugin-side worker subclass

A single small subclass that overrides one method. Spyre's attention-backend KV layout is not changed; upstream `NixlConnector` is not changed.

```python
# spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_pd_nixl.py
from typing import Any
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.spyre_paged_kv_accessor import (
    SpyrePagedKVCacheAccessor,
)


class SpyrePdNixlWorker(NixlPullConnectorWorker):
    """Worker subclass that allocates the host xfer buffer from Spyre's
    page-list KV cache. Everything else (handshake, register_memory,
    transfer-state polling) is inherited from upstream unchanged.

    initialize_host_xfer_buffer is defined on NixlBaseConnectorWorker
    (nixl/base_worker.py:645); we override it via this pull-worker
    subclass for the pull/READ mode that NixlConnector aliases."""

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, Any]) -> None:
        accessor = SpyrePagedKVCacheAccessor.try_from_kv_caches(kv_caches)
        if accessor is None:
            # Defensive: if Spyre ever switches to a Tensor-shaped layout
            # later, fall through to upstream's default behavior.
            return super().initialize_host_xfer_buffer(kv_caches)

        # Per layer, allocate a contiguous host mirror in the accessor's
        # CANONICAL BLOCK LAYOUT (§3.6): [block_size, num_kv_heads, head_size]
        # per block. Concretely, per layer:
        #   [2 (k|v), num_pages, block_size, num_kv_heads, head_size]
        # The (k|v) dimension is leading so a single page slice
        # host_buf[k|v, page_id] has the exact shape returned by
        # accessor.read_block(...) — see §5.3.
        host_buffers: dict[str, torch.Tensor] = {}
        for layer_name in accessor.layer_names:
            host_buffers[layer_name] = torch.empty(
                (2, accessor.num_pages,
                 accessor.block_size, accessor.num_kv_heads, accessor.head_dim),
                dtype=accessor.dtype, device="cpu",
            )

        self.host_xfer_buffers = host_buffers
        self._spyre_accessor = accessor


class SpyrePdNixlConnector(NixlPullConnector):
    """NixlPullConnector that constructs SpyrePdNixlWorker on the WORKER role.
    Registered with KVConnectorFactory under the name 'SpyreNixlConnector'
    from spyre_inference/__init__.py — only loads on Spyre platforms."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        # Construct upstream first so all base-class state is set up,
        # then swap in our worker on the WORKER role.
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            self.connector_worker = SpyrePdNixlWorker(
                vllm_config, self.engine_id, kv_cache_config,
            )
```

Spyre's attention backend's `SpyrePagedKVCache(k_pages, v_pages)` layout is **not modified**. The attention path keeps its current containers. Only the host-buffer allocation step looks at the geometry through `SpyrePagedKVCacheAccessor`.

(For push-mode PD, an analogous `SpyrePushPdNixlWorker(NixlPushConnectorWorker)` and `SpyrePushPdNixlConnector(NixlPushConnector)` would follow the same pattern; out of scope for M1 since pull mode is what `NixlConnector` aliases by default.)

### 5.3 The `CopyBlocksOp`

The CopyBlocksOp bridges the device side (via `SpyrePagedKVCacheAccessor.read_block` / `write_block`, which absorbs the page → block transpose from §3.6) and the host-mirror side (plain Tensor slicing).

**Shape contract at each seam** — this is the check the reviewer specifically called out:

- `accessor.read_block(layer_name, kv_kind, page_id)` returns a CPU tensor of shape `[block_size, num_kv_heads, head_size]` and dtype `accessor.dtype`.
- A single-block slice of the host mirror `host_buf[kind_idx, page_id]` has shape `[block_size, num_kv_heads, head_size]` (from §5.2: full shape `[2, num_pages, block_size, num_kv_heads, head_size]`; slicing two leading dims removes them).
- `accessor.write_block(..., values=...)` expects `values` of shape `[block_size, num_kv_heads, head_size]`.

All three match. Both `copy_()` (d2h) and `write_block(values=...)` (h2d) operate on identically-shaped tensors — no permute, no reshape, no fallback path.

```python
# spyre_inference/distributed/kv_transfer/kv_connector/v1/copy_blocks.py
from typing import Literal
import torch

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier  # PR #240 M2
from spyre_inference.distributed.kv_transfer.kv_connector.v1.spyre_paged_kv_accessor import (
    SpyrePagedKVCacheAccessor,
)


def make_spyre_copy_blocks(
    accessor: SpyrePagedKVCacheAccessor,
    copier: SpyreKvDmaCopier | None = None,
):
    copier = copier or SpyreKvDmaCopier()

    def copy_blocks(
        device_kv_caches: dict[str, object],         # Spyre page-list NamedTuples
        host_buffers: dict[str, torch.Tensor],       # contiguous host mirrors
        src_block_ids: list[int],
        dst_block_ids: list[int],
        direction: Literal["h2d", "d2h"],
    ) -> None:
        for layer_name in accessor.layer_names:
            # host_kv shape: [2 (k|v), num_pages, block_size, num_kv_heads, head_size]
            # (see §5.2 — canonical block layout, matching accessor.read_block)
            host_kv = host_buffers[layer_name]
            for kind_idx, kind in enumerate(("k", "v")):
                for src, dst in zip(src_block_ids, dst_block_ids):
                    if direction == "d2h":
                        # device page → CPU canonical block → host mirror slot
                        block = accessor.read_block(
                            layer_name=layer_name, kv_kind=kind, page_id=src,
                        )
                        # LHS slice: [block_size, num_kv_heads, head_size]
                        # RHS block: [block_size, num_kv_heads, head_size]  ✓
                        host_kv[kind_idx, dst].copy_(block)
                    else:  # "h2d"
                        # host mirror slot → CPU canonical block → device page
                        # values shape: [block_size, num_kv_heads, head_size]  ✓
                        block = host_kv[kind_idx, src]
                        accessor.write_block(
                            layer_name=layer_name, kv_kind=kind, page_id=dst,
                            values=block,
                        )

    return copy_blocks
```

`SpyrePagedKVCacheAccessor.read_block` / `write_block` already handle the device-side DMA and page-vs-block transpose. `SpyreKvDmaCopier` is consumed transitively when those page tensors are device-resident.

### 5.3.1 Host-buffer size estimate

The per-layer host mirror is `[2, num_pages, block_size, num_kv_heads, head_size]` in `accessor.dtype`, so:

```
bytes_per_layer = 2 * num_pages * block_size * num_kv_heads * head_size * dtype_bytes
bytes_total     = num_layers * bytes_per_layer
```

Two representative sizings:

| Model shape                      | num_layers | num_kv_heads | head_size | block_size | num_pages | dtype  | per-layer  | total     |
|----------------------------------|-----------:|-------------:|----------:|-----------:|----------:|--------|-----------:|----------:|
| GQA-8 8B-ish (e.g. Llama-3-8B)   | 32         | 8            | 128       | 64         | 2048      | fp16   | **512 MiB**| **16 GiB**|
| MHA 8B-ish, moderate pool        | 32         | 32           | 128       | 64         |  512      | fp16   | **512 MiB**| **16 GiB**|
| GQA-2 dense small (micro-g3.3-8B)| 40         | 2            | 128       | 64         | 1024      | fp16   |  64 MiB    |  2.5 GiB  |
| Same, larger pool                | 40         | 2            | 128       | 64         | 4096      | fp16   | 256 MiB    | 10 GiB    |

The host buffer scales linearly with `num_pages` (i.e. the KV-cache pool size), which is a deployment-time knob. For a Spyre worker with 32 Gi allocatable, 16 GiB of NIXL-registerable host mirror is comfortable but not free — deployment YAMLs (§6 M1.5) should either size the pool with this in mind or expose it as a config knob.

Two notes on where this can be reduced later, tracked as follow-ups:

- **HMPM-backed host buffer** (§10 out-of-scope, dependent on wangchen615/vllm PR #51): the host mirror moves from process-local `torch.empty` into a shared mmap region, so intra-host prefill+decode pairs share the buffer instead of allocating one each.
- **Windowed / streaming buffer**: allocate only the working set of pages that PD actively transfers at any moment, rather than mirroring the full pool. Requires connector-lifecycle changes; not in this RFC.

### 5.4 Worker-side wiring of `set_host_xfer_buffer_ops`

The subclass also wires the `CopyBlocksOp` once `register_kv_caches` has populated `host_xfer_buffers`. The simplest place is inside the same subclass, immediately after the host-buffer allocation:

```python
# Inside SpyrePdNixlWorker (extends §5.2):
def register_kv_caches(self, kv_caches):
    super().register_kv_caches(kv_caches)   # populates host_xfer_buffers via our override
    if getattr(self, "_spyre_accessor", None) is None:
        return
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.copy_blocks import (
        make_spyre_copy_blocks,
    )
    self.set_host_xfer_buffer_ops(make_spyre_copy_blocks(self._spyre_accessor))
```

(Equivalently, the model runner can wire it through `kv_connector.set_host_xfer_buffer_ops(...)` mirroring the GPU pattern at [`gpu_model_runner.py:6534`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L6534). Either placement is correct; we'll pick whichever fits more cleanly with how Spyre's worker is structured at implementation time.)

### 5.5 Backend configuration

Default `nixl_backends = ["UCX"]`. UCX auto-routes:

- Same host → `shm`, `cma`, `knem`.
- Different host, RDMA NIC present → RDMA.
- Different host, no RDMA → TCP. Force with `UCX_TLS=tcp` if needed.

### 5.6 Compatibility hash (no upstream change required for M1)

`compute_nixl_compatibility_hash` already hashes vLLM version + model + dtype + backend, which is sufficient for M1: prefill and decode on Spyre run identical code paths and identical layouts; mismatched configs already fail loud.

A future RFC could propose adding a `host_buffer_source` field to the hash if/when HMPM lands, so that an `"mmap_hmpm"` host buffer on one side doesn't silently pair with a `"torch_empty"` buffer on the other. **Out of scope for this RFC** — neither M1 acceptance nor heterogeneous-platform PD requires it.

### 5.7 Connector registration under the name `"SpyreNixlConnector"`

Upstream's `KVConnectorFactory.register_connector` raises `ValueError` on duplicate names ([factory.py:33-34](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/factory.py#L33-L34)):

```python
if name in cls._registry:
    raise ValueError(f"Connector '{name}' is already registered.")
```

So we cannot re-register `"NixlConnector"` from `spyre_inference` — the import would crash at plugin load. The Spyre subclass is registered under a distinct name:

```python
# spyre_inference/__init__.py — additive
from vllm.distributed.kv_transfer.kv_connector.v1.factory import KVConnectorFactory

KVConnectorFactory.register_connector(
    "SpyreNixlConnector",
    "spyre_inference.distributed.kv_transfer.kv_connector.v1.spyre_pd_nixl",
    "SpyrePdNixlConnector",
)
```

Operators on Spyre type `kv_connector: SpyreNixlConnector`. CUDA / TPU operators continue to type `kv_connector: NixlConnector`. The cost is one extra config token per Spyre deployment; in exchange, zero upstream patches and no risk of breaking CUDA / TPU paths. The Spyre subclass still inherits all of upstream `NixlConnector`'s lifecycle — handshake, agent metadata, transfer state polling, compatibility hash — by virtue of being a subclass. The connector-name string is just a routing label.

If unifying the user-facing name across platforms becomes important later, it can be done with a small upstream factory enhancement (e.g., `register_connector(..., allow_override=True)` or a `Platform.register_connector_overrides()` hook). That's filed as a follow-up — out of scope for M1.

## 6. File-by-file plan

### M1 new files

| File | Purpose | Approx LOC |
|---|---|---|
| `spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_pd_nixl.py` | `SpyrePdNixlConnector` + `SpyrePdNixlWorker` (subclass of upstream `NixlConnectorWorker`; overrides `initialize_host_xfer_buffer`; wires `CopyBlocksOp`). | ~150 |
| `spyre_inference/distributed/kv_transfer/kv_connector/v1/copy_blocks.py` | `make_spyre_copy_blocks` factory using `SpyrePagedKVCacheAccessor` + `SpyreKvDmaCopier`. | ~80 |

(The path `spyre_inference/distributed/kv_transfer/kv_connector/v1/` and its `__init__.py` files are already created by PR #266.)

### M1 modified files

| File | Change |
|---|---|
| `spyre_inference/__init__.py` | `KVConnectorFactory.register_connector("SpyreNixlConnector", ...)` pointing at `SpyrePdNixlConnector`. |
| `spyre_inference/platform.py` | Add classmethod `get_nixl_supported_devices(cls) -> {"spyre": ("cpu",)}`. |
| `spyre_inference/envs.py` | Add `SPYRE_NIXL_PD_DEBUG` env var (optional verbose handshake logging). |

**No changes to `spyre_inference/v1/attention/backends/spyre_attn.py`.** The KV cache layout is unchanged.

### M1 upstream patches

**None.** Upstream already provides `Platform.get_nixl_supported_devices()` ([interface.py:999](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/interface.py#L999)) for OOT-platform device admission, and `nixl/utils.py:31` consumes it. All connector-side behavior comes from a plugin-side subclass. Spyre-specific behavior stays out of upstream `NixlConnector` entirely.

### M1 tests

| File | Coverage |
|---|---|
| `tests/v1/kv_transfer/test_spyre_pd_copy_blocks.py` | Unit: `make_spyre_copy_blocks` round-trip d2h → h2d → d2h preserves bytes. **Pins the shape contract from §3.6 / §5.3**: the host-mirror slot `host_buf[kind_idx, page_id]` and `accessor.read_block(...)` return / `accessor.write_block(values=...)` accept must all be `[block_size, num_kv_heads, head_size]` (canonical block layout, not the device page layout `[num_kv_heads, block_size, head_size]`). Test uses a fake `SpyrePagedKVCacheAccessor` that returns/accepts the canonical block layout and asserts (a) the tensor shapes at both seams match, (b) a synthetic pattern round-trips byte-for-byte through the device-side page transpose. Page-vs-block layout was where the prior internal prototype found bugs; this test pins the contract. CPU-only. |
| `tests/v1/kv_transfer/test_spyre_pd_init_buffer.py` | Unit: `SpyrePdNixlWorker.initialize_host_xfer_buffer` over a fake page-list `kv_caches` produces correctly-shaped host buffers; falls through to `super().initialize_host_xfer_buffer` for non-page-list inputs. CPU-only, mocks the upstream worker base where needed. |
| `tests/v1/kv_transfer/test_spyre_pd_factory.py` | Unit: `KVConnectorFactory.create_connector_v1("SpyreNixlConnector", ...)` resolves to `SpyrePdNixlConnector` when `spyre_inference` is loaded. CPU-only. |
| `tests/v1/kv_transfer/test_pd_intra_host.py` | Integration: spawn two `vllm serve` subprocesses on one host (one prefill, one decode) with `kv_connector: SpyreNixlConnector`, submit a request, verify generated text matches a single-process baseline. Spyre runner. |
| `tests/v1/kv_transfer/test_pd_inter_host.py` | Integration: gated, two physical hosts with Spyre on each. Manual or CI-fixture. |

### M1.5 new files

| File | Purpose | Approx LOC |
|---|---|---|
| `tests/v1/kv_transfer/test_pd_with_offload_compose.py` | `MultiConnector([SpyreNixlConnector, OffloadingConnector(SpyreOffloadingSpec)])` end-to-end. PD + prefix-cache reuse. | ~200 |

### M1.5 deployment YAMLs

| File | Purpose |
|---|---|
| `deployment/spyre-pd-prefill.yaml` | k8s Deployment for prefill role (`kv_connector: SpyreNixlConnector` + `kv_role: kv_producer`). |
| `deployment/spyre-pd-decode.yaml` | k8s Deployment for decode role (`kv_connector: SpyreNixlConnector` + `kv_role: kv_consumer`). |
| `deployment/spyre-pd-with-offload.yaml` | `MultiConnector` composition: `SpyreNixlConnector` + `OffloadingConnector(SpyreOffloadingSpec)`. |

## 7. Compatibility with PR #240, PR #264, and the wider ecosystem

After M1 ships, the following work on Spyre **without further plugin code**:

- **Standalone PD via `SpyreNixlConnector`** (M1) — kv_producer/kv_consumer roles, intra-host (UCX-SHM) or inter-host (UCX-TCP/RDMA).
- **PD via `InMemorySpyreConnector` (PR #264)** — unchanged. Both connectors registered; users select via `kv_connector` config.
- **PD + prefix-cache reuse** (M1.5) — `MultiConnector([SpyreNixlConnector, OffloadingConnector(SpyreOffloadingSpec)])`. Requires PR #240 M1 landed.
- **PD + prefix-cache + tiering** — `MultiConnector([SpyreNixlConnector, OffloadingConnector(SpyreTieringOffloadingSpec)])`. Requires PR #240 M1.5.

### 7.1 Coexistence with `InMemorySpyreConnector` (PR #264)

`InMemorySpyreConnector` registers with `KVConnectorFactory` under its own name; upstream `NixlConnector` is registered by upstream vLLM. Users select one or compose via `MultiConnector`. There is no implicit dispatch.

| Operational scenario | Recommended choice |
|---|---|
| Need PD on Spyre this week | `InMemorySpyreConnector` — it's the working PR. |
| Long-term deployment with prefix-cache reuse on top of PD | `SpyreNixlConnector` (this RFC) + `OffloadingConnector(SpyreOffloadingSpec)` via `MultiConnector`. |
| Composing with LMCache or other upstream ecosystem connectors | `SpyreNixlConnector` — it's a subclass of upstream `NixlPullConnector`, so anything built on upstream `NixlConnector` semantics composes with it. |
| Heterogeneous deployment alongside CUDA / TPU workers | `SpyreNixlConnector` on the Spyre side, `NixlConnector` on CUDA / TPU. Both inherit from upstream's `NixlBaseConnector`, so the orchestrator's routing rules are unchanged. |

The two connectors are not internally compatible: a request prefilled on `InMemorySpyreConnector` cannot be decoded on `SpyreNixlConnector` and vice versa, because handshake protocols and metadata schemas differ. **Within a single deployment, pick one.**

### 7.2 Migration path from `InMemorySpyreConnector` to upstream `NixlConnector`

The migration is a `kv_transfer_config` change. Equivalence table for common settings:

| `InMemorySpyreConnector` | upstream `NixlConnector` |
|---|---|
| `kv_connector: InMemorySpyreConnector` | `kv_connector: SpyreNixlConnector` |
| `kv_role: kv_producer` | `kv_role: kv_producer` (unchanged — upstream key) |
| `kv_role: kv_consumer` | `kv_role: kv_consumer` (unchanged) |
| `kv_connector_extra_config.use_nixl: true` | implicit (always uses NIXL) |
| `kv_connector_extra_config.nixl_remote_ip: <ip>` | propagated via `kv_transfer_params.remote_host` per request (set by orchestrator) |
| `kv_connector_extra_config.nixl_port: <port>` | `VLLM_NIXL_SIDE_CHANNEL_PORT` env var |
| `VLLM_SPYRE_KV_ROLE` env override | n/a — `kv_role` is the upstream config key; no env override needed |

The acceptance criteria (§11) include a same-workload, same-output verification across both connectors.

## 8. Open questions

1. **Worker construction pattern.** §5.2's `SpyrePdNixlConnector.__init__` calls `super().__init__(...)` (which constructs upstream's `NixlPullConnectorWorker`) and then replaces `self.connector_worker` with `SpyrePdNixlWorker` on the WORKER role. This works in the current upstream layout (single worker constructed once, in `connector.py:336-340`), but assumes upstream doesn't reuse or reference the original worker between super-init and our replacement. Worth verifying at implementation — if any base-class code in `__init__` triggers worker startup (e.g., kicks off the side-channel listener), we may need to defer the swap. If upstream's construction model becomes incompatible, the fallback is a small upstream patch parametrising the worker class (e.g., `_worker_cls = NixlPullConnectorWorker` as a class attribute that subclasses can override).

2. **Unifying user-facing connector name across platforms.** §5.7 settles on `kv_connector: SpyreNixlConnector` for Spyre because upstream forbids re-registering `"NixlConnector"`. If `kv_connector: NixlConnector` everywhere is desired later, a small upstream factory enhancement (e.g., `register_connector(..., allow_override=True)` for OOT-platform overrides) would do it. Out of scope for M1.

3. **`flit_offset` stability across processes.** Shared with PR #240 §10 Q1 and (transitively) any PD design including `InMemorySpyreConnector`. The verification is the same one-day spike on real hardware. Result blocks all three workstreams; doing it once unblocks all of them.

4. **NIXL `register_memory()` on Spyre's host buffer.** PR #240 §6.2: no `cudaHostRegister` equivalent on Spyre. UCX-SHM and UCX-TCP backends generally accept plain mmap; UCX-RDMA may need pinning. Resolution: verification spike. If RDMA pinning is required and we can't satisfy it, M1 acceptance restricts to UCX-SHM (intra-host) and UCX-TCP (inter-host); RDMA becomes a follow-up.

5. **TP heterogeneity between prefill and decode.** Upstream `NixlConnector` handles TP_ratio>1; Spyre's `SpyreCommunicator` currently supports TP=2. M1 acceptance restricts to symmetric TP between prefill and decode. Whether asymmetric Spyre TP across PD works is filed as M2.

6. **Compatibility hash composition.** Existing upstream hash is sufficient for M1. If/when HMPM lands and we want to distinguish standalone vs HMPM host buffers, that's a small additive change proposed in a separate RFC — not in this one.

7. **Eviction race during PD handoff.** Resolved for this path: upstream `NixlConnector`'s host buffer is connector-owned (allocated at `initialize_host_xfer_buffer`), not pool-managed. Blocks live in the buffer until decode acks. **No additional action needed.** This was a real concern with the OffloadingConnector-based path we considered earlier and is one of the structural reasons NixlConnector fits PD better.

8. **Block-table reconciliation.** Decode allocates its own page IDs; the bytes have to land in those pages. Upstream NixlConnector handles this via per-request `kv_transfer_params.block_ids`. We verify intra-host first; the page-list layout shouldn't make this materially different from the contiguous case because the accessor abstracts page IDs.

## 9. Out of scope (filed as follow-ups)

- **Async DMA on Spyre.** Depends on torch-spyre exposing a stream/event API. Until then, PD throughput is bounded by single-stream sync DMA (~3 GB/s per PR #240 §4.1). Same dependency as PR #240 and PR #264.
- **Stable on-device KV descriptor.** Shared with PR #240 and PR #264. Until torch-spyre exposes a stable kv-region descriptor, all three depend on `flit_offset` from `perfdsc` artifacts.
- **Asymmetric TP between prefill and decode on Spyre.** Filed as M2 work, dependent on `SpyreCommunicator` evolution.
- **Cross-connector composition beyond `MultiConnector`.** PD + offload + tiering + LMCache stacking is supported through `MultiConnector` already; bespoke composition logic is out of scope.

## 10. Acceptance criteria

### M1 acceptance

**A1.1 — intra-host PD round-trip.**

```bash
# Prefill role
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "SpyreNixlConnector",
  "kv_role": "kv_producer",
  "kv_buffer_device": "cpu",
  "kv_connector_extra_config": {
    "backends": ["UCX"]
  }
}' --port 8001

# Decode role (same host)
VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "SpyreNixlConnector",
  "kv_role": "kv_consumer",
  "kv_buffer_device": "cpu",
  "kv_connector_extra_config": {
    "backends": ["UCX"]
  }
}' --port 8002
```

- [ ] Both servers boot. `register_kv_caches` reaches the Spyre worker on each side. NIXL handshake completes.
- [ ] Orchestrator (or a hand-rolled test client) dispatches a request with `do_remote_prefill=True` and matching `remote_engine_id` / `remote_host`.
- [ ] Prefill writes to Spyre device, `CopyBlocksOp` mirrors pages into the contiguous host buffer, NIXL READ pulls the bytes to decode's host buffer, `CopyBlocksOp` writes pages back to decode device.
- [ ] Generated output matches a non-PD baseline run on the same prompt, byte-identical at `temperature=0`.
- [ ] UCX selected SHM transport (verify via `UCX_LOG_LEVEL=info`) — confirms intra-host fast path, no TCP stack involvement.

**A1.2 — inter-host PD round-trip.**

Two physical hosts A (prefill) and B (decode), Spyre on each:

```bash
# Host A: prefill
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "SpyreNixlConnector",
  "kv_role": "kv_producer",
  "kv_buffer_device": "cpu"
}' --host 0.0.0.0 --port 8001

# Host B: decode
VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "SpyreNixlConnector",
  "kv_role": "kv_consumer",
  "kv_buffer_device": "cpu"
}' --host 0.0.0.0 --port 8002
```

- [ ] NIXL side-channel handshake completes between A and B.
- [ ] Decode pulls KV via NIXL READ over UCX (TCP if no RDMA NIC; RDMA if available).
- [ ] Generated output matches the intra-host run for the same prompt.
- [ ] PD throughput is non-trivially better than full re-prefill on host B for prompts ≥256 tokens of shared prefix.

**A1.3 — equivalence with `InMemorySpyreConnector` (PR #264) on the same workload.**

- [ ] Same model, same prompt set, switch only `kv_connector` between `NixlConnector` and `InMemorySpyreConnector`. Generated outputs match byte-identically at `temperature=0`.
- [ ] Documented in `deployment/spyre-inmemory-to-nixl-migration.md` — config equivalence verified.

**A1.4 — plugin-side test suite green.**

- [ ] `pytest tests/v1/kv_transfer/test_spyre_pd_*` passes on a Spyre runner.
- [ ] `pytest tests/v1/kv_transfer/test_spyre_pd_init_buffer.py` and `test_spyre_pd_factory.py` pass on CPU-only runners.

**A1.5 — no regression on PR #240, PR #264, or non-Spyre devices.**

- [ ] PR #240 M1 acceptance tests still pass — the Spyre attention backend's KV layout is unchanged.
- [ ] PR #264's `InMemorySpyreConnector` tests still pass.
- [ ] CUDA / TPU `NixlConnector` deployments are unaffected — `spyre_inference` only loads on Spyre platforms; the `"SpyreNixlConnector"` registration does not exist elsewhere; and zero upstream patches are required.
- [ ] `bash format.sh` clean.

### M1.5 acceptance

**A1.5.1 — PD + prefix-cache compose.**

```bash
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "MultiConnector",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "SpyreNixlConnector",
        "kv_role": "kv_consumer",
        "kv_buffer_device": "cpu"
      },
      {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "spec_name": "SpyreOffloadingSpec",
          "cpu_bytes_to_use": 8000000000
        }
      }
    ]
  }
}'
```

- [ ] Server boots; both child connectors register `kv_caches` against the Spyre worker.
- [ ] A request with `do_remote_prefill=True` goes through the upstream `NixlConnector` (PD path).
- [ ] A subsequent request whose prefix overlaps the prior one's prompt hits the `OffloadingConnector` prefix cache without going through PD — verified via `kv_offload_blocks_loaded` metric.
- [ ] Generated outputs match the non-composed M1 baselines for both kinds of requests.

**A1.5.2 — plugin-side test suite green.**

- [ ] `pytest tests/v1/kv_transfer/test_pd_with_offload_compose.py` passes on a Spyre runner.

**A1.5.3 — engineering budget.**

- [ ] M1 plugin-side LOC ≤ ~500, excluding tests. The whole point of this RFC vs PR #264 is the small footprint; if the implementation grows the plugin LOC by an order of magnitude, the design is wrong — pause and revise.

## 12. References

- Upstream `NixlConnector` family: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` (`connector.py`, `base_worker.py`, `pull_worker.py`, `push_worker.py`, `metadata.py`, `utils.py`).
- Upstream `_NIXL_SUPPORTED_DEVICE` + OOT-platform hook: `nixl/utils.py:18-31`.
- Upstream `Platform.get_nixl_supported_devices`: `vllm/platforms/interface.py:999`.
- Upstream `initialize_host_xfer_buffer`: `nixl/base_worker.py:645`.
- Upstream `set_host_xfer_buffer_ops`: `nixl/base_worker.py:690`.
- Upstream `compute_nixl_compatibility_hash`: `nixl/metadata.py:79`.
- Upstream `KVConnectorFactory.register_connector` (raises on duplicate names): `vllm/distributed/kv_transfer/kv_connector/factory.py:31-34`.
- Upstream `MultiConnector`: `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`
- Upstream `OffloadingConnector` (sibling track): `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- Companion RFC (PR #240): `docs/architecture/rfcs/upstream-connector-port.md`
- Tactical NIXL connector (PR #264): `spyre_inference/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- Page accessor (PR #266): `spyre_inference/distributed/kv_transfer/kv_connector/v1/spyre_paged_kv_accessor.py`
- HMPM RFC: [wangchen615/vllm PR #51](https://github.com/wangchen615/vllm/pull/51)
- Prior internal Spyre PD prototype: `llm-d-on-spyre/llm-d-pd-utils/app/pd_disagg/`
