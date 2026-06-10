# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import spyre_inference.envs as envs_spyre
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.v1.core.sched.output import SchedulerOutput

from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
    KVKind,
    SavedRequestRecord,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    SpyreConnectorStats,
    SpyreKVStoreBackend,
    StoreKey,
    build_spyre_kv_store_backend,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.heap_kv_accessor import (
    resolve_heap_kv_paths,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.heap_kv_inprocess_client import (
    InProcessHeapKVClient,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.spyre_paged_kv_accessor import (
    SpyrePagedKVCacheAccessor,
)

# NIXL imports for KV cache transfer
try:
    from nixl._api import nixl_agent, nixl_agent_config

    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False
    nixl_agent = None
    nixl_agent_config = None

NIXL_PORT = 9100

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)
_CONNECTOR_PROBE_PREFIX = "[KV_CONNECTOR_PROBE]"

_GLOBAL_STORE: SpyreKVStoreBackend | None = None


def _build_configured_store(store_backend_name: str | None = None) -> SpyreKVStoreBackend:
    backend_name = store_backend_name or envs_spyre.VLLM_SPYRE_KV_STORE_BACKEND
    return build_spyre_kv_store_backend(
        backend_name,
        max_bytes=envs_spyre.VLLM_SPYRE_KV_STORE_MAX_BYTES,
        max_saved_requests=envs_spyre.VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE,
        service_socket=envs_spyre.VLLM_SPYRE_KV_SERVICE_SOCKET,
    )


def get_global_store(store_backend_name: str | None = None) -> SpyreKVStoreBackend:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = _build_configured_store(store_backend_name)
    return _GLOBAL_STORE


def reset_global_store(store_backend_name: str | None = None) -> None:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        _GLOBAL_STORE.shutdown()
    _GLOBAL_STORE = _build_configured_store(store_backend_name)


def _emit_connector_probe(phase: str, **payload: Any) -> None:
    if not envs_spyre.VLLM_SPYRE_KV_PLACEMENT_PROBE_ENABLED:
        return
    logger.info(
        "%s %s", _CONNECTOR_PROBE_PREFIX, json.dumps({"phase": phase, **payload}, sort_keys=True)
    )


@dataclass(frozen=True)
class _PendingLoadSource:
    source: SavedRequestRecord
    matched_tokens_total: int
    num_local_computed_tokens: int


class InMemorySpyreConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
        store: SpyreKVStoreBackend | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._role_name = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        self._block_size = vllm_config.cache_config.block_size
        self._store = store if store is not None else get_global_store()

        # Step 2: Store KV role for file-based transfer
        self._kv_role = os.environ.get("VLLM_SPYRE_KV_ROLE", "")

        self._pending_requests: list[SpyreConnectorRequestMeta] = []
        self._saved_requests: OrderedDict[str, SavedRequestRecord] = OrderedDict()
        self._saved_requests_max_size = envs_spyre.VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE
        self._pending_load_sources: dict[str, _PendingLoadSource] = {}

        self._kv_caches: dict[str, torch.Tensor] = {}
        self._num_layers = 0
        self._num_kv_heads = 0
        self._head_dim = 0
        self._dtype_str = ""
        self._layer_names: list[str] = []

        self._step_stores: set[str] = set()
        self._step_loads: set[str] = set()
        self._load_error_block_ids: set[int] = set()
        self._blocks_saved = 0
        self._blocks_loaded = 0
        self._blocks_missing = 0
        self._stats = SpyreConnectorStats()
        self._use_heap_kv = bool(envs_spyre.VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE)
        self._heap_kv_strict = bool(envs_spyre.VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT)
        self._heap_kv_client: InProcessHeapKVClient | None = None
        self._heap_kv_init_error: str | None = None
        self._paged_accessor: SpyrePagedKVCacheAccessor | None = None

        # Step 2: Store prompt tokens for file transfer (before request_finished)
        self._pending_prompt_tokens: dict[str, list[int]] = {}

        # Per-request tracking of computed tokens (fixes cache hit degradation bug)
        self._request_computed_tokens: dict[str, int] = {}

        # NIXL integration for KV cache transfer
        self._use_nixl = envs_spyre.VLLM_SPYRE_ENABLE_NIXL_TRANSFER
        self._nixl_agent = None
        self._nixl_register_descs = None
        self._nixl_remote_ip = os.environ.get("VLLM_SPYRE_NIXL_REMOTE_IP", "10.130.2.89")

        # Decode-initiated pull: Prefill stores pending transfers, decode requests them
        self._pending_transfers = {}  # Map: req_id -> {metadata, descriptors, register_descs, timestamp}
        self._pending_transfers_lock = None  # Thread lock for map access
        self._pull_request_handler_thread = None  # Background thread to handle PULL_REQUEST
        self._pull_request_handler_stop = False  # Signal to stop thread

        if self._use_nixl:
            if not NIXL_AVAILABLE:
                logger.warning("[InMemorySpyreConnector] NIXL requested but not available")
                self._use_nixl = False
            else:
                # Defer NIXL agent initialization until first use to avoid metadata errors
                # when remote peer is not yet available
                logger.info(
                    "[InMemorySpyreConnector] NIXL enabled, will initialize on first transfer"
                )

                # Initialize thread lock for pending_transfers map
                import threading

                self._pending_transfers_lock = threading.Lock()

                # Register cleanup handlers for graceful shutdown
                atexit.register(self._cleanup_nixl)
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)

    def _init_nixl_agent(self) -> None:
        """Initialize NIXL agent for KV cache transfer (lazy initialization)"""
        if self._nixl_agent is not None:
            return  # Already initialized

        try:
            port = NIXL_PORT if self._kv_role == "kv_producer" else 0
            config = nixl_agent_config(True, True, port)
            role = "server" if self._kv_role == "kv_producer" else "client"
            self._nixl_agent = nixl_agent(role, config)
            logger.info(
                "[InMemorySpyreConnector] NIXL agent initialized: role=%s, port=%d", role, port
            )

            # For client role, establish connection to server
            if self._kv_role == "kv_consumer":
                logger.info(
                    "[InMemorySpyreConnector] Client connecting to server at %s:%d",
                    self._nixl_remote_ip,
                    NIXL_PORT,
                )
                self._nixl_agent.fetch_remote_metadata("server", self._nixl_remote_ip, NIXL_PORT)
                self._nixl_agent.send_local_metadata(self._nixl_remote_ip, NIXL_PORT)

                # Wait for server to register us before sending signal
                logger.info(
                    "[InMemorySpyreConnector] Waiting for server to complete metadata exchange..."
                )
                wait_count = 0
                while not self._nixl_agent.check_remote_metadata("server") and wait_count < 500:
                    time.sleep(0.01)
                    wait_count += 1

                if not self._nixl_agent.check_remote_metadata("server"):
                    logger.error(
                        "[InMemorySpyreConnector] Server metadata not available after %d checks",
                        wait_count,
                    )
                    raise RuntimeError("Server metadata exchange timeout")

                logger.info(
                    "[InMemorySpyreConnector] Server metadata available after %d checks", wait_count
                )

                # Now send READY signal to server so it knows we're about to start polling
                logger.info("[InMemorySpyreConnector] Sending READY signal to server")
                self._nixl_agent.send_notif("server", b"CLIENT_READY")
                logger.info("[InMemorySpyreConnector] CLIENT_READY signal sent successfully")
        except Exception as exc:
            logger.error("[InMemorySpyreConnector] Failed to initialize NIXL agent: %s", exc)
            self._use_nixl = False
            self._nixl_agent = None

    def _cleanup_nixl(self) -> None:
        """Cleanup NIXL resources on shutdown"""
        # Stop pull request handler thread first
        self._stop_pull_request_handler()

        # Deregister any pending memory from pending_transfers map
        if self._nixl_agent is not None and self._pending_transfers:
            with self._pending_transfers_lock:
                for req_id, transfer_data in self._pending_transfers.items():
                    try:
                        self._nixl_agent.deregister_memory(transfer_data["register_descs"])
                        logger.info(
                            "[InMemorySpyreConnector] Deregistered memory for req_id=%s", req_id
                        )
                    except Exception as exc:
                        logger.warning(
                            "[InMemorySpyreConnector] Error deregistering memory for %s: %s",
                            req_id,
                            exc,
                        )
                self._pending_transfers.clear()

        if self._nixl_agent is not None:
            try:
                peer = "client" if self._kv_role == "kv_producer" else "server"
                self._nixl_agent.remove_remote_agent(peer)
                logger.info(
                    "[InMemorySpyreConnector] NIXL agent cleaned up via atexit/signal handler"
                )
            except Exception as exc:
                logger.warning("[InMemorySpyreConnector] NIXL cleanup error: %s", exc)
            self._nixl_agent = None

    def _signal_handler(self, signum, frame):
        """Handle termination signals to cleanup NIXL resources"""
        logger.info("[InMemorySpyreConnector] Received signal %d, cleaning up NIXL...", signum)
        self._cleanup_nixl()
        # Re-raise the signal to allow normal termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _reset_nixl_transfer_state(self) -> None:
        """Reset NIXL transfer state for next request"""
        if self._nixl_agent is None:
            return

        try:
            # Clear pending notifications (drain the queue)
            drained_count = 0
            while True:
                notifs = self._nixl_agent.get_new_notifs()
                if not notifs or all(len(v) == 0 for v in notifs.values()):
                    break
                drained_count += sum(len(v) for v in notifs.values())

            if drained_count > 0:
                logger.info(
                    "[InMemorySpyreConnector] Drained %d pending notifications", drained_count
                )

            # Deregister any leftover memory registrations
            if self._nixl_register_descs is not None:
                try:
                    self._nixl_agent.deregister_memory(self._nixl_register_descs)
                    logger.info("[InMemorySpyreConnector] Deregistered previous memory")
                except Exception as exc:
                    logger.warning("[InMemorySpyreConnector] Error deregistering memory: %s", exc)
                self._nixl_register_descs = None

            logger.info("[InMemorySpyreConnector] NIXL transfer state reset complete")
        except Exception as exc:
            logger.warning("[InMemorySpyreConnector] Error resetting NIXL state: %s", exc)

    def _start_pull_request_handler(self) -> None:
        """Start background thread to handle PULL_REQUEST from decode (prefill side)"""
        if self._pull_request_handler_thread is not None:
            return  # Already started

        self._pull_request_handler_stop = False

        def pull_request_handler():
            """Background worker that handles PULL_REQUEST from decode"""
            logger.info("[InMemorySpyreConnector] Pull request handler thread started")

            cleanup_counter = 0
            while not self._pull_request_handler_stop:
                import time

                # Periodic cleanup of old pending transfers (every 1000 iterations = ~10 seconds)
                # TEMPORARILY DISABLED FOR TESTING - investigating cache hit rate degradation
                cleanup_counter += 1
                if cleanup_counter >= 1000:
                    # self._cleanup_old_pending_transfers()  # DISABLED
                    cleanup_counter = 0

                # Poll for PULL_REQUEST notifications from client
                try:
                    notifs = self._nixl_agent.get_new_notifs()
                    if not notifs or "client" not in notifs:
                        time.sleep(0.01)
                        continue

                    for notif in notifs["client"]:
                        notif_str = notif.decode("utf-8")

                        # Handle LIST_REQUEST - send available req_ids
                        if notif_str == "LIST_REQUEST":
                            with self._pending_transfers_lock:
                                available_ids = ",".join(self._pending_transfers.keys())
                            response = f"LIST:{available_ids}".encode()
                            self._nixl_agent.send_notif("client", response)
                            logger.info(
                                "[InMemorySpyreConnector] Sent LIST response: %d req_ids available",
                                len(self._pending_transfers),
                            )
                            continue

                        # Check if it's a PULL_REQUEST
                        if not notif_str.startswith("PULL:"):
                            continue

                        req_id = notif_str[5:]  # Extract req_id from "PULL:req_id"
                        logger.info(
                            "[InMemorySpyreConnector] Received PULL_REQUEST for req_id=%s", req_id
                        )

                        # Check if we have this transfer ready
                        with self._pending_transfers_lock:
                            if req_id not in self._pending_transfers:
                                logger.warning(
                                    "[InMemorySpyreConnector] PULL_REQUEST for unknown req_id=%s, sending RETRY",
                                    req_id,
                                )
                                self._nixl_agent.send_notif("client", b"RETRY")
                                continue

                            transfer_data = self._pending_transfers[req_id]

                        # Send metadata and descriptors
                        try:
                            logger.info(
                                "[InMemorySpyreConnector] Sending transfer data for req_id=%s",
                                req_id,
                            )
                            self._nixl_agent.send_notif("client", transfer_data["metadata"])
                            logger.info("[InMemorySpyreConnector] Sent metadata")
                            time.sleep(0.05)
                            self._nixl_agent.send_notif("client", transfer_data["descriptors"])
                            logger.info(
                                "[InMemorySpyreConnector] Sent descriptors for req_id=%s", req_id
                            )
                        except Exception as exc:
                            logger.error(
                                "[InMemorySpyreConnector] Error sending transfer data for req_id=%s: %s",
                                req_id,
                                exc,
                            )

                except Exception as exc:
                    logger.error("[InMemorySpyreConnector] Error in pull request handler: %s", exc)
                    time.sleep(0.1)

            logger.info("[InMemorySpyreConnector] Pull request handler thread stopped")

        import threading

        self._pull_request_handler_thread = threading.Thread(
            target=pull_request_handler, daemon=True
        )
        self._pull_request_handler_thread.start()
        logger.info("[InMemorySpyreConnector] Started pull request handler thread")

    def _stop_pull_request_handler(self) -> None:
        """Stop pull request handler thread"""
        if self._pull_request_handler_thread is None:
            return

        self._pull_request_handler_stop = True
        self._pull_request_handler_thread.join(timeout=5.0)
        self._pull_request_handler_thread = None
        logger.info("[InMemorySpyreConnector] Stopped pull request handler thread")

    def _cleanup_old_pending_transfers(self, max_age_seconds: float = 300.0) -> None:
        """Cleanup pending transfers older than max_age_seconds (default 5 minutes)"""
        if not self._pending_transfers:
            return

        import time

        current_time = time.time()
        to_remove = []

        with self._pending_transfers_lock:
            for req_id, transfer_data in self._pending_transfers.items():
                age = current_time - transfer_data["timestamp"]
                if age > max_age_seconds:
                    to_remove.append(req_id)

            for req_id in to_remove:
                transfer_data = self._pending_transfers[req_id]
                try:
                    self._nixl_agent.deregister_memory(transfer_data["register_descs"])
                    logger.info(
                        "[InMemorySpyreConnector] Cleaned up stale transfer: req_id=%s, age=%.1fs",
                        req_id,
                        current_time - transfer_data["timestamp"],
                    )
                except Exception as exc:
                    logger.warning(
                        "[InMemorySpyreConnector] Error deregistering stale transfer %s: %s",
                        req_id,
                        exc,
                    )
                del self._pending_transfers[req_id]

        if to_remove:
            logger.info("[InMemorySpyreConnector] Cleaned up %d stale transfers", len(to_remove))

    def _prune_saved_request(self, req_id: str, *, remove_store: bool = False) -> bool:
        if self._store.has_persistent_saved_requests():
            removed = self._store.remove_saved_request(req_id)
        else:
            removed = self._saved_requests.pop(req_id, None) is not None
        if remove_store:
            self._store.remove_by_req(req_id)
        return removed

    def _load_saved_requests(self) -> list[SavedRequestRecord]:
        if not self._store.has_persistent_saved_requests():
            # Check in-memory saved requests first
            if self._saved_requests:
                return list(self._saved_requests.values())

            # For consumer role, try NIXL transfer first, then file transfer
            if self._kv_role == "kv_consumer":
                if self._use_nixl:
                    try:
                        return self._load_saved_requests_nixl()
                    except Exception as exc:
                        logger.warning(
                            "[InMemorySpyreConnector] NIXL load failed, falling back to file: %s",
                            exc,
                        )

                if envs_spyre.VLLM_SPYRE_ENABLE_FILE_TRANSFER:
                    kv_cache_file = (
                        envs_spyre.VLLM_SPYRE_KV_CACHE_FILE_PATH
                        or f"/tmp/{os.getpid()}/kv_cache.pt"
                    )
                    if os.path.exists(kv_cache_file):
                        try:
                            kv_data = torch.load(kv_cache_file, weights_only=False)
                            if "prompt_token_ids" in kv_data and kv_data["prompt_token_ids"]:
                                saved = SavedRequestRecord(
                                    req_id=kv_data["req_id"],
                                    prompt_token_ids=tuple(kv_data["prompt_token_ids"]),
                                    block_ids=kv_data["block_ids"],
                                    num_tokens=kv_data["token_count"],
                                )

                                # CRITICAL FIX: Load KV cache blocks into store NOW
                                # This ensures available_prefix_blocks() returns correct count
                                from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
                                    StoreKey,
                                    KVKind,
                                )

                                layer_data = kv_data.get("layer_data", {})
                                blocks_loaded = 0

                                # Iterate over layer_data keys directly (layer_names may not be set yet)
                                for layer_idx in sorted(layer_data.keys()):
                                    layer_blocks = layer_data.get(layer_idx, [])
                                    # Each block is a tuple of (key_tensor, value_tensor)
                                    for block_idx, block_tuple in enumerate(layer_blocks):
                                        if isinstance(block_tuple, tuple) and len(block_tuple) == 2:
                                            # Get block_id from saved request
                                            block_id = (
                                                saved.block_ids[block_idx]
                                                if block_idx < len(saved.block_ids)
                                                else block_idx
                                            )

                                            # Store key tensor
                                            key_store_key = StoreKey(
                                                req_id=saved.req_id,
                                                layer_idx=layer_idx,
                                                block_id=block_id,
                                                kv_kind=KVKind.K,
                                            )
                                            self._store.put(key_store_key, block_tuple[0])
                                            blocks_loaded += 1

                                            # Store value tensor
                                            value_store_key = StoreKey(
                                                req_id=saved.req_id,
                                                layer_idx=layer_idx,
                                                block_id=block_id,
                                                kv_kind=KVKind.V,
                                            )
                                            self._store.put(value_store_key, block_tuple[1])
                                            blocks_loaded += 1

                                logger.info(
                                    "[InMemorySpyreConnector] Loaded saved request from file: "
                                    "req_id=%s, prompt_tokens=%d, block_ids=%s, blocks_loaded=%d",
                                    saved.req_id,
                                    len(saved.prompt_token_ids),
                                    saved.block_ids,
                                    blocks_loaded,
                                )
                                return [saved]
                        except Exception as exc:
                            logger.warning(
                                "[InMemorySpyreConnector] Failed to load saved request from file %s: %s",
                                kv_cache_file,
                                exc,
                            )

            return list(self._saved_requests.values())

        records = [
            SavedRequestRecord(
                req_id=str(record.req_id),
                prompt_token_ids=tuple(record.prompt_token_ids),
                block_ids=list(record.block_ids),
                num_tokens=int(record.num_tokens),
            )
            for record in self._store.get_saved_requests()
        ]
        self._saved_requests = OrderedDict((record.req_id, record) for record in records)
        return records

    def _nixl_receive_dtype(self) -> torch.dtype:
        """Dtype for NIXL receive buffers and descriptor-size math.

        Spyre paged pages are fp16, so descriptor byte counts must use the
        registered cache dtype — fp32 sizing halves the element count and
        NIXL rejects the transfer with NIXL_ERR_INVALID_PARAM.
        """
        if self._paged_accessor is not None:
            return self._paged_accessor.dtype
        if self._kv_caches:
            first_cache = next(iter(self._kv_caches.values()))
            dtype = getattr(first_cache, "dtype", None)
            if isinstance(dtype, torch.dtype):
                return dtype
        if self._dtype_str.startswith("torch."):
            dtype = getattr(torch, self._dtype_str.removeprefix("torch."), None)
            if isinstance(dtype, torch.dtype):
                return dtype
        return torch.float32

    def _load_saved_requests_nixl(self) -> list[SavedRequestRecord]:
        """Load saved requests via NIXL transfer (consumer/decode side)"""
        # Ensure NIXL agent is initialized (lazy init)
        if self._nixl_agent is None:
            self._init_nixl_agent()

        if self._nixl_agent is None:
            raise RuntimeError("NIXL agent initialization failed")

        # DON'T reset NIXL state on decode side - it drains notifications we need!

        logger.info("[InMemorySpyreConnector] Starting NIXL KV cache transfer (client)")

        # Step 1: Request list of available req_ids from prefill
        import time
        import json

        logger.info("[InMemorySpyreConnector] Sending LIST_REQUEST to prefill")
        self._nixl_agent.send_notif("server", b"LIST_REQUEST")

        # Step 2: Poll for LIST response
        max_wait_seconds = 60
        wait_count = 0
        available_req_ids = []

        while wait_count < max_wait_seconds * 100:
            notifs = self._nixl_agent.get_new_notifs()
            if len(notifs) > 0 and "server" in notifs and len(notifs["server"]) > 0:
                for notif in notifs["server"]:
                    try:
                        notif_str = notif.decode("utf-8")
                        if notif_str.startswith("LIST:"):
                            ids_str = notif_str[5:]  # Extract after "LIST:"
                            available_req_ids = [
                                rid.strip() for rid in ids_str.split(",") if rid.strip()
                            ]
                            logger.info(
                                "[InMemorySpyreConnector] Received LIST response: %d req_ids available",
                                len(available_req_ids),
                            )
                            break
                    except UnicodeDecodeError:
                        # Skip binary notifications (e.g., tensor descriptors from previous transfers)
                        logger.debug(
                            "[InMemorySpyreConnector] Skipping non-UTF-8 notification (likely binary data)"
                        )
                        continue
                if available_req_ids:
                    break
            time.sleep(0.01)
            wait_count += 1

        if not available_req_ids:
            raise RuntimeError(
                f"Timeout waiting for LIST response from prefill (waited {max_wait_seconds}s)"
            )

        # Step 3: Select LAST (most recent) req_id to avoid pulling stale cached data
        # CRITICAL FIX: Use [-1] to get newest request, not [0] which gets oldest
        req_id = available_req_ids[-1]
        logger.info(
            "[InMemorySpyreConnector] Selected req_id=%s from available list (total: %d)",
            req_id,
            len(available_req_ids),
        )

        # Step 4: Send PULL_REQUEST for selected req_id
        logger.info("[InMemorySpyreConnector] Sending PULL_REQUEST for req_id=%s", req_id)

        # First, receive metadata (req_id, prompt_tokens, block_ids)
        import time

        # Send PULL_REQUEST to prefill
        pull_request_msg = f"PULL:{req_id}".encode()
        self._nixl_agent.send_notif("server", pull_request_msg)
        logger.info("[InMemorySpyreConnector] Sent PULL_REQUEST")

        # Poll for response with exponential backoff
        max_wait_seconds = 60
        wait_count = 0
        backoff_delay = 0.01  # Start with 10ms
        max_backoff = 1.0  # Max 1 second
        notifs = {}
        retry_count = 0

        while wait_count < max_wait_seconds * 100:  # 100 iterations per second
            notifs = self._nixl_agent.get_new_notifs()

            if len(notifs) > 0 and "server" in notifs and len(notifs["server"]) > 0:
                # Check if it's a RETRY response
                first_notif = notifs["server"][0]
                if first_notif == b"RETRY":
                    retry_count += 1
                    logger.info(
                        "[InMemorySpyreConnector] Received RETRY, attempt %d, waiting %.2fs before retry",
                        retry_count,
                        backoff_delay,
                    )
                    time.sleep(backoff_delay)
                    # Exponential backoff
                    backoff_delay = min(backoff_delay * 2, max_backoff)
                    # Resend PULL_REQUEST
                    self._nixl_agent.send_notif("server", pull_request_msg)
                    wait_count += int(backoff_delay * 100)
                    continue
                else:
                    # Got actual data
                    logger.info(
                        "[InMemorySpyreConnector] Received transfer data after %d iterations, %d retries",
                        wait_count,
                        retry_count,
                    )
                    break

            time.sleep(0.01)
            wait_count += 1

        if len(notifs) == 0 or "server" not in notifs or len(notifs["server"]) == 0:
            raise RuntimeError(
                f"Timeout waiting for NIXL transfer data from server (waited {max_wait_seconds}s, retries={retry_count})"
            )

        # Check if we received multiple notifications at once
        server_notifs = notifs["server"]
        logger.info(
            "[InMemorySpyreConnector] Received %d notifications from server", len(server_notifs)
        )

        # First notification should be metadata
        metadata_bytes = server_notifs[0]
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        req_id = metadata["req_id"]
        prompt_token_ids = tuple(metadata["prompt_token_ids"])
        block_ids = metadata["block_ids"]
        num_tokens = metadata["num_tokens"]
        num_layers = metadata["num_layers"]

        logger.info(
            "[InMemorySpyreConnector] Received metadata: req_id=%s, prompt_tokens=%d, blocks=%d",
            req_id,
            len(prompt_token_ids),
            len(block_ids),
        )

        # Check if descriptor is already in the same batch
        remote_xfer_descs = None
        if len(server_notifs) > 1:
            logger.info("[InMemorySpyreConnector] Descriptor already received in same batch")
            try:
                remote_xfer_descs = self._nixl_agent.deserialize_descs(server_notifs[1])
            except Exception as e:
                logger.warning("[InMemorySpyreConnector] Failed to deserialize second notif: %s", e)

        # If not received yet, wait for it
        if remote_xfer_descs is None:
            logger.info("[InMemorySpyreConnector] Waiting for tensor descriptors...")
            max_wait = 30
            wait_count = 0

            while remote_xfer_descs is None and wait_count < max_wait * 100:
                notifs = self._nixl_agent.get_new_notifs()
                if len(notifs) > 0 and "server" in notifs:
                    try:
                        remote_xfer_descs = self._nixl_agent.deserialize_descs(notifs["server"][0])
                        logger.info("[InMemorySpyreConnector] Received tensor descriptors")
                        break
                    except Exception as e:
                        logger.warning("[InMemorySpyreConnector] Failed to deserialize: %s", e)
                time.sleep(0.01)
                wait_count += 1

            if remote_xfer_descs is None:
                raise RuntimeError("Timeout waiting for tensor descriptors from server")

        while not self._nixl_agent.check_remote_metadata("server"):
            time.sleep(0.01)

        # Reallocate buffers based on remote descriptors
        # CRITICAL: Must match the producer's registered block layout.
        desc_count = remote_xfer_descs.descCount()
        tensors = []
        recv_dtype = self._nixl_receive_dtype()
        recv_itemsize = recv_dtype.itemsize

        # Get KV cache shape - if not yet registered, infer from descriptor size
        if self._paged_accessor is not None:
            # The producer registers store-resident blocks in heap-block layout
            # (see _save_kv_bulk -> SpyrePagedKVCacheAccessor.read_block), so the
            # receive buffers must match: [block_size, num_kv_heads, head_dim].
            kv_block_shape = self._paged_accessor.block_shape
        elif self._kv_caches:
            first_cache = next(iter(self._kv_caches.values()))
            kv_block_shape = (
                first_cache.shape[2],
                self._block_size,
                first_cache.shape[4],
            )  # [num_kv_heads, block_size, head_dim]
        else:
            # KV caches not yet registered - infer shape from first descriptor
            # CRITICAL: KV cache format is [block_size, num_kv_heads, head_dim], NOT [num_kv_heads, block_size, head_dim]
            desc = remote_xfer_descs[0]
            desc_len_bytes = desc[1]
            num_elements = desc_len_bytes // recv_itemsize
            # num_elements = block_size * num_kv_heads * head_dim
            # For granite: 64 * 8 * 128 = 65536
            num_kv_heads = num_elements // (self._block_size * 128)
            head_dim = 128
            kv_block_shape = (self._block_size, num_kv_heads, head_dim)  # CORRECT ORDER
            logger.info(
                "[InMemorySpyreConnector] Inferred KV shape from descriptor: %s (num_elements=%d)",
                kv_block_shape,
                num_elements,
            )

        expected_elements = kv_block_shape[0] * kv_block_shape[1] * kv_block_shape[2]

        for i in range(desc_count):
            desc = remote_xfer_descs[i]
            desc_len_bytes = desc[1]
            num_elements = desc_len_bytes // recv_itemsize

            if num_elements != expected_elements:
                logger.warning(
                    "[InMemorySpyreConnector] Descriptor %d size mismatch: got %d elements, expected %d",
                    i,
                    num_elements,
                    expected_elements,
                )

            # Allocate with correct shape and the registered cache dtype so the
            # byte count matches the producer's registered pages (fp16 on Spyre).
            tensor = torch.zeros(kv_block_shape, dtype=recv_dtype, device="cpu")
            tensors.append(tensor)

        # Re-register with actual buffers
        descs = self._nixl_agent.get_reg_descs(tensors)
        register_descs = self._nixl_agent.register_memory(descs)
        local_xfer_descs = register_descs.trim()

        # Initialize and execute transfer with unique tag per request
        transfer_tag = f"KV_TRANSFER_{req_id}"
        transfer_handle = self._nixl_agent.initialize_xfer(
            "READ", local_xfer_descs, remote_xfer_descs, "server", transfer_tag
        )

        logger.info("[InMemorySpyreConnector] Starting NIXL transfer with tag=%s", transfer_tag)
        state = self._nixl_agent.transfer(transfer_handle)
        logger.info("[InMemorySpyreConnector] Initial transfer state: %s", state)
        if state == "ERR":
            raise RuntimeError("NIXL transfer initiation failed")

        check_count = 0
        while True:
            state = self._nixl_agent.check_xfer_state(transfer_handle)
            if check_count % 1000 == 0:  # Log every second
                logger.info(
                    "[InMemorySpyreConnector] Transfer check %d: state=%s", check_count, state
                )
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")
            if state == "DONE":
                logger.info(
                    "[InMemorySpyreConnector] Transfer completed after %d checks", check_count
                )
                break
            time.sleep(0.001)
            check_count += 1

        logger.info(
            "[InMemorySpyreConnector] NIXL transfer complete, received %d tensors", len(tensors)
        )

        # Log first received tensor for debugging
        if len(tensors) > 0:
            first_tensor = tensors[0]
            logger.info(
                "[InMemorySpyreConnector] First received tensor: shape=%s, dtype=%s, min=%.4f, max=%.4f, mean=%.4f",
                tuple(first_tensor.shape),
                first_tensor.dtype,
                first_tensor.min().item(),
                first_tensor.max().item(),
                first_tensor.mean().item(),
            )

        # Use metadata received from server
        tensor_idx = 0
        blocks_loaded = 0

        for layer_idx in range(num_layers):
            for block_id in block_ids:
                key_tensor = tensors[tensor_idx]
                value_tensor = tensors[tensor_idx + 1]
                tensor_idx += 2

                # Log first block for debugging
                if layer_idx == 0 and block_id == block_ids[0]:
                    logger.info(
                        "[InMemorySpyreConnector] Storing first block: layer=%d, block=%d, key_shape=%s, value_shape=%s",
                        layer_idx,
                        block_id,
                        tuple(key_tensor.shape),
                        tuple(value_tensor.shape),
                    )

                key_store_key = StoreKey(req_id, layer_idx, block_id, KVKind.K)
                value_store_key = StoreKey(req_id, layer_idx, block_id, KVKind.V)

                # Receive buffers are allocated in heap-block layout (the
                # producer's registered layout), so they are stored directly.
                self._store.put(key_store_key, key_tensor)
                self._store.put(value_store_key, value_tensor)

                # Verify what was stored by reading it back
                if layer_idx == 0 and block_id == block_ids[0]:
                    readback_key = self._store.get(key_store_key)
                    readback_value = self._store.get(value_store_key)
                    key_data = readback_key.data if hasattr(readback_key, "data") else readback_key
                    value_data = (
                        readback_value.data if hasattr(readback_value, "data") else readback_value
                    )
                    logger.info(
                        "[InMemorySpyreConnector] Readback verification: key_shape=%s, value_shape=%s, key_min=%.4f, key_max=%.4f",
                        tuple(key_data.shape),
                        tuple(value_data.shape),
                        key_data.min().item(),
                        key_data.max().item(),
                    )

                blocks_loaded += 2

        saved = SavedRequestRecord(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            block_ids=block_ids,
            num_tokens=num_tokens,
        )

        # CRITICAL FIX: Store in _saved_requests dict to accumulate across multiple loads
        # Without this, each load replaces previous entries, causing cache lookup failures
        self._saved_requests[req_id] = saved

        logger.info(
            "[InMemorySpyreConnector] NIXL load complete: req_id=%s, blocks_loaded=%d, total_saved=%d",
            req_id,
            blocks_loaded,
            len(self._saved_requests),
        )

        # Return full list of all saved requests, not just the newly loaded one
        return list(self._saved_requests.values())

    def _save_request_nixl(self, record: SavedRequestRecord) -> None:
        """Save request via NIXL transfer (producer/prefill side)"""
        # Ensure NIXL agent is initialized (lazy init)
        if self._nixl_agent is None:
            self._init_nixl_agent()

        if self._nixl_agent is None:
            raise RuntimeError("NIXL agent initialization failed")

        # Check if blocking mode is enabled
        blocking_mode = envs_spyre.VLLM_SPYRE_NIXL_BLOCKING_TRANSFER

        # Only reset state in blocking mode (non-blocking uses background thread for cleanup)
        if blocking_mode:
            self._reset_nixl_transfer_state()

        logger.info("[InMemorySpyreConnector] Starting NIXL KV cache transfer (server)")

        # Collect all KV tensors for this request from self._store
        # NOTE: Called from _save_kv_bulk AFTER blocks are stored
        tensors = []
        for layer_idx in range(self._num_layers):
            for block_id in record.block_ids:
                key_store_key = StoreKey(record.req_id, layer_idx, block_id, KVKind.K)
                value_store_key = StoreKey(record.req_id, layer_idx, block_id, KVKind.V)

                key_entry = self._store.get(key_store_key)
                value_entry = self._store.get(value_store_key)

                if key_entry is None or value_entry is None:
                    logger.warning(
                        "[InMemorySpyreConnector] Missing entry in store: layer=%d block=%d",
                        layer_idx,
                        block_id,
                    )
                    continue

                # Extract tensor from HostMemoryKVEntry
                key_tensor = key_entry.data if hasattr(key_entry, "data") else key_entry
                value_tensor = value_entry.data if hasattr(value_entry, "data") else value_entry

                if not key_tensor.is_contiguous():
                    key_tensor = key_tensor.contiguous()
                if not value_tensor.is_contiguous():
                    value_tensor = value_tensor.contiguous()

                # Log first tensor shape for debugging
                if layer_idx == 0 and block_id == record.block_ids[0]:
                    logger.info(
                        "[InMemorySpyreConnector] First tensor shapes: key=%s, value=%s, dtype=%s",
                        tuple(key_tensor.shape),
                        tuple(value_tensor.shape),
                        key_tensor.dtype,
                    )

                tensors.append(key_tensor)
                tensors.append(value_tensor)

        if not tensors:
            logger.warning("[InMemorySpyreConnector] No tensors to transfer via NIXL")
            return

        # Register memory
        descs = self._nixl_agent.get_reg_descs(tensors)
        register_descs = self._nixl_agent.register_memory(descs)

        import json
        import time

        # Prepare metadata (req_id, prompt_tokens, block_ids) and descriptors
        # up front; only blocking mode pushes them to a connected client here.
        metadata = {
            "req_id": record.req_id,
            "prompt_token_ids": list(record.prompt_token_ids),
            "block_ids": list(record.block_ids),
            "num_tokens": record.num_tokens,
            "num_layers": self._num_layers,
        }
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        local_xfer_descs = register_descs.trim()
        desc = self._nixl_agent.get_serialized_descs(local_xfer_descs)

        if blocking_mode:
            # BLOCKING MODE: wait for the client, push metadata + descriptors,
            # then wait for transfer completion (original behavior).
            while not self._nixl_agent.check_remote_metadata("client"):
                time.sleep(0.01)

            self._nixl_agent.send_notif("client", metadata_bytes)
            # Give the client a beat to consume metadata before descriptors.
            time.sleep(0.1)
            self._nixl_agent.send_notif("client", desc)

            logger.info("[InMemorySpyreConnector] NIXL transfer in BLOCKING mode")
            while not self._nixl_agent.check_remote_xfer_done("client", b"KV_TRANSFER"):
                time.sleep(0.001)

            # Cleanup immediately after transfer completes
            self._nixl_agent.deregister_memory(register_descs)
            self._nixl_register_descs = None

            logger.info(
                "[InMemorySpyreConnector] NIXL transfer complete (blocking): req_id=%s, tensors=%d",
                record.req_id,
                len(tensors),
            )
        else:
            # NON-BLOCKING MODE: Store in pending_transfers map, decode will pull when ready
            logger.info(
                "[InMemorySpyreConnector] NIXL transfer in NON-BLOCKING mode - storing in pending_transfers"
            )

            # Store transfer data in map for decode to pull
            with self._pending_transfers_lock:
                self._pending_transfers[record.req_id] = {
                    "metadata": metadata_bytes,
                    "descriptors": desc,
                    "register_descs": register_descs,
                    "timestamp": time.time(),
                }

            # Start pull request handler if not already running
            if self._pull_request_handler_thread is None:
                self._start_pull_request_handler()

            logger.info(
                "[InMemorySpyreConnector] NIXL transfer ready (non-blocking): req_id=%s, tensors=%d, stored in pending_transfers",
                record.req_id,
                len(tensors),
            )

    def _save_request_record(self, record: SavedRequestRecord) -> None:
        if self._store.has_persistent_saved_requests():
            self._store.save_request_record(record)
            self._load_saved_requests()
            return

        if record.req_id in self._saved_requests:
            self._saved_requests.move_to_end(record.req_id)
        self._saved_requests[record.req_id] = record

        if self._saved_requests_max_size > 0:
            while len(self._saved_requests) > self._saved_requests_max_size:
                oldest_req_id, _ = self._saved_requests.popitem(last=False)
                self._store.remove_by_req(oldest_req_id)
                self._stats.record("evictions")

    def _saved_request_count(self) -> int:
        if self._store.has_persistent_saved_requests():
            return self._store.saved_request_count()
        return len(self._saved_requests)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        if not kv_caches:
            return

        self._num_layers = len(kv_caches)
        self._layer_names = sorted(kv_caches.keys())

        # The Spyre attention backend registers SpyrePagedKVCache page lists
        # (a NamedTuple of k/v page lists), not monolithic staging tensors.
        self._paged_accessor = SpyrePagedKVCacheAccessor.try_from_kv_caches(kv_caches)
        if self._paged_accessor is not None:
            self._num_kv_heads = self._paged_accessor.num_kv_heads
            self._head_dim = self._paged_accessor.head_dim
            self._dtype_str = str(self._paged_accessor.dtype)
            if self._block_size != self._paged_accessor.block_size:
                logger.warning(
                    "[InMemorySpyreConnector] Config block_size=%d disagrees with "
                    "registered page block_size=%d; using page geometry",
                    self._block_size,
                    self._paged_accessor.block_size,
                )
                self._block_size = self._paged_accessor.block_size
            logger.info(
                "[InMemorySpyreConnector] Paged KV cache registered: %s",
                self._paged_accessor.describe(),
            )
            return

        first_tensor = next(iter(kv_caches.values()))
        self._dtype_str = str(first_tensor.dtype)
        if first_tensor.dim() >= 4:
            self._num_kv_heads = first_tensor.shape[-2]
            self._head_dim = first_tensor.shape[-1]

    @property
    def uses_heap_kv(self) -> bool:
        return self._use_heap_kv

    def heap_kv_active(self) -> bool:
        return self._ensure_heap_kv_client() is not None

    def paged_kv_active(self) -> bool:
        return self._paged_accessor is not None

    def get_paged_kv_status(self) -> dict[str, Any]:
        if self._paged_accessor is None:
            return {"active": False}
        return {"active": True, **self._paged_accessor.describe()}

    def get_heap_kv_status(self) -> dict[str, Any]:
        active = self._heap_kv_client is not None
        return {
            "requested": bool(self._use_heap_kv),
            "strict": bool(self._heap_kv_strict),
            "active": active,
            "mode": "in_process" if active else None,
            "init_error": self._heap_kv_init_error,
        }

    def _ensure_heap_kv_client(self) -> InProcessHeapKVClient | None:
        if not self._use_heap_kv:
            return None
        if self._paged_accessor is not None:
            # The heap accessor addresses compiled-graph HBM offsets; with
            # page-list caches the pages are directly accessible tensors and
            # the heap path is the fallback for older/unknown layouts only.
            logger.warning_once(
                "[InMemorySpyreConnector] Heap KV requested but paged KV cache "
                "is registered; using paged accessor"
            )
            return None
        if self._heap_kv_client is not None:
            return self._heap_kv_client
        try:
            if not self._kv_caches:
                raise RuntimeError("KV cache geometry is unavailable before register_kv_caches")
            first_tensor = next(iter(self._kv_caches.values()))
            perfdsc_dir, export_dir = resolve_heap_kv_paths(
                perfdsc_dir=envs_spyre.VLLM_SPYRE_HEAP_KV_PERFDSC_DIR or None,
                export_dir=envs_spyre.VLLM_SPYRE_HEAP_KV_EXPORT_DIR or None,
            )
            client = InProcessHeapKVClient(
                kv_heads=self._num_kv_heads,
                block_size=self._block_size,
                head_dim=self._head_dim,
                dtype=first_tensor.dtype,
                perfdsc_dir=perfdsc_dir,
                export_dir=export_dir,
            )
            probe = client.probe()
            self._heap_kv_client = client
            logger.info(
                "[InMemorySpyreConnector] Experimental in-process heap KV enabled "
                "perfdsc_dir=%s export_dir=%s probe=%s",
                perfdsc_dir,
                export_dir,
                probe,
            )
            return self._heap_kv_client
        except Exception as exc:
            self._heap_kv_init_error = f"{type(exc).__name__}: {exc}"
            if self._heap_kv_strict:
                raise RuntimeError(
                    "Experimental in-process heap KV strict mode is enabled and "
                    f"initialization failed: {self._heap_kv_init_error}"
                ) from exc
            logger.warning(
                "[InMemorySpyreConnector] Experimental in-process heap KV unavailable, "
                "falling back to staging path: %s",
                self._heap_kv_init_error,
            )
            return None

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if isinstance(connector_metadata, SpyreConnectorMeta):
            connector_metadata.validate()
        self._step_stores.clear()
        self._step_loads.clear()
        self._load_error_block_ids.clear()
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        # Extract dynamic remote connection info from metadata (for llm-d routing proxy)
        # Falls back to environment variable for manual deployments
        remote_ip = None
        remote_port = None
        for req_meta in meta.requests:
            if not req_meta.is_store and req_meta.remote:
                remote_ip = req_meta.remote.host
                remote_port = req_meta.remote.port
                logger.info(
                    "[InMemorySpyreConnector] Using dynamic remote connection: host=%s, port=%d (from routing proxy)",
                    remote_ip,
                    remote_port,
                )
                break

        # Update NIXL remote IP if provided dynamically
        if remote_ip:
            self._nixl_remote_ip = remote_ip
            logger.info(
                "[InMemorySpyreConnector] Updated NIXL remote IP to %s (dynamic from metadata)",
                self._nixl_remote_ip,
            )
        else:
            # Use environment variable (manual deployment)
            logger.info(
                "[InMemorySpyreConnector] Using NIXL remote IP from environment: %s",
                self._nixl_remote_ip,
            )

        total_load = 0
        total_miss = 0
        self._load_error_block_ids.clear()

        has_load_requests = any(not req_meta.is_store for req_meta in meta.requests)
        heap_client = self._ensure_heap_kv_client() if has_load_requests else None
        if has_load_requests and self._paged_accessor is not None:
            total_load, total_miss = self._load_via_paged_accessor(meta, self._paged_accessor)
        elif heap_client is not None:
            total_load, total_miss = self._load_via_heap_helper(meta, heap_client)
        else:
            for layer_idx, layer_name in enumerate(self._layer_names):
                layer_load, layer_miss = self._load_layer(
                    meta,
                    layer_idx,
                    layer_name,
                )
                total_load += layer_load
                total_miss += layer_miss

        self._blocks_loaded += total_load
        self._blocks_missing += total_miss
        self._stats.record("loaded_blocks", total_load)
        self._stats.record("load_misses", total_miss)

        for req_meta in meta.requests:
            if not req_meta.is_store:
                self._step_loads.add(req_meta.req_id)

    def _load_layer(
        self,
        meta: SpyreConnectorMeta,
        layer_idx: int,
        layer_name: str,
    ) -> tuple[int, int]:
        staging = self._kv_caches.get(layer_name)
        if staging is None:
            return 0, 0

        # Step 2: Try loading from file first if file transfer is enabled
        if envs_spyre.VLLM_SPYRE_ENABLE_FILE_TRANSFER:
            kv_cache_file = (
                envs_spyre.VLLM_SPYRE_KV_CACHE_FILE_PATH or f"/tmp/{os.getpid()}/kv_cache.pt"
            )
            if os.path.exists(kv_cache_file):
                try:
                    kv_data = torch.load(kv_cache_file, weights_only=False)
                    layer_data = kv_data.get("layer_data", {}).get(layer_idx, [])

                    if layer_data:
                        load_count = 0
                        saved_req_id = kv_data.get("req_id")
                        saved_block_ids = kv_data.get("block_ids", [])

                        logger.info(
                            "[InMemorySpyreConnector] Layer %d: Attempting to load from file. "
                            "Saved req_id=%s, saved_blocks=%s, num_requests=%d",
                            layer_idx,
                            saved_req_id,
                            saved_block_ids,
                            len(meta.requests),
                        )

                        for req_meta in meta.requests:
                            logger.info(
                                "[InMemorySpyreConnector] Layer %d: Processing req_meta: "
                                "req_id=%s, is_store=%s, source_req_id=%s, block_ids=%s, block_mapping=%s",
                                layer_idx,
                                req_meta.req_id,
                                req_meta.is_store,
                                req_meta.source_req_id,
                                req_meta.block_ids,
                                req_meta.block_mapping,
                            )

                            if req_meta.is_store:
                                logger.info(
                                    "[InMemorySpyreConnector] Layer %d: Skipping req_id=%s (is_store=True)",
                                    layer_idx,
                                    req_meta.req_id,
                                )
                                continue

                            # Get the source request ID to match saved data
                            source_req = req_meta.source_req_id or req_meta.req_id

                            # Check if this matches the saved request
                            if source_req != saved_req_id:
                                logger.info(
                                    "[InMemorySpyreConnector] Layer %d: Skipping req_id=%s "
                                    "(source_req=%s doesn't match saved_req_id=%s)",
                                    layer_idx,
                                    req_meta.req_id,
                                    source_req,
                                    saved_req_id,
                                )
                                continue

                            mapping = (
                                list(req_meta.block_mapping)
                                if req_meta.block_mapping
                                else [(block_id, block_id) for block_id in req_meta.block_ids]
                            )

                            logger.info(
                                "[InMemorySpyreConnector] Layer %d: Loading blocks for req_id=%s, mapping=%s",
                                layer_idx,
                                req_meta.req_id,
                                mapping,
                            )

                            # Load blocks: layer_data is indexed by position in saved block list
                            for src_block_id, dest_bid in mapping:
                                if dest_bid < 0 or dest_bid >= staging.shape[1]:
                                    logger.warning(
                                        "[InMemorySpyreConnector] Layer %d: Invalid dest_bid=%d (staging.shape[1]=%d)",
                                        layer_idx,
                                        dest_bid,
                                        staging.shape[1],
                                    )
                                    continue

                                # Find the position of src_block_id in saved_block_ids
                                try:
                                    block_idx = saved_block_ids.index(src_block_id)
                                    if block_idx < len(layer_data):
                                        k_block, v_block = layer_data[block_idx]
                                        staging[0][dest_bid].copy_(k_block.to(staging.device))
                                        staging[1][dest_bid].copy_(v_block.to(staging.device))
                                        load_count += 2  # K and V
                                        logger.info(
                                            "[InMemorySpyreConnector] Layer %d: Loaded block src=%d->dest=%d (idx=%d)",
                                            layer_idx,
                                            src_block_id,
                                            dest_bid,
                                            block_idx,
                                        )
                                    else:
                                        logger.warning(
                                            "[InMemorySpyreConnector] Layer %d: block_idx=%d >= len(layer_data)=%d",
                                            layer_idx,
                                            block_idx,
                                            len(layer_data),
                                        )
                                except ValueError:
                                    logger.warning(
                                        "[InMemorySpyreConnector] Layer %d: src_block_id=%d not in saved_block_ids=%s",
                                        layer_idx,
                                        src_block_id,
                                        saved_block_ids,
                                    )
                                    continue

                        logger.info(
                            "[InMemorySpyreConnector] Loaded layer %d from file %s: %d blocks",
                            layer_idx,
                            kv_cache_file,
                            load_count // 2,
                        )
                        return load_count, 0
                except Exception as exc:
                    logger.warning(
                        "[InMemorySpyreConnector] Failed to load from file %s: %s",
                        kv_cache_file,
                        exc,
                    )

        # Fall back to existing store-based loading
        load_count = 0
        miss_count = 0

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id
            mapping = (
                list(req_meta.block_mapping)
                if req_meta.block_mapping
                else [(block_id, block_id) for block_id in req_meta.block_ids]
            )

            for src_block_id, dest_bid in mapping:
                if dest_bid < 0 or dest_bid >= staging.shape[1]:
                    miss_count += 1
                    self._load_error_block_ids.add(dest_bid)
                    continue

                for kv_kind, kv_dim in ((KVKind.K, 0), (KVKind.V, 1)):
                    store_key = StoreKey(
                        req_id=source_req,
                        layer_idx=layer_idx,
                        block_id=src_block_id,
                        kv_kind=kv_kind,
                    )

                    # Log first load attempt for debugging
                    if layer_idx == 0 and src_block_id == mapping[0][0] and kv_kind == KVKind.K:
                        target_tensor = staging[kv_dim][dest_bid]
                        logger.info(
                            "[InMemorySpyreConnector] Loading into staging: layer=%d, src_block=%d, dest_block=%d, target_shape=%s, target_device=%s",
                            layer_idx,
                            src_block_id,
                            dest_bid,
                            tuple(target_tensor.shape),
                            target_tensor.device,
                        )
                        # Check what's in store before load
                        stored_entry = self._store.get(store_key)
                        if stored_entry:
                            stored_data = (
                                stored_entry.data if hasattr(stored_entry, "data") else stored_entry
                            )
                            logger.info(
                                "[InMemorySpyreConnector] Store contains: shape=%s, device=%s, min=%.4f, max=%.4f",
                                tuple(stored_data.shape),
                                stored_data.device,
                                stored_data.min().item(),
                                stored_data.max().item(),
                            )

                    if self._store.load_into(store_key, staging[kv_dim][dest_bid]):
                        load_count += 1

                        # Verify loaded data
                        if layer_idx == 0 and src_block_id == mapping[0][0] and kv_kind == KVKind.K:
                            loaded_tensor = staging[kv_dim][dest_bid]
                            logger.info(
                                "[InMemorySpyreConnector] After load_into: shape=%s, min=%.4f, max=%.4f, mean=%.4f",
                                tuple(loaded_tensor.shape),
                                loaded_tensor.min().item(),
                                loaded_tensor.max().item(),
                                loaded_tensor.mean().item(),
                            )
                    else:
                        miss_count += 1
                        self._load_error_block_ids.add(dest_bid)

        return load_count, miss_count

    def _load_via_heap_helper(
        self,
        meta: SpyreConnectorMeta,
        heap_client: InProcessHeapKVClient,
    ) -> tuple[int, int]:
        block_values: dict[tuple[int, str, int], torch.Tensor] = {}
        load_count = 0
        miss_count = 0
        dtype = next(iter(self._kv_caches.values())).dtype

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id
            mapping = (
                list(req_meta.block_mapping)
                if req_meta.block_mapping
                else [(block_id, block_id) for block_id in req_meta.block_ids]
            )

            for src_block_id, dest_bid in mapping:
                for layer_idx in range(self._num_layers):
                    for kv_kind in (KVKind.K, KVKind.V):
                        store_key = StoreKey(
                            req_id=source_req,
                            layer_idx=layer_idx,
                            block_id=src_block_id,
                            kv_kind=kv_kind,
                        )
                        cpu_block = torch.empty(
                            (self._block_size, self._num_kv_heads, self._head_dim),
                            dtype=dtype,
                            device="cpu",
                        )
                        if self._store.load_into(store_key, cpu_block):
                            block_values[(layer_idx, kv_kind.value.lower(), dest_bid)] = cpu_block
                            load_count += 1
                        else:
                            miss_count += 1
                            self._load_error_block_ids.add(dest_bid)

        if block_values:
            heap_client.write_blocks(block_values)
        return load_count, miss_count

    def _load_via_paged_accessor(
        self,
        meta: SpyreConnectorMeta,
        accessor: SpyrePagedKVCacheAccessor,
    ) -> tuple[int, int]:
        """Load stored blocks directly into the registered K/V page tensors."""
        load_count = 0
        miss_count = 0
        cpu_block = torch.empty(accessor.block_shape, dtype=accessor.dtype, device="cpu")

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id
            mapping = (
                list(req_meta.block_mapping)
                if req_meta.block_mapping
                else [(block_id, block_id) for block_id in req_meta.block_ids]
            )

            for src_block_id, dest_bid in mapping:
                if not 0 <= dest_bid < accessor.num_pages:
                    miss_count += 1
                    self._load_error_block_ids.add(dest_bid)
                    continue
                for layer_idx, layer_name in enumerate(accessor.layer_names):
                    for kv_kind in (KVKind.K, KVKind.V):
                        store_key = StoreKey(
                            req_id=source_req,
                            layer_idx=layer_idx,
                            block_id=src_block_id,
                            kv_kind=kv_kind,
                        )
                        if self._store.load_into(store_key, cpu_block):
                            accessor.write_block(
                                layer_name=layer_name,
                                kv_kind=kv_kind.value.lower(),
                                page_id=dest_bid,
                                values=cpu_block,
                            )
                            load_count += 1
                        else:
                            miss_count += 1
                            self._load_error_block_ids.add(dest_bid)
        return load_count, miss_count

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def _save_kv_bulk(self) -> None:
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        save_count = 0
        heap_client = self._ensure_heap_kv_client()
        for req_meta in meta.requests:
            if not req_meta.is_store:
                continue

            logger.info(
                "[InMemorySpyreConnector] save_kv_bulk req=%s token_count=%d "
                "block_ids=%s store_stats_before=%s",
                req_meta.req_id,
                req_meta.token_count,
                req_meta.block_ids,
                self._store.stats(),
            )

            heap_blocks: dict[tuple[int, str, int], torch.Tensor] = {}
            if heap_client is not None:
                block_refs = [
                    (layer_idx, kv_kind.value.lower(), block_id)
                    for layer_idx in range(self._num_layers)
                    for block_id in req_meta.block_ids
                    for kv_kind in (KVKind.K, KVKind.V)
                ]
                if block_refs:
                    heap_blocks = heap_client.read_blocks(block_refs)

            paged = self._paged_accessor
            for layer_idx, layer_name in enumerate(self._layer_names):
                staging = self._kv_caches.get(layer_name)
                if staging is None and not heap_blocks:
                    continue

                for block_id in req_meta.block_ids:
                    if paged is not None and not 0 <= block_id < paged.num_pages:
                        logger.warning(
                            "[InMemorySpyreConnector] save_kv_bulk skipping out-of-range "
                            "block %d (num_pages=%d)",
                            block_id,
                            paged.num_pages,
                        )
                        continue
                    for kv_kind, kv_dim in ((KVKind.K, 0), (KVKind.V, 1)):
                        store_key = StoreKey(
                            req_id=req_meta.req_id,
                            layer_idx=layer_idx,
                            block_id=block_id,
                            kv_kind=kv_kind,
                        )
                        if heap_blocks:
                            cpu_block = heap_blocks[(layer_idx, kv_kind.value.lower(), block_id)]
                        elif paged is not None:
                            cpu_block = paged.read_block(
                                layer_name=layer_name,
                                kv_kind=kv_kind.value.lower(),
                                page_id=block_id,
                            )
                        else:
                            assert staging is not None
                            cpu_block = staging[kv_dim][block_id]
                        self._store.put(
                            store_key,
                            cpu_block,
                            source_req=req_meta.req_id,
                        )
                        save_count += 1

            logger.info(
                "[InMemorySpyreConnector] save_kv_bulk req=%s store_stats_after=%s",
                req_meta.req_id,
                self._store.stats(),
            )
            self._step_stores.add(req_meta.req_id)

            # Step 2: Save to file if file transfer is enabled
            if envs_spyre.VLLM_SPYRE_ENABLE_FILE_TRANSFER:
                kv_cache_file = (
                    envs_spyre.VLLM_SPYRE_KV_CACHE_FILE_PATH or f"/tmp/{os.getpid()}/kv_cache.pt"
                )
                try:
                    # Get prompt tokens from req_meta (passed from scheduler instance)
                    prompt_tokens = getattr(req_meta, "prompt_token_ids", [])

                    logger.info(
                        "[InMemorySpyreConnector] Saving to file: req_id=%s, prompt_tokens_count=%d, "
                        "has_prompt_token_ids_attr=%s",
                        req_meta.req_id,
                        len(prompt_tokens),
                        hasattr(req_meta, "prompt_token_ids"),
                    )

                    kv_data = {
                        "req_id": req_meta.req_id,
                        "token_count": req_meta.token_count,
                        "block_ids": list(req_meta.block_ids),
                        "prompt_token_ids": prompt_tokens,
                        "layer_data": {},
                    }

                    for layer_idx, layer_name in enumerate(self._layer_names):
                        staging = self._kv_caches.get(layer_name)
                        if staging is None:
                            continue

                        num_blocks = paged.num_pages if paged is not None else staging.shape[1]
                        layer_blocks = []
                        for block_id in req_meta.block_ids:
                            if block_id < num_blocks:
                                k_block = staging[0][block_id].cpu().clone()
                                v_block = staging[1][block_id].cpu().clone()
                                layer_blocks.append((k_block, v_block))

                        kv_data["layer_data"][layer_idx] = layer_blocks

                    torch.save(kv_data, kv_cache_file)
                    logger.info(
                        "[InMemorySpyreConnector] Saved KV cache to file %s: req=%s, %d layers, %d blocks, %d prompt_tokens",
                        kv_cache_file,
                        req_meta.req_id,
                        len(kv_data["layer_data"]),
                        len(req_meta.block_ids),
                        len(prompt_tokens),
                    )

                    # Don't clean up yet - might be called multiple times for chunked prefill
                    # self._pending_prompt_tokens.pop(req_meta.req_id, None)
                except Exception as exc:
                    logger.error(
                        "[InMemorySpyreConnector] Failed to save KV cache to file %s: %s",
                        kv_cache_file,
                        exc,
                    )

            # Step 3: Transfer via NIXL if enabled (after blocks are stored)
            if self._use_nixl and self._kv_role == "kv_producer":
                try:
                    # Create SavedRequestRecord for NIXL transfer
                    prompt_tokens = getattr(req_meta, "prompt_token_ids", [])
                    saved_record = SavedRequestRecord(
                        req_id=req_meta.req_id,
                        prompt_token_ids=tuple(prompt_tokens) if prompt_tokens else tuple(),
                        block_ids=list(req_meta.block_ids),
                        num_tokens=req_meta.token_count,
                    )
                    self._save_request_nixl(saved_record)
                    logger.info(
                        "[InMemorySpyreConnector] NIXL transfer completed for req=%s",
                        req_meta.req_id,
                    )
                except Exception as exc:
                    logger.error(
                        "[InMemorySpyreConnector] NIXL transfer failed for req=%s: %s",
                        req_meta.req_id,
                        exc,
                    )

        self._blocks_saved += save_count
        self._stats.record("saved_blocks", save_count)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def wait_for_save(self) -> None:
        self._save_kv_bulk()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        finished_sending = self._step_stores & finished_req_ids
        finished_recving = self._step_loads & finished_req_ids
        return (
            finished_sending if finished_sending else None,
            finished_recving if finished_recving else None,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self._load_error_block_ids)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        prompt = request.prompt_token_ids
        saved_requests = self._load_saved_requests()

        logger.info(
            "[InMemorySpyreConnector] get_num_new_matched_tokens: req_id=%s, "
            "prompt_len=%d, num_computed_tokens=%d, saved_requests_count=%d",
            request.request_id,
            len(prompt) if prompt else 0,
            num_computed_tokens,
            len(saved_requests),
        )

        if not prompt or not saved_requests:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            logger.info(
                "[InMemorySpyreConnector] get_num_new_matched_tokens: No match - "
                "prompt=%s, saved_requests=%s",
                bool(prompt),
                bool(saved_requests),
            )
            return 0, False

        prompt_tuple = tuple(prompt)
        best_match: SavedRequestRecord | None = None
        best_tokens_total = 0
        stale_request_ids: list[str] = []

        for saved in saved_requests:
            saved_len = len(saved.prompt_token_ids)
            logger.info(
                "[InMemorySpyreConnector] Checking saved req_id=%s, "
                "saved_prompt_len=%d, saved_block_ids=%s",
                saved.req_id,
                saved_len,
                saved.block_ids,
            )

            if saved_len == 0:
                logger.info(
                    "[InMemorySpyreConnector] Skipping saved req_id=%s (empty prompt)",
                    saved.req_id,
                )
                continue

            available_blocks = self._store.available_prefix_blocks(
                saved.req_id,
                saved.block_ids,
            )
            logger.info(
                "[InMemorySpyreConnector] Saved req_id=%s: available_blocks=%d, required_blocks=%d",
                saved.req_id,
                available_blocks,
                len(saved.block_ids),
            )

            if available_blocks < len(saved.block_ids):
                logger.warning(
                    "[InMemorySpyreConnector] Marking req_id=%s as stale "
                    "(available_blocks=%d < required=%d)",
                    saved.req_id,
                    available_blocks,
                    len(saved.block_ids),
                )
                stale_request_ids.append(saved.req_id)
                continue

            common_len = 0
            for prompt_token, saved_token in zip(prompt_tuple, saved.prompt_token_ids):
                if prompt_token != saved_token:
                    break
                common_len += 1

            # Allow partial block matching for prompts shorter than one block
            if common_len >= self._block_size:
                # Full blocks only for longer prompts
                aligned = (common_len // self._block_size) * self._block_size
            elif common_len == len(saved.prompt_token_ids) and common_len == len(prompt_tuple):
                # Allow full match for short prompts (< block_size)
                aligned = common_len
            else:
                # Partial match within a block - align to block boundary
                aligned = (common_len // self._block_size) * self._block_size

            logger.info(
                "[InMemorySpyreConnector] Saved req_id=%s: common_len=%d, aligned=%d, "
                "saved_len=%d, prompt_len=%d, block_size=%d",
                saved.req_id,
                common_len,
                aligned,
                len(saved.prompt_token_ids),
                len(prompt_tuple),
                self._block_size,
            )

            if aligned > best_tokens_total:
                best_tokens_total = aligned
                best_match = saved
                logger.info(
                    "[InMemorySpyreConnector] New best match: req_id=%s, tokens=%d, "
                    "common_len=%d, saved_prompt_first_10=%s, request_prompt_first_10=%s",
                    saved.req_id,
                    aligned,
                    common_len,
                    list(saved.prompt_token_ids[:10]),
                    list(prompt_tuple[:10]),
                )

        for req_id in stale_request_ids:
            logger.info(
                "[InMemorySpyreConnector] Pruning stale request: %s",
                req_id,
            )
            self._prune_saved_request(req_id, remove_store=True)

        if best_match is None or best_tokens_total == 0:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            logger.info(
                "[InMemorySpyreConnector] No valid match found: best_match=%s, "
                "best_tokens_total=%d",
                best_match.req_id if best_match else None,
                best_tokens_total,
            )
            return 0, False

        # CRITICAL FIX: Prefill should NEVER load external tokens
        # Only decode (consumer) should load from external cache
        if self._kv_role == "kv_producer":
            logger.info(
                "[InMemorySpyreConnector] Prefill side: returning 0 external tokens (req_id=%s)",
                request.request_id,
            )
            return 0, False

        # Per-request tracking: Use tracked value instead of scheduler's accumulated value
        # This fixes the bug where num_computed_tokens incorrectly accumulates across requests
        if request.request_id not in self._request_computed_tokens:
            # New request - initialize to 0
            self._request_computed_tokens[request.request_id] = 0
            logger.info(
                "[InMemorySpyreConnector] New request detected (req_id=%s), initializing num_computed_tokens=0 (scheduler passed=%d)",
                request.request_id,
                num_computed_tokens,
            )

            # CRITICAL FIX: If scheduler passed non-zero num_computed_tokens for a NEW request,
            # it's using stale data from a previous request. Return 0 to avoid assertion failure.
            if num_computed_tokens > 0:
                logger.warning(
                    "[InMemorySpyreConnector] Scheduler passed num_computed_tokens=%d for NEW request %s, "
                    "returning 0 external tokens to avoid assertion failure (scheduler will compute locally)",
                    num_computed_tokens,
                    request.request_id,
                )
                self._pending_load_sources.pop(request.request_id, None)
                self._stats.record("match_attempts")
                return 0, False

            num_computed_tokens = 0
        else:
            # Existing request - use our tracked value, ignore scheduler's value
            tracked_value = self._request_computed_tokens[request.request_id]
            logger.info(
                "[InMemorySpyreConnector] Existing request (req_id=%s), using tracked num_computed_tokens=%d (scheduler passed=%d)",
                request.request_id,
                tracked_value,
                num_computed_tokens,
            )
            num_computed_tokens = tracked_value

        num_local = max(0, num_computed_tokens)

        # Calculate external tokens based on matched tokens and what's already computed
        # Only load tokens that are:
        # 1. Part of the match (up to best_tokens_total)
        # 2. Not already computed locally (subtract num_local)
        # 3. Block-aligned (must be multiple of block_size)

        logger.info(
            "[InMemorySpyreConnector] Calculating num_external: num_local=%d, "
            "best_tokens_total=%d, block_size=%d",
            num_local,
            best_tokens_total,
            self._block_size,
        )

        if num_local == 0:
            # First call: no tokens computed yet
            # CRITICAL: Must satisfy TWO constraints:
            # 1. num_external must be block-aligned (divisible by block_size)
            # 2. num_external < prompt_len (leave at least 1 token for computation)

            prompt_len = len(prompt_tuple)

            # For prompts < block_size: no external cache benefit
            if prompt_len < self._block_size:
                num_external = 0
                logger.info(
                    "[InMemorySpyreConnector] First call (num_local=0): num_external=0 "
                    "(prompt_len=%d < block_size=%d, no cache benefit)",
                    prompt_len,
                    self._block_size,
                )
            else:
                # For prompts >= block_size: load all complete blocks that fit
                # CRITICAL: Must leave at least 1 token for local computation
                # The scheduler requires num_new_tokens = prompt_len - num_external - num_local > 0

                # Calculate how many complete blocks we can load
                # We must ensure: prompt_len - (blocks * block_size) - num_local > 0
                # Since num_local = 0 on first call: prompt_len - (blocks * block_size) > 0
                # Therefore: blocks * block_size < prompt_len
                # Maximum blocks: (prompt_len - 1) // block_size

                max_blocks_to_load = (prompt_len - 1) // self._block_size

                # Limit by available blocks
                available_blocks = len(best_match.block_ids)
                blocks_to_load = min(max_blocks_to_load, available_blocks)

                num_external = blocks_to_load * self._block_size

                logger.info(
                    "[InMemorySpyreConnector] First call (num_local=0): num_external=%d "
                    "(matched=%d, block_size=%d, prompt_len=%d, max_blocks=%d, available_blocks=%d, loading_blocks=%d)",
                    num_external,
                    best_tokens_total,
                    self._block_size,
                    prompt_len,
                    max_blocks_to_load,
                    available_blocks,
                    blocks_to_load,
                )
        else:
            # Subsequent call: some tokens already computed
            # This happens with chunked prefill
            # Don't load anything - already processed
            num_external = 0
            logger.warning(
                "[InMemorySpyreConnector] Subsequent call (num_local=%d > 0): "
                "Setting num_external=0 (assuming chunked prefill already processed)",
                num_local,
            )

        logger.info(
            "[InMemorySpyreConnector] External token calculation: "
            "best_tokens_total=%d, num_local=%d, num_external=%d, "
            "prompt_len=%d, block_size=%d",
            best_tokens_total,
            num_local,
            num_external,
            len(prompt_tuple),
            self._block_size,
        )

        if num_external == 0:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            return 0, False

        self._pending_load_sources[request.request_id] = _PendingLoadSource(
            source=best_match,
            matched_tokens_total=best_tokens_total,
            num_local_computed_tokens=num_local,
        )
        _emit_connector_probe(
            "match_selected",
            req_id=request.request_id,
            source_req_id=best_match.req_id,
            prompt_token_count=len(prompt_tuple),
            matched_tokens_total=best_tokens_total,
            num_local_computed_tokens=num_local,
            num_external_tokens=num_external,
            source_block_ids=list(best_match.block_ids),
            source_num_tokens=best_match.num_tokens,
        )
        self._stats.record("match_attempts")
        self._stats.record("matched_tokens", num_external)

        # CRITICAL FIX: Update tracked computed tokens to include external tokens we're loading
        # This prevents the scheduler from thinking we need to compute these tokens again
        self._request_computed_tokens[request.request_id] = num_local + num_external
        logger.info(
            "[InMemorySpyreConnector] Updated tracked computed tokens for req_id=%s: %d (was %d, added %d external)",
            request.request_id,
            num_local + num_external,
            num_local,
            num_external,
        )

        return num_external, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if blocks is None:
            block_id_lists = ()
        elif hasattr(blocks, "get_block_ids"):
            block_id_lists = blocks.get_block_ids()
        else:
            block_id_lists = tuple(list(group) for group in blocks)
        if block_id_lists:
            assert len(block_id_lists) == 1, (
                "InMemorySpyreConnector assumes a single KV cache group"
            )

        flat_block_ids: list[int] = []
        for group in block_id_lists:
            flat_block_ids.extend(group)

        # Step 2: Capture prompt tokens early for file transfer
        logger.info(
            "[InMemorySpyreConnector] update_state_after_alloc: req_id=%s, "
            "has_prompt_token_ids=%s, prompt_len=%d, num_external_tokens=%d",
            request.request_id,
            hasattr(request, "prompt_token_ids") and request.prompt_token_ids is not None,
            len(request.prompt_token_ids)
            if hasattr(request, "prompt_token_ids") and request.prompt_token_ids
            else 0,
            num_external_tokens,
        )
        if hasattr(request, "prompt_token_ids") and request.prompt_token_ids:
            self._pending_prompt_tokens[request.request_id] = list(request.prompt_token_ids)
            logger.info(
                "[InMemorySpyreConnector] Captured %d prompt tokens for req_id=%s",
                len(request.prompt_token_ids),
                request.request_id,
            )

        # CRITICAL: Track processed requests to prevent duplicate calls from overwriting load metadata
        # Use a separate tracking dict since _pending_requests is appended at the end
        if not hasattr(self, "_processed_alloc_requests"):
            self._processed_alloc_requests = {}

        if request.request_id in self._processed_alloc_requests:
            prev_num_external = self._processed_alloc_requests[request.request_id]
            logger.info(
                "[InMemorySpyreConnector] update_state_after_alloc: Already processed req_id=%s "
                "(prev_num_external=%d, current_num_external=%d), skipping duplicate call",
                request.request_id,
                prev_num_external,
                num_external_tokens,
            )
            return

        # Mark as processed with the num_external_tokens value
        self._processed_alloc_requests[request.request_id] = num_external_tokens

        if num_external_tokens > 0:
            pending = self._pending_load_sources.get(request.request_id)
            logger.info(
                "[InMemorySpyreConnector] update_state_after_alloc: num_external_tokens=%d, "
                "has_pending_source=%s, pending_source_req_id=%s",
                num_external_tokens,
                pending is not None,
                pending.source.req_id if pending else None,
            )
            if pending is None:
                logger.warning(
                    "[InMemorySpyreConnector] No pending source for req_id=%s despite num_external_tokens=%d. "
                    "Setting is_store=True (will not load from cache)",
                    request.request_id,
                    num_external_tokens,
                )
                req_meta = SpyreConnectorRequestMeta(
                    req_id=request.request_id,
                    block_ids=flat_block_ids,
                    is_store=True,
                    token_count=len(request.all_token_ids),
                )
            else:
                num_external_blocks = num_external_tokens // self._block_size
                local_blocks = pending.num_local_computed_tokens // self._block_size
                source = pending.source
                block_mapping: list[tuple[int, int]] = []

                if num_external_tokens % self._block_size != 0:
                    num_external_blocks += 1

                # CRITICAL FIX: Map ALL blocks that the prompt uses, not just external blocks
                # For 67 tokens with num_external=64 (1 block), we still need blocks [1,2]
                # because the prompt spans into block 2 (tokens 64-66)
                total_prompt_blocks = len(flat_block_ids)

                for i in range(total_prompt_blocks):
                    src_idx = local_blocks + i
                    dest_idx = local_blocks + i
                    if src_idx >= len(source.block_ids) or dest_idx >= len(flat_block_ids):
                        break
                    block_mapping.append((source.block_ids[src_idx], flat_block_ids[dest_idx]))

                # CRITICAL: Validate against total_prompt_blocks, not num_external_blocks
                # We map ALL blocks the prompt uses, even if only some are "external"
                if len(block_mapping) != total_prompt_blocks:
                    logger.warning(
                        "[InMemorySpyreConnector] Block mapping incomplete: mapped=%d, expected=%d, setting is_store=True",
                        len(block_mapping),
                        total_prompt_blocks,
                    )
                    req_meta = SpyreConnectorRequestMeta(
                        req_id=request.request_id,
                        block_ids=flat_block_ids,
                        is_store=True,
                        token_count=len(request.all_token_ids),
                    )
                else:
                    req_meta = SpyreConnectorRequestMeta(
                        req_id=request.request_id,
                        block_ids=flat_block_ids,
                        is_store=False,
                        token_count=num_external_tokens,
                        source_req_id=source.req_id,
                        block_mapping=block_mapping,
                    )
        else:
            req_meta = SpyreConnectorRequestMeta(
                req_id=request.request_id,
                block_ids=flat_block_ids,
                is_store=True,
                token_count=len(request.all_token_ids),
            )

        # Step 2: Add prompt tokens to metadata for cross-instance transfer
        if hasattr(request, "prompt_token_ids") and request.prompt_token_ids:
            req_meta.prompt_token_ids = list(request.prompt_token_ids)
            logger.info(
                "[InMemorySpyreConnector] Added %d prompt_token_ids to req_meta for req_id=%s",
                len(request.prompt_token_ids),
                req_meta.req_id,
            )

        _emit_connector_probe(
            "pending_request_meta",
            req_id=req_meta.req_id,
            is_store=req_meta.is_store,
            token_count=req_meta.token_count,
            num_external_tokens=num_external_tokens,
            block_ids=list(req_meta.block_ids),
            source_req_id=req_meta.source_req_id,
            block_mapping=[list(pair) for pair in req_meta.block_mapping],
        )
        self._pending_requests.append(req_meta)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        logger.info(
            "[InMemorySpyreConnector] build_connector_meta: pending_prompt_tokens_keys=%s, size=%d",
            list(self._pending_prompt_tokens.keys()),
            len(self._pending_prompt_tokens),
        )
        meta = SpyreConnectorMeta(
            requests=list(self._pending_requests),
            layer_names=list(self._layer_names),
            block_size=self._block_size,
            dtype=self._dtype_str,
            layout="NHD",
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
        self._pending_requests.clear()
        self._pending_load_sources.clear()
        # Don't clear pending_prompt_tokens here
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        prompt = request.prompt_token_ids
        if prompt and block_ids:
            num_prompt_blocks = (len(prompt) + self._block_size - 1) // self._block_size
            prompt_block_ids = list(block_ids[:num_prompt_blocks])
            if not prompt_block_ids:
                logger.info(
                    "[InMemorySpyreConnector] request_finished req=%s prompt_tokens=%d "
                    "received no prompt block ids from %s",
                    request.request_id,
                    len(prompt),
                    block_ids,
                )
                return False, None

            available_blocks = self._store.available_prefix_blocks(
                request.request_id,
                prompt_block_ids,
            )
            if available_blocks < len(prompt_block_ids):
                logger.info(
                    "[InMemorySpyreConnector] request_finished prune req=%s "
                    "prompt_tokens=%d block_ids=%s prompt_block_ids=%s "
                    "available_blocks=%d store_stats=%s",
                    request.request_id,
                    len(prompt),
                    block_ids,
                    prompt_block_ids,
                    available_blocks,
                    self._store.stats(),
                )
                self._store.remove_by_req(request.request_id)
                return False, None

            saved = SavedRequestRecord(
                req_id=request.request_id,
                prompt_token_ids=tuple(prompt),
                block_ids=prompt_block_ids,
                num_tokens=len(prompt),
            )
            self._save_request_record(saved)
            _emit_connector_probe(
                "request_finished_saved",
                req_id=request.request_id,
                prompt_token_count=len(prompt),
                all_block_ids=list(block_ids),
                prompt_block_ids=prompt_block_ids,
            )
            logger.info(
                "[InMemorySpyreConnector] request_finished saved req=%s "
                "prompt_tokens=%d prompt_block_ids=%s store_stats=%s",
                request.request_id,
                len(prompt),
                prompt_block_ids,
                self._store.stats(),
            )

        # Cleanup per-request tracking
        self._request_computed_tokens.pop(request.request_id, None)
        logger.info(
            "[InMemorySpyreConnector] Cleaned up tracking for req_id=%s",
            request.request_id,
        )

        return False, None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self._stats.is_empty():
            return None
        snapshot = SpyreConnectorStats(data=dict(self._stats.data))
        self._stats.reset()
        return snapshot

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        if data is not None:
            return SpyreConnectorStats(data=data)
        return SpyreConnectorStats()

    def get_cumulative_metrics(self) -> dict[str, int]:
        return {
            "blocks_saved": self._blocks_saved,
            "blocks_loaded": self._blocks_loaded,
            "blocks_missing": self._blocks_missing,
            "saved_requests_count": self._saved_request_count(),
        }

    def get_store(self) -> SpyreKVStoreBackend:
        return self._store

    def reset_probe_state(
        self,
        *,
        clear_store: bool = True,
        clear_saved_requests: bool = True,
        clear_metrics: bool = True,
    ) -> None:
        self._pending_requests.clear()
        self._pending_load_sources.clear()
        self._step_stores.clear()
        self._step_loads.clear()
        self._load_error_block_ids.clear()
        # Don't clear pending_prompt_tokens - they're needed for file save
        # self._pending_prompt_tokens.clear()

        if clear_saved_requests:
            if self._store.has_persistent_saved_requests():
                self._saved_requests.clear()
                if not clear_store:
                    self._store.clear_saved_requests()
            else:
                self._saved_requests.clear()

        if clear_metrics:
            self._blocks_saved = 0
            self._blocks_loaded = 0
            self._blocks_missing = 0
            self._stats.reset()

        if clear_store:
            self._store.clear()

    def shutdown(self) -> None:
        logger.info("[InMemorySpyreConnector] Shutdown. Store stats: %s", self._store.stats())

        # Cleanup NIXL agent
        if self._nixl_agent is not None:
            try:
                peer = "client" if self._kv_role == "kv_producer" else "server"
                self._nixl_agent.remove_remote_agent(peer)
                logger.info("[InMemorySpyreConnector] NIXL agent cleaned up")
            except Exception as exc:
                logger.warning("[InMemorySpyreConnector] NIXL cleanup error: %s", exc)
            self._nixl_agent = None

        self._store.shutdown()
