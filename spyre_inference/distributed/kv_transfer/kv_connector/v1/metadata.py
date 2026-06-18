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

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import io
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.local_transport_store import (
    UDSProcessKVTransport,
)
from spyre_inference.distributed.kv_transfer.kv_connector.v1.persistent_kv_service import (
    PersistentKVServiceClient,
)
import contextlib


class KVKind(str, Enum):  # noqa: UP042 — str() semantics relied on by store keys
    K = "K"
    V = "V"


@dataclass(frozen=True)
class StoreKey:
    req_id: str
    layer_idx: int
    block_id: int
    kv_kind: KVKind

    @property
    def layer_name(self) -> str:
        return f"model.layers.{self.layer_idx}.self_attn"


@dataclass
class HostMemoryKVEntry:
    data: torch.Tensor
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(self, expected_shape: tuple[int, ...], expected_dtype: str) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


@dataclass
class SerializedHostMemoryKVEntry:
    payload: bytes
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(self, expected_shape: tuple[int, ...], expected_dtype: str) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


@dataclass
class SerializedUDSProcessKVEntry:
    payload_size: int
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(self, expected_shape: tuple[int, ...], expected_dtype: str) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


@dataclass
class SerializedSharedMemoryKVEntry:
    shm_name: str
    payload_size: int
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(self, expected_shape: tuple[int, ...], expected_dtype: str) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


def _unregister_shared_memory_attachment(entry_name: str, shm: shared_memory.SharedMemory) -> None:
    # This process is only attaching to a segment owned elsewhere. Unregister
    # the attachment so Python does not try to unlink it again at interpreter
    # shutdown.
    tracker_name = str(getattr(shm, "_name", entry_name))
    with contextlib.suppress(Exception):
        resource_tracker.unregister(tracker_name, "shared_memory")


class SpyreKVStoreBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def current_bytes(self) -> int: ...

    @property
    @abstractmethod
    def max_bytes(self) -> int: ...

    @property
    @abstractmethod
    def evictions(self) -> int: ...

    @abstractmethod
    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]: ...

    @abstractmethod
    def get(self, key: StoreKey) -> HostMemoryKVEntry | None: ...

    @abstractmethod
    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool: ...

    @abstractmethod
    def contains(self, key: StoreKey) -> bool: ...

    @abstractmethod
    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int: ...

    @abstractmethod
    def remove_by_req(self, req_id: str) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...

    def has_persistent_saved_requests(self) -> bool:
        return False

    def save_request_record(self, record: SavedRequestRecord) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support persistent saved requests"
        )

    def get_saved_requests(self) -> list[SavedRequestRecord]:
        return []

    def remove_saved_request(self, req_id: str) -> bool:
        return False

    def clear_saved_requests(self) -> None:
        return

    def saved_request_count(self) -> int:
        return 0


@dataclass
class SpyreRemoteMeta:
    """Remote connection metadata for KV transfer (similar to GPU NixlConnector)"""

    block_ids: list[int]
    host: str  # Remote prefill IP (dynamic from routing proxy or env var)
    port: int  # Remote prefill port
    engine_id: str  # Remote engine identifier
    request_id: str  # Request ID on remote side


@dataclass
class SpyreConnectorRequestMeta:
    req_id: str
    block_ids: list[int] = field(default_factory=list)
    is_store: bool = True
    token_count: int = 0
    source_req_id: str = ""
    block_mapping: list[tuple[int, int]] = field(default_factory=list)
    remote: SpyreRemoteMeta | None = None  # Remote connection info for decode-side loads


@dataclass(frozen=True)
class SavedRequestRecord:
    req_id: str
    prompt_token_ids: tuple[int, ...]
    block_ids: list[int]
    num_tokens: int


@dataclass
class SpyreConnectorMeta(KVConnectorMetadata):
    schema_version: int = 1
    requests: list[SpyreConnectorRequestMeta] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    block_size: int = 0
    dtype: str = ""
    layout: str = "NHD"
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0

    _SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})
    _KNOWN_LAYOUTS: frozenset[str] = frozenset({"NHD"})

    def add_store_request(
        self,
        req_id: str,
        block_ids: list[int],
        token_count: int = 0,
    ) -> None:
        self.requests.append(
            SpyreConnectorRequestMeta(
                req_id=req_id,
                block_ids=block_ids,
                is_store=True,
                token_count=token_count,
            )
        )

    def add_load_request(
        self,
        req_id: str,
        block_ids: list[int],
        source_req_id: str,
        token_count: int = 0,
        block_mapping: list[tuple[int, int]] | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> None:
        """Add a load request with optional remote connection parameters.

        Args:
            req_id: Local request ID
            block_ids: Local block IDs to load into
            source_req_id: Source request ID (for local cache lookup)
            token_count: Number of tokens
            block_mapping: Optional block mapping
            kv_transfer_params: Optional dict with remote connection info:
                - remote_host: Prefill IP (from routing proxy or env var)
                - remote_port: Prefill port
                - remote_engine_id: Remote engine ID
                - remote_request_id: Request ID on remote side
                - remote_block_ids: Block IDs on remote side
        """
        remote_meta = None
        if kv_transfer_params:
            # Extract remote connection parameters (similar to GPU NixlConnector)
            remote_host = kv_transfer_params.get("remote_host")
            remote_port = kv_transfer_params.get("remote_port")
            remote_engine_id = kv_transfer_params.get("remote_engine_id")
            remote_request_id = kv_transfer_params.get("remote_request_id")
            remote_block_ids = kv_transfer_params.get("remote_block_ids", block_ids)

            if remote_host and remote_port:
                remote_meta = SpyreRemoteMeta(
                    block_ids=remote_block_ids,
                    host=remote_host,
                    port=remote_port,
                    engine_id=remote_engine_id or "",
                    request_id=remote_request_id or source_req_id,
                )

        self.requests.append(
            SpyreConnectorRequestMeta(
                req_id=req_id,
                block_ids=block_ids,
                is_store=False,
                token_count=token_count,
                source_req_id=source_req_id,
                block_mapping=block_mapping or [],
                remote=remote_meta,
            )
        )

    def validate_block_mapping(self) -> None:
        all_dest_blocks: list[int] = []
        for req in self.requests:
            if not req.is_store and req.block_mapping:
                for _, dest_block in req.block_mapping:
                    all_dest_blocks.append(dest_block)
            elif not req.is_store:
                all_dest_blocks.extend(req.block_ids)

        seen: set[int] = set()
        for block_id in all_dest_blocks:
            if block_id in seen:
                raise ValueError(
                    f"Duplicate destination block ID {block_id} in load requests. "
                    "This would cause data corruption."
                )
            seen.add(block_id)

    def validate(self) -> None:
        if self.schema_version not in self._SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version}. "
                f"Supported: {sorted(self._SUPPORTED_VERSIONS)}"
            )

        if self.layout and self.layout not in self._KNOWN_LAYOUTS:
            raise ValueError(f"Unknown layout '{self.layout}'")

        for req in self.requests:
            if not req.req_id:
                raise ValueError("Request with empty req_id in metadata")
            if not req.is_store and not req.source_req_id and req.block_mapping:
                raise ValueError(
                    f"Load request {req.req_id} has block_mapping but no source_req_id"
                )
            for bid in req.block_ids:
                if bid < 0:
                    raise ValueError(f"Request {req.req_id} has negative block_id {bid}")
            for src, dest in req.block_mapping:
                if src < 0 or dest < 0:
                    raise ValueError(f"Request {req.req_id} block_mapping contains negative IDs")

        if self.num_layers < 0 or self.num_kv_heads < 0 or self.head_dim < 0:
            raise ValueError("Connector metadata dimensions must be non-negative")
        if self.block_size < 0:
            raise ValueError("block_size must be non-negative")

        self.validate_block_mapping()

    @staticmethod
    def make_layer_names(num_layers: int) -> list[str]:
        return [f"model.layers.{i}.self_attn" for i in range(num_layers)]


class HostMemoryKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, HostMemoryKVEntry] = {}
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "host_memory"

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    @staticmethod
    def _entry_bytes(entry: HostMemoryKVEntry) -> int:
        return entry.data.nelement() * entry.data.element_size()

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)

        entry = HostMemoryKVEntry(
            data=data.detach().clone().cpu(),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        entry_size = self._entry_bytes(entry)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        return self._store.get(key)

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self.get(key)
        if entry is None:
            return False

        try:
            dest.copy_(entry.data)
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
        self._request_order.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }

    def shutdown(self) -> None:
        self.clear()


class SerializedHostMemoryKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, SerializedHostMemoryKVEntry] = {}
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "serialized_host_memory"

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    @staticmethod
    def _entry_bytes(entry: SerializedHostMemoryKVEntry) -> int:
        return len(entry.payload)

    @staticmethod
    def _serialize_tensor(data: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(data.detach().clone().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _materialize_tensor(entry: SerializedHostMemoryKVEntry) -> torch.Tensor:
        buffer = io.BytesIO(entry.payload)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)

        entry = SerializedHostMemoryKVEntry(
            payload=self._serialize_tensor(data),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        entry_size = self._entry_bytes(entry)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        return HostMemoryKVEntry(
            data=self._materialize_tensor(entry),
            dtype=entry.dtype,
            shape=entry.shape,
            version=entry.version,
            source_req=entry.source_req,
        )

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False

        try:
            dest.copy_(self._materialize_tensor(entry))
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
        self._request_order.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }

    def shutdown(self) -> None:
        self.clear()


class SerializedSharedMemoryKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, SerializedSharedMemoryKVEntry] = {}
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "serialized_shared_memory"

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    @staticmethod
    def _entry_bytes(entry: SerializedSharedMemoryKVEntry) -> int:
        return entry.payload_size

    @staticmethod
    def _serialize_tensor(data: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(data.detach().clone().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _materialize_tensor(payload: bytes) -> torch.Tensor:
        buffer = io.BytesIO(payload)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    @staticmethod
    def _write_payload(payload: bytes) -> str:
        shm = shared_memory.SharedMemory(create=True, size=max(len(payload), 1))
        try:
            buf = shm.buf
            assert buf is not None  # buf is always set on an open SharedMemory
            buf[: len(payload)] = payload
            return shm.name
        finally:
            shm.close()

    @staticmethod
    def _read_payload(entry: SerializedSharedMemoryKVEntry) -> bytes | None:
        try:
            shm = shared_memory.SharedMemory(name=entry.shm_name, create=False)
        except FileNotFoundError:
            return None

        with contextlib.suppress(Exception):
            _unregister_shared_memory_attachment(entry.shm_name, shm)

        try:
            buf = shm.buf
            assert buf is not None  # buf is always set on an open SharedMemory
            return bytes(buf[: entry.payload_size])
        finally:
            shm.close()

    @staticmethod
    def _unlink_entry(entry: SerializedSharedMemoryKVEntry) -> None:
        try:
            shm = shared_memory.SharedMemory(name=entry.shm_name, create=False)
        except FileNotFoundError:
            return

        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)
            self._unlink_entry(old)

        payload = self._serialize_tensor(data)
        entry_size = len(payload)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        shm_name = self._write_payload(payload)
        entry = SerializedSharedMemoryKVEntry(
            shm_name=shm_name,
            payload_size=entry_size,
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )

        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None

        payload = self._read_payload(entry)
        if payload is None:
            return None

        return HostMemoryKVEntry(
            data=self._materialize_tensor(payload),
            dtype=entry.dtype,
            shape=entry.shape,
            version=entry.version,
            source_req=entry.source_req,
        )

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False

        payload = self._read_payload(entry)
        if payload is None:
            return False

        try:
            dest.copy_(self._materialize_tensor(payload))
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
                self._unlink_entry(entry)
        self._request_order.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        for entry in list(self._store.values()):
            self._unlink_entry(entry)
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def shutdown(self) -> None:
        self.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


class SerializedSharedMemoryServiceKVStoreBackend(SpyreKVStoreBackend):
    def __init__(
        self,
        max_bytes: int = 0,
        *,
        socket_path: str = "/tmp/spyre-kv-persistent.sock",
        max_saved_requests: int = 1024,
    ) -> None:
        self._socket_path = socket_path
        self._client = PersistentKVServiceClient(socket_path)
        self._client.configure(
            max_bytes=max(0, max_bytes),
            max_saved_requests=max(0, max_saved_requests),
        )

    @property
    def backend_name(self) -> str:
        return "serialized_shared_memory_service"

    def has_persistent_saved_requests(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return int(self.stats().get("total_entries", 0))

    @property
    def current_bytes(self) -> int:
        return int(self.stats().get("memory_estimate_bytes", 0))

    @property
    def max_bytes(self) -> int:
        return int(self.stats().get("max_bytes", 0))

    @property
    def evictions(self) -> int:
        return int(self.stats().get("evictions", 0))

    @staticmethod
    def _serialize_tensor(data: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(data.detach().clone().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _materialize_tensor(payload: bytes) -> torch.Tensor:
        buffer = io.BytesIO(payload)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> SerializedSharedMemoryKVEntry:
        return SerializedSharedMemoryKVEntry(
            shm_name=str(entry["shm_name"]),
            payload_size=int(entry["payload_size"]),
            dtype=str(entry["dtype"]),
            shape=tuple(entry["shape"]),
            version=int(entry.get("version", 1)),
            source_req=str(entry.get("source_req", "")),
        )

    @staticmethod
    def _read_payload(entry: SerializedSharedMemoryKVEntry) -> bytes | None:
        try:
            shm = shared_memory.SharedMemory(name=entry.shm_name, create=False)
        except FileNotFoundError:
            return None

        with contextlib.suppress(Exception):
            _unregister_shared_memory_attachment(entry.shm_name, shm)

        try:
            buf = shm.buf
            assert buf is not None  # buf is always set on an open SharedMemory
            return bytes(buf[: entry.payload_size])
        finally:
            shm.close()

    @staticmethod
    def _normalize_saved_request(record: Any) -> SavedRequestRecord:
        if isinstance(record, SavedRequestRecord):
            return record
        if isinstance(record, dict):
            return SavedRequestRecord(
                req_id=str(record["req_id"]),
                prompt_token_ids=tuple(record["prompt_token_ids"]),
                block_ids=list(record["block_ids"]),
                num_tokens=int(record["num_tokens"]),
            )
        raise TypeError(f"Unsupported saved request record type: {type(record)!r}")

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        payload = self._serialize_tensor(data)
        return self._client.put(
            key,
            payload,
            str(data.dtype),
            tuple(data.shape),
            source_req=source_req,
        )

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        entry_dict = self._client.get_entry(key)
        if entry_dict is None:
            return None

        entry = self._normalize_entry(entry_dict)
        payload = self._read_payload(entry)
        if payload is None:
            return None

        return HostMemoryKVEntry(
            data=self._materialize_tensor(payload),
            dtype=entry.dtype,
            shape=entry.shape,
            version=entry.version,
            source_req=entry.source_req,
        )

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry_dict = self._client.get_entry(key)
        if entry_dict is None:
            return False

        entry = self._normalize_entry(entry_dict)
        payload = self._read_payload(entry)
        if payload is None:
            return False

        try:
            dest.copy_(self._materialize_tensor(payload))
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return self._client.contains(key)

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        return self._client.available_prefix_blocks(req_id, block_ids)

    def remove_by_req(self, req_id: str) -> int:
        return self._client.remove_by_req(req_id)

    def clear(self) -> None:
        self._client.clear()

    def shutdown(self) -> None:
        self._client.close()

    def stats(self) -> dict[str, Any]:
        return self._client.stats()

    def save_request_record(self, record: SavedRequestRecord) -> None:
        self._client.save_request_record(record)

    def get_saved_requests(self) -> list[SavedRequestRecord]:
        return [
            self._normalize_saved_request(record) for record in self._client.get_saved_requests()
        ]

    def remove_saved_request(self, req_id: str) -> bool:
        return self._client.remove_saved_request(req_id)

    def clear_saved_requests(self) -> None:
        self._client.clear_saved_requests()

    def saved_request_count(self) -> int:
        return int(self.stats().get("saved_requests_count", 0))


class SerializedUDSProcessKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, SerializedUDSProcessKVEntry] = {}
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._transport = UDSProcessKVTransport()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "serialized_uds_process_store"

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    @staticmethod
    def _entry_bytes(entry: SerializedUDSProcessKVEntry) -> int:
        return entry.payload_size

    @staticmethod
    def _serialize_tensor(data: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(data.detach().clone().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _materialize_tensor(payload: bytes) -> torch.Tensor:
        buffer = io.BytesIO(payload)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)

        payload = self._serialize_tensor(data)
        entry = SerializedUDSProcessKVEntry(
            payload_size=len(payload),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        entry_size = self._entry_bytes(entry)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        self._transport.put(key, payload)
        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        payload = self._transport.get(key)
        if payload is None:
            return None
        return HostMemoryKVEntry(
            data=self._materialize_tensor(payload),
            dtype=entry.dtype,
            shape=entry.shape,
            version=entry.version,
            source_req=entry.source_req,
        )

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False

        payload = self._transport.get(key)
        if payload is None:
            return False

        try:
            dest.copy_(self._materialize_tensor(payload))
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        removed = 0
        if keys:
            removed = self._transport.delete_keys(keys)
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
        self._request_order.pop(req_id, None)
        return removed

    def clear(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0
        self._transport.clear()

    def shutdown(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0
        self._transport.shutdown()

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


_STORE_BACKEND_TYPES: dict[str, type[SpyreKVStoreBackend]] = {
    "host_memory": HostMemoryKVStoreBackend,
    # Compatibility alias: VLLM_SPYRE_KV_STORE_BACKEND defaults to "heap"
    # in envs.py and prior runbooks/scripts use that name. Map it to the
    # host-memory backend, matching the InMemoryKVStore = HostMemoryKVStoreBackend
    # alias maintained below.
    "heap": HostMemoryKVStoreBackend,
    "serialized_host_memory": SerializedHostMemoryKVStoreBackend,
    "serialized_shared_memory": SerializedSharedMemoryKVStoreBackend,
    "serialized_shared_memory_service": SerializedSharedMemoryServiceKVStoreBackend,
    "serialized_uds_process_store": SerializedUDSProcessKVStoreBackend,
}


def build_spyre_kv_store_backend(
    backend_name: str,
    *,
    max_bytes: int = 0,
    max_saved_requests: int = 1024,
    service_socket: str | None = None,
) -> SpyreKVStoreBackend:
    backend_key = backend_name.strip().lower()
    backend_type = _STORE_BACKEND_TYPES.get(backend_key)
    if backend_type is None:
        supported = ", ".join(sorted(_STORE_BACKEND_TYPES))
        raise ValueError(
            f"Unknown Spyre KV store backend '{backend_name}'. Supported backends: {supported}"
        )
    # issubclass narrows the registry's base type to the concrete
    # constructor signatures.
    if issubclass(backend_type, SerializedSharedMemoryServiceKVStoreBackend):
        return backend_type(
            max_bytes=max_bytes,
            socket_path=service_socket or "/tmp/spyre-kv-persistent.sock",
            max_saved_requests=max_saved_requests,
        )
    return backend_type(max_bytes=max_bytes)


# Backward-compatible aliases for the current slice, tests, and examples.
InMemoryKVEntry = HostMemoryKVEntry
InMemoryKVStore = HostMemoryKVStoreBackend


_STATS_KEYS = (
    "matched_tokens",
    "loaded_blocks",
    "saved_blocks",
    "load_misses",
    "evictions",
    "match_attempts",
)


@dataclass
class SpyreConnectorStats(KVConnectorStats):
    # Mirrors the base KVConnectorStats field so the generated dataclass
    # __init__ accepts `data=` even when vLLM types are unresolved.
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data = {key: 0 for key in _STATS_KEYS}

    def record(self, key: str, value: int = 1) -> None:
        self.data[key] = self.data.get(key, 0) + value

    def aggregate(self, other: KVConnectorStats) -> SpyreConnectorStats:
        for key in _STATS_KEYS:
            self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)
        return self

    def reduce(self) -> dict[str, int | float]:
        return {key: self.data.get(key, 0) for key in _STATS_KEYS}

    def is_empty(self) -> bool:
        return all(self.data.get(key, 0) == 0 for key in _STATS_KEYS)
