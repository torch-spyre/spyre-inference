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

from collections import OrderedDict
import os
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Listener
import time
from typing import Any


def _unlink_socket(socket_path: str) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)


class _SharedMemoryKVServiceState:
    def __init__(self) -> None:
        self._entries: dict[Any, dict[str, Any]] = {}
        self._request_keys: dict[str, set[Any]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._saved_requests: OrderedDict[str, Any] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = 0
        self._max_saved_requests = 1024
        self._current_bytes = 0
        self._evictions = 0

    def configure(
        self, max_bytes: int | None = None, max_saved_requests: int | None = None
    ) -> None:
        if max_bytes is not None:
            self._max_bytes = max(0, int(max_bytes))
        if max_saved_requests is not None:
            self._max_saved_requests = max(0, int(max_saved_requests))

    def _entry_bytes(self, entry: dict[str, Any]) -> int:
        return int(entry["payload_size"])

    def _request_id_for(self, key: Any, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: Any) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def _write_payload(self, payload: bytes) -> str:
        shm = shared_memory.SharedMemory(create=True, size=max(len(payload), 1))
        try:
            buf = shm.buf
            assert buf is not None  # buf is always set on an open SharedMemory
            buf[: len(payload)] = payload
            return shm.name
        finally:
            shm.close()

    def _unlink_entry(self, entry: dict[str, Any]) -> None:
        shm_name = entry["shm_name"]
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
            return

        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def put(
        self,
        key: Any,
        payload: bytes,
        dtype: str,
        shape: tuple[int, ...],
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._entries

        if was_overwrite:
            old = self._entries[key]
            self._current_bytes -= self._entry_bytes(old)
            self._unlink_entry(old)

        entry_size = len(payload)
        if self._max_bytes > 0:
            while self._entries and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        shm_name = self._write_payload(payload)
        entry = {
            "shm_name": shm_name,
            "payload_size": entry_size,
            "dtype": dtype,
            "shape": tuple(shape),
            "version": version,
            "source_req": source_req,
        }
        self._entries[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def get_entry(self, key: Any) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return dict(entry)

    def contains(self, key: Any) -> bool:
        return key in self._entries

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        present = {
            (
                key.layer_idx,
                key.block_id,
                getattr(key.kv_kind, "value", key.kv_kind),
            )
            for key in req_keys
        }

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in ("K", "V"):
                    if (layer_idx, block_id, kv_kind) not in present:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def save_request_record(self, record: Any) -> None:
        req_id = record.req_id
        if req_id in self._saved_requests:
            self._saved_requests.move_to_end(req_id)
        self._saved_requests[req_id] = record

        if self._max_saved_requests > 0:
            while len(self._saved_requests) > self._max_saved_requests:
                oldest_req_id, _ = self._saved_requests.popitem(last=False)
                self.remove_by_req(oldest_req_id)

    def get_saved_requests(self) -> list[Any]:
        return list(self._saved_requests.values())

    def remove_saved_request(self, req_id: str) -> bool:
        return self._saved_requests.pop(req_id, None) is not None

    def clear_saved_requests(self) -> None:
        self._saved_requests.clear()

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
                self._unlink_entry(entry)
        self._request_order.pop(req_id, None)
        self._saved_requests.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        for entry in list(self._entries.values()):
            self._unlink_entry(entry)
        self._entries.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._saved_requests.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": "serialized_shared_memory_service",
            "total_entries": len(self._entries),
            "unique_requests": len(self._request_keys),
            "saved_requests_count": len(self._saved_requests),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


def run_persistent_kv_service(socket_path: str) -> None:
    socket_dir = os.path.dirname(socket_path)
    if socket_dir:
        os.makedirs(socket_dir, exist_ok=True)
    _unlink_socket(socket_path)

    state = _SharedMemoryKVServiceState()
    listener = Listener(address=socket_path, family="AF_UNIX")
    try:
        while True:
            conn = listener.accept()
            try:
                while True:
                    try:
                        message = conn.recv()
                    except EOFError:
                        break

                    op = message["op"]
                    if op == "ping":
                        conn.send({"ok": True})
                    elif op == "configure":
                        state.configure(
                            max_bytes=message.get("max_bytes"),
                            max_saved_requests=message.get("max_saved_requests"),
                        )
                        conn.send({"ok": True})
                    elif op == "put":
                        version, was_overwrite = state.put(
                            message["key"],
                            message["payload"],
                            message["dtype"],
                            tuple(message["shape"]),
                            source_req=message.get("source_req", ""),
                        )
                        conn.send(
                            {
                                "ok": True,
                                "version": version,
                                "was_overwrite": was_overwrite,
                            }
                        )
                    elif op == "get_entry":
                        conn.send({"ok": True, "entry": state.get_entry(message["key"])})
                    elif op == "contains":
                        conn.send({"ok": True, "contains": state.contains(message["key"])})
                    elif op == "available_prefix_blocks":
                        conn.send(
                            {
                                "ok": True,
                                "available": state.available_prefix_blocks(
                                    message["req_id"],
                                    list(message["block_ids"]),
                                ),
                            }
                        )
                    elif op == "save_request_record":
                        state.save_request_record(message["record"])
                        conn.send({"ok": True})
                    elif op == "get_saved_requests":
                        conn.send({"ok": True, "records": state.get_saved_requests()})
                    elif op == "remove_saved_request":
                        conn.send(
                            {
                                "ok": True,
                                "removed": state.remove_saved_request(message["req_id"]),
                            }
                        )
                    elif op == "clear_saved_requests":
                        state.clear_saved_requests()
                        conn.send({"ok": True})
                    elif op == "remove_by_req":
                        conn.send(
                            {
                                "ok": True,
                                "removed": state.remove_by_req(message["req_id"]),
                            }
                        )
                    elif op == "clear":
                        state.clear()
                        conn.send({"ok": True})
                    elif op == "stats":
                        conn.send({"ok": True, "stats": state.stats()})
                    elif op == "shutdown_service":
                        state.clear()
                        conn.send({"ok": True})
                        return
                    else:
                        conn.send({"ok": False, "error": f"Unknown op: {op}"})
            finally:
                conn.close()
    finally:
        listener.close()
        _unlink_socket(socket_path)


class PersistentKVServiceClient:
    def __init__(self, socket_path: str, *, connect_timeout_s: float = 30.0) -> None:
        self._socket_path = socket_path
        self._connect_timeout_s = connect_timeout_s

    def _connect(self):
        deadline = time.time() + self._connect_timeout_s
        while True:
            try:
                conn = Client(address=self._socket_path, family="AF_UNIX")
                conn.send({"op": "ping"})
                response = conn.recv()
                if response.get("ok"):
                    return conn
                conn.close()
            except FileNotFoundError:
                pass
            except ConnectionRefusedError:
                pass
            except OSError:
                pass
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out connecting to persistent KV service at {self._socket_path}"
                )
            time.sleep(0.05)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.send(message)
            response = conn.recv()
        finally:
            conn.close()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Unknown persistent KV service error"))
        return response

    def configure(
        self, *, max_bytes: int | None = None, max_saved_requests: int | None = None
    ) -> None:
        self._request(
            {
                "op": "configure",
                "max_bytes": max_bytes,
                "max_saved_requests": max_saved_requests,
            }
        )

    def put(
        self,
        key: Any,
        payload: bytes,
        dtype: str,
        shape: tuple[int, ...],
        *,
        source_req: str = "",
    ) -> tuple[int, bool]:
        response = self._request(
            {
                "op": "put",
                "key": key,
                "payload": payload,
                "dtype": dtype,
                "shape": tuple(shape),
                "source_req": source_req,
            }
        )
        return int(response["version"]), bool(response["was_overwrite"])

    def get_entry(self, key: Any) -> dict[str, Any] | None:
        response = self._request({"op": "get_entry", "key": key})
        return response.get("entry")

    def contains(self, key: Any) -> bool:
        response = self._request({"op": "contains", "key": key})
        return bool(response.get("contains", False))

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        response = self._request(
            {"op": "available_prefix_blocks", "req_id": req_id, "block_ids": block_ids}
        )
        return int(response.get("available", 0))

    def save_request_record(self, record: Any) -> None:
        self._request({"op": "save_request_record", "record": record})

    def get_saved_requests(self) -> list[Any]:
        response = self._request({"op": "get_saved_requests"})
        return list(response.get("records", []))

    def remove_saved_request(self, req_id: str) -> bool:
        response = self._request({"op": "remove_saved_request", "req_id": req_id})
        return bool(response.get("removed", False))

    def clear_saved_requests(self) -> None:
        self._request({"op": "clear_saved_requests"})

    def remove_by_req(self, req_id: str) -> int:
        response = self._request({"op": "remove_by_req", "req_id": req_id})
        return int(response.get("removed", 0))

    def clear(self) -> None:
        self._request({"op": "clear"})

    def stats(self) -> dict[str, Any]:
        response = self._request({"op": "stats"})
        return dict(response.get("stats", {}))

    def shutdown_service(self) -> None:
        self._request({"op": "shutdown_service"})

    def close(self) -> None:
        return
