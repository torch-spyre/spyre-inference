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

import os
import time
import uuid
from multiprocessing import get_context
from multiprocessing.connection import Client, Listener
from typing import Any
import contextlib


def _run_transport_store_server(socket_path: str) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    store: dict[Any, bytes] = {}
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
                    elif op == "put":
                        store[message["key"]] = message["payload"]
                        conn.send({"ok": True})
                    elif op == "get":
                        conn.send({"ok": True, "payload": store.get(message["key"])})
                    elif op == "delete_keys":
                        removed = 0
                        for key in message["keys"]:
                            if key in store:
                                del store[key]
                                removed += 1
                        conn.send({"ok": True, "removed": removed})
                    elif op == "clear":
                        store.clear()
                        conn.send({"ok": True})
                    elif op == "shutdown":
                        store.clear()
                        conn.send({"ok": True})
                        return
                    else:
                        conn.send({"ok": False, "error": f"Unknown op: {op}"})
            finally:
                conn.close()
    finally:
        listener.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


class UDSProcessKVTransport:
    def __init__(self) -> None:
        self._socket_path = os.path.join(
            "/tmp",
            f"spyre-kv-store-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock",
        )
        start_method = "fork" if os.name == "posix" else "spawn"
        self._ctx = get_context(start_method)
        self._process = self._ctx.Process(
            target=_run_transport_store_server,
            args=(self._socket_path,),
            daemon=True,
        )
        self._process.start()
        self._conn = self._connect_with_retry()

    def _connect_with_retry(self):
        deadline = time.time() + 30.0
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
                    f"Timed out connecting to local KV transport server at {self._socket_path}"
                )
            time.sleep(0.05)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._conn.send(message)
        response = self._conn.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Unknown local KV transport error"))
        return response

    def put(self, key: Any, payload: bytes) -> None:
        self._request({"op": "put", "key": key, "payload": payload})

    def get(self, key: Any) -> bytes | None:
        response = self._request({"op": "get", "key": key})
        return response.get("payload")

    def delete_keys(self, keys: list[Any]) -> int:
        response = self._request({"op": "delete_keys", "keys": keys})
        return int(response.get("removed", 0))

    def clear(self) -> None:
        self._request({"op": "clear"})

    def shutdown(self) -> None:
        conn = getattr(self, "_conn", None)
        process = getattr(self, "_process", None)
        try:
            if conn is not None:
                try:
                    conn.send({"op": "shutdown"})
                    conn.recv()
                except Exception:
                    pass
                conn.close()
        finally:
            self._conn = None

        try:
            if process is not None:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
        finally:
            self._process = None
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
