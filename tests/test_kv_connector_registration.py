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

"""Smoke tests for the Spyre KV connector registration and import path."""

import builtins
import importlib
import sys

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM required for connector tests")

CONNECTOR_MODULE = (
    "spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector"
)


def test_register_kv_connector_with_factory():
    """register_kv_connector registers InMemorySpyreConnector by name."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    import spyre_inference

    spyre_inference.register_kv_connector()
    assert "InMemorySpyreConnector" in KVConnectorFactory._registry

    # Idempotent on second call.
    spyre_inference.register_kv_connector()


def test_connector_module_imports():
    """The ported connector module imports and exposes the connector class."""
    mod = importlib.import_module(CONNECTOR_MODULE)
    assert hasattr(mod, "InMemorySpyreConnector")


def test_connector_imports_without_nixl(monkeypatch):
    """The connector module must not crash when nixl is not importable."""
    real_import = builtins.__import__

    def block_nixl(name, *args, **kwargs):
        if name == "nixl" or name.startswith("nixl."):
            raise ImportError(f"nixl blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_nixl)
    for key in [k for k in sys.modules if k == "nixl" or k.startswith("nixl.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.delitem(sys.modules, CONNECTOR_MODULE, raising=False)

    mod = importlib.import_module(CONNECTOR_MODULE)
    assert mod.NIXL_AVAILABLE is False
    assert hasattr(mod, "InMemorySpyreConnector")


def test_build_spyre_kv_store_backend_heap_alias():
    """`heap` and `host_memory` must both build a HostMemoryKVStoreBackend.

    This is the runtime counterpart to the structure-test guard against the
    env-default-vs-registry mismatch. `heap` is the compatibility alias; the
    canonical name is `host_memory`.
    """
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
        HostMemoryKVStoreBackend,
        build_spyre_kv_store_backend,
    )

    heap_backend = build_spyre_kv_store_backend("heap")
    host_memory_backend = build_spyre_kv_store_backend("host_memory")
    assert isinstance(heap_backend, HostMemoryKVStoreBackend)
    assert isinstance(host_memory_backend, HostMemoryKVStoreBackend)


def test_build_spyre_kv_store_backend_default_env_value():
    """The current envs.py default for VLLM_SPYRE_KV_STORE_BACKEND must build."""
    from spyre_inference import envs as envs_spyre
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
        HostMemoryKVStoreBackend,
        build_spyre_kv_store_backend,
    )

    default = envs_spyre.VLLM_SPYRE_KV_STORE_BACKEND
    backend = build_spyre_kv_store_backend(default)
    assert isinstance(backend, HostMemoryKVStoreBackend), (
        f"default backend {default!r} did not build a HostMemoryKVStoreBackend; "
        f"got {type(backend).__name__}"
    )


def test_build_spyre_kv_store_backend_invalid_raises():
    """Invalid backend names must still raise ValueError with the supported list."""
    from spyre_inference.distributed.kv_transfer.kv_connector.v1.metadata import (
        build_spyre_kv_store_backend,
    )

    with pytest.raises(ValueError, match="Unknown Spyre KV store backend"):
        build_spyre_kv_store_backend("definitely_not_a_real_backend_xyz")
