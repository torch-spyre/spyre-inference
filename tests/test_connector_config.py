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

"""Tests for connector setting resolution from env vars and vLLM config.

The pure resolvers are imported by file path so they run without vLLM,
NIXL, or AIU. A vLLM-gated test confirms the connector wires them up.
"""

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CONNECTOR_DIR = REPO / "spyre_inference" / "distributed" / "kv_transfer" / "kv_connector" / "v1"


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load_by_path("connector_config", CONNECTOR_DIR / "connector_config.py")


# --- role -------------------------------------------------------------------


def test_role_from_config_when_no_env():
    assert cfg.resolve_kv_role(None, "kv_producer") == "kv_producer"
    assert cfg.resolve_kv_role(None, "kv_consumer") == "kv_consumer"


def test_role_empty_env_falls_back_to_config():
    # An empty (but present) env var is not an override.
    assert cfg.resolve_kv_role("", "kv_consumer") == "kv_consumer"


def test_role_env_overrides_config():
    assert cfg.resolve_kv_role("kv_consumer", "kv_producer") == "kv_consumer"


def test_role_default_empty():
    assert cfg.resolve_kv_role(None, None) == ""
    assert cfg.resolve_kv_role("", None) == ""


# --- use_nixl ---------------------------------------------------------------


def test_use_nixl_from_config_when_no_env():
    assert cfg.resolve_use_nixl(None, {"use_nixl": True}) is True
    assert cfg.resolve_use_nixl(None, {"use_nixl": False}) is False


def test_use_nixl_defaults_false():
    assert cfg.resolve_use_nixl(None, None) is False
    assert cfg.resolve_use_nixl(None, {}) is False


def test_use_nixl_env_overrides_config():
    assert cfg.resolve_use_nixl("1", {"use_nixl": False}) is True
    assert cfg.resolve_use_nixl("0", {"use_nixl": True}) is False


# --- remote ip / port -------------------------------------------------------


def test_remote_ip_env_overrides_all():
    assert (
        cfg.resolve_nixl_remote_ip("1.2.3.4", {"nixl_remote_ip": "5.6.7.8"}, "9.9.9.9") == "1.2.3.4"
    )


def test_remote_ip_extra_config_then_kv_ip_then_default():
    assert cfg.resolve_nixl_remote_ip(None, {"nixl_remote_ip": "5.6.7.8"}, "9.9.9.9") == "5.6.7.8"
    assert cfg.resolve_nixl_remote_ip(None, {}, "9.9.9.9") == "9.9.9.9"
    assert cfg.resolve_nixl_remote_ip(None, None, None) == cfg.DEFAULT_NIXL_REMOTE_IP


def test_port_extra_config_else_default():
    assert cfg.resolve_nixl_port({"nixl_port": 9200}, 9100) == 9200
    assert cfg.resolve_nixl_port({}, 9100) == 9100
    assert cfg.resolve_nixl_port(None, 9100) == 9100


# --- connector wiring (vLLM-gated) -----------------------------------------


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_connector_role_from_config_without_env(kv_role, monkeypatch):
    pytest.importorskip("vllm", reason="vLLM required for connector construction")
    monkeypatch.delenv("VLLM_SPYRE_KV_ROLE", raising=False)
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)

    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    kv_cfg = KVTransferConfig(
        kv_connector="InMemorySpyreConnector",
        kv_role=kv_role,
        kv_connector_extra_config={"use_nixl": False},
    )
    connector = InMemorySpyreConnector(
        VllmConfig(kv_transfer_config=kv_cfg), KVConnectorRole.WORKER
    )
    assert connector._kv_role == kv_role  # from config, no env
    assert connector._use_nixl is False  # from extra_config, no env


def test_connector_use_nixl_from_config_without_env(monkeypatch):
    pytest.importorskip("vllm", reason="vLLM required for connector construction")
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)
    monkeypatch.setenv("VLLM_SPYRE_KV_ROLE", "kv_producer")

    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    # extra_config requests NIXL; with NIXL not installed locally the
    # connector logs and disables it, so assert via the resolver instead.
    assert cfg.resolve_use_nixl(None, {"use_nixl": True}) is True

    kv_cfg = KVTransferConfig(
        kv_connector="InMemorySpyreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"nixl_remote_ip": "10.0.0.5", "nixl_port": 9300},
    )
    connector = InMemorySpyreConnector(
        VllmConfig(kv_transfer_config=kv_cfg), KVConnectorRole.WORKER
    )
    assert connector._nixl_remote_ip == "10.0.0.5"  # from extra_config, no env
    assert connector._nixl_port == 9300  # from extra_config, no env


def test_connector_env_overrides_config(monkeypatch):
    pytest.importorskip("vllm", reason="vLLM required for connector construction")
    monkeypatch.setenv("VLLM_SPYRE_KV_ROLE", "kv_consumer")
    monkeypatch.setenv("VLLM_SPYRE_NIXL_REMOTE_IP", "192.168.1.1")
    monkeypatch.delenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", raising=False)

    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from spyre_inference.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    kv_cfg = KVTransferConfig(
        kv_connector="InMemorySpyreConnector",
        kv_role="kv_producer",  # overridden by env -> kv_consumer
        kv_connector_extra_config={"nixl_remote_ip": "5.6.7.8"},  # overridden by env
    )
    connector = InMemorySpyreConnector(
        VllmConfig(kv_transfer_config=kv_cfg), KVConnectorRole.WORKER
    )
    assert connector._kv_role == "kv_consumer"  # env wins
    assert connector._nixl_remote_ip == "192.168.1.1"  # env wins
