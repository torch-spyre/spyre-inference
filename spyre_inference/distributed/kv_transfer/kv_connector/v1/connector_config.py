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

"""Pure resolution of InMemorySpyreConnector runtime settings.

These helpers derive role, NIXL enablement, and the NIXL remote endpoint
from the two sources the connector accepts:

1. Environment variables (the baked-image/manual deployment path). An
   explicitly set env var always wins, for backwards compatibility.
2. vLLM ``kv_transfer_config`` (the ``vllm serve --kv-transfer-config``
   path): ``kv_role`` for the role, ``kv_connector_extra_config`` for
   NIXL knobs (``use_nixl``, ``nixl_remote_ip``, ``nixl_port``), and
   ``kv_ip`` as a remote-IP fallback.

The functions take plain values (no vLLM types) so they unit-test without
vLLM, NIXL, or AIU installed.
"""

from __future__ import annotations

from typing import Any

DEFAULT_NIXL_REMOTE_IP = "10.130.2.89"


def _parse_env_bool(value: str) -> bool:
    """Match the existing ``bool(int(...))`` env parsing (1/0)."""
    return bool(int(value))


def resolve_kv_role(env_role: str | None, config_role: str | None) -> str:
    """Non-empty ``VLLM_SPYRE_KV_ROLE`` wins, else config role, else ''."""
    if env_role:
        return env_role
    return config_role or ""


def resolve_use_nixl(env_value: str | None, extra_config: dict[str, Any] | None) -> bool:
    """Explicitly set env wins, else ``extra_config['use_nixl']``, else False.

    ``env_value`` is the raw ``os.environ.get`` result: ``None`` means the
    env var was not set, so config is consulted.
    """
    if env_value is not None:
        return _parse_env_bool(env_value)
    if extra_config and "use_nixl" in extra_config:
        return bool(extra_config["use_nixl"])
    return False


def resolve_nixl_remote_ip(
    env_value: str | None,
    extra_config: dict[str, Any] | None,
    config_ip: str | None,
    default: str = DEFAULT_NIXL_REMOTE_IP,
) -> str:
    """Env override wins, else ``extra_config['nixl_remote_ip']``, else
    the connector ``kv_ip``, else the legacy default."""
    if env_value is not None:
        return env_value
    if extra_config and extra_config.get("nixl_remote_ip"):
        return str(extra_config["nixl_remote_ip"])
    if config_ip:
        return str(config_ip)
    return default


def resolve_nixl_port(extra_config: dict[str, Any] | None, default: int) -> int:
    """``extra_config['nixl_port']`` if set, else the module default.

    ``kv_port`` is intentionally not used as a fallback: vLLM auto-assigns
    it (e.g. 14579), which would silently break the established listen-port
    contract. Operators wanting a non-default port set ``nixl_port``.
    """
    if extra_config and extra_config.get("nixl_port"):
        return int(extra_config["nixl_port"])
    return default
