# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre model registry — typed wrappers around model_configs.yaml.

Usage::

    from spyre_inference.config import model_registry, lookup_config

    entry = model_registry()["google/gemma-3-1b-it"]
    cfg = lookup_config("google/gemma-3-1b-it", tp_size=1, max_model_len=32768)
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

_YAML_PATH = Path(__file__).with_name("model_configs.yaml")

_VALID_PLATFORMS = frozenset({"x86_64", "s390x", "ppc64le"})


def current_platform() -> str:
    """Return the normalised machine architecture for registry platform checks.

    Maps ``platform.machine()`` to the registry's canonical names:
    ``x86_64``, ``s390x``, or ``ppc64le``.  Unknown architectures are
    returned as-is so they never match a restricted platform list.
    """
    machine = platform.machine()
    _aliases = {"amd64": "x86_64", "x86": "x86_64", "s390": "s390x"}
    return _aliases.get(machine.lower(), machine)


@dataclass
class DeviceConfig:
    """Spyre-specific overrides applied on top of the serving config."""

    env_vars: dict[str, Any] = field(default_factory=dict)
    num_gpu_blocks_override: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceConfig:
        return cls(
            env_vars=d.get("env_vars") or {},
            num_gpu_blocks_override=d.get("num_gpu_blocks_override"),
        )


@dataclass
class ContinuousBatchingConfig:
    tp_size: int
    max_model_len: int
    max_num_seqs: int
    device_config: DeviceConfig = field(default_factory=DeviceConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContinuousBatchingConfig:
        return cls(
            tp_size=d["tp_size"],
            max_model_len=d["max_model_len"],
            max_num_seqs=d["max_num_seqs"],
            device_config=DeviceConfig.from_dict(d.get("device_config") or {}),
        )


@dataclass
class ModelEntry:
    model_id: str
    platforms: list[str] | None
    continuous_batching_configs: list[ContinuousBatchingConfig]

    def supports_platform(self, machine: str) -> bool:
        return self.platforms is None or machine in self.platforms

    @classmethod
    def from_dict(cls, model_id: str, d: dict[str, Any]) -> ModelEntry:
        raw_platforms = d.get("platforms")
        cb_configs = [
            ContinuousBatchingConfig.from_dict(c)
            for c in (d.get("continuous_batching_configs") or [])
        ]
        return cls(
            model_id=model_id,
            platforms=list(raw_platforms) if raw_platforms is not None else None,
            continuous_batching_configs=cb_configs,
        )


@lru_cache(maxsize=1)
def model_registry() -> dict[str, ModelEntry]:
    """Return the parsed model registry, cached after first load."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load the Spyre model registry. "
            "Install it with: pip install pyyaml"
        ) from exc

    raw: dict[str, Any] = yaml.safe_load(_YAML_PATH.read_text())
    models_raw: dict[str, Any] = raw.get("models") or {}
    return {
        model_id: ModelEntry.from_dict(model_id, entry) for model_id, entry in models_raw.items()
    }


def lookup_config(
    model_id: str,
    tp_size: int,
    max_model_len: int,
    machine: str | None = None,
) -> ContinuousBatchingConfig | None:
    """Return the first ``ContinuousBatchingConfig`` matching ``tp_size``,
    ``max_model_len``, and the current platform for ``model_id``.

    Returns ``None`` when no match is found — either the model is unknown,
    the platform is not supported, or no config matches the requested
    tp/len combination.

    ``machine`` defaults to ``current_platform()`` and can be overridden
    in tests.
    """
    if machine is None:
        machine = current_platform()

    entry = model_registry().get(model_id)
    if entry is None:
        return None
    if not entry.supports_platform(machine):
        return None
    for cfg in entry.continuous_batching_configs:
        if cfg.tp_size == tp_size and cfg.max_model_len == max_model_len:
            return cfg
    return None
