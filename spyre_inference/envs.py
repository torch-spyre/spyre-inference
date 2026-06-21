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

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    SPYRE_ATTN_IMPL: str = "default"
    SPYRE_SCATTER_USE_OVERWRITE: bool = False
    VLLM_SPYRE_ENABLE_FILE_TRANSFER: bool = False
    VLLM_SPYRE_ENABLE_NIXL_TRANSFER: bool = False
    VLLM_SPYRE_NIXL_BLOCKING_TRANSFER: bool = True
    VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE: bool = True
    VLLM_SPYRE_KV_CACHE_FILE_PATH: str | None = None
    VLLM_SPYRE_NIXL_REMOTE_IP: str = "10.130.2.89"
    VLLM_SPYRE_KV_ROLE: str = ""
    VLLM_SPYRE_KV_STORE_BACKEND: str = "heap"
    VLLM_SPYRE_KV_STORE_MAX_BYTES: int = 0
    VLLM_SPYRE_KV_SERVICE_SOCKET: str | None = None
    VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE: int = 100
    VLLM_SPYRE_KV_PLACEMENT_PROBE_ENABLED: bool = False
    VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE: bool = False
    VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT: bool = False
    VLLM_SPYRE_HEAP_KV_EXPORT_DIR: str | None = None
    VLLM_SPYRE_HEAP_KV_PERFDSC_DIR: str | None = None

_cache: dict[str, Any] = {}


def override(name: str, value: str) -> None:
    if name not in environment_variables:
        raise ValueError(f"The variable {name} is not a known setting and cannot be overridden")
    os.environ[name] = value
    _cache[name] = environment_variables[name]()


def clear_env_cache() -> None:
    _cache.clear()


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Selects the attention backend implementation registered for the
    # CUSTOM backend. "exp" selects the experimental on-device KV cache
    # backend (spyre_attn_exp.py); any other value uses the default
    # backend (spyre_attn.py).
    "SPYRE_ATTN_IMPL": lambda: os.getenv("SPYRE_ATTN_IMPL", "default"),
    # If set, the experimental on-device KV cache scatter uses a per-token
    # spyre.overwrite_f path instead of the default two-bmm placement.
    # Requires PR #2084 (specialize_int=True) applied to torch-spyre or
    # the kernel will reuse the first call's offsets.
    "SPYRE_SCATTER_USE_OVERWRITE": lambda: bool(int(os.getenv("SPYRE_SCATTER_USE_OVERWRITE", "0"))),
    # Enable file-based KV cache transfer for disaggregated prefill-decode.
    "VLLM_SPYRE_ENABLE_FILE_TRANSFER": lambda: bool(
        int(os.getenv("VLLM_SPYRE_ENABLE_FILE_TRANSFER", "0"))
    ),
    # Enable NIXL-based KV cache transfer for disaggregated prefill-decode.
    "VLLM_SPYRE_ENABLE_NIXL_TRANSFER": lambda: bool(
        int(os.getenv("VLLM_SPYRE_ENABLE_NIXL_TRANSFER", "0"))
    ),
    # Use blocking mode for NIXL transfers. Set to 0 for non-blocking
    # async transfers.
    "VLLM_SPYRE_NIXL_BLOCKING_TRANSFER": lambda: bool(
        int(os.getenv("VLLM_SPYRE_NIXL_BLOCKING_TRANSFER", "1"))
    ),
    # Enable the worker-side KV connector bridge for disaggregated
    # prefill-decode.
    "VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE": lambda: bool(
        int(os.getenv("VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE", "1"))
    ),
    # Path to KV cache file for file-based transfer.
    "VLLM_SPYRE_KV_CACHE_FILE_PATH": lambda: os.getenv("VLLM_SPYRE_KV_CACHE_FILE_PATH"),
    # Remote IP address for NIXL transfer (decode connects to prefill).
    # Overridden per-request by kv_transfer_params from a routing proxy.
    "VLLM_SPYRE_NIXL_REMOTE_IP": lambda: os.getenv("VLLM_SPYRE_NIXL_REMOTE_IP", "10.130.2.89"),
    # KV role: 'kv_producer' (prefill) or 'kv_consumer' (decode).
    "VLLM_SPYRE_KV_ROLE": lambda: os.getenv("VLLM_SPYRE_KV_ROLE", ""),
    # KV store backend type ('heap' or 'host_memory').
    "VLLM_SPYRE_KV_STORE_BACKEND": lambda: os.getenv("VLLM_SPYRE_KV_STORE_BACKEND", "heap"),
    # Maximum bytes for the KV store (0 = unlimited).
    "VLLM_SPYRE_KV_STORE_MAX_BYTES": lambda: int(os.getenv("VLLM_SPYRE_KV_STORE_MAX_BYTES", "0")),
    # Unix socket path for the persistent KV service.
    "VLLM_SPYRE_KV_SERVICE_SOCKET": lambda: os.getenv("VLLM_SPYRE_KV_SERVICE_SOCKET"),
    # Maximum number of saved requests in the KV reuse registry.
    "VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE": lambda: int(
        os.getenv("VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE", "100")
    ),
    # Emit structured KV placement probe log lines.
    "VLLM_SPYRE_KV_PLACEMENT_PROBE_ENABLED": lambda: bool(
        int(os.getenv("VLLM_SPYRE_KV_PLACEMENT_PROBE_ENABLED", "0"))
    ),
    # Experimental: read/write KV directly from the Spyre heap.
    "VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE": lambda: bool(
        int(os.getenv("VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE", "0"))
    ),
    # Experimental: fail instead of falling back when heap KV is unavailable.
    "VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT": lambda: bool(
        int(os.getenv("VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT", "0"))
    ),
    # Directory to export heap KV debug dumps.
    "VLLM_SPYRE_HEAP_KV_EXPORT_DIR": lambda: os.getenv("VLLM_SPYRE_HEAP_KV_EXPORT_DIR"),
    # Directory containing heap KV performance descriptors.
    "VLLM_SPYRE_HEAP_KV_PERFDSC_DIR": lambda: os.getenv("VLLM_SPYRE_HEAP_KV_PERFDSC_DIR"),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str) -> Any:
    if name in _cache:
        return _cache[name]

    if name in environment_variables:
        value = environment_variables[name]()
        _cache[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
