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
