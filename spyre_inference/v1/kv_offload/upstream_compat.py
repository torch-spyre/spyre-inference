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

"""Locate `OffloadingConnectorWorker` across upstream module moves.

The class name is stable; its module is not. At vLLM v0.26.0 it lived in
`...v1/offloading/worker.py`; post-0.26 `main` also exposes it from
`...v1/offloading_connector.py`. Import by name from whichever module has it so
an upstream move is a clear ImportError here rather than an obscure failure at
serve time.
"""

from importlib import import_module

_CANDIDATE_MODULES = (
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
)

OffloadingConnectorWorker = None
for _module_name in _CANDIDATE_MODULES:
    try:
        _module = import_module(_module_name)
    except ImportError:
        continue
    OffloadingConnectorWorker = getattr(_module, "OffloadingConnectorWorker", None)
    if OffloadingConnectorWorker is not None:
        break

if OffloadingConnectorWorker is None:
    raise ImportError(
        "OffloadingConnectorWorker not found in any of "
        f"{_CANDIDATE_MODULES}; upstream vLLM moved or renamed it"
    )

__all__ = ["OffloadingConnectorWorker"]
