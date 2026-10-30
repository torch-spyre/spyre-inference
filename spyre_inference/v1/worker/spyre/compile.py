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

"""MRV2: suppress MRV2's whole-model stock compile.

Under STOCK_TORCH_COMPILE, upstream's load_model ends with
compile_model_with_stock_torch(self.model), which would trace the whole model over
the per-block graphs load.py has already installed, through vLLM's own backend
rather than Spyre's inductor. Spyre compiles in load.py through the shared
compile_for_spyre instead.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

from vllm.v1.worker.gpu import model_runner as upstream_model_runner


@contextlib.contextmanager
def no_stock_compile() -> Iterator[None]:
    """Make upstream's compile_model_with_stock_torch a no-op for the duration."""
    stock_compile = upstream_model_runner.compile_model_with_stock_torch

    def skip(model: Any, vllm_config: Any) -> None:
        pass

    upstream_model_runner.compile_model_with_stock_torch = skip  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        upstream_model_runner.compile_model_with_stock_torch = stock_compile
