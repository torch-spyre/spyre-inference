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

"""MRV2: scoped bypass of vllm.triton_utils.HAS_TRITON / VllmConfig._validate_v2_model_runner.

vllm.triton_utils binds real triton/tl or placeholders once, at first
import, from HAS_TRITON, and MRV2 decorates its kernels with @triton.jit at
that same import; open() must therefore run before vllm.triton_utils is
first imported in the process. Nothing in spyre/ can call it at the right
time: worker.py and model_runner.py already import
vllm.v1.worker.gpu.model_runner (which imports vllm.triton_utils) at
module scope. #1237's opt-in switch owns the real call site, from
spyre_inference/__init__.py, before anything imports this package.
"""


def open() -> None:
    """Make HAS_TRITON / _validate_v2_model_runner accept triton-cpu on x86."""
    raise NotImplementedError


def close() -> None:
    """Undo open(), restoring the stock gate."""
    raise NotImplementedError
