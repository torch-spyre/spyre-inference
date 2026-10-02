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

"""Loads the host sampling kernels built from csrc/ (``torch.ops._spyre_C``)."""

import platform

import torch

# x86 ships an AVX512 build and an AVX2 fallback, as vLLM's CPU backend does.
if platform.machine() == "x86_64" and not torch.cpu._is_avx512_supported():
    import spyre_inference._C_AVX2  # noqa: F401  # ty: ignore[unresolved-import]
else:
    import spyre_inference._C  # noqa: F401  # ty: ignore[unresolved-import]
