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
from vllm.logger import init_logger


def sampling_extension_name(machine: str, capability: str) -> str:
    """Which csrc build this host may load: ``_C`` or ``_C_AVX2``.

    ATen reports ``AVX2`` only when FMA3 is present too, and ``AVX512`` only
    with FMA3 as well. An AVX2 host without FMA therefore comes back as
    ``NO AVX`` and gets no x86 extension, instead of ``_C_AVX2``. Power and Z
    always use ``_C``.
    """
    if machine == "x86_64":
        if capability == "AVX512":
            return "_C"
        if capability == "AVX2":
            return "_C_AVX2"
        raise ImportError(f"host CPU capability {capability!r} cannot run the x86 sampling kernels")
    return "_C"


try:
    _extension = sampling_extension_name(
        platform.machine(), torch.backends.cpu.get_cpu_capability()
    )
    if _extension == "_C_AVX2":
        import spyre_inference._C_AVX2  # noqa: F401  # ty: ignore[unresolved-import]
    else:
        import spyre_inference._C  # noqa: F401  # ty: ignore[unresolved-import]
    _HAS_SAMPLING_KERNELS = True
except (ImportError, AttributeError) as e:
    _HAS_SAMPLING_KERNELS = False
    init_logger(__name__).warning(
        "Failed to import the sampling kernels extension (spyre_inference._C): %s. "
        "Ordinary sampling falls back to PyTorch; rejection sampling requires these kernels.",
        e,
    )


def has_sampling_kernels() -> bool:
    """Whether the csrc/ sampling kernels are built and loaded."""
    return _HAS_SAMPLING_KERNELS


# An older _C (greedy and Gumbel only) still imports. The top-k ops arrived
# later, so decide once here instead of raising on the first filter call.
_HAS_TOPK_TOPP_KERNELS = _HAS_SAMPLING_KERNELS and hasattr(
    torch.ops._spyre_C, "cpu_topk_topp_sampling"
)
if _HAS_SAMPLING_KERNELS and not _HAS_TOPK_TOPP_KERNELS:
    init_logger(__name__).warning(
        "The loaded spyre_inference._C has no cpu_topk_topp_sampling. "
        "Rebuild the csrc/ CPU extension to use the top-k / top-p kernels; "
        "until then those filters stay on the PyTorch path."
    )


def has_topk_topp_kernels() -> bool:
    """Whether this _C build has the top-k / top-p filter ops.

    False on a stale extension, so top-k / top-p fall back to PyTorch while
    greedy and the Gumbel kernel keep using the ops that build does have."""
    return _HAS_TOPK_TOPP_KERNELS
