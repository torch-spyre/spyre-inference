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
from collections.abc import Mapping

import torch
from vllm.logger import init_logger

# What cmake/cpu_extension.cmake passes for _C. AVX512F alone is Knights
# Landing, which has no VL/BW/DQ and would SIGILL in this build.
_AVX512_BUILD_FLAGS = ("avx512_f", "avx512_vl", "avx512_bw", "avx512_dq")


def isa_from_capabilities(caps: Mapping[str, object]) -> tuple[bool, bool]:
    """``(avx512, avx2_fma)`` from ``torch.cpu.get_capabilities()``.

    That map comes from cpuinfo and ignores ``ATEN_CPU_CAPABILITY``. Pinning
    ATen's dispatch level must not hide an extension the CPU can run: rejection
    sampling has no Python path. ``avx512`` is the four flags ``_C`` is compiled
    with. ``avx2_fma`` is AVX2 plus FMA3, which ``_C_AVX2`` needs.
    """
    avx512 = all(bool(caps.get(flag)) for flag in _AVX512_BUILD_FLAGS)
    avx2_fma = bool(caps.get("avx2") and caps.get("fma3"))
    return avx512, avx2_fma


def sampling_extension_name(machine: str, *, avx512: bool, avx2_fma: bool) -> str:
    """Which csrc build this host may load: ``_C`` or ``_C_AVX2``.

    ``avx512`` and ``avx2_fma`` are hardware facts. ``_C_AVX2`` needs AVX2 and
    FMA; without FMA its exp would SIGILL. Power and Z always use ``_C``.
    """
    if machine == "x86_64":
        if avx512:
            return "_C"
        if avx2_fma:
            return "_C_AVX2"
        raise ImportError("host CPU cannot run the x86 sampling kernels")
    return "_C"


try:
    _avx512, _avx2_fma = isa_from_capabilities(torch.cpu.get_capabilities())
    _extension = sampling_extension_name(
        platform.machine(),
        avx512=_avx512,
        avx2_fma=_avx2_fma,
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
