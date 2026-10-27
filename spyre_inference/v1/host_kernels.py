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

"""Loads the vLLM host kernels compiled from vLLM's csrc/ (``torch.ops._C``)."""

import platform

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    # x86 ships an AVX512 build and an AVX2 fallback, as vLLM's CPU backend does.
    if platform.machine() == "x86_64" and not torch.cpu._is_avx512_supported():
        import spyre_inference._C_AVX2  # noqa: F401  # ty: ignore[unresolved-import]
    else:
        import spyre_inference._C  # noqa: F401  # ty: ignore[unresolved-import]
    _HAS_HOST_KERNELS = True
except (ImportError, AttributeError) as e:
    _HAS_HOST_KERNELS = False
    logger.warning(
        "Failed to import the host kernels extension (spyre_inference._C): %s. "
        "Falling back to PyTorch sampling, which is slower.",
        e,
    )

_SLOT_MAPPING_KERNEL = "vllm.v1.worker.block_table.ComputeSlotMappingKernel.kernel"
_slot_mapping_registered = False


def has_host_kernels() -> bool:
    return _HAS_HOST_KERNELS


# Named like the Triton kernel's parameters, so the dispatcher forwards by name.
def _compute_slot_mapping(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    slot_mapping_ptr,
    KV_CACHE_BLOCK_SIZE,
    TOTAL_CP_WORLD_SIZE,
    TOTAL_CP_RANK,
    CP_KV_CACHE_INTERLEAVE_SIZE,
    PAD_ID,
    BLOCK_SIZE,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context parallelism is not supported on Spyre."
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc_ptr,
        positions_ptr,
        block_table_ptr,
        slot_mapping_ptr,
        KV_CACHE_BLOCK_SIZE,
    )
    # The Triton kernel pads the tail; the C++ op only writes the scheduled tokens.
    slot_mapping_ptr[num_tokens:max_num_tokens].fill_(PAD_ID)


def register_slot_mapping_kernel() -> None:
    # vLLM 0.32's own CPU override passes the arguments out of order (vllm#61992).
    global _slot_mapping_registered
    if _slot_mapping_registered:
        return
    if not _HAS_HOST_KERNELS:
        raise RuntimeError(
            "spyre_inference._C is not built: the slot mapping kernel has no other "
            "implementation on Spyre. Reinstall spyre-inference and check the CMake output."
        )
    from vllm.triton_utils.dispatcher import register_kernels

    register_kernels({_SLOT_MAPPING_KERNEL: _compute_slot_mapping})
    _slot_mapping_registered = True
