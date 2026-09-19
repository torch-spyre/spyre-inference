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

"""Unit tests for the pure-PyTorch slot-mapping patch.

``HAS_TRITON`` is always False on Spyre, so vLLM's slot-mapping Triton kernel is a
placeholder -- a bare undecorated function whose grid subscript raises ``TypeError:
'function' object is not subscriptable``. That happens at the *first decode step*, long
after startup, so these tests pin the patch to the attribute vLLM actually launches
rather than trusting it to have landed.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import (
    _SLOT_MAPPING_KERNEL_ATTR,
    _compute_slot_mapping_kernel,
    _patch_compute_slot_mapping,
)


def test_patched_attribute_is_the_one_vllm_launches():
    """A patch on a name nothing reads any more would bind a fresh module attribute and
    leave the real launch site on the Triton placeholder. vLLM has already moved this
    kernel twice, so assert the launch site still reads the name we patch."""
    from vllm.v1.worker import block_table

    launch_src = inspect.getsource(block_table.BlockTable.compute_slot_mapping)
    assert _SLOT_MAPPING_KERNEL_ATTR in launch_src, (
        f"block_table.BlockTable.compute_slot_mapping no longer launches "
        f"{_SLOT_MAPPING_KERNEL_ATTR}; update _patch_compute_slot_mapping"
    )

    register_src = inspect.getsource(block_table.BlockTable.__init__)
    assert f"{_SLOT_MAPPING_KERNEL_ATTR}.register_warmup" in register_src, (
        "warmup registration moved off the patched attribute; it would try to compile"
    )


def test_patch_installs_the_shim():
    from vllm.v1.worker import block_table

    _patch_compute_slot_mapping()

    assert getattr(block_table, _SLOT_MAPPING_KERNEL_ATTR) is _compute_slot_mapping_kernel


def test_patch_raises_when_the_kernel_moves(monkeypatch):
    """Silently skipping the patch defers the failure to the first decode step."""
    from vllm.v1.worker import block_table

    monkeypatch.delattr(block_table, _SLOT_MAPPING_KERNEL_ATTR)

    with pytest.raises(RuntimeError, match="Cannot find vLLM's slot-mapping kernel"):
        _patch_compute_slot_mapping()


def test_shim_matches_vllms_positional_launch_order():
    """vLLM launches the kernel with 14 positional arguments. The shim names its last
    five in Triton's constexpr style, so only their *order* keeps them aligned."""
    from vllm.v1.worker import block_table

    launched = inspect.signature(block_table.ComputeSlotMappingKernel.__call__).parameters
    shim = inspect.signature(_compute_slot_mapping_kernel.__call__).parameters

    # Compare position by position; the names differ in case by design.
    assert len(shim) == len(launched) - 1  # `self` is bound on the shim instance
    assert [name.lower() for name in shim] == [name for name in launched if name != "self"]


def test_shim_computes_the_slot_mapping_through_a_real_block_table():
    """End-to-end through vLLM's own BlockTable: slot = block_id * block_size + offset."""
    from vllm.v1.worker import block_table

    _patch_compute_slot_mapping()

    block_size, max_num_reqs, max_blocks_per_req, max_tokens = 16, 2, 4, 8
    table = block_table.BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_blocks_per_req,
        max_num_batched_tokens=max_tokens,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=block_size,
        cp_kv_cache_interleave_size=1,
    )

    # One request whose logical blocks live in physical blocks 7 and 3.
    table.add_row([7, 3], row_idx=0)
    table.commit_block_table(num_reqs=1)

    # Positions straddling the block boundary: 15 is the last slot of block 7,
    # 16 the first of block 3.
    positions = torch.tensor([14, 15, 16, 17], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)

    table.compute_slot_mapping(1, query_start_loc, positions)

    expected = torch.tensor(
        [7 * block_size + 14, 7 * block_size + 15, 3 * block_size, 3 * block_size + 1],
        dtype=torch.int64,
    )
    torch.testing.assert_close(table.slot_mapping.gpu[:4], expected)
