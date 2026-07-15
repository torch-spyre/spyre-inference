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

"""DeviceCommunicator override for IBM Spyre devices.

libspyre_comms natively implements: `barrier`, `broadcast`, `send`/`recv`,
list-form `allgather`, and `gather`. The `allreduce` and `reduce` overrides
on `SpyreCommsContext` are still throw-stubs. The `_allgather_base` entry
point on the torch-spyre spyreccl backend side is also still stubbed,
so `dist.all_gather_into_tensor` does not work.

This class supplies:
  - A hand-rolled `all_reduce` for TP=2 using send/recv + broadcast (native
    allreduce is not yet available).
  - An `all_gather` that uses native list-form `dist.all_gather` because
    the base class routes through `dist.all_gather_into_tensor` (blocked
    by the `_allgather_base` stub).

This class supplies a hand-rolled fallback for `all_reduce` that works for
TP=2 by using send/recv + broadcast (all of which ARE implemented), so the
TP forward path can run end-to-end on two ranks without waiting for the
upstream comms-side implementations to land. Other collectives raise a
clear NotImplementedError describing what's needed to unblock them.

Remaining per-op blockers (REPLACE-WITH-NATIVE markers below):
  - all_reduce : libspyre_comms native allreduce.
  - all_gather : torch-spyre spyreccl `_allgather_base` (so the base class
                 can use `dist.all_gather_into_tensor` directly).
  - reduce     : libspyre_comms native reduce (not on the TP forward path).

The companion test file `tests/test_spyre_comms_native_probes.py` runs
each native collective on a real spyreccl device_group and is xfail-strict.
When a comms RPM lands an impl, the corresponding probe flips to passing,
the strict-xfail fails CI, and that's the signal to delete the override
here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


# libspyre_comms enforces a per-message minimum size (128 bytes). Every
# TP all_reduce we expect to see in inference is on a hidden-state shard
# that comfortably exceeds this threshold (hidden_size=1024 float16 =
# 2048 bytes, well above the 128-byte threshold), so we don't pad here.
# If something smaller hits this path, the fallback will surface the
# comms-layer error verbatim and we add padding then.


def _spyre_collective_unsupported_message(
    op_name: str, world_size: int, blocker: str | None = None
) -> str:
    parts = [
        f"SpyreCommunicator: {op_name} is not natively available in the "
        "installed libspyre_comms (the corresponding "
        f"SpyreCommsContext::{op_name} method throws). ",
    ]
    if blocker is not None:
        parts.append(f"Blocked on: {blocker}. ")
    parts.append(
        f"No fallback is implemented for world_size={world_size}. Either "
        "wait for the upstream implementation to land + a comms RPM rebuild, "
        "or extend SpyreCommunicator with an additional manual fallback."
    )
    return "".join(parts)


class SpyreCommunicator(DeviceCommunicatorBase):
    """Spyre-specific DeviceCommunicator with manual fallbacks.

    See the module docstring for the full picture. In short:
      - `all_reduce` is overridden with a TP=2-only manual reduce-to-root
        + broadcast that uses send/recv. TP>2 raises.
      - All other broken collectives raise NotImplementedError describing
        what's needed to unblock them.
    """

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # World size 1 is a no-op for all_reduce; the base class doesn't
        # short-circuit here, so do it ourselves.
        if self.world_size == 1:
            return input_

        # TP all_reduce, staged through the gloo CPU group.
        #
        # Why not spyreccl directly: the spyreccl backend has no native
        # allreduce (SpyreCommsContext::allreduce throws). A hand-rolled
        # send/recv/broadcast on the spyreccl `device_group` works for
        # *eagerly-allocated* Spyre tensors, but the tensors that reach this
        # method under torch.compile are compiled-graph intermediate buffers
        # allocated by the Inductor/scratchpad allocator. Their device storage
        # is NOT registered in libspyre_comms' symbol table, so the collective
        # aborts with "The Nth envelope at rank R missing symbolic address".
        #
        # The gloo CPU group has a real, working allreduce and does not care
        # about Spyre device-address registration. Staging through host memory
        # (D2H -> gloo allreduce -> H2D) is correct for any Spyre tensor,
        # eager or compiled, at any world size. The hidden-state shards reduced
        # here are small (e.g. [16, 4096] fp16 = 128 KiB), so the extra copies
        # are cheap relative to the layer compute.
        #
        # `.to("cpu")` produces a fresh host tensor, so the returned value is a
        # new tensor (the custom-op all_reduce contract is out-of-place; we do
        # NOT mutate `input_`).
        #
        # REPLACE-WITH-NATIVE: once libspyre_comms gains a native allreduce
        # (and, for the compiled path, registers Inductor-allocated buffers in
        # its symbol table), delete this override and let the base class call
        # `dist.all_reduce(input_, group=self.device_group)`.
        cpu_input = input_.to("cpu")
        dist.all_reduce(  # ty: ignore[possibly-missing-attribute]
            cpu_input, group=self.cpu_group
        )
        return cpu_input.to(input_.device)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # The base class uses dist.all_gather_into_tensor which needs
        # _allgather_base in spyreccl is still stubbed. Use list-form
        # dist.all_gather instead (natively supported).
        # REPLACE-WITH-NATIVE: when torch-spyre wires up _allgather_base,
        # delete this override and let the base class handle it.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)
        output_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(  # ty: ignore[possibly-missing-attribute]
            output_list, input_, group=self.device_group
        )
        return torch.cat(output_list, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_collective_unsupported_message("reduce_scatter", self.world_size)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms, so we leave them alone.
