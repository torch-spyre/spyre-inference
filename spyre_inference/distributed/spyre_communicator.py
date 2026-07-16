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

`all_reduce` and `all_gather` are overridden to run on the gloo CPU group
(D2H -> gloo collective -> H2D) rather than the spyreccl `device_group`. Two
reasons the spyreccl group can't be used directly:
  - libspyre_comms has no native `allreduce` (SpyreCommsContext::allreduce
    throws), and torch-spyre's spyreccl `_allgather_base` is stubbed (so
    `dist.all_gather_into_tensor`, which the base class uses, does not work).
  - Even the collectives libspyre_comms does implement require the tensor's
    device storage to be registered in its symbol table. Tensors that reach
    these methods under torch.compile are Inductor/scratchpad-allocated
    buffers that are NOT registered, so a spyreccl collective on them aborts
    with "The Nth envelope at rank R missing symbolic address".

Staging through gloo sidesteps both: gloo has working collectives and needs
no Spyre device-address registration, and it is correct for any Spyre tensor
(eager or compiled) at any world size. `reduce_scatter` is not on the TP
forward path and raises a clear NotImplementedError.

"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from spyre_inference.custom_ops.utils import convert


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
    """Spyre-specific DeviceCommunicator with gloo-staged fallbacks.

    - `all_reduce` and `all_gather` run on the gloo CPU group (CPU fallback;
      D2H -> gloo -> H2D), which works for any Spyre tensor at any world size.
    - `reduce_scatter` (not on the TP forward path) raises
      NotImplementedError describing what's needed to unblock it.
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
        cpu_input = convert(input_, device="cpu")
        dist.all_reduce(  # ty: ignore[possibly-missing-attribute]
            cpu_input, group=self.cpu_group
        )
        return convert(cpu_input, device=input_.device)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # All-gather, staged through the gloo CPU group.
        #
        # Two reasons we don't all_gather on the spyreccl `device_group`:
        #   1. The base class routes through `dist.all_gather_into_tensor`, whose
        #      spyreccl `_allgather_base` entry point is still stubbed.
        #   2. Even list-form `dist.all_gather` on the spyreccl group fails for
        #      the tensors that reach this method: they are compiled-graph /
        #      Inductor-allocated (or otherwise not eagerly-registered) Spyre
        #      buffers whose storage is NOT in libspyre_comms' symbol table, so
        #      the collective aborts with "The Nth envelope at rank R missing
        #      symbolic address" (same failure mode as all_reduce).
        #
        # The gloo CPU group has a working all_gather and needs no Spyre
        # device-address registration. Staging through host memory (D2H -> gloo
        # all_gather -> H2D) is correct for any Spyre tensor, eager or compiled,
        # at any world size. `dist.all_gather` mutates the output list in place,
        # so we build a fresh CPU output list and move the concatenation back.
        #
        # REPLACE-WITH-NATIVE: when torch-spyre wires up spyreccl
        # `_allgather_base` AND registers Inductor-allocated buffers in its
        # symbol table, delete this override and let the base class handle it.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)
        cpu_input = convert(input_, device="cpu")
        output_list = [torch.empty_like(cpu_input) for _ in range(self.world_size)]
        dist.all_gather(  # ty: ignore[possibly-missing-attribute]
            output_list, cpu_input, group=self.cpu_group
        )
        return convert(torch.cat(output_list, dim=dim), device=input_.device)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_collective_unsupported_message("reduce_scatter", self.world_size)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms, so we leave them alone.
