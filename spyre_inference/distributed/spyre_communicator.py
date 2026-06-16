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

The currently-installed `libspyre_comms.so` only has `barrier`, `broadcast`,
single-tensor `allgather(Tensor&,Tensor&)`, and pairwise `send`/`recv`
actually implemented. The list-form `allgather`, `allreduce`, `gather`,
and `reduce` overrides on `SpyreCommsContext` are throw-stubs. The
`_allgather_base` entry point on the torch-spyre side is also stubbed
regardless of the comms lib.

That means the base `DeviceCommunicatorBase` collectives that route through
`dist.all_reduce`, `dist.all_gather_into_tensor`, `dist.all_gather` (list),
and `dist.gather` will all throw at runtime when the device_group's backend
is `spyreccl`.

This class supplies a hand-rolled fallback for `all_reduce` that works for
TP=2 by using send/recv + broadcast (all of which ARE implemented), so the
TP forward path can run end-to-end on two ranks without waiting for the
upstream comms-side implementations to land. Other collectives raise a
clear NotImplementedError describing what's needed to unblock them.

REPLACE-WITH-NATIVE markers below identify each fallback that should be
removed once libspyre_comms gains the corresponding native impl. The
intent is that when the comms library catches up, this whole file can
be deleted (or reduced to a couple of overrides for ops we genuinely
want to do differently for perf), and the platform's
`get_device_communicator_cls` can revert to the base class.

Per-op blockers:
  - all_reduce        : libspyre_comms native allreduce.
  - gather            : libspyre_comms native gather.
  - all_gather (list) : libspyre_comms list-form allgather. The single-
                        tensor `allgather(Tensor&,Tensor&)` IS implemented
                        in libspyre_comms but is not exposed via
                        torch-spyre's spyreccl backend.
  - reduce            : libspyre_comms native reduce. Not on the standard
                        TP forward path; we don't need it for #137.

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

        # REPLACE-WITH-NATIVE: when libspyre_comms gains a native allreduce
        # and a comms RPM containing it is available, drop this entire
        # override and let the base class call
        # `dist.all_reduce(input_, group=self.device_group)`.
        #
        # Bounce-through-CPU fallback. The send+recv+broadcast pattern this
        # used to use is unreliable on the spyreccl backend: those primitives
        # appear to return before data lands in the destination tensor, so
        # follow-up reads see stale values. The multi-backend device_group
        # (cpu:gloo,spyre:spyreccl) dispatches CPU tensors to gloo, which is
        # well-tested. Copy to CPU, all-reduce there, copy back.
        if input_.device.type == "spyre":
            cpu_input = input_.cpu()
            dist.all_reduce(cpu_input, group=self.device_group)
            input_.copy_(cpu_input)
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # CPU tensors dispatch through the gloo half of the multi-backend
        # `cpu:gloo,spyre:spyreccl` device_group, which implements allgather
        # fine. Only spyre tensors hit the spyreccl gap, so guard on device.
        # REPLACE-WITH-NATIVE: when libspyre_comms exposes the list-form
        # SpyreCommsContext::allgather (and torch-spyre's spyreccl backend
        # wires up `_allgather_base`), delete this override and let the
        # base class use `dist.all_gather_into_tensor`.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)
        raise NotImplementedError(
            _spyre_collective_unsupported_message(
                "allgather",
                self.world_size,
                blocker="libspyre_comms list-form allgather + torch-spyre _allgather_base",
            )
        )

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        # CPU tensors dispatch through gloo on the multi-backend device_group;
        # only spyre tensors need the spyreccl native gather we don't have yet.
        # REPLACE-WITH-NATIVE: when libspyre_comms supports gather natively,
        # delete this override.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().gather(input_, dst, dim)
        raise NotImplementedError(
            _spyre_collective_unsupported_message(
                "gather",
                self.world_size,
                blocker="libspyre_comms native gather impl",
            )
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_collective_unsupported_message("reduce_scatter", self.world_size)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms, so we leave them alone.
