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

        if self.world_size != 2:
            # REPLACE-WITH-NATIVE: once libspyre_comms supports allreduce
            # natively, delete this entire `all_reduce` override and let
            # the base class call `dist.all_reduce(input_, group=self.device_group)`.
            raise NotImplementedError(
                _spyre_collective_unsupported_message(
                    "allreduce",
                    self.world_size,
                    blocker="libspyre_comms native allreduce impl",
                )
            )

        # We hand `input_` directly to dist.send/dist.recv, which require
        # contiguous storage. The base class's dist.all_reduce path has
        # its own internal handling, but ours doesn't.
        assert input_.is_contiguous(), (
            "SpyreCommunicator.all_reduce requires a contiguous input tensor; "
            f"got tensor with shape={tuple(input_.shape)} stride={input_.stride()}"
        )

        # TP=2 manual all_reduce.
        #
        # We verified on this pod that:
        #   - dist.send / dist.recv on the spyreccl group work for paired
        #     ranks at world_size=2.
        #   - dist.broadcast works on the spyreccl group at any world size.
        #   - The Spyre comms message matcher requires every p2p message
        #     to have an immediate matching counterpart across all ranks;
        #     only world_size=2 trivially satisfies that constraint with a
        #     single send/recv pair.
        #
        # Pattern:
        #   Rank 1 -> Rank 0 (send/recv).
        #   Rank 0 sums in place.
        #   Rank 0 -> all (broadcast).
        #
        # REPLACE-WITH-NATIVE: when libspyre_comms gains a native allreduce
        # and a comms RPM containing it is available, drop this branch.
        other = 1 - self.rank_in_group
        if self.rank_in_group == 0:
            scratch = torch.empty_like(input_)
            dist.recv(scratch, src=self.ranks[other], group=self.device_group)
            input_.add_(scratch)
        else:
            dist.send(input_, dst=self.ranks[other], group=self.device_group)
        dist.broadcast(input_, src=self.ranks[0], group=self.device_group)
        return input_

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        # REPLACE-WITH-NATIVE: when libspyre_comms supports gather natively,
        # delete this override.
        if self.world_size == 1:
            return input_
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
