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

The installed `libspyre_comms.so` (1.0.0-0.main.1+27.5f812da on this pod)
has only `barrier`, `broadcast`, single-tensor `allgather(Tensor&,Tensor&)`,
and pairwise `send`/`recv` actually implemented. The list-form `allgather`,
`allreduce`, `gather`, and `reduce` are throw-stubs in
`spyre_comms_internal::SpyreCommsContext`. The `_allgather_base` entry point
on the torch-spyre side is also stubbed regardless of the comms lib.

That means the base `DeviceCommunicatorBase` collectives that route through
`dist.all_reduce`, `dist.all_gather_into_tensor`, `dist.all_gather` (list),
and `dist.gather` will all throw at runtime when the device_group's backend
is `spyreccl`.

This class supplies a hand-rolled fallback for `all_reduce` that works for
TP=2 by using send/recv + broadcast (all of which ARE implemented), so the
TP forward path can run end-to-end on two ranks without waiting for the
upstream comms-side implementations to land. Other collectives raise a
clear NotImplementedError that names the upstream PR they're blocked on.

REPLACE-WITH-NATIVE markers below identify each fallback that should be
removed once the corresponding upstream change lands. The intent is that
when the comms library catches up, this whole file can be deleted (or
reduced to a couple of overrides for ops we genuinely want to do
differently for perf), and the platform's
`get_device_communicator_cls` can revert to the base class.

Upstream tracking:
  - all_reduce        : github.ibm.com/ai-chip-toolchain/spyre-comms PR #97
                        (dsolt/allreduce_work). Currently *draft* and
                        conflicting with main.
  - gather            : github.ibm.com/ai-chip-toolchain/spyre-comms PR #101
                        (dsolt/concat_based_gather). Open, non-draft, awaiting
                        review + Jenkins CI.
  - all_gather (list) : no PR identified at time of writing. The single-
                        tensor `allgather(Tensor&,Tensor&)` IS implemented
                        in libspyre_comms.so but is not exposed via
                        torch-spyre's spyre_ccl backend.
  - reduce            : no PR. `src/coll/reduce.cpp` doesn't exist either,
                        so the underlying algorithm is missing too. Note
                        that `reduce` is not on the standard TP forward
                        path; we don't need it for #137.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


# Spyre comms enforces a per-message minimum size (matches
# `InvalidTensorSizeException(message_size_bytes, 128)` in
# spyre-comms/src/context.cpp). Every TP all_reduce we expect to see in
# inference is on a hidden-state shard that comfortably exceeds this
# threshold (hidden_size float16 = 2 KB for hidden_size=1024), so we
# don't pad here. If something smaller hits this path, the fallback
# will surface the comms-layer error verbatim and we add padding then.


def _spyre_p2p_unsupported_message(op_name: str, world_size: int, pr: str | None) -> str:
    parts = [
        f"SpyreCommunicator: {op_name} is not natively available in the "
        "installed libspyre_comms.so (the corresponding "
        f"SpyreCommsContext::{op_name} method throws). ",
    ]
    if pr is not None:
        parts.append(f"Upstream tracking: {pr}. ")
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
      - All other broken collectives raise NotImplementedError with a
        pointer to the upstream PR that will fix them.
    """

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # World size 1 is a no-op for all_reduce; the base class doesn't
        # short-circuit here, so do it ourselves.
        if self.world_size == 1:
            return input_

        if self.world_size != 2:
            # REPLACE-WITH-NATIVE (spyre-comms PR #97): once `dist.all_reduce`
            # works on the spyreccl-backed device_group, delete this entire
            # `all_reduce` override and let the base class call
            # `dist.all_reduce(input_, group=self.device_group)`.
            raise NotImplementedError(
                _spyre_p2p_unsupported_message(
                    "allreduce",
                    self.world_size,
                    "github.ibm.com/ai-chip-toolchain/spyre-comms#97",
                )
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
        # REPLACE-WITH-NATIVE: spyre-comms PR #97 makes
        # SpyreCommsContext::allreduce real; once a comms RPM containing
        # that change is available, drop this branch.
        other = 1 - self.rank_in_group
        if self.rank_in_group == 0:
            scratch = torch.empty_like(input_)
            dist.recv(scratch, src=self.ranks[other], group=self.device_group)
            input_.add_(scratch)
        else:
            dist.send(input_, dst=self.ranks[other], group=self.device_group)
        dist.broadcast(input_, src=self.ranks[0], group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # REPLACE-WITH-NATIVE: when SpyreCommsContext::allgather(vector<>)
        # is implemented (no PR yet) and a comms RPM exposes it, delete
        # this override and let the base class use
        # `dist.all_gather_into_tensor`. Note the base class also relies
        # on `_allgather_base` being implemented in spyre_ccl.cpp itself,
        # which is currently a separate stub on the torch-spyre side.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_p2p_unsupported_message("allgather", self.world_size, None)
        )

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        # REPLACE-WITH-NATIVE: spyre-comms PR #101 wires up
        # SpyreCommsContext::gather. Once landed + RPM rebuilt, delete
        # this override.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_p2p_unsupported_message(
                "gather",
                self.world_size,
                "github.ibm.com/ai-chip-toolchain/spyre-comms#101",
            )
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_p2p_unsupported_message("reduce_scatter", self.world_size, None)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms.so, so we leave them alone.
