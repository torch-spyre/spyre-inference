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

libspyre_comms natively implements: `barrier`, `broadcast`, `send`/`recv`.
The `allreduce` and `reduce` overrides on `SpyreCommsContext` are still
throw-stubs. The `_allgather_base` entry point on the torch-spyre spyreccl
backend side is also still stubbed, so `dist.all_gather_into_tensor` does
not work. List-form `allgather` and `gather` are wired up but leave a
zero-size device output on the current build (a D2H copy of the result
fails with "device_address size must be greater than zero"), so they are
not usable natively yet either -- see the xfail-strict probes.

This class supplies:
  - A bounce-through-CPU `all_reduce`. Native allreduce is a throw-stub, and
    an on-device manual reduce (send/recv + broadcast + `add_`) is unreliable
    on the spyreccl backend, so we copy to CPU, all-reduce on gloo, copy back.
  - An `all_gather` that uses native list-form `dist.all_gather` and then
    concatenates on CPU (the base class routes through
    `dist.all_gather_into_tensor`, blocked by the `_allgather_base` stub).

Together these let the TP forward path run end-to-end without waiting for
the upstream comms-side implementations to land. `reduce_scatter` (not on
the TP forward path) raises a clear NotImplementedError.

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
      - `all_reduce` bounces through CPU and reduces on gloo (native
        allreduce is a throw-stub; an on-device manual reduce is unreliable).
      - `all_gather` uses native list-form allgather + a CPU-side concat.
      - `reduce_scatter` (not on the TP forward path) raises
        NotImplementedError describing what's needed to unblock it.
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
        # used to use is unreliable on the spyreccl backend: the on-device
        # `input_.add_(scratch)` reduction compiles to a fused kernel that
        # fails with "Interleaved not supported yet", and the p2p primitives
        # appear to return before data lands in the destination tensor, so
        # follow-up reads see stale values. The multi-backend device_group
        # (cpu:gloo,spyre:spyreccl) dispatches CPU tensors to gloo, which is
        # well-tested. Copy to CPU, all-reduce there, copy back. This also
        # removes the world_size==2-only restriction the manual reduce had.
        if input_.device.type == "spyre":
            cpu_input = input_.cpu()
            dist.all_reduce(cpu_input, group=self.device_group)
            input_.copy_(cpu_input)
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # The base class uses dist.all_gather_into_tensor; spyreccl's
        # _allgather_base is intentionally unimplemented, so use list-form
        # dist.all_gather instead.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)
        output_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(output_list, input_, group=self.device_group)
        # torch.cat on Spyre shifts the second operand by -(|A| mod 64) slots
        # along the concat dim when |A| is not stick-aligned; round-trip
        # through CPU until the upstream kernel is fixed.
        output_list_cpu = [t.to("cpu") for t in output_list]
        return torch.cat(output_list_cpu, dim=dim).to(input_.device)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            _spyre_collective_unsupported_message("reduce_scatter", self.world_size)
        )

    # `broadcast`, `send`, `recv` from DeviceCommunicatorBase route through
    # ops that are implemented in libspyre_comms, so we leave them alone.
