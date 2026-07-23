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
list-form `allgather`, `gather`, and `allreduce`. The `_allgather_base`
entry point on the torch-spyre spyreccl backend side is still stubbed,
so `dist.all_gather_into_tensor` does not work.

This class supplies:
  - An `all_gather` that uses native list-form `dist.all_gather` because
    the base class routes through `dist.all_gather_into_tensor` (blocked
    by the `_allgather_base` stub).

Remaining per-op blockers (REPLACE-WITH-NATIVE markers below):
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
      - `all_gather` is overridden to use list-form `dist.all_gather`
        because `dist.all_gather_into_tensor` is blocked by a spyreccl stub.
      - Other broken collectives raise NotImplementedError describing
        what's needed to unblock them.
    """

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
        dist.all_gather(output_list, input_, group=self.device_group)
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
