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

"""Byte-exact KV-page DMA between a Spyre device tensor and a shared host pool."""

import torch
import torch_spyre._C as torch_spyre_c  # ty: ignore[unresolved-import]


class SpyreKvDmaCopier:
    def copy_d2h(
        self,
        dev_tensor: torch.Tensor,
        pool: torch_spyre_c.SharedHostPool,
        slot_id: int,
        *,
        non_blocking: bool = False,
    ) -> None:
        torch_spyre_c.copy_tensor_raw(
            dev_tensor, pool, slot_id, to_device=False, non_blocking=non_blocking
        )

    def copy_h2d(
        self,
        dev_tensor: torch.Tensor,
        pool: torch_spyre_c.SharedHostPool,
        slot_id: int,
        *,
        non_blocking: bool = False,
    ) -> None:
        torch_spyre_c.copy_tensor_raw(
            dev_tensor, pool, slot_id, to_device=True, non_blocking=non_blocking
        )

    @staticmethod
    def slot_bytes_for(dev_tensor: torch.Tensor) -> int:
        return torch_spyre_c.get_composite_address(dev_tensor).total_size

    @staticmethod
    def create_or_attach_pool(
        name: str, num_slots: int, slot_bytes: int
    ) -> torch_spyre_c.SharedHostPool:
        return torch_spyre_c.SharedHostPool.create_or_attach(name, num_slots, slot_bytes)
