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

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

from spyre_inference.distributed.kv_transfer.kv_connector.v1.heap_kv_accessor import (
    HeapKVAccessor,
)


class InProcessHeapKVClient:
    def __init__(
        self,
        *,
        kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        perfdsc_dir: str | Path,
        export_dir: str | Path,
    ) -> None:
        self._kv_heads = int(kv_heads)
        self._block_size = int(block_size)
        self._head_dim = int(head_dim)
        self._dtype = dtype
        self._perfdsc_dir = str(perfdsc_dir)
        self._export_dir = str(export_dir)

        self._runtime_ready = False
        self._probe_info: dict[str, object] | None = None
        self._accessor: HeapKVAccessor | None = None

    def _ensure_runtime_ready(self) -> dict[str, object]:
        if self._probe_info is not None:
            return self._probe_info

        torch_spyre = importlib.import_module("torch_spyre")
        autoload = getattr(torch_spyre, "_autoload", None)
        if callable(autoload):
            autoload()
        C = importlib.import_module("torch_spyre._C")

        orig_mode = os.environ.get("COMPILATION_MODE")
        os.environ["COMPILATION_MODE"] = "offline"
        try:
            x_cpu = torch.arange(256, dtype=torch.float16).reshape(4, 64)
            x_spyre = x_cpu.to("spyre")
            addr = C.get_dmpa(x_spyre)
            roundtrip = C.spyre_from_blob(
                addr,
                size=tuple(x_spyre.shape),
                stride=tuple(x_spyre.stride()),
                dtype=torch.float16,
            ).to("cpu")
        finally:
            if orig_mode is None:
                os.environ.pop("COMPILATION_MODE", None)
            else:
                os.environ["COMPILATION_MODE"] = orig_mode

        self._runtime_ready = True
        self._probe_info = {
            "torch_version": torch.__version__,
            "torch_file": torch.__file__,
            "torch_spyre": getattr(torch_spyre, "__file__", "<extension>"),
            "torch_spyre_c": getattr(C, "__file__", "<extension>"),
            "backend_name": str(torch._C._get_privateuse1_backend_name()),
            "device": str(x_spyre.device),
            "equal": bool(torch.equal(roundtrip, x_cpu)),
        }
        return self._probe_info

    def _ensure_accessor(self) -> HeapKVAccessor:
        if self._accessor is not None:
            return self._accessor

        self._ensure_runtime_ready()
        self._accessor = HeapKVAccessor.from_metadata(
            kv_heads=self._kv_heads,
            block_size=self._block_size,
            head_dim=self._head_dim,
            dtype=self._dtype,
            perfdsc_dir=self._perfdsc_dir,
            export_dir=self._export_dir,
        )
        return self._accessor

    def probe(self) -> dict[str, object]:
        return dict(self._ensure_runtime_ready())

    def read_blocks(
        self,
        block_refs: list[tuple[int, str, int]],
    ) -> dict[tuple[int, str, int], torch.Tensor]:
        accessor = self._ensure_accessor()
        return {
            (layer_idx, kv_kind, block_id): accessor.read_block(
                layer_idx=layer_idx,
                kv_kind=kv_kind,
                block_id=block_id,
            )
            for layer_idx, kv_kind, block_id in block_refs
        }

    def write_blocks(
        self,
        block_values: dict[tuple[int, str, int], torch.Tensor],
    ) -> None:
        accessor = self._ensure_accessor()
        for (layer_idx, kv_kind, block_id), values in block_values.items():
            accessor.write_block(
                layer_idx=layer_idx,
                kv_kind=kv_kind,
                block_id=block_id,
                values=values,
            )
