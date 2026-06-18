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
import json
import mmap
import os
import re
from pathlib import Path
from typing import Any

import torch


def resolve_heap_kv_paths(
    *,
    perfdsc_dir: str | None = None,
    export_dir: str | None = None,
) -> tuple[Path, Path]:
    perfdsc_root = perfdsc_dir or os.getenv("VLLM_SPYRE_HEAP_KV_PERFDSC_DIR", "")
    export_root = export_dir or os.getenv("VLLM_SPYRE_HEAP_KV_EXPORT_DIR", "")

    if not perfdsc_root:
        base = os.getenv("PERFDSC_DUMP_DIR", "perfdsc_dumps")
        base_path = Path(base)
        if base_path.name.startswith("execute_itr"):
            perfdsc_root = str(base_path)
        else:
            perfdsc_root = str(base_path / "execute_itr0")

    if not export_root:
        export_root = "export_dtcompiler/r0_1/export_deeprt"

    return Path(perfdsc_root), Path(export_root)


def get_layer_kv_tensor_info(perfdsc_dump_dir: str | Path) -> dict[str, dict[str, Any]]:
    perfdsc_dump_dir = str(perfdsc_dump_dir)
    node_tensor_prop_json = os.path.join(
        perfdsc_dump_dir, "sengraph", "nodeTensorProp_after_rcuOpt.json"
    )
    sdsc_dir = os.path.join(perfdsc_dump_dir, "sdsc")
    if not os.path.exists(node_tensor_prop_json) or not os.path.exists(sdsc_dir):
        raise FileNotFoundError(
            f"Missing perfdsc metadata under {perfdsc_dump_dir}: "
            f"expected {node_tensor_prop_json} and {sdsc_dir}"
        )

    with open(node_tensor_prop_json) as f:
        nodes = json.load(f)["NodeTensorProperties"]

    kv_tensors: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"paged_attn_store_*([0-9]*)-Scatter([KV])")

    for node in nodes:
        name = node["name"]
        if not name.startswith("paged_attn_store") or "OutputTensors" not in node:
            continue

        output_tensor = None
        for tensor in node["OutputTensors"]:
            if tensor.get("isKvCacheTensor"):
                output_tensor = tensor
                break
        if output_tensor is None:
            continue

        matches = pattern.findall(name)
        if not matches:
            continue
        layer_idx = int(matches[0][0]) if matches[0][0] else 0
        kv_kind = matches[0][1].lower()

        tensor_json = os.path.join(sdsc_dir, f"{name}.json")
        if not os.path.exists(tensor_json):
            raise FileNotFoundError(f"Missing tensor metadata file {tensor_json}")

        with open(tensor_json) as g:
            tensor_meta = json.load(g)[name]["datadscs_"]

        kv_tensor: dict[str, Any] = {"name": name}
        for item in tensor_meta:
            if name not in item:
                continue
            data = item[name]
            for labeled in data["labeledDs_"]:
                if labeled["ldsName_"] != f"{name}_out":
                    continue
                phys_addr = int(labeled["hbmStartAddress_"]["data_"]["[0]"])
                phys_size = int(labeled["hbmSize_"])
                kv_tensor["flit_offset"] = phys_addr - 4 * (1 << 27)
                kv_tensor["size"] = phys_size
                break
            if "flit_offset" in kv_tensor:
                break

        if "flit_offset" not in kv_tensor:
            raise RuntimeError(f"Failed to find flit_offset for {name}")

        kv_tensors[f"{layer_idx}.{kv_kind}"] = kv_tensor

    if not kv_tensors:
        raise RuntimeError(f"No paged_attn_store KV metadata found in {perfdsc_dump_dir}")
    return kv_tensors


def get_heap_hbm_start_bytes(segment_size_json: str | Path, *, flit_size: int = 128) -> int:
    with open(segment_size_json) as f:
        seg_dict = json.load(f)["memReqPerSegment"]

    total_flits = 1
    permanent_flits = 0
    for seg_idx in ("2", "3", "7"):
        permanent_flits += seg_dict[seg_idx]["size"]

    page_flits = mmap.PAGESIZE // flit_size
    total_flits += (permanent_flits + page_flits - 1) // page_flits * page_flits

    for seg_idx in ("0", "1"):
        total_flits += seg_dict[seg_idx]["size"]

    return total_flits * flit_size


class HeapKVAccessor:
    def __init__(
        self,
        *,
        kv_heads: int,
        offsets: dict[str, dict[str, Any]],
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        start_of_heap_bytes: int,
        flit_size: int = 128,
    ) -> None:
        if dtype != torch.float16:
            raise ValueError(f"Experimental heap accessor only supports torch.float16, got {dtype}")

        self._offsets = offsets
        self._flit_size = flit_size
        self._heap_addr = int(start_of_heap_bytes)
        self._block_size = int(block_size)
        self._head_dim = int(head_dim)
        self._kv_heads = int(kv_heads)
        self._dtype = dtype
        self._nbytes = torch.tensor([], dtype=dtype).element_size()

        self._C = importlib.import_module("torch_spyre._C")

    @classmethod
    def from_metadata(
        cls,
        *,
        kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        perfdsc_dir: str | Path,
        export_dir: str | Path,
    ) -> HeapKVAccessor:
        offsets = get_layer_kv_tensor_info(perfdsc_dir)
        start_of_heap_bytes = get_heap_hbm_start_bytes(Path(export_dir) / "segment_size.json")
        return cls(
            kv_heads=kv_heads,
            offsets=offsets,
            block_size=block_size,
            head_dim=head_dim,
            dtype=dtype,
            start_of_heap_bytes=start_of_heap_bytes,
        )

    def _block_addr(self, *, layer_idx: int, kv_kind: str, block_id: int) -> int:
        flit_offset = int(self._offsets[f"{layer_idx}.{kv_kind}"]["flit_offset"])
        addr = self._heap_addr
        addr += flit_offset * self._flit_size
        addr += block_id * self._kv_heads * self._block_size * self._head_dim * self._nbytes
        return addr

    def read_block(self, *, layer_idx: int, kv_kind: str, block_id: int) -> torch.Tensor:
        addr = self._block_addr(layer_idx=layer_idx, kv_kind=kv_kind, block_id=block_id)
        block = self._C.spyre_from_blob(
            addr,
            size=(self._kv_heads, self._block_size, self._head_dim),
            stride=(self._block_size * self._head_dim, self._head_dim, 1),
            dtype=self._dtype,
        )
        return block.to("cpu").permute(1, 0, 2).contiguous()

    def write_block(
        self,
        *,
        layer_idx: int,
        kv_kind: str,
        block_id: int,
        values: torch.Tensor,
    ) -> None:
        if values.shape != (self._block_size, self._kv_heads, self._head_dim):
            raise ValueError(
                "Expected block tensor shape "
                f"{(self._block_size, self._kv_heads, self._head_dim)}, got {tuple(values.shape)}"
            )

        addr = self._block_addr(layer_idx=layer_idx, kv_kind=kv_kind, block_id=block_id)
        block = self._C.spyre_from_blob(
            addr,
            size=(self._kv_heads, self._block_size, self._head_dim),
            stride=(self._block_size * self._head_dim, self._head_dim, 1),
            dtype=self._dtype,
        )
        block.copy_(values.permute(1, 0, 2).contiguous())
