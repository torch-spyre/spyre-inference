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

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_heap_kv_accessor_module() -> Any:
    module_path = Path(__file__).with_name("heap_kv_accessor.py")
    spec = importlib.util.spec_from_file_location("_spyre_heap_kv_accessor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load heap_kv_accessor module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HEAP_KV_ACCESSOR = _load_heap_kv_accessor_module()
_RUNTIME_WARMED = False


def _ensure_spyre_runtime_ready() -> dict[str, Any]:
    global _RUNTIME_WARMED
    if _RUNTIME_WARMED:
        return {
            "runtime_ready": True,
            "runtime_warmup_performed": False,
        }

    torch_spyre = importlib.import_module("torch_spyre")
    torch_spyre_c = importlib.import_module("torch_spyre._C")
    autoload = getattr(torch_spyre, "_autoload", None)
    if callable(autoload):
        autoload()

    orig_mode = os.environ.get("COMPILATION_MODE")
    os.environ["COMPILATION_MODE"] = "offline"
    try:
        x_cpu = torch.arange(256, dtype=torch.float16).reshape(4, 64)
        x_spyre = x_cpu.to("spyre")
        addr = torch_spyre_c.get_dmpa(x_spyre)
        roundtrip = torch_spyre_c.spyre_from_blob(
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

    _RUNTIME_WARMED = True
    return {
        "runtime_ready": True,
        "runtime_warmup_performed": True,
        "runtime_backend_name": str(torch._C._get_privateuse1_backend_name()),
        "runtime_warmup_device": str(x_spyre.device),
        "runtime_warmup_shape": list(x_spyre.shape),
        "runtime_warmup_equal": bool(torch.equal(roundtrip, x_cpu)),
        "torch_spyre": getattr(torch_spyre, "__file__", "<extension>"),
    }


def _load_plan(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_accessor(plan: dict[str, Any]) -> Any:
    _ensure_spyre_runtime_ready()
    dtype_name = str(plan["dtype"])
    if dtype_name != "torch.float16":
        raise ValueError(f"Unsupported dtype {dtype_name}; expected torch.float16")

    return _HEAP_KV_ACCESSOR.HeapKVAccessor.from_metadata(
        kv_heads=int(plan["kv_heads"]),
        block_size=int(plan["block_size"]),
        head_dim=int(plan["head_dim"]),
        dtype=torch.float16,
        perfdsc_dir=plan["perfdsc_dir"],
        export_dir=plan["export_dir"],
    )


def _probe() -> dict[str, Any]:
    torch_spyre = importlib.import_module("torch_spyre")
    torch_spyre_c = importlib.import_module("torch_spyre._C")
    result = {
        "action": "probe",
        "torch_version": torch.__version__,
        "torch_spyre": getattr(torch_spyre, "__file__", "<extension>"),
        "torch_spyre_c": getattr(torch_spyre_c, "__file__", "<extension>"),
    }
    result.update(_ensure_spyre_runtime_ready())
    return result


def _read(plan: dict[str, Any], data_path: str | Path) -> dict[str, Any]:
    accessor = _build_accessor(plan)
    arrays: dict[str, np.ndarray] = {}
    for block in plan["blocks"]:
        key = str(block["key"])
        tensor = accessor.read_block(
            layer_idx=int(block["layer_idx"]),
            kv_kind=str(block["kv_kind"]),
            block_id=int(block["block_id"]),
        )
        arrays[key] = tensor.numpy()

    np.savez(data_path, **arrays)
    return {
        "action": "read",
        "block_count": len(plan["blocks"]),
        "data_path": str(data_path),
    }


def _write(plan: dict[str, Any], data_path: str | Path) -> dict[str, Any]:
    accessor = _build_accessor(plan)
    with np.load(data_path) as payloads:
        for block in plan["blocks"]:
            key = str(block["key"])
            values = torch.from_numpy(np.array(payloads[key], copy=True))
            accessor.write_block(
                layer_idx=int(block["layer_idx"]),
                kv_kind=str(block["kv_kind"]),
                block_id=int(block["block_id"]),
                values=values,
            )

    return {
        "action": "write",
        "block_count": len(plan["blocks"]),
        "data_path": str(data_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental helper boundary for old-stack heap-backed KV block IO. "
            "This script is intended to run inside the dt-inductor env."
        )
    )
    parser.add_argument(
        "--action",
        choices=("probe", "read", "write"),
        required=True,
    )
    parser.add_argument("--plan", required=False, help="Path to the JSON plan file.")
    parser.add_argument(
        "--data",
        required=False,
        help="Path to the .npz payload file for read/write operations.",
    )
    args = parser.parse_args()

    if args.action == "probe":
        print(json.dumps(_probe(), sort_keys=True))
        return 0

    if not args.plan or not args.data:
        parser.error("--plan and --data are required for read/write")

    plan = _load_plan(args.plan)
    if args.action == "read":
        result = _read(plan, args.data)
    else:
        result = _write(plan, args.data)

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
