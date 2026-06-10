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

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def make_block_key(*, layer_idx: int, kv_kind: str, block_id: int) -> str:
    return f"layer{layer_idx}_{kv_kind}_block{block_id}"


class HeapKVHelperClient:
    def __init__(
        self,
        *,
        kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        perfdsc_dir: str | Path,
        export_dir: str | Path,
        dti_root: str | Path,
        dev_env_script: str | Path,
        timeout_s: int,
    ) -> None:
        self._kv_heads = int(kv_heads)
        self._block_size = int(block_size)
        self._head_dim = int(head_dim)
        self._dtype = dtype
        self._perfdsc_dir = str(perfdsc_dir)
        self._export_dir = str(export_dir)
        self._dti_root = str(dti_root)
        self._dev_env_script = str(dev_env_script)
        self._timeout_s = int(timeout_s)

        self._repo_root = Path(__file__).resolve().parents[5]
        self._helper_script = Path(__file__).with_name("heap_kv_helper.py")

    def _plan_dict(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "kv_heads": self._kv_heads,
            "block_size": self._block_size,
            "head_dim": self._head_dim,
            "dtype": str(self._dtype),
            "perfdsc_dir": self._perfdsc_dir,
            "export_dir": self._export_dir,
            "blocks": blocks,
        }

    def _run_helper(
        self,
        *,
        action: str,
        plan_path: str | Path | None = None,
        data_path: str | Path | None = None,
    ) -> str:
        home = os.environ.get("HOME", "")
        plan_arg = f" --plan {shlex.quote(str(plan_path))}" if plan_path else ""
        data_arg = f" --data {shlex.quote(str(data_path))}" if data_path else ""
        script = (
            "set -euo pipefail\n"
            f"export HOME={shlex.quote(home)}\n"
            f"export DTI_PROJECT_ROOT={shlex.quote(self._dti_root)}\n"
            f"source {shlex.quote(self._dev_env_script)}\n"
            "export TORCH_DEVICE_BACKEND_AUTOLOAD=0\n"
            f"export PYTHONPATH={shlex.quote(str(self._repo_root))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
            f"python {shlex.quote(str(self._helper_script))} --action {shlex.quote(action)}"
            f"{plan_arg}{data_arg}\n"
        )
        result = subprocess.run(
            ["bash", "-lc", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            raise RuntimeError(
                f"heap helper failed action={action} rc={result.returncode} "
                f"stdout={stdout!r} stderr={stderr!r}"
            )
        return result.stdout.strip()

    def probe(self) -> dict[str, Any]:
        stdout = self._run_helper(action="probe")
        return json.loads(stdout.splitlines()[-1])

    def read_blocks(
        self,
        block_refs: list[tuple[int, str, int]],
    ) -> dict[tuple[int, str, int], torch.Tensor]:
        blocks = [
            {
                "key": make_block_key(layer_idx=layer_idx, kv_kind=kv_kind, block_id=block_id),
                "layer_idx": int(layer_idx),
                "kv_kind": str(kv_kind),
                "block_id": int(block_id),
            }
            for layer_idx, kv_kind, block_id in block_refs
        ]
        with tempfile.TemporaryDirectory(prefix="spyre-heap-read-") as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            data_path = Path(tmpdir) / "payloads.npz"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(self._plan_dict(blocks), f)
            self._run_helper(action="read", plan_path=plan_path, data_path=data_path)
            tensors: dict[tuple[int, str, int], torch.Tensor] = {}
            with np.load(data_path) as payloads:
                for layer_idx, kv_kind, block_id in block_refs:
                    key = make_block_key(
                        layer_idx=layer_idx,
                        kv_kind=kv_kind,
                        block_id=block_id,
                    )
                    tensors[(layer_idx, kv_kind, block_id)] = torch.from_numpy(
                        np.array(payloads[key], copy=True)
                    )
            return tensors

    def write_blocks(
        self,
        block_values: dict[tuple[int, str, int], torch.Tensor],
    ) -> None:
        blocks: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        for (layer_idx, kv_kind, block_id), tensor in block_values.items():
            key = make_block_key(layer_idx=layer_idx, kv_kind=kv_kind, block_id=block_id)
            blocks.append(
                {
                    "key": key,
                    "layer_idx": int(layer_idx),
                    "kv_kind": str(kv_kind),
                    "block_id": int(block_id),
                }
            )
            arrays[key] = tensor.detach().cpu().numpy()

        with tempfile.TemporaryDirectory(prefix="spyre-heap-write-") as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            data_path = Path(tmpdir) / "payloads.npz"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(self._plan_dict(blocks), f)
            np.savez(data_path, **arrays)
            self._run_helper(action="write", plan_path=plan_path, data_path=data_path)
