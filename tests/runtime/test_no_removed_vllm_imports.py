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

"""Nothing of ours reaches the vLLM modules 0.34 deletes with the V1 model runner.

Scanned rather than imported with those modules blocked: on 0.33, vLLM's own MRV2 code
still imports some of them, so a runtime block fails inside vLLM. The scan also sees
imports inside functions and the dotted paths patches name as strings.
"""

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_REMOVED = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.cpu_model_runner",
    "vllm.v1.worker.block_table",
    "vllm.v1.sample.sampler",
    "vllm.v1.sample.metadata",
    "vllm.v1.sample.logits_processor",
    "vllm.v1.sample.rejection_sampler",
    "vllm.v1.spec_decode",
)


def _removed(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in _REMOVED)


def _references(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        else:
            continue
        found += [(node.lineno, name) for name in names if _removed(name)][:1]
    return found


def test_no_reference_to_a_module_vllm_034_removes() -> None:
    this = pathlib.Path(__file__).resolve()
    found = {
        str(path.relative_to(_REPO_ROOT)): refs
        for root in ("spyre_inference", "tests")
        for path in sorted((_REPO_ROOT / root).rglob("*.py"))
        if path != this and (refs := _references(ast.parse(path.read_text(), filename=str(path))))
    }
    assert not found, f"references to modules vLLM 0.34 removes: {found}"


def test_the_scan_sees_imports_inside_functions_and_patch_targets() -> None:
    source = (
        "def f():\n"
        "    from vllm.v1.worker import block_table\n"
        "patch = 'vllm.v1.sample.rejection_sampler.expand_kernel'\n"
    )
    assert [line for line, _ in _references(ast.parse(source))] == [2, 3]
