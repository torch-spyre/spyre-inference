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

"""kernels.bind() covers every Triton launch site MRV2 can reach on the pinned vLLM,
with each host op taking its kernel's arguments, so a new, renamed or re-signed
kernel fails here on the bump, not mid-serving."""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from spyre_inference.v1.worker.spyre import kernels

from .kernels import parity_cases, spec_decode_cases  # noqa: F401  (registers cases)

# Modules MRV2's runner reaches outside vllm/v1/worker/gpu.
_OUTSIDE_GPU = ("utils", "mamba_utils")
# The speculators Spyre builds on V2 (XPress, over DFlash), plus the shared verifier.
_SPEC_DECODE = "gpu.spec_decode"
_SPEC_DECODE_BOUND = tuple(
    f"{_SPEC_DECODE}.{m}"
    for m in (
        "rejection_sampler",
        "rejection_sampler_utils",
        "acceptance_estimator",
        "dflash.speculator",
        "xpress.kernels",
    )
)


def _is_jit(decorator: ast.expr) -> bool:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return isinstance(target, ast.Attribute) and target.attr == "jit"


def _launch_sites() -> set[str]:
    root = Path(importlib.util.find_spec("vllm.v1.worker").origin).parent  # ty: ignore
    files = [*(root / "gpu").rglob("*.py"), *(root / f"{m}.py" for m in _OUTSIDE_GPU)]
    sources = {".".join(p.relative_to(root).with_suffix("").parts): p.read_text() for p in files}
    # Parsing and walking dominate the runtime (the tests/mrv2 budget is 1s), so each module
    # is parsed at most once and walked only if its text has a launch.
    trees = {module: ast.parse(source) for module, source in sources.items() if "jit" in source}
    # Only a module-level kernel is a global kernels.bind() can replace.
    jitted = {
        node.name
        for tree in trees.values()
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and any(_is_jit(d) for d in node.decorator_list)
    }
    # Kernels can be launched from a module that imports them by name.
    launch = re.compile(rf"\b(?:{'|'.join(map(re.escape, jitted))})\[")
    launching = {
        module
        for module, source in sources.items()
        if (not module.startswith(_SPEC_DECODE) or module in _SPEC_DECODE_BOUND)
        and launch.search(source)
    }
    return {
        f"{module}.{node.func.value.id}"
        for module in launching
        for node in ast.walk(trees.get(module) or ast.parse(sources[module]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in jitted
    }


def test_every_launch_site_is_bound() -> None:
    bound = set(kernels.HOST_KERNELS) | set(kernels.UNSUPPORTED_KERNELS)
    sites = _launch_sites()
    assert sites == bound, f"unbound: {sorted(sites - bound)}, gone: {sorted(bound - sites)}"


@pytest.mark.parametrize("name", sorted(kernels.HOST_KERNELS))
def test_host_op_takes_the_kernel_arguments(name: str) -> None:
    kernel = getattr(kernels._module(name), name.rpartition(".")[2])
    if isinstance(kernel, kernels.HostKernel):  # bound by an earlier test in this process
        kernel = kernels._originals[name]
    assert kernels.op_params(kernels.HOST_KERNELS[name]) == kernels.kernel_params(kernel)


def test_every_host_kernel_has_a_parity_case() -> None:
    covered = {k for case in parity_cases.CASES.values() for k in case.kernels}
    assert set(kernels.HOST_KERNELS) <= covered, sorted(set(kernels.HOST_KERNELS) - covered)
