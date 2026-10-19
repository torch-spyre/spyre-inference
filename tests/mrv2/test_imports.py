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

"""Every spyre/ module imports against the pinned vLLM, and importing it has
no side effects: no torch.cuda shimming, no kernel binding, no Triton gate.
Those only run from SpyreWorkerV2.init_device, or from plugin load.
"""

import importlib
import pkgutil
import subprocess
import sys
import textwrap

import pytest

import spyre_inference.v1.worker.spyre as _spyre_pkg


def _module_names() -> list[str]:
    return [
        mod.name for mod in pkgutil.walk_packages(_spyre_pkg.__path__, _spyre_pkg.__name__ + ".")
    ]


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports_against_pinned_vllm(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.subprocess_isolated
def test_importing_the_package_has_no_side_effects() -> None:
    """Runs in a fresh interpreter: within this process, some other test
    module (e.g. test_worker.py) has already imported spyre/ by collection
    time, so a before/after check here would compare an already-mutated
    torch.cuda against itself and pass regardless of a real side effect."""
    script = textwrap.dedent(f"""
        import importlib
        import torch
        from vllm import triton_utils

        # device_shims.install() is the only thing allowed to touch these four.
        _cuda_names = ("Stream", "Event", "stream", "current_stream")
        cuda_before = {{name: getattr(torch.cuda, name) for name in _cuda_names}}
        has_triton_before = triton_utils.HAS_TRITON

        # kernels.bind()/triton_gate.open() only run from init_device or plugin
        # load; patched to raise so a future module that calls either of them
        # at import time fails this test instead of passing it silently.
        from spyre_inference.v1.worker.spyre import kernels, triton_gate

        def _raise(*args, **kwargs):
            raise AssertionError("importing must not bind kernels or open the Triton gate")

        kernels.bind = _raise
        triton_gate.open = _raise

        for name in {_module_names()!r}:
            importlib.import_module(name)

        for name, before in cuda_before.items():
            after = getattr(torch.cuda, name)
            assert after is before, f"importing must not install device_shims ({{name}})"
        assert triton_utils.HAS_TRITON == has_triton_before, (
            "importing must not open the Triton gate"
        )
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
