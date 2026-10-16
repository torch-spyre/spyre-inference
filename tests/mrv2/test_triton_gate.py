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

"""The Triton gate, checked in fresh interpreters: it acts once, at the first
import of vllm.triton_utils, so this process (whose plugin load has already
closed it) can't show either state."""

import importlib.util
import os
import platform
import subprocess
import sys
import textwrap

import pytest

from spyre_inference.v1.worker.spyre import triton_gate

_needs_triton_cpu = pytest.mark.skipif(
    platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None,
    reason="triton-cpu is x86-only and not installed",
)

_PROBE = textwrap.dedent("""
    import vllm
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.triton_utils import HAS_TRITON, triton
    from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

    config = VllmConfig()
    config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    print(HAS_TRITON)
    print(type(triton).__name__)
    print(type(_gather_block_tables_kernel).__name__)
    print("stock torch.compile" in config._get_v2_model_runner_unsupported_features())
""")


def _env() -> dict[str, str]:
    """Without the VLLM_USE_V2_MODEL_RUNNER an earlier in-process VllmConfig pinned."""
    return {k: v for k, v in os.environ.items() if k != "VLLM_USE_V2_MODEL_RUNNER"}


def _probe(**env: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={**_env(), **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()[-4:]


@_needs_triton_cpu
@pytest.mark.subprocess_isolated
def test_v2_binds_real_triton_and_accepts_stock_compile() -> None:
    assert _probe(SPYRE_MODEL_RUNNER="v2") == ["True", "module", "JITFunction", "False"]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("cuda_visible_devices", [None, ""])
def test_v1_stays_triton_free(cuda_visible_devices: str | None) -> None:
    """An empty CUDA_VISIBLE_DEVICES is where vLLM's own check keeps Triton on;
    with triton-cpu installed, so does its single active CPU driver."""
    env = {"SPYRE_MODEL_RUNNER": "v1"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    assert _probe(**env) == [
        "False",
        "TritonPlaceholder",
        "function",
        "True",
    ]


@pytest.mark.subprocess_isolated
def test_v2_drops_stock_compile_from_mrv2_unsupported_without_triton() -> None:
    """The open() path minus triton-cpu itself, so CI covers it without the wheel."""
    script = textwrap.dedent("""
        import sys

        import vllm
        import vllm.config.vllm as config_module
        from vllm.config.compilation import CompilationMode

        from spyre_inference.v1.worker.spyre import triton_gate

        # Stands in for triton-cpu, so MRV2's own validation passes.
        config_module.HAS_TRITON = True
        config = config_module.VllmConfig()
        config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
        print(triton_gate._state)
        print("stock torch.compile" in config._get_v2_model_runner_unsupported_features())
        print(any(isinstance(f, triton_gate._HookFinder) for f in sys.meta_path))
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**_env(), "SPYRE_MODEL_RUNNER": "v2"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-3:] == ["open", "False", "False"]


def test_gate_cannot_switch_once_set() -> None:
    state = triton_gate._state
    if state is None:
        pytest.skip("the Spyre plugin did not load in this process")
    same, other = (
        (triton_gate.close, triton_gate.open)
        if state == "closed"
        else (triton_gate.open, triton_gate.close)
    )
    same()
    with pytest.raises(RuntimeError, match=f"already {state}"):
        other()


def test_hook_finder_leaves_meta_path_once_used() -> None:
    if triton_gate._state is None:
        pytest.skip("the Spyre plugin did not load in this process")
    assert not any(isinstance(f, triton_gate._HookFinder) for f in sys.meta_path)
