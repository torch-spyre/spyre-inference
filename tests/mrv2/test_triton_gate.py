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

"""The Triton gate, in fresh interpreters: vllm.triton_utils binds HAS_TRITON once."""

import importlib.util
import os
import platform
import subprocess
import sys
import textwrap
import types

import pytest
import vllm.platforms

from spyre_inference.v1.worker.spyre import triton_gate

_HAS_TRITON_CPU = triton_gate.has_triton_cpu()

# What runs before the first vllm.triton_utils import. The first two resolve the
# platform, and so run register(), inside that import.
_FIRST_IMPORTS = {
    "bare_import": "import vllm",
    "triton_utils_first": "",
    "vllm_config": "from vllm.config import VllmConfig",
    "llm": "from vllm import LLM",
    "current_platform": "from vllm.platforms import current_platform\ncurrent_platform.device_type",
}

_PROBE = """
{first}
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

print(HAS_TRITON, type(triton).__name__, type(_gather_block_tables_kernel).__name__)
"""


def test_x86_has_triton_cpu() -> None:
    """Without the wheel HAS_TRITON is False either way, so on x86 every gate test below
    would pass without exercising the gate. The dev group installs it there."""
    assert platform.machine() != "x86_64" or _HAS_TRITON_CPU


def _run(script: str, **env: str) -> list[str]:
    # Without the VLLM_USE_V2_MODEL_RUNNER an earlier in-process VllmConfig pinned.
    base = {k: v for k, v in os.environ.items() if k != "VLLM_USE_V2_MODEL_RUNNER"}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**base, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def _probe(first: str, **env: str) -> list[str]:
    return _run(_PROBE.format(first=first), **env)[-3:]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("cuda_visible_devices", [None, ""])
@pytest.mark.parametrize("first", _FIRST_IMPORTS)
def test_v1_stays_triton_free(first: str, cuda_visible_devices: str | None) -> None:
    """An empty CUDA_VISIBLE_DEVICES is where vLLM's own check keeps Triton on."""
    env = {"SPYRE_MODEL_RUNNER": "v1"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    assert _probe(_FIRST_IMPORTS[first], **env) == ["False", "TritonPlaceholder", "function"]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("first", _FIRST_IMPORTS)
def test_v2_on_host_kernels_stays_triton_free(first: str) -> None:
    """The default: kernels.bind() replaces every kernel, so vLLM never sees Triton."""
    assert _probe(_FIRST_IMPORTS[first], SPYRE_MODEL_RUNNER="v2") == [
        "False",
        "TritonPlaceholder",
        "function",
    ]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("first", _FIRST_IMPORTS)
def test_v2_on_triton_binds_real_triton_when_triton_cpu_is_installed(first: str) -> None:
    want = (
        ["True", "module", "JITFunction"]
        if _HAS_TRITON_CPU
        else ["False", "TritonPlaceholder", "function"]
    )
    env = {"SPYRE_MODEL_RUNNER": "v2", "SPYRE_MRV2_KERNELS": "triton"}
    assert _probe(_FIRST_IMPORTS[first], **env) == want


_V2_CONFIG_WITHOUT_TRITON = """
import importlib.util
import platform

platform.machine = lambda: {machine!r}
real = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a: None if name == "triton" else real(name, *a)

from vllm.config import VllmConfig

try:
    config = VllmConfig()
    print(config.use_v2_model_runner, config.parallel_config.worker_cls)
except Exception as e:
    print("refused", "requires Triton" in str(e) or "needs triton-cpu" in str(e))
"""


_V2_WORKER = ["True", "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2"]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("machine", ["ppc64le", "s390x", "x86_64"])
def test_v2_on_host_kernels_builds_without_triton(machine: str) -> None:
    out = _run(_V2_CONFIG_WITHOUT_TRITON.format(machine=machine), SPYRE_MODEL_RUNNER="v2")
    assert out[-2:] == _V2_WORKER


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("machine", ["ppc64le", "x86_64"])
def test_v2_on_triton_refuses_without_triton_cpu(machine: str) -> None:
    out = _run(
        _V2_CONFIG_WITHOUT_TRITON.format(machine=machine),
        SPYRE_MODEL_RUNNER="v2",
        SPYRE_MRV2_KERNELS="triton",
    )
    assert out[-2:] == ["refused", "True"]


def test_v2_worker_init_keeps_upstreams_other_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.v1.worker.gpu_worker as worker_module

    from spyre_inference.v1.worker.spyre.worker import SpyreWorkerV2

    calls = []

    def init(self, *args, **kwargs):
        calls.append(worker_module.HAS_TRITON)
        raise ValueError("Model Runner V2 does not yet support: pipeline parallelism")

    monkeypatch.setattr(worker_module.Worker, "__init__", init)
    monkeypatch.setattr(worker_module, "HAS_TRITON", False)

    with pytest.raises(ValueError, match="pipeline parallelism"):
        SpyreWorkerV2.__new__(SpyreWorkerV2).__init__()

    assert calls == [True], "upstream's __init__ ran without seeing HAS_TRITON=True"
    assert worker_module.HAS_TRITON is False


def _fresh_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unmarked find_spec: register() already patched the real one in this process."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: name)


def test_close_hides_triton_only_from_vllm_importing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_find_spec(monkeypatch)
    triton_gate.close()
    find_spec = importlib.util.find_spec

    lookup = "find_spec('triton'), find_spec('pytorch-triton-xpu'), find_spec('torch')"
    assert eval(lookup, {"__name__": triton_gate._IMPORTING, "find_spec": find_spec}) == (
        None,
        None,
        "torch",
    )
    assert eval(lookup, {"__name__": "other", "find_spec": find_spec}) == (
        "triton",
        "pytorch-triton-xpu",
        "torch",
    )


def test_patches_apply_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_find_spec(monkeypatch)
    triton_gate.close()
    once = importlib.util.find_spec
    triton_gate.close()
    assert importlib.util.find_spec is once


def test_close_reaches_a_half_imported_importing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_find_spec(monkeypatch)
    half = types.ModuleType(triton_gate._IMPORTING)
    half.find_spec = importlib.util.find_spec
    monkeypatch.setitem(sys.modules, triton_gate._IMPORTING, half)

    triton_gate.close()

    assert eval("find_spec('triton')", {"__name__": triton_gate._IMPORTING, **vars(half)}) is None


def test_open_reaches_a_half_imported_importing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm.platforms, "vllm_version_matches_substr", lambda substr: False)
    half = types.ModuleType(triton_gate._IMPORTING)
    half.vllm_version_matches_substr = vllm.platforms.vllm_version_matches_substr
    monkeypatch.setitem(sys.modules, triton_gate._IMPORTING, half)

    triton_gate.open()

    lookup = "vllm_version_matches_substr('cpu'), vllm_version_matches_substr('cuda')"
    assert eval(lookup, {**vars(half), "__name__": triton_gate._IMPORTING}) == (True, False)


def test_open_leaves_the_platform_probes_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm.platforms, "vllm_version_matches_substr", lambda substr: False)
    triton_gate.open()

    lookup = "vllm_version_matches_substr('cpu')"
    assert eval(lookup, vars(vllm.platforms)) is False
    assert eval(lookup, {**vars(vllm.platforms), "__name__": triton_gate._IMPORTING}) is True


def test_check_restores_the_patched_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once HAS_TRITON is bound, every later lookup in the process skips the wrappers,
    including the copies a half-imported importing.py holds."""
    find_spec = lambda name, package=None: name  # noqa: E731
    matches = lambda substr: False  # noqa: E731
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(vllm.platforms, "vllm_version_matches_substr", matches)
    half = types.ModuleType(triton_gate._IMPORTING)
    half.find_spec = find_spec
    half.vllm_version_matches_substr = matches
    monkeypatch.setitem(sys.modules, triton_gate._IMPORTING, half)
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", False)
    triton_gate.close()
    triton_gate.open()
    assert half.find_spec is not find_spec
    assert half.vllm_version_matches_substr is not matches

    triton_gate.check("v1", "host")

    assert importlib.util.find_spec is half.find_spec is find_spec
    assert vllm.platforms.vllm_version_matches_substr is half.vllm_version_matches_substr is matches
