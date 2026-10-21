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
    from vllm.triton_utils import HAS_TRITON, triton
    from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

    print(HAS_TRITON)
    print(type(triton).__name__)
    print(type(_gather_block_tables_kernel).__name__)
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
    return result.stdout.split()[-3:]


def _run(script: str, **env: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**_env(), **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


@_needs_triton_cpu
@pytest.mark.subprocess_isolated
def test_v2_binds_real_triton() -> None:
    assert _probe(SPYRE_MODEL_RUNNER="v2") == ["True", "module", "JITFunction"]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("cuda_visible_devices", [None, ""])
def test_v1_stays_triton_free(cuda_visible_devices: str | None) -> None:
    """An empty CUDA_VISIBLE_DEVICES is where vLLM's own check keeps Triton on;
    with triton-cpu installed, so does its single active CPU driver."""
    env = {"SPYRE_MODEL_RUNNER": "v1"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    assert _probe(**env) == ["False", "TritonPlaceholder", "function"]


@pytest.mark.subprocess_isolated
def test_v2_opens_the_gate_without_triton() -> None:
    """The open() path minus triton-cpu itself, so CI covers it without the wheel."""
    out = _run(
        """
        import sys

        from vllm.config import VllmConfig
        from vllm.triton_utils import HAS_TRITON

        from spyre_inference.v1.worker.spyre import triton_gate

        print(triton_gate.state(), HAS_TRITON)
        print(any(isinstance(f, triton_gate._HookFinder) for f in sys.meta_path))
        """,
        SPYRE_MODEL_RUNNER="v2",
    )
    assert out[-3:] == ["open", "False", "False"]


def test_gate_cannot_switch_once_set() -> None:
    state = triton_gate.state()
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
    if triton_gate.state() is None:
        pytest.skip("the Spyre plugin did not load in this process")
    assert not any(isinstance(f, triton_gate._HookFinder) for f in sys.meta_path)


@pytest.mark.subprocess_isolated
def test_a_bare_find_spec_does_not_consume_the_hook() -> None:
    """Only executing the module consumes the hook, so a speculative lookup doesn't
    leave the gate reporting a state it never applied."""
    out = _run(
        """
        import importlib.util
        import os
        import sys

        from spyre_inference.v1.worker.spyre import triton_gate

        hide = triton_gate._hide_triton
        ran = []
        triton_gate._hide_triton = lambda module: ran.append(module.__name__) or hide(module)
        triton_gate.close()
        # Not importlib.util.find_spec, which imports the parent package and so the
        # module itself.
        (finder,) = [f for f in sys.meta_path if isinstance(f, triton_gate._HookFinder)]
        (vllm_dir,) = importlib.util.find_spec("vllm").submodule_search_locations
        path = [os.path.join(vllm_dir, "triton_utils")]
        assert finder.find_spec("vllm.triton_utils.importing", path) is not None
        import vllm.triton_utils

        print(triton_gate.state(), ran)
        """,
        VLLM_PLUGINS="",
    )
    assert out[-2:] == ["closed", "['vllm.triton_utils.importing']"]


@pytest.mark.subprocess_isolated
def test_a_missed_hook_leaves_the_gate_unset() -> None:
    out = _run(
        """
        import sys

        from spyre_inference.v1.worker.spyre import triton_gate

        triton_gate.close()
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, triton_gate._HookFinder)]
        import vllm.triton_utils.importing

        print(triton_gate.state())
        """,
        VLLM_PLUGINS="",
    )
    assert out[-1] == "None"


def test_hide_triton_only_hides_it_from_vllm_importing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other callers, e.g. modules importing.py pulls in, keep seeing triton."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: name)
    module = type(sys)("probe")

    with triton_gate._hide_triton(module):
        find_spec = importlib.util.find_spec
        lookup = "find_spec('triton'), find_spec('torch')"
        assert eval(lookup, {"__name__": triton_gate._IMPORTING, "find_spec": find_spec}) == (
            None,
            "torch",
        )
        assert eval(lookup, {"__name__": "other", "find_spec": find_spec}) == ("triton", "torch")

    assert importlib.util.find_spec("triton") == "triton"
    assert module.find_spec is importlib.util.find_spec


@pytest.mark.subprocess_isolated
def test_a_failing_hook_leaves_the_gate_unapplied() -> None:
    out = _run(
        """
        import sys
        import types

        from spyre_inference.v1.worker.spyre import triton_gate

        def _fail(module):
            raise AttributeError(module.__name__)

        sys.modules[triton_gate._IMPORTING] = types.ModuleType(triton_gate._IMPORTING)
        sys.modules[triton_gate._IMPORTING].HAS_TRITON = False
        triton_gate._hide_triton = _fail
        try:
            triton_gate.close()
        except AttributeError:
            pass
        print(triton_gate.state(), triton_gate._hooks)
        """,
        VLLM_PLUGINS="",
    )
    assert out[-2:] == ["None", "{}"]
