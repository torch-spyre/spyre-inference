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

"""MRV2: set vllm.triton_utils.HAS_TRITON per runner by patching the two names
vllm/triton_utils/importing.py reads. register() can run inside that import (it
resolves the platform), so each patch is also applied to the half-executed module.
"""

import contextlib
import importlib.util
import platform
import sys
from collections.abc import Callable, Iterator
from typing import Any

import vllm.platforms

_IMPORTING = "vllm.triton_utils.importing"
_MARK = "_spyre_triton_gate"


def _patch(module: Any, name: str, wrap: Callable[[Any], Any]) -> None:
    # Once: vLLM calls register() on every plugin probe.
    original = getattr(module, name)
    if hasattr(original, _MARK):
        return
    replacement = wrap(original)
    setattr(replacement, _MARK, original)
    setattr(module, name, replacement)
    importing = sys.modules.get(_IMPORTING)
    if importing is not None and hasattr(importing, name):
        setattr(importing, name, replacement)


def _unpatch(module: Any, name: str) -> None:
    replacement = getattr(module, name)
    original = getattr(replacement, _MARK, None)
    if original is None:
        return
    setattr(module, name, original)
    importing = sys.modules.get(_IMPORTING)
    if getattr(importing, name, None) is replacement:
        setattr(importing, name, original)


def _from_importing() -> bool:
    # Frame 0 is this function, 1 the wrapper, 2 the wrapper's caller.
    return sys._getframe(2).f_globals.get("__name__") == _IMPORTING


def open() -> None:
    """V2: report a CPU build to importing.py only, so it takes vLLM's own
    triton-cpu branch while the platform probes still see the real build."""

    def wrap(matches: Callable[[str], bool]) -> Callable[[str], bool]:
        def cpu_for_importing(substr: str) -> bool:
            return (substr == "cpu" and _from_importing()) or matches(substr)

        return cpu_for_importing

    _patch(vllm.platforms, "vllm_version_matches_substr", wrap)


def close() -> None:
    """V1: hide triton from importing.py only; torch has already imported it, and
    triton-cpu's CPU driver would otherwise pass vLLM's check."""

    def wrap(find_spec: Callable[..., Any]) -> Callable[..., Any]:
        def hidden(name: str, package: str | None = None) -> Any:
            if name in ("triton", "pytorch-triton-xpu") and _from_importing():
                return None
            return find_spec(name, package)

        return hidden

    _patch(importlib.util, "find_spec", wrap)


def has_triton_cpu() -> bool:
    if platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None:
        return False
    try:
        return importlib.util.find_spec("triton.backends.cpu") is not None
    except ImportError:
        return False


def wants_triton(runner: str, mrv2_kernels: str) -> bool:
    """Only V2 on SPYRE_MRV2_KERNELS=triton runs Triton, and only on triton-cpu."""
    return runner == "v2" and mrv2_kernels == "triton"


def check(runner: str, mrv2_kernels: str) -> None:
    """Raise unless HAS_TRITON matches the runner and kernel source. V2 on the host
    kernels gets past vLLM's "requires Triton" check in Worker.__init__ on every
    machine; V2 on SPYRE_MRV2_KERNELS=triton without triton-cpu fails here."""
    from vllm.triton_utils import HAS_TRITON

    # importing.py has bound HAS_TRITON, so the patches have done their job.
    _unpatch(importlib.util, "find_spec")
    _unpatch(vllm.platforms, "vllm_version_matches_substr")
    triton = wants_triton(runner, mrv2_kernels)
    if triton and not has_triton_cpu():
        raise ValueError(
            "SPYRE_MRV2_KERNELS=triton needs triton-cpu, an x86-only dev dependency "
            "(`uv sync --group dev`); unset it to run MRV2 on the host kernels."
        )
    if triton != HAS_TRITON:
        raise ValueError(
            f"SPYRE_MODEL_RUNNER={runner} SPYRE_MRV2_KERNELS={mrv2_kernels} needs "
            f"HAS_TRITON={triton}, but vLLM resolved {HAS_TRITON}: set both before vLLM "
            "is imported, and the same in every process."
        )


@contextlib.contextmanager
def v2_worker_init() -> Iterator[None]:
    """Around upstream's Worker.__init__ for V2: its "Model Runner V2 requires Triton"
    check sees HAS_TRITON=True, since kernels.bind() replaces every kernel MRV2 launches.
    Entered by the worker itself, in whichever process builds it (the engine core does
    not rerun check_and_update_config)."""
    import vllm.v1.worker.gpu_worker as worker_module

    saved = worker_module.HAS_TRITON
    worker_module.HAS_TRITON = True
    try:
        yield
    finally:
        worker_module.HAS_TRITON = saved
