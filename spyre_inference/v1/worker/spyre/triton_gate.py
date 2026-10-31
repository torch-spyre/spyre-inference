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
from collections.abc import Callable
from typing import Any

import vllm.platforms

_IMPORTING = "vllm.triton_utils.importing"
# Triton-free; V2 runs there on the replacement kernels kernels.py binds (#1238).
_NO_TRITON_MACHINES = ("ppc64le", "s390x")
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


def check(runner: str) -> None:
    """Raise unless HAS_TRITON matches runner. On Power and Z, V2 gets past vLLM's
    "requires Triton" check; x86 V2 without triton-cpu still fails it."""
    from vllm.triton_utils import HAS_TRITON

    # importing.py has bound HAS_TRITON, so the patches have done their job.
    _unpatch(importlib.util, "find_spec")
    _unpatch(vllm.platforms, "vllm_version_matches_substr")
    want = runner == "v2" and has_triton_cpu()
    if want != HAS_TRITON:
        raise ValueError(
            f"SPYRE_MODEL_RUNNER={runner} needs HAS_TRITON={want}, but vLLM resolved "
            f"{HAS_TRITON}: set SPYRE_MODEL_RUNNER before vLLM is imported, and the same "
            "in every process."
        )
    if runner == "v2" and not HAS_TRITON and platform.machine() in _NO_TRITON_MACHINES:
        _allow_v2_without_triton()


def _allow_v2_without_triton() -> None:
    import vllm.config.vllm as config_module

    validate = config_module.VllmConfig._validate_v2_model_runner
    if hasattr(validate, _MARK):
        return

    def without_triton(self: Any) -> None:
        # Upstream's own check, with its HAS_TRITON read made true.
        with _has_triton(config_module):
            validate(self)

    setattr(without_triton, _MARK, validate)
    config_module.VllmConfig._validate_v2_model_runner = without_triton  # ty: ignore[invalid-assignment]


@contextlib.contextmanager
def _has_triton(module: Any) -> Any:
    saved = module.HAS_TRITON
    module.HAS_TRITON = True
    try:
        yield
    finally:
        module.HAS_TRITON = saved
