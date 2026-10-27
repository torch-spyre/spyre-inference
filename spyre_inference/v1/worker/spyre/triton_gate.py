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

"""MRV2: scoped bypass of vllm.triton_utils.HAS_TRITON.

vllm.triton_utils binds real triton or placeholders once, at first import, so
the gate must act before then: register() calls it during platform resolution,
which runs earlier in every process. A meta-path hook rewrites HAS_TRITON as
vllm/triton_utils/importing.py executes.

open() (V2) accepts triton-cpu on x86. close() (V1) hides triton from vLLM:
triton-cpu's CPU driver would otherwise pass vLLM's own check.
"""

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import platform
import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

_IMPORTING = "vllm.triton_utils.importing"

_state: str | None = None
_hooks: dict[str, Callable[[ModuleType], contextlib.AbstractContextManager[None]]] = {}


class _HookLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, hook: Callable[..., Any]) -> None:
        self._loader = loader
        self._hook = hook

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        # Not in find_spec, so a lookup that never executes leaves the hook pending.
        _hooks.pop(module.__name__, None)
        try:
            with self._hook(module):
                self._loader.exec_module(module)
        finally:
            # Not from find_spec: importlib may still be iterating sys.meta_path there.
            if not _hooks:
                sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _HookFinder)]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _HookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        hook = _hooks.get(fullname)
        if hook is None:
            return None
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _HookLoader(spec.loader, hook)
                return spec
        return None


def _has_triton_cpu() -> bool:
    if platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None:
        return False
    try:
        return importlib.util.find_spec("triton.backends.cpu") is not None
    except ImportError:
        return False


@contextlib.contextmanager
def _accept_triton_cpu(module: ModuleType) -> Iterator[None]:
    yield
    module.HAS_TRITON = module.HAS_TRITON or _has_triton_cpu()


@contextlib.contextmanager
def _hide_triton(module: ModuleType) -> Iterator[None]:
    find_spec = importlib.util.find_spec

    # Hidden from importing.py's own frame only.
    def hidden(name: str, package: str | None = None) -> Any:
        if (
            name in ("triton", "pytorch-triton-xpu")
            and sys._getframe(1).f_globals.get("__name__") == _IMPORTING
        ):
            return None
        return find_spec(name, package)

    importlib.util.find_spec = hidden  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        importlib.util.find_spec = find_spec
        module.find_spec = find_spec


def _apply(
    state: str, hooks: dict[str, Callable[..., Any]], has_triton: Callable[[], bool]
) -> None:
    global _state
    if _state == state:
        return
    if _state is not None:
        raise RuntimeError(f"Triton gate is already {_state}; cannot switch it to {state}")
    importing = sys.modules.get(_IMPORTING)
    if importing is not None:
        if not hasattr(importing, "HAS_TRITON"):
            raise RuntimeError(
                f"{_IMPORTING} is mid-import, so the Triton gate cannot be {state} in this process"
            )
        if has_triton() != importing.HAS_TRITON:
            raise RuntimeError(
                f"{_IMPORTING} was imported before the Spyre plugin loaded, so the Triton "
                f"gate cannot be {state} in this process"
            )
    try:
        for name, hook in hooks.items():
            if name in sys.modules:
                with hook(sys.modules[name]):
                    pass
            else:
                _hooks[name] = hook
        if _hooks and not any(isinstance(f, _HookFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _HookFinder())
    except BaseException:
        _hooks.clear()
        raise
    _state = state


def state() -> str | None:
    """ "open" or "closed" once applied, else None, including when a hooked module
    was imported without the hook running."""
    if any(name in sys.modules for name in _hooks):
        return None
    return _state


def open() -> None:
    """Make HAS_TRITON accept triton-cpu on x86."""
    _apply("open", {_IMPORTING: _accept_triton_cpu}, _has_triton_cpu)


def close() -> None:
    """Keep vLLM Triton-free for the V1 runner, whatever triton is installed."""
    _apply("closed", {_IMPORTING: _hide_triton}, lambda: False)
