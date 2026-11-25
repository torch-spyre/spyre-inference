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

"""Load compiled attention kernels from disk instead of tracing them again.

The attention kernels are module-level ``torch.compile(fullgraph=True,
dynamic=False)`` functions. Each new set of input shapes is a new Dynamo trace,
and a restart traces every one of them again even when the FX graph and
AOTAutograd caches hold the compiled code. ``AotKernel`` skips that: the first
time it sees a set of inputs it compiles them with ``aot_compile`` (Dynamo +
AOTAutograd + Inductor, as before), saves the result, and on the next start
loads it with ``torch.compiler.load_compiled_function``, which runs neither
Dynamo nor AOTAutograd.

Which file to load is decided by our own key, not by Dynamo's tensor guards
(the Spyre layout guard cannot be saved). The key holds every tensor's dtype,
device, shape, strides, storage offset, Spyre device layout, sharding and
``dynamic=`` range, which arguments share a storage, every non-tensor argument
by type and value, and the Spyre core cap. The file name also holds the
kernel's module and name, and a stamp of torch, torch-spyre, this package, the
Spyre compiler versions and the torch and torch-spyre configs, so a change to
any of them misses. The shape guards on the ``dynamic=`` dims (range, step,
equal sizes) are kept in the file and checked on every call.

Anything that fails falls back to the plain ``torch.compile`` function: a call
that cannot be keyed, a file that does not load, a failed guard check, a failed
compile, or a compiled kernel that raises. A kernel that failed is not used
again in this process and its file is renamed to ``*.bad``.
"""

import contextlib
import functools
import hashlib
import importlib.util
import inspect
import os
import platform
import sys
import threading
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Any

import torch
import torch._functorch.config
from vllm.logger import init_logger

logger = init_logger(__name__)

# Open compile_only() blocks in any thread of this process.
_compile_only_depth = 0
_compile_only_lock = threading.Lock()


@contextlib.contextmanager
def compile_only() -> Iterator[None]:
    """Mark a parallel precompile pool as open: no kernel is compiled or saved inside.

    The pool hands back stand-in graphs whose real code is compiled in a child
    process. The pool is process-wide, so this mark is too.
    """
    global _compile_only_depth
    with _compile_only_lock:
        _compile_only_depth += 1
    try:
        yield
    finally:
        with _compile_only_lock:
            _compile_only_depth -= 1


def _pool_open() -> bool:
    """True while any thread has a parallel precompile pool open."""
    if _compile_only_depth:
        return True
    # torch-spyre's fork_inner_compiles swaps this function for the whole process while
    # its pool is open, also for a caller that did not use compile_only().
    module = sys.modules.get("torch_spyre._inductor")
    inner = getattr(module, "_spyre_inner_compile", None)
    return inner is not None and getattr(inner, "__qualname__", "") != "_spyre_inner_compile"


def _keep_shape_guards(entries: Any) -> list[bool]:
    """Keep only the shape guards on dynamic dims; the key covers the rest."""
    return [entry.guard_type == "SHAPE_ENV" for entry in entries]


def _guard_check(fn: Any) -> Callable[[tuple, dict], bool] | None:
    """The shape guard check of ``fn`` as ``check(args, kwargs)``, or None when it has no
    guards. For keyword-only kernels it skips ``inspect.Signature.bind``, which costs
    more than the check itself on every call."""

    def bind_and_check(args: tuple, kwargs: dict) -> bool:
        return fn.guard_check(*args, **kwargs)

    try:
        manager = fn._artifacts.guard_manager
        root = manager.root
        if not (
            root.get_leaf_guards() or root.get_child_managers() or root.get_epilogue_lambda_guards()
        ):
            return None
        params = list(fn._artifacts.signature.parameters.values())
        closure = fn._artifacts.runtime_env.closure
    except Exception:
        return bind_and_check
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    var_keyword = inspect.Parameter.VAR_KEYWORD
    if closure or any(
        p.kind not in (keyword_only, var_keyword) or p.default is not p.empty for p in params
    ):
        return bind_and_check
    named = [p.name for p in params if p.kind == keyword_only]
    rest = next((p.name for p in params if p.kind == var_keyword), None)

    def check(args: tuple, kwargs: dict) -> bool:
        if args or not all(name in kwargs for name in named):
            return bind_and_check(args, kwargs)
        # The same locals Signature.bind would give.
        f_locals = {name: kwargs[name] for name in named}
        if rest is not None:
            f_locals[rest] = {k: v for k, v in kwargs.items() if k not in f_locals}
        elif len(kwargs) != len(named):
            return bind_and_check(args, kwargs)
        return manager.check(f_locals)

    return check


@contextlib.contextmanager
def _compile_mode() -> Iterator[None]:
    """Compile with inference mode off, as a plain ``torch.compile`` call does.

    ``torch.compile`` turns inference mode off for the compile (no_grad instead) and
    drops the inference flag from its fake inputs; ``aot_compile`` does neither. The
    flag is part of the FX graph cache key, so without this the graphs a precompile
    pool compiled under inference mode never match. The key of this cache is taken
    before, in the caller's mode, and the kernel still runs in the caller's mode.
    """
    from torch._dynamo.utils import (
        maybe_disable_inference_mode,
        maybe_disable_inference_mode_for_fake_prop,
    )

    with maybe_disable_inference_mode(), maybe_disable_inference_mode_for_fake_prop():
        yield


class AotKernel:
    """A ``torch.compile``'d kernel whose compiled code is kept on disk per input key."""

    def __init__(self, fn: FunctionType, compiled: Callable[..., Any]) -> None:
        self._fn = fn
        self._compiled = compiled
        # A second compile object for aot_compile only, which keeps just the shape
        # guards. The plain object keeps all its guards for the fallback.
        self._aot: Any = torch.compile(
            fn,
            dynamic=False,
            fullgraph=True,
            options={"guard_filter_fn": _keep_shape_guards},
        )
        # key -> (compiled function, its file or None, its shape guard check or None)
        self._loaded: dict[tuple, tuple[Any, str | None, Any]] = {}
        self._broken: set[tuple] = set()
        self._dir: str | None = None
        self._stamp: str | None = None
        # warmup.py resets the frame of this callable after the parallel precompile.
        self._torchdynamo_orig_callable = fn
        self.__name__ = getattr(fn, "__name__", "kernel")
        self._qualname = f"{fn.__module__}.{fn.__qualname__}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            key = _key(args, kwargs)
        except Exception:
            logger.warning_once("Attention AOT cache: cannot key a call, tracing instead.")
            return self._compiled(*args, **kwargs)
        entry = self._loaded.get(key)
        if entry is None:
            if key in self._broken:
                return self._compiled(*args, **kwargs)
            entry = self._load(key)
            if entry is not None:
                self._loaded[key] = entry
        if entry is not None:
            ran, out = self._run(key, entry, args, kwargs)
            return out if ran else self._compiled(*args, **kwargs)
        if key in self._broken or _pool_open():
            # Inside a precompile pool a compile would only give a stand-in graph.
            # The plain path is served by the pool; the kernel is saved on a later call.
            return self._compiled(*args, **kwargs)
        return self._compile_and_run(key, args, kwargs)

    def _run(self, key: tuple, entry: tuple, args: tuple, kwargs: dict) -> tuple[bool, Any]:
        """Run a compiled function. Returns (False, None) when the plain path must run."""
        fn, path, check = entry
        try:
            if check is not None and not check(args, kwargs):
                # A shape fact the kernel was traced under does not hold for this call.
                return False, None
            return True, fn(*args, **kwargs)
        except Exception:
            logger.warning(
                "Attention AOT cache: a compiled %s failed, tracing it instead.",
                self.__name__,
                exc_info=True,
            )
            self._drop(key, path)
            return False, None

    def _compile_and_run(self, key: tuple, args: tuple, kwargs: dict) -> Any:
        try:
            with _compile_mode():
                fn = self._aot.aot_compile((args, kwargs))
            # _run checks the guards itself, so a failed check takes the plain path.
            fn.disable_guard_check()
        except Exception:
            logger.warning(
                "Attention AOT cache: aot_compile failed for %s, tracing it instead.",
                self.__name__,
                exc_info=True,
            )
            self._broken.add(key)
            return self._compiled(*args, **kwargs)
        ran, out = self._run(key, (fn, None, _guard_check(fn)), args, kwargs)
        if not ran:
            return self._compiled(*args, **kwargs)
        if not _pool_open():
            self._loaded[key] = (fn, self._save(key, fn), _guard_check(fn))
        return out

    def _drop(self, key: tuple, path: str | None) -> None:
        """Never use this key's compiled function again, here or after a restart."""
        self._broken.add(key)
        self._loaded.pop(key, None)
        if path is not None:
            with contextlib.suppress(OSError):
                os.replace(path, path + ".bad")

    def _path(self, key: tuple) -> str:
        if self._dir is None:
            from torch._inductor.runtime.cache_dir_utils import cache_dir

            self._dir = os.path.join(cache_dir(), "spyre_attn_aot")
            os.makedirs(self._dir, exist_ok=True)
        if self._stamp is None:
            self._stamp = _stamp()
        digest = hashlib.sha256(
            repr((self._stamp, _configs(), self._qualname, key)).encode()
        ).hexdigest()
        return os.path.join(self._dir, f"{self.__name__}-{digest}.bin")

    def _load(self, key: tuple) -> tuple[Any, str, Any] | None:
        path = None
        try:
            path = self._path(key)
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                fn = torch.compiler.load_compiled_function(f, f_globals=self._fn.__globals__)
            saved, ours = fn._artifacts.original_code, self._fn.__code__
            if (saved.co_filename, saved.co_qualname) != (ours.co_filename, ours.co_qualname):
                raise ValueError(f"the file holds {saved.co_qualname}, not {ours.co_qualname}")
            fn.disable_guard_check()
            return fn, path, _guard_check(fn)
        except Exception:
            logger.warning(
                "Attention AOT cache: could not load %s, tracing it instead.",
                self.__name__,
                exc_info=True,
            )
            self._drop(key, path)
            return None

    def _save(self, key: tuple, fn: Any) -> str | None:
        try:
            path = self._path(key)
            fn.save_compiled_function(path)
            return path
        except Exception as e:
            # warning_once takes no exc_info: the error type keeps the line useful.
            logger.warning_once(
                "Attention AOT cache: could not save a compiled kernel (%s), it stays in "
                "memory for this run only.",
                type(e).__name__,
            )
            return None


def _key(args: tuple, kwargs: dict) -> tuple:
    """Everything the compiled code depends on, for one call. It runs on every call,
    so it stays one plain loop."""
    names = tuple(sorted(kwargs))
    parts: list[Any] = [names, len(args)]
    first_of: dict[int, int] = {}
    storage_of: dict[int, int] = {}
    helpers = _torch_spyre()
    for v in (*args, *(kwargs[name] for name in names)):
        if not isinstance(v, torch.Tensor):
            parts.append(_value_key(v))
            continue
        # Same tensor passed twice, or two views of one storage.
        same = first_of.setdefault(id(v), len(first_of))
        ptr = v.untyped_storage().data_ptr()
        # Empty storages all share address zero and alias nothing.
        slot = storage_of.setdefault(ptr, same) if ptr else same
        # A dim marked dynamic compiles once for its whole range, like Dynamo does.
        # Dynamo still treats sizes 0 and 1 apart, so they stay in the key. The shape
        # guards saved with the kernel check the rest of what Dynamo knew.
        dynamic = getattr(v, "_dynamo_dynamic_indices", None)
        shape = (
            tuple(s if i not in dynamic or s in (0, 1) else None for i, s in enumerate(v.shape))
            if dynamic
            else tuple(v.shape)
        )
        device = v.device
        parts.append(
            (
                same,
                slot,
                v.dtype,
                device,
                shape,
                # Always the full strides: tensors with the same layout can differ in
                # the strides of their size-one dims, and Dynamo's tensor guard sees it.
                v.stride(),
                v.storage_offset(),
                v.requires_grad,
                v.is_inference(),
                type(v),
                None if device.type in ("cpu", "meta") else _spyre_tensor_key(v, helpers),
            )
        )
    parts.append(
        (
            None if helpers is None else int(helpers[0].sencores),
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.get_default_dtype(),
        )
    )
    return tuple(parts)


@functools.cache
def _torch_spyre() -> tuple | None:
    """torch-spyre's config and tensor key helpers, or None without torch-spyre."""
    try:
        from torch_spyre._inductor import config
        from torch_spyre._monkey_patch import _reserved_dims_key, _sharding_key
    except ImportError:
        return None
    return config, _sharding_key, _reserved_dims_key


def _spyre_tensor_key(t: torch.Tensor, helpers: tuple | None) -> tuple:
    """What torch-spyre's own tensor guard checks: device layout, sharding, ranges."""
    if helpers is None:
        raise TypeError("a device tensor without torch-spyre")
    layout = t.device_tensor_layout()
    layout_key = (
        None
        if layout is None
        else (
            tuple(layout.device_size),
            tuple(layout.stride_map),
            str(layout.device_dtype),
            str(layout.element_arrangement),
        )
    )
    return layout_key, helpers[1](t), helpers[2](t)


def _value_key(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return (type(v).__name__, v)
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, (tuple, list)):
        return (type(v).__name__, tuple(_value_key(x) for x in v))
    if isinstance(v, dict):
        return ("dict", tuple((k, _value_key(x)) for k, x in sorted(v.items())))
    if isinstance(v, (torch.dtype, torch.device)):
        return (type(v).__name__, str(v))
    raise TypeError(f"no AOT cache key for a {type(v).__name__} argument")


def _stamp() -> str:
    """What the saved files depend on beyond one call and the configs: versions and
    sources. Taken once per process."""
    from torch._inductor.codecache import torch_key

    parts: list[Any] = [
        platform.python_version(),
        torch_key().hex(),
        sorted((k, v) for k, v in os.environ.items() if k.startswith("TORCH_SPYRE_")),
        _backend_versions(),
    ]
    for package in ("torch_spyre", "spyre_inference"):
        spec = importlib.util.find_spec(package)
        locations = list(spec.submodule_search_locations or []) if spec else []
        parts.append((package, [_files_stamp(root) for root in locations]))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _configs() -> Any:
    """The torch and torch-spyre configs a compile reads, minus the per-call core cap."""
    parts = [
        torch._inductor.config.get_config_copy(),
        torch._dynamo.config.get_config_copy(),
        torch._functorch.config.get_config_copy(),
    ]
    helpers = _torch_spyre()
    if helpers is not None:
        config = helpers[0].get_config_copy()
        parts.append({k: v for k, v in config.items() if k != "sencores"})
    return _plain(parts)


def _backend_versions() -> str | None:
    """The Spyre compiler versions file (deeptools, flex), when the image has one."""
    path = os.environ.get("LIB_VERSION_FILE")
    try:
        with open(path or "", "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _files_stamp(root: str) -> str:
    """Content of every source file and size and change time of every binary under
    ``root``."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if name.endswith(".py"):
                with open(path, "rb") as f:
                    h.update(f"{path}:".encode() + hashlib.sha256(f.read()).digest())
            elif name.endswith(".so"):
                st = os.stat(path)
                h.update(f"{path}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    """``value`` with anything but plain data replaced by its type name, so the
    stamp holds no memory addresses."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(repr(_plain(v)) for v in value)
    if isinstance(value, dict):
        return sorted((str(k), _plain(v)) for k, v in value.items())
    return type(value).__qualname__
