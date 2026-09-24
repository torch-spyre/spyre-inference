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

"""Flag compiles of registered callables that should already have been compiled.

``watch`` registers a callable whose compiles are all expected to have happened by
the time ``arm`` is called; from then on a compile of one is reported, because it
costs a full Inductor compile wherever it lands. Callers decide when that is --
``TorchSpyreWorker`` arms once warmup returns.

``on_compile_start`` gives a ``compile_id`` whose last component is the frame's
compile count, so a first compile and a guard-triggered recompile are
distinguishable. The traced code object is what identifies the callable; Dynamo
passes no frame to start callbacks, so ``_traced_code`` reads it from Dynamo's own
frame. A watched kernel compiled without ``fullgraph`` also reaches the callback as
the resume frames a graph break creates, which ``_resume_label`` maps back to the
kernel that owns them.

torch-spyre executes eager aten ops by registering ``torch.compile(op)`` as the
``PrivateUse1`` kernel, so a new shape on any eager op compiles legitimately and
keeps doing so for the whole run. Those all trace through one shared torch frame
(``_EAGER_OP_TRACE_FILE``), which is how they are told apart from a block or kernel
compile and dropped.
"""

from __future__ import annotations

import enum
import sys
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from types import CodeType
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompileGuardLevel(enum.Enum):
    """How to react to a compile of a watched callable while armed."""

    OFF = "off"
    """Do not install the callback at all. Default."""
    WARN = "warn"
    """Log each distinct violation."""
    ERROR = "error"
    """Raise ``UnexpectedCompileError``."""


def parse_level(value: str) -> CompileGuardLevel:
    """Parse ``SPYRE_COMPILE_GUARD``, raising on a typo rather than ignoring it."""
    try:
        return CompileGuardLevel(value.strip().lower())
    except ValueError:
        valid = ", ".join(level.value for level in CompileGuardLevel)
        raise ValueError(f"Invalid SPYRE_COMPILE_GUARD={value!r}. Valid values: {valid}.") from None


class CompileKind(enum.Enum):
    """What kind of code object Dynamo was asked to trace."""

    EAGER_OP = "eager_op"
    """torch-spyre's per-aten-op compile. Expected for the whole run."""
    WATCHED = "watched"
    """A callable registered via ``watch``."""
    UNKNOWN = "unknown"
    """Neither. Reported but never fatal."""


class UnexpectedCompileError(AssertionError, RuntimeError):
    """A watched callable compiled while the guard was armed at ``error`` level.

    ``AssertionError`` is load-bearing: Dynamo re-raises whatever a start callback
    throws as ``InternalTorchDynamoError`` unless the type is on its passthrough
    list, which ``AssertionError`` is on and ``RuntimeError`` is not.
    """


# torch-spyre runs every eager aten op through its own torch.compile, and all of them
# are traced from this one torch-internal trampoline. Matching the file rather than
# the function name keeps this working if the function inside it is renamed.
_EAGER_OP_TRACE_FILE = "torch/_dynamo/external_utils.py"

try:
    from torch._dynamo.resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX as _RESUME_PREFIX
except ImportError:  # pragma: no cover - a torch refactor moving the constant
    # Same string, hard-coded: a wrong guess only costs resume-frame attribution,
    # which degrades to the UNKNOWN reporting this replaced.
    _RESUME_PREFIX = "torch_dynamo_resume_in"


class _CompileGuard:
    """Process-wide state; ``torch._dynamo``'s own state is per-process too."""

    def __init__(self) -> None:
        # Guards the registry and the reported-once set against the callback firing
        # on several threads at once. Held for dict and set operations only, never
        # across a compile: the callback runs at compile *start* and returns before
        # Inductor is invoked. It therefore cannot serialise torch-spyre's async
        # compiles, which in any case run DXP in a subprocess pool
        # (torch_spyre/execution/async_compile.py) that never re-enters Dynamo.
        self._lock = threading.Lock()
        self._watched: dict[CodeType, str] = {}
        # (co_filename, co_name) -> label, for resolving the resume frames a graph
        # break inside a watched callable creates. Keyed by name because those code
        # objects are synthesized at compile time, so `watch` never saw them.
        self._watched_names: dict[tuple[str, str], str] = {}
        self._level = CompileGuardLevel.OFF
        self._armed = False
        self._callback: Callable[[Any], None] | None = None
        # Per-thread, so allow_compile() on one thread does not blind the guard to a
        # compile triggered on another. Not a hard guarantee by itself: torch fires
        # start callbacks only when its process-wide pending counter goes 0 -> 1, so a
        # compile that begins while another is in flight is not reported at all. It
        # under-reports rather than over-reports, which is the safe direction.
        self._suppressed = threading.local()
        # One report per (label, is_recompile). A recompiling shape usually repeats
        # every step, and the point is to name it once, not to flood the log.
        self._reported: set[tuple[str, bool]] = set()

    def watch(self, target: object, label: str) -> None:
        """Register ``target``'s code object, so a compile of it is reported.

        Accepts a module (uses ``type(target).forward``), a bound or plain function,
        or a code object, and unwraps ``torch.compile`` results so registering
        before or after compiling both work.
        """
        with self._lock:
            for code in _code_objects(target):
                # First label wins: identical blocks share one code object, and the
                # first registration carries the more general name.
                self._watched.setdefault(code, label)
                # Graph breaks split a kernel into resume frames Dynamo synthesizes
                # fresh, so they are not in `_watched` and have to be matched by
                # (file, name) instead. See `_classify`.
                self._watched_names.setdefault((code.co_filename, code.co_name), label)

    def watched_labels(self) -> dict[CodeType, str]:
        with self._lock:
            return dict(self._watched)

    def arm(self, level: CompileGuardLevel) -> None:
        """Start reporting at ``level``. Idempotent; re-arming at a new level is allowed.

        ``OFF`` disarms, so ``arm`` always leaves the guard at the level asked for
        rather than keeping an earlier one.
        """
        if level is CompileGuardLevel.OFF:
            if self._armed:
                logger.debug("Compile guard disarmed by arm(OFF)")
                self.disarm()
                return
            logger.debug("Compile guard disabled (SPYRE_COMPILE_GUARD=off)")
            return

        with self._lock:
            from torch._dynamo.callback import callback_handler

            self._level = level
            self._reported.clear()
            watched = len(self._watched)
            # Not `self._armed` alone: torch._dynamo.reset() clears the handler's
            # list, so an armed guard can have lost its registration. Re-register
            # whenever the callback is not actually installed.
            registered = (
                self._callback is not None and self._callback in callback_handler.start_callbacks
            )
            already_armed = self._armed and registered
            if not registered:
                self._callback = self._on_compile_start
                callback_handler.register_start_callback(self._callback)
                self._armed = True

        if already_armed:
            logger.debug("Compile guard re-armed at level %s", level.value)
            return
        logger.info(
            "Compile guard armed at level '%s': %d watched compile sites. A further "
            "compile of one is %s.",
            level.value,
            watched,
            "a hard error" if level is CompileGuardLevel.ERROR else "logged",
        )

    def disarm(self) -> None:
        """Stop reporting. Safe to call when not armed."""
        with self._lock:
            self._level = CompileGuardLevel.OFF
            if not self._armed:
                return
            from torch._dynamo.callback import callback_handler

            if self._callback is not None:
                # torch._dynamo.reset() calls callback_handler.clear(), dropping the
                # registration without telling us, so the callback may already be
                # gone. Removal is by value, and it is not an error for it to have
                # been unregistered underneath us.
                try:
                    callback_handler.remove_start_callback(self._callback)
                except ValueError:
                    logger.debug("Compile guard callback was already unregistered")
                self._callback = None
            self._armed = False

    def reset(self) -> None:
        """Full teardown, including the registry. For tests."""
        self.disarm()
        with self._lock:
            self._watched.clear()
            self._watched_names.clear()
            self._reported.clear()
        self._suppressed.depth = 0

    def clear_reported(self) -> None:
        """Forget which violations were already logged, keeping the registry.

        One report per (label, kind) is what keeps a recompiling shape from flooding
        the log, which across tests would mute a report the next one expects.
        """
        with self._lock:
            self._reported.clear()

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def level(self) -> CompileGuardLevel:
        return self._level

    @contextmanager
    def allow_compile(self, reason: str) -> Generator[None, None, None]:
        """Permit compiles on this thread, for a phase that compiles on purpose."""
        depth = getattr(self._suppressed, "depth", 0)
        self._suppressed.depth = depth + 1
        if self._armed and depth == 0:
            logger.debug("Compile guard suspended: %s", reason)
        try:
            yield
        finally:
            # Restore the saved depth rather than decrementing whatever is there
            # now: `reset()` inside the block sets it to 0, and decrementing that
            # would leave -1, which suppresses nothing and never recovers because
            # thread-locals outlive the call.
            self._suppressed.depth = depth

    def _on_compile_start(self, args: Any) -> None:
        """Classify and report one compile. Runs inside Dynamo, so it stays cheap and
        raises only for a deliberate violation."""
        if getattr(self._suppressed, "depth", 0) > 0:
            return

        compile_id = str(getattr(args, "compile_id", "?"))
        kind, label = self._classify(_traced_code())
        if kind is CompileKind.EAGER_OP:
            return

        is_recompile = _is_recompile(compile_id)
        with self._lock:
            level = self._level
            first_report = (label, is_recompile) not in self._reported
            self._reported.add((label, is_recompile))

        what = "recompiled" if is_recompile else "compiled"
        described = f"{label} {what} unexpectedly (compile_id={compile_id})"

        # An unknown frame is never fatal: were a torch upgrade to move the eager-op
        # trampoline, every eager op would land here, and killing the engine over
        # that is far worse than a noisy log.
        if level is CompileGuardLevel.ERROR and kind is CompileKind.WATCHED:
            raise UnexpectedCompileError(
                f"{described}. Every compile of this callable was expected to have "
                "happened already, so this costs a full Inductor compile here. "
                "Re-run with TORCH_LOGS=recompiles to see which guard failed, or set "
                "SPYRE_COMPILE_GUARD=warn to downgrade this to a log line."
            )
        if first_report:
            logger.warning(
                "%s. This costs a full Inductor compile here; re-run with "
                "TORCH_LOGS=recompiles to see which guard failed.",
                described,
            )

    def _classify(self, code: CodeType | None) -> tuple[CompileKind, str]:
        if code is None:
            return CompileKind.UNKNOWN, "<unknown frame>"
        with self._lock:
            label = self._watched.get(code)
            if label is None:
                label = self._resume_label(code)
        if label is not None:
            return CompileKind.WATCHED, label
        # Normalised so the match holds on Windows-style separators too.
        if code.co_filename.replace("\\", "/").endswith(_EAGER_OP_TRACE_FILE):
            return CompileKind.EAGER_OP, "torch-spyre eager op"
        return CompileKind.UNKNOWN, f"{code.co_filename}:{code.co_name}"

    def _resume_label(self, code: CodeType) -> str | None:
        """The watched label owning ``code``, when it is a resume frame of one.

        A watched kernel compiled without ``fullgraph`` splits at every graph break
        into ``torch_dynamo_resume_in_<parent>_at_<lineno>`` code objects that Dynamo
        synthesizes, so ``watch`` cannot have registered them. Left unresolved they
        classify as ``UNKNOWN``, which is never fatal -- so a post-warmup recompile
        confined to the resume half of a watched kernel would slip past ``error``.

        Unwraps repeatedly: resuming a resume frame nests the prefix again.

        Caller holds the lock.
        """
        name = code.co_name
        # Bounded by the nesting actually present in the name, and each turn strips
        # at least the prefix, so this cannot spin.
        while name.startswith(_RESUME_PREFIX):
            # One separator underscore, then the parent name, then "_at_<lineno>";
            # rpartition so a parent whose own name contains "_at_" survives.
            name, _, _ = name[len(_RESUME_PREFIX) + 1 :].rpartition("_at_")
            if not name:
                return None
            label = self._watched_names.get((code.co_filename, name))
            if label is not None:
                return label
        return None


def _is_recompile(compile_id: str) -> bool:
    """True when Dynamo's compile id says the frame had already been compiled.

    ``CompileId.__str__`` emits ``"<frame>/<frame_compile>"``, or
    ``"!<autograd>/<frame>/<frame_compile>"`` when compiling a compiled-autograd
    graph, so the *last* component is the per-frame compile count. Anything
    unparsable is treated as a first compile, which only ever under-reports
    severity.
    """
    body = compile_id[1:].partition("/")[2] if compile_id.startswith("!") else compile_id
    try:
        return int(body.rpartition("/")[2]) > 0
    except ValueError:
        return False


def _traced_code() -> CodeType | None:
    """The code object Dynamo is about to trace, read from its own frame.

    Best-effort: a torch refactor yields ``None``, which classifies as ``UNKNOWN``
    and is reported rather than fatal.
    """
    frame: Any = sys._getframe()
    while frame is not None:
        code = frame.f_code
        if code.co_name == "_compile" and code.co_filename.replace("\\", "/").endswith(
            "torch/_dynamo/convert_frame.py"
        ):
            # `code`, not `frame.f_code`: _compile takes the code object as its first
            # required parameter, while its `frame` parameter is optional and None for
            # the debug replay() helper.
            traced = frame.f_locals.get("code")
            return traced if isinstance(traced, CodeType) else None
        frame = frame.f_back
    return None


def _code_objects(target: object) -> Iterable[CodeType]:
    """Code objects to register for ``target``, unwrapping compile wrappers.

    Dynamo guards on the *original* function's frame, so that is what the callback
    sees. ``OptimizedModule`` keeps it at ``_orig_mod``; a compiled function keeps it
    at ``_torchdynamo_orig_callable``.
    """
    if isinstance(target, CodeType):
        yield target
        return

    seen: set[int] = set()
    queue = [target]
    while queue:
        obj = queue.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))

        for attr in ("_orig_mod", "_torchdynamo_orig_callable", "__wrapped__", "__func__"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                queue.append(inner)

        code = getattr(obj, "__code__", None)
        if isinstance(code, CodeType):
            yield code
            continue

        # A module: Dynamo traces the entry point, looked up on the class, so every
        # instance of that class shares the code object registered here. `forward` on
        # an OptimizedModule is an *instance* attribute, so type(obj).forward falls
        # through to nn.Module's abstract placeholder; skipping it keeps that
        # never-traced function out of the registry (and out of arm's site count), and
        # falls back to __call__ for a module that implements that instead. The queue
        # still has to be drained either way: an OptimizedModule yields nothing here
        # and is resolved through the `_orig_mod` entry queued above.
        for name in ("forward", "__call__"):
            candidate = getattr(type(obj), name, None)
            code = getattr(candidate, "__code__", None)
            if isinstance(code, CodeType) and not _is_placeholder(code):
                yield code
                break


def _is_placeholder(code: CodeType) -> bool:
    """Whether ``code`` is a shared dispatcher rather than a layer's own kernel.

    ``nn.Module.forward`` is ``_forward_unimplemented`` and ``__call__`` is the
    ``_wrapped_call_impl`` dispatcher every module shares, so registering either says
    nothing about the layer being watched. ``OptimizedModule.__call__`` is likewise
    torch's compile wrapper, not the traced kernel: the real one is reached through
    ``_orig_mod``.
    """
    import torch.nn as nn
    from torch._dynamo.eval_frame import OptimizedModule

    stubs = (
        nn.Module.forward,
        getattr(nn.Module, "_call_impl", None),
        nn.Module.__call__,
        OptimizedModule.__call__,
    )
    return any(code is getattr(stub, "__code__", None) for stub in stubs if stub is not None)


_guard = _CompileGuard()


def watch(target: object, label: str) -> None:
    """Register a callable whose compiles should all happen before ``arm``."""
    _guard.watch(target, label)


def arm(level: CompileGuardLevel) -> None:
    """Begin reporting compiles of watched callables."""
    _guard.arm(level)


def disarm() -> None:
    """Stop reporting."""
    _guard.disarm()


def reset() -> None:
    """Tear down completely, registry included. For tests."""
    _guard.reset()


def clear_reported() -> None:
    """Forget already-logged violations, keeping the registry. For tests."""
    _guard.clear_reported()


def is_armed() -> bool:
    return _guard.armed


def level() -> CompileGuardLevel:
    return _guard.level


def allow_compile(reason: str) -> Any:
    """Suspend the guard on this thread, for a phase that compiles on purpose.

    Reserved escape hatch: nothing in ``spyre_inference`` calls this today, because the
    guard is armed only once warmup has finished, so every deliberate compile phase
    (whole-model or per-block compile, attention recording) has already run by then,
    and a layer that recompiles per input shape by contract opts out through
    ``CompileOutermost.allow_inference_recompiles`` instead. Kept for a future phase
    that has to compile after arming.

    Suppression is per-thread, but that cannot be a guarantee on its own: torch runs
    start callbacks only when its *process-wide* pending-compile counter goes 0 -> 1,
    so a compile beginning while another is in flight fires no callback at all. That
    direction fails safe -- a missed report, never a false one.
    """
    return _guard.allow_compile(reason)
