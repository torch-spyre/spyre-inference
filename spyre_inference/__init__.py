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

import contextlib
import importlib.metadata
import logging
import os
import signal
import threading

__version__ = importlib.metadata.version("spyre_inference")


def register():
    """Register the Spyre platform."""
    # vLLM resolves the platform during `import vllm`, before vllm.triton_utils is
    # imported, which is the last point the Triton gate can act.
    from vllm.logger import init_logger

    from spyre_inference import envs
    from spyre_inference.v1.worker.spyre import triton_gate

    # vLLM's plugin probe swallows exceptions here and falls back to another platform,
    # so errors are only logged; check_and_update_config rejects the resulting state.
    try:
        if envs.SPYRE_MODEL_RUNNER == "v2":
            triton_gate.open()
        else:
            triton_gate.close()
    except Exception:
        init_logger(__name__).exception("Spyre Triton gate could not be applied")
    # VllmConfig.__post_init__ reads use_v2_model_runner before the platform hook.
    pin_model_runner(envs.SPYRE_MODEL_RUNNER)
    return "spyre_inference.platform.TorchSpyrePlatform"


def pin_model_runner(runner: str) -> None:
    """Point VLLM_USE_V2_MODEL_RUNNER at SPYRE_MODEL_RUNNER's choice.

    Pinned for v1 too: use_v2_model_runner returns an explicit value before its
    HAS_TRITON fallback, and vLLM's own heuristics can otherwise still pick V2.
    """
    want = "1" if runner == "v2" else "0"
    current = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if current not in (None, "", want):
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "SPYRE_MODEL_RUNNER=%s picks the model runner; overriding "
            "VLLM_USE_V2_MODEL_RUNNER=%s to %s.",
            runner,
            current,
            want,
        )
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = want


@contextlib.contextmanager
def keep_sigterm_handler():
    """Restore the SIGTERM handler that loading senlib (via ``torch_spyre._C``) replaces.

    senlib's handler exits at once, while vLLM's shuts the engine's workers down; without
    it they outlive the engine and keep the card. By default ``import torch`` loads senlib
    before vLLM installs its handler, so this only matters with
    ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``, where senlib loads inside these blocks.
    senlib's crash handlers (SIGSEGV, SIGABRT, ...) are left in place.
    """
    on_main_thread = threading.current_thread() is threading.main_thread()
    handler = signal.getsignal(signal.SIGTERM) if on_main_thread else None
    try:
        yield
    finally:
        if handler is not None:
            signal.signal(signal.SIGTERM, handler)


def register_ops():
    """Register the Spyre OOT custom ops and model adaptations."""
    from spyre_inference.custom_ops import register_all
    from spyre_inference.models import register_models

    with keep_sigterm_handler():
        register_all()
        register_models()


def _route_logging() -> None:
    """Parent ``spyre_inference`` under ``vllm``, unless a logging config gave it
    handlers of its own: it then owns its routing, and propagating into ``vllm`` too
    would print each record twice."""
    ours = logging.getLogger("spyre_inference")
    ours.parent = logging.root if ours.handlers else logging.getLogger("vllm")


def _init_logging():
    """Route our loggers through vLLM's ``vllm`` logger.

    vLLM configures logging lazily (``vllm.logger.configure_logging``, from the engine,
    workers and CLI) and only for the ``vllm`` logger, whose format needs a field its
    record factory adds. Parenting ours under it inherits whatever handlers and level
    vLLM installs, whenever it does; configuring our own at import would format records
    before that factory exists.

    ``dictConfig`` only spares the existing loggers a config names, or their children
    by dotted name, so a custom config that leaves ``disable_existing_loggers`` at its
    default would still disable ours. Every entry point configures through
    ``_configure_vllm_root_logger``, so it is wrapped to re-enable them afterwards.
    """
    import vllm.logger as vllm_logger

    _route_logging()

    configure = vllm_logger._configure_vllm_root_logger
    if getattr(configure, "_spyre_patched", False):
        return

    def _configure(*args, **kwargs):
        ours = [
            lg
            for name, lg in logging.root.manager.loggerDict.items()
            if isinstance(lg, logging.Logger)
            and (name == "spyre_inference" or name.startswith("spyre_inference."))
            and not lg.disabled
        ]
        configure(*args, **kwargs)
        for lg in ours:
            lg.disabled = False
        _route_logging()

    _configure._spyre_patched = True
    vllm_logger._configure_vllm_root_logger = _configure


_init_logging()
