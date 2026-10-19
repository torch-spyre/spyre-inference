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

import importlib.metadata
import logging
import os

# Defer torch_spyre's autoload until we explicitly trigger it inside
# `TorchSpyreWorker.init_device`. Autoload loads `libspyre_comms.so`,
# which captures `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`LOCAL_WORLD_SIZE`
# at dlopen time — those env vars are only known per-worker, so the
# library can't load before init_device runs.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

__version__ = importlib.metadata.version("spyre_inference")


def register():
    """Register the Spyre platform."""
    return "spyre_inference.platform.TorchSpyrePlatform"


def register_ops():
    """Register the Spyre OOT custom ops and model adaptations."""
    from spyre_inference.custom_ops import register_all
    from spyre_inference.models import register_models

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
