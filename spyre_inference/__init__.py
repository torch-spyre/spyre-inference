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


def _init_logging():
    """Route our loggers through vLLM's ``vllm`` logger.

    Since 0.31 vLLM configures logging lazily (``vllm.logger.configure_logging``,
    from the engine, workers and CLI) and only for the ``vllm`` logger, whose format
    needs a field its record factory adds. Parenting ours under it inherits whatever
    handlers and level vLLM installs, whenever it does; configuring our own at import
    would format records before that factory exists.
    """
    logging.getLogger("spyre_inference").parent = logging.getLogger("vllm")


_init_logging()
