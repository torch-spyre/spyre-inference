# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logging utilities for pastamachine.

Uses Python's standard ``logging`` module (PyTorch-compatible).
All pastamachine loggers live under the ``pastamachine`` namespace so they
can be configured externally via ``logging.getLogger("pastamachine")``.

The default handler prints to stderr with a ``[pastamachine::<module>]``
prefix.  It is only installed on the root ``pastamachine`` logger so it
does not interfere with application-level logging configuration.
"""

import logging

_FMT = "[pastamachine::%(module)s] %(message)s"


def _setup_root_logger():
    root = logging.getLogger("pastamachine")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return root


# Eagerly set up so that importing any pastamachine submodule activates
# the handler (users can still override via the root logger).
_setup_root_logger()


def getLogger(name: str) -> logging.Logger:
    """Return a child logger under the ``pastamachine`` namespace."""
    return logging.getLogger(f"pastamachine.{name}")


# Made with Bob
