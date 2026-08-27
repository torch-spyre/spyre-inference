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

"""Compile a layer's kernel when no enclosing graph already covers it.

Blocks compile one at a time, so layers inside a block are already in a graph; the
same classes outside them (input embedding, final norm) are not. A layer running
while nothing is tracing has no enclosing graph, so it is the outermost one --
``torch.compiler.is_compiling()`` is that test, and Dynamo folds it while tracing.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar, cast

import torch

from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

F = TypeVar("F", bound=Callable)


class CompileOutermost:
    """Base for layers with one ``@compile_when_outermost`` kernel.

    The mode is sampled here because construction is the only point where the vLLM
    config context is live; ``enforce_eager`` arrives as mode ``NONE``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = get_cached_compilation_config().mode
        self.spyre_compile_enabled = mode is not CompilationMode.NONE
        self.spyre_compiled_kernel: Callable | None = None


def compile_when_outermost(method: F) -> F:
    """Compile ``method`` on its first call that no other graph is already tracing."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if torch.compiler.is_compiling() or not self.spyre_compile_enabled:
            return method(self, *args, **kwargs)
        if self.spyre_compiled_kernel is None:
            logger.info_once(
                "Compiling %s.%s as its own graph: no enclosing graph covers it.",
                type(self).__name__,
                method.__name__,
            )
            # dynamic=False is mandatory: the Spyre backend rejects SymInt shapes.
            self.spyre_compiled_kernel = torch.compile(
                method.__get__(self),
                backend=current_platform.simple_compile_backend,
                fullgraph=True,
                dynamic=False,
            )
        return self.spyre_compiled_kernel(*args, **kwargs)

    return cast(F, wrapper)
