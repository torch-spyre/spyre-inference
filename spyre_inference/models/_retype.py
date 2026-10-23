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

"""Retype upstream-built submodules that provide no subclass hook."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from torch import nn

_ModuleT = TypeVar("_ModuleT", bound="nn.Module")


def retype(module: object, spyre: type[_ModuleT]) -> _ModuleT:
    """Retype a module; the Spyre subclass's final base must be its exact current type."""
    upstream = spyre.__bases__[-1]
    if type(module) is not upstream:
        raise RuntimeError(
            f"expected {upstream.__name__}, got {type(module).__name__}; the Spyre "
            f"{spyre.__name__} adaptation needs updating for this vLLM version."
        )
    module.__class__ = spyre
    return cast("_ModuleT", module)
