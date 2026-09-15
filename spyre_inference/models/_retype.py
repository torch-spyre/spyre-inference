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

"""Retyping an upstream-built submodule to its Spyre subclass.

Shared machinery rather than one architecture's adaptation, hence the private name.

A vLLM model that hardcodes the class of a submodule it builds offers no
``embedding_class``-style hook to pass a subclass through, so the Spyre adaptation
lets ``super().__init__()`` build the tree and then swaps the class of the one
submodule it needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from torch import nn

_ModuleT = TypeVar("_ModuleT", bound="nn.Module")


def retype(module: object, spyre: type[_ModuleT]) -> _ModuleT:
    """Retype ``module`` to the Spyre subclass ``spyre``, and hand it back.

    ``spyre`` must name its mixins before the upstream class it adapts
    (``class SpyreX(SomeMixin, UpstreamX)``): its last base is the class ``module``
    is required to be an exact instance of.
    """
    upstream = spyre.__bases__[-1]
    if type(module) is not upstream:
        raise RuntimeError(
            f"expected {upstream.__name__}, got {type(module).__name__}; the Spyre "
            f"{spyre.__name__} adaptation needs updating for this vLLM version."
        )
    module.__class__ = spyre
    return cast("_ModuleT", module)
