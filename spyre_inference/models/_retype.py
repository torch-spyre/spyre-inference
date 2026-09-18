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

A vLLM model that hardcodes a submodule's class offers no ``embedding_class``-style hook
to pass a subclass through, so ``super().__init__()`` builds the tree and this swaps the
class afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from torch import nn

_ModuleT = TypeVar("_ModuleT", bound="nn.Module")


def retype(module: object, spyre: type[_ModuleT]) -> _ModuleT:
    """Retype ``module`` to the Spyre subclass ``spyre``, and hand it back.

    ``spyre`` must name its mixins first (``class SpyreX(SomeMixin, UpstreamX)``): its last
    base is the class ``module`` must be an exact instance of.
    """
    upstream = spyre.__bases__[-1]
    if type(module) is not upstream:
        raise RuntimeError(
            f"expected {upstream.__name__}, got {type(module).__name__}; the Spyre "
            f"{spyre.__name__} adaptation needs updating for this vLLM version."
        )
    module.__class__ = spyre
    return cast("_ModuleT", module)
