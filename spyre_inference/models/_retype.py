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
submodule it needs to change. The alternatives are worse: rebuilding the submodule
allocates its weights a second time and re-registers its attention layers, and
skipping ``super().__init__()`` to inline upstream's body duplicates whatever churns
there. The subclass adds no ``__init__``, no parameters and no children, so the
built tree is already exactly what it would have constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from torch import nn

_ModuleT = TypeVar("_ModuleT", bound="nn.Module")


def retype(module: object, spyre: type[_ModuleT]) -> _ModuleT:
    """Retype ``module`` to the Spyre subclass ``spyre``, and hand it back.

    ``module`` is annotated ``object`` because the static type of a built submodule
    differs by call site — a plain mixin where one is in play — and it is ``spyre``
    that has to be an ``nn.Module`` subclass. The check below is the real constraint.

    The class ``module`` is expected to be is ``spyre``'s last base, so a Spyre
    subclass must name its mixins first (``class SpyreX(SomeMixin, UpstreamX)``);
    written the other way round the check below rejects every module.

    Returns:
        ``module``, typed as ``spyre`` so callers can reach what the subclass adds.

    Raises:
        RuntimeError: if ``module`` is not exactly an instance of that base. A
            subclass is refused too: retyping one would drop whatever it overrides.
    """
    upstream = spyre.__bases__[-1]
    if type(module) is not upstream:
        raise RuntimeError(
            f"expected {upstream.__name__}, got {type(module).__name__}; the Spyre "
            f"{spyre.__name__} adaptation needs updating for this vLLM version."
        )
    module.__class__ = spyre
    return cast("_ModuleT", module)
