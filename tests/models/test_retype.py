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

"""``retype`` moves an already-built module onto its Spyre subclass.

The wrapper classes (``*ForSequenceClassification`` and friends) hardcode the class
they build, so the Spyre adaptation is applied after the fact by reassigning
``__class__``: the same instance, parameters and module tree, but the Spyre
``forward``. The type check turns an upstream class change into a clear error
instead of a module that silently keeps the unadapted behavior.
"""

import pytest
import torch
from torch import nn

from spyre_inference.models._retype import retype


class _Upstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0))
        self.child = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _SpyreForward:
    """Stands in for a Spyre mixin ahead of the upstream base."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 2


class _Spyre(_SpyreForward, _Upstream):
    """The Spyre subclass shape: mixins first, the upstream class last."""


def test_retype_swaps_the_class_in_place():
    module = _Upstream()
    retyped = retype(module, _Spyre)

    assert retyped is module
    assert type(module) is _Spyre
    assert isinstance(module, _Upstream)
    # The swap must not rebuild anything the constructor set up.
    assert module.weight is retyped.weight
    assert module.child is retyped.child


def test_retype_runs_the_spyre_forward():
    module = _Upstream()
    retype(module, _Spyre)

    assert module(torch.zeros(2)).tolist() == [2.0, 2.0]


def test_retype_rejects_a_subclass_of_the_upstream_class():
    """Only the exact upstream type can be retyped: a subclass has its own layout,
    and a vLLM version change that introduces one must be looked at, not papered
    over."""

    class _Subclassed(_Upstream):
        pass

    module = _Subclassed()
    with pytest.raises(RuntimeError, match="expected _Upstream, got _Subclassed"):
        retype(module, _Spyre)
    assert type(module) is _Subclassed


def test_retype_rejects_an_unrelated_module():
    class _Unrelated(nn.Module):
        pass

    module = _Unrelated()
    with pytest.raises(RuntimeError, match="the Spyre _Spyre adaptation needs updating"):
        retype(module, _Spyre)
    assert type(module) is _Unrelated
