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

"""``spyre_models()`` has to track vLLM's registry.

``ModelRegistry.register_model`` accepts any architecture string, so a rename or
a typo would leave the architecture resolving to the unadapted vLLM class. These
tests fail instead: every key is a real vLLM architecture, every value names a
class that exists and subclasses the one it replaces, and the derivation that
covers the encoders still finds them.
"""

import importlib

import pytest
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.registry import _VLLM_MODELS

from spyre_inference.models import _ADAPTED_ARCHS, _ADAPTED_MODULES, register_models
from spyre_inference.models import spyre_models as _spyre_models

SPYRE_MODELS = _spyre_models()


def _load(target: str) -> type:
    module_name, _, cls_name = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name, None)
    assert cls is not None, f"{module_name} has no {cls_name}; add the Spyre subclass"
    return cls


def _upstream_class(arch: str) -> type:
    module, cls_name = _VLLM_MODELS[arch]
    return getattr(importlib.import_module(f"vllm.model_executor.models.{module}"), cls_name)


def test_every_spyre_arch_is_known_to_vllm():
    """A key vLLM does not know is a silent no-op, not an error."""
    unknown = sorted(set(SPYRE_MODELS) - set(ModelRegistry.get_supported_archs()))
    assert not unknown, f"not registered by this vLLM: {unknown}"


@pytest.mark.parametrize("arch", sorted(SPYRE_MODELS))
def test_spyre_model_subclasses_the_arch_it_replaces(arch):
    """The lazy ``"module:Class"`` half is unvalidated until vLLM resolves it.

    Also the reverse check for the derived encoders: a new bert/roberta
    architecture upstream is registered here whether or not this package has a
    ``Spyre`` subclass for it, and ``_load`` is what says which one is missing.
    """
    spyre_cls = _load(SPYRE_MODELS[arch])
    upstream_cls = _upstream_class(arch)
    assert issubclass(spyre_cls, upstream_cls), (
        f"{spyre_cls.__name__} does not subclass {upstream_cls.__name__}"
    )


def test_the_encoder_derivation_still_finds_the_encoders():
    """Upstream moving bert/roberta elsewhere would silently adapt nothing."""
    upstream = {arch for arch, (module, _) in _VLLM_MODELS.items() if module in _ADAPTED_MODULES}
    assert upstream, f"no architectures left in vLLM's {_ADAPTED_MODULES} modules"
    assert upstream <= set(SPYRE_MODELS)


def test_register_models_installs_every_arch_lazily():
    """Registration must point at the Spyre target without importing it."""
    register_models()
    installed = {
        arch: f"{model.module_name}:{model.class_name}"
        for arch, model in ModelRegistry.models.items()
        if arch in SPYRE_MODELS
    }
    assert installed == SPYRE_MODELS


def test_register_models_rejects_an_unknown_arch(monkeypatch):
    monkeypatch.setitem(_ADAPTED_ARCHS, "NotAnArchitecture", "spyre_inference.models.bert:Nope")
    with pytest.raises(RuntimeError, match="NotAnArchitecture"):
        register_models()
