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
import tomllib
from pathlib import Path

import pytest
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.registry import _VLLM_MODELS

from spyre_inference.models import (
    _ADAPTED_ARCHS,
    _ADAPTED_MODULES,
    check_spyre_model,
    register_models,
)
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


def _unbuilt(cls: type) -> object:
    """An instance for ``isinstance`` alone: building a real model needs a vLLM config."""
    return object.__new__(cls)


@pytest.mark.parametrize("arch", sorted(SPYRE_MODELS))
def test_check_accepts_the_spyre_class(arch):
    spyre_cls = _load(SPYRE_MODELS[arch])
    check_spyre_model(_unbuilt(spyre_cls), arch)
    # vLLM wraps a resolved class for some runners (e.g. as_embedding_model).
    check_spyre_model(_unbuilt(type("Wrapped", (spyre_cls,), {})), arch)


@pytest.mark.parametrize("arch", sorted(SPYRE_MODELS))
def test_check_rejects_the_unadapted_vllm_class(arch):
    """What a worker builds when it resolved the architecture before registration."""
    with pytest.raises(RuntimeError, match=arch):
        check_spyre_model(_unbuilt(_upstream_class(arch)), arch)


def test_check_ignores_an_architecture_without_an_adaptation():
    assert "LlamaForCausalLM" not in SPYRE_MODELS
    check_spyre_model(_unbuilt(_upstream_class("LlamaForCausalLM")), "LlamaForCausalLM")


def test_platform_plugin_name_also_loads_the_model_registration():
    """``VLLM_PLUGINS`` filters each group by name, so ``spyre_inference`` alone must
    reach ``register_ops`` too: otherwise a worker resolves a multimodal architecture
    (CLIP) before the Spyre classes are registered."""
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    entry_points = pyproject["project"]["entry-points"]
    platform_names = set(entry_points["vllm.platform_plugins"])
    general = entry_points["vllm.general_plugins"]
    assert platform_names <= set(general)
    assert {general[name] for name in platform_names} == {"spyre_inference:register_ops"}
