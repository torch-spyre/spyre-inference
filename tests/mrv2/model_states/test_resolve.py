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

"""resolve_spyre_model_state puts the Spyre mixin over the class upstream picks."""

import pytest
import torch
import torch.nn as nn
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from spyre_inference.v1.worker.spyre.model_states import (
    resolve_spyre_model_state,
    spyre_model_state_cls,
)
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)

from .state_fakes import make_config


def test_dedicated_spyre_states():
    assert spyre_model_state_cls(DefaultModelState) is SpyreDefaultModelState
    assert spyre_model_state_cls(EncoderOnlyModelState) is SpyreEncoderOnlyModelState


def test_other_states_get_the_mixin_over_them():
    cls = spyre_model_state_cls(MambaHybridModelState)
    assert cls.__mro__[1:3] == (SpyreModelStateMixin, MambaHybridModelState)
    assert spyre_model_state_cls(MambaHybridModelState) is cls


def test_constructs_with_both_devices():
    spyre_device = torch.device("meta")
    state = resolve_spyre_model_state(
        make_config(),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        spyre_device,
    )
    assert type(state) is SpyreDefaultModelState
    assert state.device == torch.device("cpu")
    assert state.spyre_device == spyre_device
    # Upstream's hooks still run: the mixin defines none it does not implement.
    assert state.add_request.__func__ is DefaultModelState.add_request


def _prompt_embeds_config():
    config = make_config()
    config.model_config.enable_prompt_embeds = True
    return config


def test_prompt_embeds_reject_a_state_that_would_ignore_them():
    class _NoEmbedsState(DefaultModelState):
        supports_prompt_embeds = False

    class _Model(nn.Module):
        @staticmethod
        def get_model_state_cls():
            return _NoEmbedsState

    with pytest.raises(ValueError, match="enable-prompt-embeds not supported with _NoEmbedsState"):
        resolve_spyre_model_state(
            _prompt_embeds_config(),
            _Model(),
            None,
            torch.device("cpu"),
            torch.device("cpu"),
        )


def test_prompt_embeds_allow_a_state_that_supports_them():
    state = resolve_spyre_model_state(
        _prompt_embeds_config(),
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    assert type(state) is SpyreDefaultModelState
