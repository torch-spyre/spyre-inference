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

"""The Spyre EncoderRunner's staging buffer, and the model state that installs it."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.model_states import interface as upstream_interface

from spyre_inference.v1.worker.spyre.mm.encoder_runner import SpyreEncoderRunner
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

from .fakes import FakeModel

_TOKENS, _HIDDEN = 64, 8
_META = torch.device("meta")


def _wrapper() -> _SpyreModelWrapper:
    return _SpyreModelWrapper(FakeModel(), torch.device("cpu"), model_dtype=torch.float16)


def _runner(wrapper, device=_META) -> SpyreEncoderRunner:
    return SpyreEncoderRunner(
        wrapper, _TOKENS, _HIDDEN, None, torch.float16, torch.device("cpu"), spyre_device=device
    )  # ty: ignore[invalid-argument-type]


def test_the_staging_buffer_lives_on_the_spyre_device() -> None:
    """Upstream's buffer is on the runner's device, the host here, so a dummy run would
    hand the model a host tensor beside device weights."""
    runner = _runner(_wrapper())
    assert runner.inputs_embeds.device == torch.device("meta")
    assert tuple(runner.inputs_embeds.shape) == (_TOKENS, _HIDDEN)
    assert runner.inputs_embeds.dtype == torch.float16


def test_the_wrapper_stages_into_the_runners_buffer() -> None:
    """The wrapper's own buffer, not a forwarded attribute on the model."""
    wrapper = _wrapper()
    runner = _runner(wrapper)
    assert wrapper._inputs_embeds_buffer is runner.inputs_embeds
    assert not hasattr(wrapper._model, "_inputs_embeds_buffer")


def test_upstreams_copy_into_the_buffer_is_a_self_copy(monkeypatch) -> None:
    """get_inputs_embeds does inputs_embeds[:n] = x; with x already the buffer's prefix
    view, that copy is one torch-spyre skips instead of a d2d compiled per n."""
    wrapper = _wrapper()
    runner = _runner(wrapper, device=torch.device("cpu"))
    monkeypatch.setattr(
        wrapper._model,
        "embed_input_ids",
        lambda ids, *a, **kw: torch.ones(ids.shape[0], _HIDDEN, dtype=torch.float16),
        raising=False,
    )
    bucketer = SimpleNamespace(find_bucket=lambda n: 16)
    object.__setattr__(wrapper, "_shape_bucketer", bucketer)

    ids, is_mm = torch.zeros(10, dtype=torch.int32), torch.zeros(10, dtype=torch.bool)
    out = runner.get_inputs_embeds(ids, [], is_mm)

    assert out.data_ptr() == runner.inputs_embeds.data_ptr()
    assert torch.equal(runner.inputs_embeds[:10], torch.ones(10, _HIDDEN, dtype=torch.float16))


class _UpstreamState:
    """What ModelState.__init__ does that matters here: build its EncoderRunner by name."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        self.model = model
        if encoder_cache is not None:
            self.encoder_runner = upstream_interface.EncoderRunner(
                model=model,
                max_num_tokens=_TOKENS,
                hidden_size=_HIDDEN,
                encoder_cache=encoder_cache,
                dtype=torch.float16,
                device=device,
            )


class _State(SpyreModelStateMixin, _UpstreamState):
    pass


def test_a_multimodal_state_builds_the_spyre_encoder_runner() -> None:
    before = upstream_interface.EncoderRunner
    state = _State(None, _wrapper(), object(), torch.device("cpu"), torch.device("meta"))
    assert isinstance(state.encoder_runner, SpyreEncoderRunner)
    assert state.encoder_runner.inputs_embeds.device == torch.device("meta")
    assert upstream_interface.EncoderRunner is before


def test_the_swap_is_undone_when_upstream_raises() -> None:
    before = upstream_interface.EncoderRunner

    class Upstream:
        def __init__(self, *args):
            assert upstream_interface.EncoderRunner is not before
            raise RuntimeError("boom")

    class State(SpyreModelStateMixin, Upstream):
        pass

    with pytest.raises(RuntimeError, match="boom"):
        State(None, _wrapper(), object(), torch.device("cpu"), torch.device("meta"))
    assert upstream_interface.EncoderRunner is before
