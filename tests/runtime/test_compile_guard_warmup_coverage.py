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

"""The compile guard catches a request shape that warmup did not cover.

Drives the real ``TorchSpyreModelRunner.warming_up_model`` over a stub runner (the
harness ``test_warmup_logits_widths`` uses), with ``_dummy_run`` wired to a genuinely
compiled block so warmup compiles the bucket shapes for real. Arming after that and
then serving an off-bucket shape reproduces the production failure this guard exists
to catch: ``SpyreShapeBucketer.find_bucket`` returns ``None``, nothing pads the
batch, and the block recompiles mid-request.

CPU with ``backend="eager"``: the guard keys on the code object Dynamo traces, which
is decided before any backend runs, so a real Spyre compile would only add minutes.
"""

from __future__ import annotations

import copy
import types
import unittest.mock

import pytest
import torch
from torch._dynamo.utils import counters
from vllm.config import CompilationMode
from vllm.model_executor.layers.attention.attention import Attention

from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.compile_guard import (
    CompileGuardLevel,
    UnexpectedCompileError,
)
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

HIDDEN = 8
BODY_BUCKETS = [4, 8]
MAX_NUM_REQS = 8
UNWARMED_TOKENS = 6
"""Strictly between two buckets, so no warmed graph covers it."""


class _Block(torch.nn.Module):
    """One transformer block's stand-in: compiled once, guarded on its row count.

    Carries a real ``Attention`` so ``_repeated_block_lists`` recognises the list as a
    decoder stack, which is what makes ``_compile_blocks`` compile and register it.
    """

    def __init__(self, attn: Attention) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        # Never called: it exists so the block looks like a decoder layer. Its own
        # forward would need a KV cache and an attention metadata object.
        self.self_attn = attn

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.linear(hidden))


class _Decoder(torch.nn.Module):
    """A model shaped the way ``_repeated_block_lists`` expects: one ModuleList."""

    def __init__(self, attn: Attention) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(attn)])


class _Bucketer:
    def __init__(self, bucket_sizes: list[int]) -> None:
        self.bucket_sizes = bucket_sizes
        self.warmed_up = False

    def mark_warmed_up(self) -> None:
        self.warmed_up = True


@pytest.fixture(autouse=True)
def isolated_dynamo_state():
    saved = copy.deepcopy(counters)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    counters.clear()
    counters.update(saved)


@pytest.fixture(autouse=True)
def clean_guard():
    compile_guard.reset()
    yield
    compile_guard.reset()


@pytest.fixture
def warmed_runner(default_vllm_config):
    """A runner whose warmup really compiled ``BODY_BUCKETS``, plus its block.

    Blocks are compiled and registered by the production ``_compile_blocks``, not by
    the test: hand-registering here would keep passing if that call site ever stopped
    watching, which is the regression this file exists to catch.
    """
    # NONE keeps warmup off the attention recorder, which needs a real KV cache.
    compilation_config = types.SimpleNamespace(
        compile_sizes=list(BODY_BUCKETS),
        inductor_compile_config={},
        static_forward_context={},
        mode=CompilationMode.NONE,
    )
    runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
    runner.model_config = types.SimpleNamespace(runner_type="generate")
    runner.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=False),
        compilation_config=compilation_config,
    )
    runner.compilation_config = compilation_config
    runner._spyre_device = torch.device("cpu")
    runner.spyre_shape_bucketer = _Bucketer(list(BODY_BUCKETS))
    runner.max_num_reqs = MAX_NUM_REQS

    attn = Attention(
        num_heads=1,
        head_size=HIDDEN,
        scale=1.0,
        prefix="layers.0.self_attn",
    )
    runner.model = _Decoder(attn)
    block = runner.model.layers[0]

    # The real wiring: compiles the block in place and registers it with the guard.
    # The backend is forced to "eager" so this stays card-free; the guard keys on the
    # traced code object, which is decided before any backend runs. Bound to the
    # original method first, or the patch would re-enter itself.
    real_compile = torch.nn.Module.compile
    with unittest.mock.patch.object(
        torch.nn.Module,
        "compile",
        autospec=True,
        side_effect=lambda self, **kw: real_compile(self, **{**kw, "backend": "eager"}),
    ):
        assert runner._compile_blocks() == 1, "the stub decoder was not recognised"

    def dummy_run(size, *args, **kwargs):
        hidden = torch.zeros(size, HIDDEN)
        return None, block(hidden)

    runner._dummy_run = dummy_run
    runner._dummy_sampler_run = lambda hidden_states: torch.tensor([])

    runner.warming_up_model()
    return runner, block


def test_warmup_compiles_every_bucket_so_a_warmed_shape_is_quiet(warmed_runner):
    """The guard must not fire on the shapes warmup did cover, or it is useless."""
    _, block = warmed_runner
    compile_guard.arm(CompileGuardLevel.ERROR)

    for size in BODY_BUCKETS:
        block(torch.zeros(size, HIDDEN))


def test_an_unwarmed_shape_is_caught(warmed_runner):
    _, block = warmed_runner
    compile_guard.arm(CompileGuardLevel.ERROR)

    with pytest.raises(UnexpectedCompileError, match="_Block .* recompiled unexpectedly"):
        block(torch.zeros(UNWARMED_TOKENS, HIDDEN))


def test_the_report_names_the_block_and_the_diagnostic(warmed_runner):
    _, block = warmed_runner
    compile_guard.arm(CompileGuardLevel.ERROR)

    with pytest.raises(UnexpectedCompileError) as excinfo:
        block(torch.zeros(UNWARMED_TOKENS, HIDDEN))

    message = str(excinfo.value)
    assert "_Block (transformer block)" in message
    assert "TORCH_LOGS=recompiles" in message


def test_warn_level_logs_the_unwarmed_shape_and_keeps_serving(warmed_runner, caplog):
    """Serving must survive at ``warn``: the compile is slow, not wrong."""
    _, block = warmed_runner
    compile_guard.arm(CompileGuardLevel.WARN)

    with caplog.at_level("WARNING"):
        out = block(torch.zeros(UNWARMED_TOKENS, HIDDEN))

    assert out.shape == (UNWARMED_TOKENS, HIDDEN)
    assert any("recompiled unexpectedly" in r.getMessage() for r in caplog.records)


def test_without_the_guard_the_unwarmed_shape_compiles_silently(warmed_runner, caplog):
    """The regression this guards against: today's default is silence."""
    _, block = warmed_runner
    compile_guard.arm(CompileGuardLevel.OFF)

    with caplog.at_level("WARNING"):
        block(torch.zeros(UNWARMED_TOKENS, HIDDEN))

    assert not any("unexpectedly" in r.getMessage() for r in caplog.records)
