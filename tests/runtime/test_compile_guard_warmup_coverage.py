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

The stub also carries a real token embedding behind the serving path's wrapper: it sits
outside the blocks, and ``compile_when_outermost`` compiles it on its own.

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
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.custom_ops import lazy_compile
from spyre_inference.custom_ops.vocab_parallel_embedding import (
    SpyreVocabParallelEmbedding,
)
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.compile_guard import (
    CompileGuardLevel,
    UnexpectedCompileError,
)
from spyre_inference.v1.worker.spyre_model_runner import (
    TorchSpyreModelRunner,
    _SpyreModelWrapper,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer

HIDDEN = 8
VOCAB = 128
BODY_BUCKETS = [4, 8]
MAX_NUM_REQS = 8
UNWARMED_TOKENS = 6
"""Strictly between two buckets, so no warmed graph covers it."""
SERVED_TOKEN_COUNTS = [1, 3, 4, 5, 8]
"""Token counts a request can arrive with: each pads onto a bucket."""


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
    """A model shaped the way ``_repeated_block_lists`` expects: one ModuleList.

    ``embed_input_ids`` mirrors a multimodal model's text-only early return.
    """

    def __init__(self, attn: Attention, embedding: torch.nn.Module) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(attn)])
        self.embed_tokens = embedding
        self.embed_calls: list[int] = []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ) -> torch.Tensor:
        self.embed_calls.append(input_ids.shape[0])
        return self.embed_tokens(input_ids)


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


@pytest.fixture(autouse=True)
def eager_outermost_backend(monkeypatch):
    """Card-free, for the reason the module docstring gives for the block backend."""
    monkeypatch.setattr(
        lazy_compile,
        "current_platform",
        types.SimpleNamespace(simple_compile_backend="eager"),
    )


@pytest.fixture
def build_runner(default_vllm_config, tp_group, monkeypatch):
    """Build and warm up a runner; ``warmup_embeddings=False`` drops the embedding pass
    so a test can see what the guard reported before it existed."""

    def build(
        *,
        supports_mm_inputs: bool = True,
        warmup_embeddings: bool = True,
        bucket_sizes: tuple[int, ...] = tuple(BODY_BUCKETS),
        mm_encoder_only: bool = False,
        runner_type: str = "generate",
    ):
        # NONE keeps warmup off the attention recorder, which needs a real KV cache.
        compilation_config = types.SimpleNamespace(
            compile_sizes=list(bucket_sizes),
            inductor_compile_config={},
            static_forward_context={},
            mode=CompilationMode.NONE,
        )
        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
        runner.model_config = types.SimpleNamespace(
            runner_type=runner_type,
            is_encoder_decoder=False,
            max_model_len=max(bucket_sizes, default=64),
            multimodal_config=(
                types.SimpleNamespace(mm_encoder_only=True) if mm_encoder_only else None
            ),
        )
        runner.vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                enforce_eager=False,
                max_model_len=runner.model_config.max_model_len,
            ),
            compilation_config=compilation_config,
        )
        runner.compilation_config = compilation_config
        runner._spyre_device = torch.device("cpu")
        runner.spyre_shape_bucketer = (
            SpyreShapeBucketer(runner.vllm_config) if bucket_sizes else None
        )
        runner.max_num_reqs = MAX_NUM_REQS
        runner.scheduler_config = types.SimpleNamespace(
            max_num_batched_tokens=64,
            max_num_seqs=MAX_NUM_REQS,
        )
        runner.vllm_config.scheduler_config = runner.scheduler_config
        runner.supports_mm_inputs = supports_mm_inputs
        runner._spyre_kv_caches = {}
        runner._encoder_budget = max(bucket_sizes, default=64)
        runner._encoder_rectangles = []
        runner._pooling_on_spyre = False

        attn = Attention(
            num_heads=1,
            head_size=HIDDEN,
            scale=1.0,
            prefix="layers.0.self_attn",
        )
        # The layer samples the compile mode at construction.
        monkeypatch.setattr(
            lazy_compile,
            "get_cached_compilation_config",
            lambda: types.SimpleNamespace(mode=CompilationMode.STOCK_TORCH_COMPILE),
        )
        embedding = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.float32)
        assert isinstance(embedding, SpyreVocabParallelEmbedding), "OOT dispatch is off"
        decoder = _Decoder(attn, embedding)
        runner.model = decoder
        block = decoder.layers[0]

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

        # Wrapped after compile, as ``load_model`` does.
        runner.model = _SpyreModelWrapper(
            decoder,
            runner._spyre_device,
            shape_bucketer=runner.spyre_shape_bucketer,
            model_dtype=torch.float32,
        )

        def base_dummy_run(_self, size, *args, **kwargs):
            hidden = torch.zeros(size, HIDDEN)
            return None, block(hidden)

        runner._dummy_sampler_run = lambda hidden_states: torch.tensor([])
        runner._dummy_pooler_run = lambda hidden_states: None
        runner._record_attention_graphs = lambda: None
        if not warmup_embeddings:
            runner._warmup_input_embedding = lambda num_tokens: None

        # The *base* method is patched, not `runner._dummy_run`: the Spyre override is
        # what drives the embedding, so stubbing it out would skip the thing under test.
        with unittest.mock.patch.object(
            GPUModelRunner, "_dummy_run", autospec=True, side_effect=base_dummy_run
        ):
            runner.warming_up_model()
        return runner, block, decoder

    return build


@pytest.fixture
def warmed_runner(build_runner):
    """A runner whose warmup really compiled ``BODY_BUCKETS``, plus its block.

    Blocks are compiled and registered by the production ``_compile_blocks``, not by
    the test: hand-registering here would keep passing if that call site ever stopped
    watching, which is the regression this file exists to catch.
    """
    runner, block, _ = build_runner()
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


def test_warmup_covers_the_token_embedding(build_runner):
    """Every token count a request can arrive with is already compiled."""
    runner, _, decoder = build_runner()
    assert decoder.embed_calls == sorted(BODY_BUCKETS, reverse=True)
    compile_guard.arm(CompileGuardLevel.ERROR)

    for num_tokens in SERVED_TOKEN_COUNTS:
        out = runner.model.embed_input_ids(torch.zeros(num_tokens, dtype=torch.int32))
        assert out.shape == (num_tokens, HIDDEN)


def test_without_the_embedding_pass_the_first_request_compiles_it(build_runner):
    """What the guard reported before the pass existed; it must stay reproducible."""
    runner, _, decoder = build_runner(warmup_embeddings=False)
    assert decoder.embed_calls == []
    compile_guard.arm(CompileGuardLevel.ERROR)

    with pytest.raises(
        UnexpectedCompileError,
        match="SpyreVocabParallelEmbedding.forward compiled unexpectedly",
    ):
        runner.model.embed_input_ids(torch.zeros(BODY_BUCKETS[0], dtype=torch.int32))


def test_a_text_only_model_is_not_embedded_during_warmup(build_runner):
    """Its ``embed_input_ids`` takes no multimodal arguments, so the call would raise."""
    _, _, decoder = build_runner(supports_mm_inputs=False)

    assert decoder.embed_calls == []


def test_an_mm_encoder_only_model_is_not_embedded_during_warmup(build_runner):
    """Its language model is absent, and upstream skips the dummy LM forward too."""
    _, _, decoder = build_runner(mm_encoder_only=True)

    assert decoder.embed_calls == []


def test_pooling_warms_each_embedding_bucket_once(build_runner):
    """Attention-shape dummy runs must not replay compiled embedding collectives."""
    _, _, decoder = build_runner(runner_type="pooling")

    assert decoder.embed_calls == sorted(BODY_BUCKETS, reverse=True)


def test_single_pass_without_buckets_still_warms_the_embedding(build_runner):
    """An empty compile-size list warms the single configured dummy width."""
    runner, _, decoder = build_runner(bucket_sizes=[])
    warmup_tokens = min(max(16, MAX_NUM_REQS), runner.scheduler_config.max_num_batched_tokens)

    assert decoder.embed_calls == [warmup_tokens]
    compile_guard.arm(CompileGuardLevel.ERROR)

    out = runner.model.embed_input_ids(torch.zeros(warmup_tokens, dtype=torch.int32))
    assert out.shape == (warmup_tokens, HIDDEN)
