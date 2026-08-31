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

"""A layer deciding for itself whether it needs compiling.

Tracing resolves the accelerator stream, which opens the single contested Spyre card,
so the two tests that trace carry the ``compile`` marker.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from vllm.config import CompilationMode

from spyre_inference.custom_ops import lazy_compile
from spyre_inference.custom_ops.lazy_compile import (
    CompileOutermost,
    compile_when_outermost,
)


@pytest.fixture
def compile_calls(monkeypatch) -> list:
    """Record torch.compile calls instead of really compiling."""
    calls: list = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


@pytest.fixture
def mode(monkeypatch):
    """Set the compile mode that CompileOutermost samples at construction."""

    def set_mode(value) -> None:
        monkeypatch.setattr(
            lazy_compile,
            "get_cached_compilation_config",
            lambda: types.SimpleNamespace(mode=value),
        )

    return set_mode


class _Norm(CompileOutermost, nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    @compile_when_outermost
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * 2


class _NoMixin(nn.Module):
    """Decorated but never samples a policy."""

    @compile_when_outermost
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class _Tail(CompileOutermost, nn.Module):
    """Stands in for the norm after the last block: real ops, really compiled."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))

    @compile_when_outermost
    def kernel(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = x + residual
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * self.weight, x


def test_compiles_itself_on_the_first_outermost_call(compile_calls, mode) -> None:
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    norm = _Norm()

    assert torch.equal(norm.kernel(torch.ones(2)), torch.full((2,), 2.0))
    assert len(compile_calls) == 1
    assert compile_calls[0]["fullgraph"] is True
    # SymInt shapes are rejected by the Spyre backend.
    assert compile_calls[0]["dynamic"] is False


def test_compiles_once_and_reuses_the_artifact(compile_calls, mode) -> None:
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    norm = _Norm()

    for _ in range(3):
        norm.kernel(torch.ones(2))

    assert len(compile_calls) == 1
    assert norm.calls == 3


def test_does_not_compile_while_another_graph_is_tracing(compile_calls, mode, monkeypatch) -> None:
    """The in-block case: an enclosing block graph already covers this layer."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    norm = _Norm()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert torch.equal(norm.kernel(torch.ones(2)), torch.full((2,), 2.0))
    assert compile_calls == []


def test_eager_mode_compiles_nothing(compile_calls, mode) -> None:
    """enforce_eager reaches the layer as mode NONE."""
    mode(CompilationMode.NONE)
    norm = _Norm()

    norm.kernel(torch.ones(2))

    assert compile_calls == []


def test_a_class_without_the_mixin_is_a_loud_error(compile_calls) -> None:
    """Forgetting the mixin must not silently fall back to eager."""
    with pytest.raises(AttributeError, match="spyre_compile_enabled"):
        _NoMixin().kernel(torch.ones(2))
    assert compile_calls == []


def test_the_artifact_does_not_become_a_submodule(compile_calls, mode) -> None:
    """A callable in _modules would reach state_dict and break weight save/reload."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    norm = _Norm()
    before = dict(norm.named_modules())

    norm.kernel(torch.ones(2))

    assert dict(norm.named_modules()) == before
    assert list(norm.state_dict()) == []


def test_parameter_names_are_untouched(compile_calls, mode) -> None:
    """Compiling the module rather than the bound method would reparent weights
    under _orig_mod and break reload_weights."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    tail = _Tail()
    before = [name for name, _ in tail.named_parameters()]

    tail.kernel(torch.ones(4, 8), torch.ones(4, 8))

    assert [name for name, _ in tail.named_parameters()] == before == ["weight"]


def test_each_instance_compiles_its_own_kernel(compile_calls, mode) -> None:
    """Caching per class would hand one instance another's graph."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)
    first, second = _Norm(), _Norm()

    first.kernel(torch.ones(2))
    second.kernel(torch.ones(2))

    assert len(compile_calls) == 2


def test_the_real_layers_opt_in() -> None:
    """Guards against the decorator or mixin being dropped from a layer."""
    from spyre_inference.custom_ops.gemma_rms_norm import SpyreGemmaRMSNorm
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm, SpyreTPAwareRMSNorm
    from spyre_inference.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    for cls, method in (
        (SpyreRMSNorm, "forward_oot"),
        (SpyreGemmaRMSNorm, "forward_oot"),
        (SpyreTPAwareRMSNorm, "forward_oot"),
        (SpyreVocabParallelEmbedding, "forward"),
    ):
        assert issubclass(cls, CompileOutermost), cls.__name__
        wrapped = getattr(cls, method)
        assert getattr(wrapped, "__wrapped__", None) is not None, f"{cls.__name__}.{method}"


@pytest.mark.compile
def test_an_enclosing_graph_absorbs_the_layer_without_breaking(mode) -> None:
    """A decorated layer inside a fullgraph block is traced into the block's graph."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)

    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import fresh_cache

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = _Norm()
            self.proj = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(self.norm.kernel(x))

    dynamo.reset()
    counters.clear()
    with fresh_cache():
        block = Block()
        block.compile(backend="inductor", fullgraph=True, dynamic=False)
        x = torch.ones(4, 8)
        with torch.inference_mode():
            out = block(x)

        # fullgraph=True would already have raised on a graph break.
        assert block.norm.spyre_compiled_kernel is None
        assert counters["stats"]["unique_graphs"] == 1

    with torch.inference_mode():
        expected = block.proj(x * 2)
    torch.testing.assert_close(out, expected)


@pytest.mark.compile
def test_a_real_compile_of_an_outermost_kernel_matches_eager(mode) -> None:
    """The head/tail path itself: torch.compile of a bound method, really traced."""
    mode(CompilationMode.STOCK_TORCH_COMPILE)

    import torch._dynamo as dynamo
    from torch._inductor.utils import fresh_cache

    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)

    dynamo.reset()
    with fresh_cache():
        tail = _Tail()
        with torch.inference_mode():
            got, got_residual = tail.kernel(x, residual)
        assert tail.spyre_compiled_kernel is not None

        eager = _Tail()
        with torch.inference_mode():
            eager.spyre_compile_enabled = False
            want, want_residual = eager.kernel(x, residual)

    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_residual, want_residual)
