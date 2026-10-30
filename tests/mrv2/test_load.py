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

"""load.py and compile.py: the wrapped loader, the swapped upstream names, and a real
SpyreModelRunnerV2.load_model through upstream's own body."""

import os
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import CompilationMode
from vllm.v1.worker.gpu import model_runner as upstream
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre import compile, load
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

from .fakes import FakeModel

_SWAPPED = ("get_model_loader", "init_model_state", "PoolingRunner")


def _snapshot() -> dict[str, object]:
    return {name: getattr(upstream, name) for name in (*_SWAPPED, "compile_model_with_stock_torch")}


def _runner(**attrs) -> SimpleNamespace:
    defaults = {
        "spyre_device": torch.device("cpu"),
        "max_num_reqs": 4,
        "max_num_tokens": 2048,
        "lora_config": None,
        "speculator": None,
    }
    return SimpleNamespace(**{**defaults, **attrs})


def _vllm_config(runner_type: str = "generate", max_logprobs: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.STOCK_TORCH_COMPILE, static_forward_context={}
        ),
        model_config=SimpleNamespace(
            enforce_eager=False,
            runner_type=runner_type,
            dtype=torch.float16,
            max_logprobs=max_logprobs,
        ),
    )


class _Loader:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.load_config = "load-config"

    def load_model(self, *, vllm_config, model_config) -> nn.Module:
        return self.model


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the shared load steps _SpyreModelLoader runs, in order."""
    calls: list[str] = []

    def record(name, result=None):
        return lambda *args, **kwargs: calls.append(name) or result

    for name in (
        "install_pad_loaders",
        "check_spyre_model",
        "fix_padded_model",
        "resolve_kv_scales",
        "move_model_to_spyre",
        "apply_multimodal_patches",
    ):
        monkeypatch.setattr(load, name, record(name))
    monkeypatch.setattr(load, "get_model_architecture", lambda config: (None, "FakeModel"))
    monkeypatch.setattr(load, "configure_pooling_for_spyre", record("configure_pooling", True))
    monkeypatch.setattr(load, "encoder_len_ladder", lambda config: [])
    monkeypatch.setattr(
        load, "compile_for_spyre", lambda model, config: calls.append("compile") or (model, 3)
    )
    monkeypatch.setattr(load.SpyreShapeBucketer, "for_config", staticmethod(lambda config: None))
    return calls


def test_the_loader_runs_the_shared_steps_and_returns_the_wrapper(steps) -> None:
    model = FakeModel()
    runner = _runner()
    config = _vllm_config()

    wrapper = load._SpyreModelLoader(_Loader(model), runner).load_model(
        vllm_config=config, model_config=config.model_config
    )

    assert steps == [
        "install_pad_loaders",
        "check_spyre_model",
        "fix_padded_model",
        "resolve_kv_scales",
        "move_model_to_spyre",
        "apply_multimodal_patches",
        "compile",
    ]
    assert isinstance(wrapper, _SpyreModelWrapper) and wrapper._model is model
    assert runner.pooling_on_spyre is False and runner.compiled_tower_blocks == 3
    assert wrapper._keep_outputs_on_device is False


def test_a_pooling_model_keeps_its_outputs_on_spyre(steps) -> None:
    runner = _runner()
    config = _vllm_config("pooling")

    wrapper = load._SpyreModelLoader(_Loader(FakeModel()), runner).load_model(
        vllm_config=config, model_config=config.model_config
    )

    assert "configure_pooling" in steps
    assert runner.pooling_on_spyre is True and wrapper._keep_outputs_on_device is True


@pytest.mark.parametrize(
    ("runner_type", "max_logprobs", "max_num_tokens", "want"),
    [
        pytest.param("generate", 20, 2048, [4, 1024], id="prompt-logprobs-chunk"),
        pytest.param("generate", 20, 512, [4, 512], id="chunk-capped-at-the-batch"),
        pytest.param("generate", 0, 2048, [4], id="logprobs-disabled"),
        pytest.param("pooling", 20, 2048, [4], id="pooling"),
    ],
)
def test_row_buckets_cover_the_prompt_logprobs_chunk(
    steps, monkeypatch, runner_type, max_logprobs, max_num_tokens, want
) -> None:
    """MRV2 projects prompt logprobs 1024 rows at a time, wider than any sampled-row
    bucket, so compiled serving would compile that width mid-request."""
    bucketer = SimpleNamespace(bucket_sizes=[64, 128, 2048])
    monkeypatch.setattr(load.SpyreShapeBucketer, "for_config", staticmethod(lambda c: bucketer))
    config = _vllm_config(runner_type, max_logprobs)

    wrapper = load._SpyreModelLoader(
        _Loader(FakeModel()), _runner(max_num_tokens=max_num_tokens)
    ).load_model(vllm_config=config, model_config=config.model_config)

    assert wrapper._logits_row_buckets == want


def test_the_loader_forwards_everything_else() -> None:
    loader = load._SpyreModelLoader(_Loader(FakeModel()), _runner())
    assert loader.load_config == "load-config"


def test_swapped_names_are_restored_after_a_raise() -> None:
    before = _snapshot()
    with pytest.raises(RuntimeError, match="boom"), load._patched(_runner()):
        during = _snapshot()
        assert all(during[name] is not before[name] for name in during)
        assert upstream.PoolingRunner is SpyrePoolingRunner
        raise RuntimeError("boom")
    assert _snapshot() == before


def test_upstreams_stock_compile_is_a_no_op_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, **kw: compiled.append(self))
    with compile.no_stock_compile():
        upstream.compile_model_with_stock_torch(FakeModel(), _vllm_config())
    assert compiled == []


def test_load_refuses_lora_spec_decode_and_caller_built_models() -> None:
    with pytest.raises(NotImplementedError, match="LoRA"):
        load.load_and_wrap(_runner(lora_config=object()))
    with pytest.raises(NotImplementedError, match="Speculative decoding"):
        load.load_and_wrap(_runner(speculator=object()))
    with pytest.raises(NotImplementedError, match="caller-built"):
        load.load_and_wrap(_runner(), model=FakeModel())


def test_load_wires_the_pooling_runner_to_the_model_state(monkeypatch) -> None:
    pooling_runner = SpyrePoolingRunner.__new__(SpyrePoolingRunner)
    model_state = SimpleNamespace(encoder_plan="plan")

    def upstream_load_model(runner, *args, **kwargs):
        runner.model_state, runner.pooling_runner = model_state, pooling_runner
        runner.pooling_on_spyre = True
        runner.model_memory_usage = 123

    monkeypatch.setattr(GPUModelRunner, "load_model", upstream_load_model)
    runner = _runner(spyre_device=torch.device("meta"))

    load.load_and_wrap(runner)

    assert runner.model_memory_usage == 0
    assert pooling_runner.pooling_on_spyre is True
    assert pooling_runner.spyre_device == torch.device("meta")
    assert pooling_runner.encoder_plan_source() == "plan"


def test_unwrap_reaches_the_bare_model() -> None:
    model = FakeModel()
    compiled = SimpleNamespace(_orig_mod=model)
    wrapper = _SpyreModelWrapper(compiled, torch.device("cpu"), model_dtype=torch.float16)  # ty: ignore[invalid-argument-type]
    assert load.unwrap(wrapper) is model
    assert load.unwrap(model) is model


def test_generation_tasks_are_read_off_the_bare_model() -> None:
    """Upstream's is_text_generation_model needs a vllm_config kwarg on __init__,
    which the wrapper lacks, so a generative model would report no tasks."""
    model = FakeModel()

    class Upstream:
        def get_supported_generation_tasks(self):
            return (self.model,)

    class State(SpyreModelStateMixin, Upstream):
        pass

    state = State.__new__(State)
    wrapper = _SpyreModelWrapper(model, torch.device("cpu"), model_dtype=torch.float16)
    state.model = wrapper

    assert state.get_supported_generation_tasks() == (model,)
    assert state.model is wrapper


@pytest.mark.subprocess_isolated
def test_load_model_through_upstreams_body() -> None:
    """A real SpyreModelRunnerV2 and upstream's load_model, on a dummy-weight decoder
    kept on the host, in a fresh process for its distributed groups and device shims."""
    script = textwrap.dedent("""
        import torch
        from vllm.config import (
            CacheConfig, CompilationConfig, CompilationMode, LoadConfig, ModelConfig,
            VllmConfig, set_current_vllm_config,
        )
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.v1.worker.gpu import model_runner as upstream

        from spyre_inference.v1.worker.spyre import device_shims
        from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2
        from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

        stock = []
        upstream.compile_model_with_stock_torch = lambda model, config: stock.append(model)
        device_shims.install()
        config = VllmConfig(
            model_config=ModelConfig(
                model="ibm-ai-platform/micro-g3.3-8b-instruct-1b", max_model_len=256,
                enforce_eager=True,
            ),
            cache_config=CacheConfig(block_size=64),
            load_config=LoadConfig(load_format="dummy"),
            compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE),
        )
        with set_current_vllm_config(config):
            init_distributed_environment(1, 0, "file://{store}", 0, "gloo")
            ensure_model_parallel_initialized(1, 1)
            runner = SpyreModelRunnerV2(config, torch.device("cpu"))
            runner.load_model()
            print("WRAPPER", isinstance(runner.model, _SpyreModelWrapper))
            print("STATE", isinstance(runner.model_state, SpyreModelStateMixin))
            print("STATE_HOLDS_WRAPPER", runner.model_state.model is runner.model)
            print("SAMPLER", runner.sampler is not None)
            print("UNWRAPPED", type(runner.get_model()).__name__)
            print("TASKS", runner.get_supported_tasks())
            print("STOCK_COMPILES", len(stock))
            print("MEMORY", runner.model_memory_usage)
    """)
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-c", script.replace("{store}", os.path.join(tmp, "store"))],
            capture_output=True,
            text=True,
            env={**os.environ, "SPYRE_MODEL_RUNNER": "v2"},
            timeout=600,
        )
    assert result.returncode == 0, result.stderr[-4000:]
    got = dict(
        line.split(" ", 1) for line in result.stdout.splitlines() if line.split(" ")[0].isupper()
    )
    assert got == {
        "WRAPPER": "True",
        "STATE": "True",
        "STATE_HOLDS_WRAPPER": "True",
        "SAMPLER": "True",
        "UNWRAPPED": "GraniteForCausalLM",
        "TASKS": "('generate',)",
        "STOCK_COMPILES": "0",
        "MEMORY": "0",
    }, result.stdout[-3000:]
