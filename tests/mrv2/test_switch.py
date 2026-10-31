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

"""SPYRE_MODEL_RUNNER picks the worker and turns on vLLM's V2 runner flag."""

import copy
import os

import pytest
import vllm.logger
from vllm.config import VllmConfig

import spyre_inference
from spyre_inference import envs
from spyre_inference.platform import TorchSpyrePlatform
from spyre_inference.v1.worker.spyre import triton_gate


@pytest.fixture(scope="session")
def _base_config() -> VllmConfig:
    """Session-scoped, so the first VllmConfig()'s lazy imports stay out of the
    tests/mrv2 per-test budget."""
    return VllmConfig()


@pytest.fixture
def vllm_config(monkeypatch: pytest.MonkeyPatch, _base_config: VllmConfig) -> VllmConfig:
    """Built before the test sets SPYRE_MODEL_RUNNER, under this process's real gate."""
    config = copy.deepcopy(_base_config)
    config.parallel_config.worker_cls = "auto"
    # setenv first, so teardown also removes a value the platform hook adds.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    return config


def _set_runner(monkeypatch: pytest.MonkeyPatch, runner: str | None) -> None:
    if runner is None:
        monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("SPYRE_MODEL_RUNNER", runner)
    envs.clear_env_cache()
    monkeypatch.setattr(triton_gate, "check", lambda runner: None)


@pytest.mark.parametrize(
    ("runner", "worker_cls", "use_v2"),
    [
        (None, "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", "0"),
        ("v1", "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", "0"),
        ("v2", "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2", "1"),
    ],
)
def test_runner_switch(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: VllmConfig,
    runner: str | None,
    worker_cls: str,
    use_v2: str,
) -> None:
    _set_runner(monkeypatch, runner)

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.parallel_config.worker_cls == worker_cls
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == use_v2


def test_unknown_runner_is_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, "v3")

    with pytest.raises(ValueError, match="SPYRE_MODEL_RUNNER='v3' is not v1 or v2"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("runner", "has_triton", "triton_cpu"),
    [
        pytest.param("v2", False, True, id="v2_got_placeholders"),
        # The bug the gate fixes: V1 with Triton on.
        pytest.param("v1", True, True, id="v1_got_triton"),
    ],
)
def test_gate_must_match_the_runner(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: VllmConfig,
    runner: str,
    has_triton: bool,
    triton_cpu: bool,
) -> None:
    """E.g. SPYRE_MODEL_RUNNER set after `import vllm`, or a gate error at plugin load."""
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", runner)
    envs.clear_env_cache()
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", has_triton)
    monkeypatch.setattr(triton_gate, "has_triton_cpu", lambda: triton_cpu)

    with pytest.raises(
        ValueError, match=f"SPYRE_MODEL_RUNNER={runner} needs HAS_TRITON={not has_triton}"
    ):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(("runner", "requested", "pinned"), [("v2", "0", "1"), ("v1", "1", "0")])
def test_conflicting_vllm_use_v2_model_runner_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: VllmConfig,
    runner: str,
    requested: str,
    pinned: str,
) -> None:
    _set_runner(monkeypatch, runner)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == pinned
    assert vllm_config.use_v2_model_runner is (runner == "v2")


def test_runner_changed_after_import_is_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    """The env cache still holds the value register() saw; the check must not trust it."""
    monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    envs.clear_env_cache()
    assert envs.SPYRE_MODEL_RUNNER == "v1"
    monkeypatch.setattr(triton_gate, "has_triton_cpu", lambda: True)
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", "v2")

    with pytest.raises(ValueError, match="SPYRE_MODEL_RUNNER=v2 needs HAS_TRITON=True"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_register_logs_a_gate_error_and_keeps_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM's plugin probe would swallow a raise and silently pick another platform."""

    def _fail() -> None:
        raise ModuleNotFoundError("triton.backends")

    logged = []
    monkeypatch.setattr(triton_gate, "close", _fail)
    monkeypatch.setattr(triton_gate, "open", _fail)
    monkeypatch.setattr(
        vllm.logger,
        "init_logger",
        lambda name: type("L", (), {"exception": lambda self, msg: logged.append(msg)})(),
    )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")

    assert spyre_inference.register() == "spyre_inference.platform.TorchSpyrePlatform"
    assert logged == ["Spyre Triton gate could not be applied"]


@pytest.mark.parametrize(("runner", "pinned"), [(None, "0"), ("v2", "1")])
def test_register_pins_the_runner_before_any_config(
    monkeypatch: pytest.MonkeyPatch, runner: str | None, pinned: str
) -> None:
    """VllmConfig.__post_init__ reads use_v2_model_runner before the platform hook runs."""
    _set_runner(monkeypatch, runner)
    monkeypatch.setattr(triton_gate, "close", lambda: None)
    monkeypatch.setattr(triton_gate, "open", lambda: None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")

    spyre_inference.register()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == pinned
