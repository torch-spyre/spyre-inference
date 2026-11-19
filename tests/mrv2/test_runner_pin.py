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

"""The platform pins vLLM's V2 runner flag, picks the V2 worker, and rejects what V2
lacks, a mismatched Triton gate, and the removed SPYRE_MODEL_RUNNER=v1."""

import copy
import os
from types import SimpleNamespace

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
    """Built under this process's real gate."""
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
    monkeypatch.setattr(triton_gate, "check", lambda mrv2_kernels: None)


@pytest.mark.parametrize("runner", [None, "v2"])
def test_the_v2_worker_is_picked_and_pinned(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig, runner: str | None
) -> None:
    _set_runner(monkeypatch, runner)

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert (
        vllm_config.parallel_config.worker_cls
        == "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2"
    )
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"


@pytest.mark.parametrize("runner", ["v1", "v3"])
def test_any_other_runner_is_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig, runner: str
) -> None:
    _set_runner(monkeypatch, runner)

    with pytest.raises(ValueError, match=f"SPYRE_MODEL_RUNNER='{runner}': the V1 model runner"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("feature", "configure"),
    [
        pytest.param(
            "speculative decoding other than DFlash/XPress",
            lambda c: setattr(c, "speculative_config", SimpleNamespace(use_dflash=lambda: False)),
            id="spec-decode-non-dflash",
        ),
        pytest.param(
            "batch-sharded sampling",
            lambda c: setattr(c.parallel_config, "enable_batch_sharded_sampling", True),
            id="batch-sharded-sampling",
        ),
        pytest.param(
            "cudagraph_mm_encoder",
            lambda c: setattr(c.compilation_config, "cudagraph_mm_encoder", True),
            id="cudagraph-mm-encoder",
        ),
        pytest.param(
            "prefill context parallelism",
            lambda c: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
            id="pcp",
        ),
    ],
)
def test_unsupported_v2_features_fail_at_startup(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig, feature: str, configure
) -> None:
    _set_runner(monkeypatch, "v2")
    configure(vllm_config)

    with pytest.raises(ValueError, match=f"V2 model runner does not support {feature}"):
        TorchSpyrePlatform._reject_unsupported_v2_features(vllm_config)


def test_a_plain_v2_config_passes_the_feature_check(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, "v2")
    TorchSpyrePlatform._reject_unsupported_v2_features(vllm_config)


def test_v2_check_and_update_config_runs_the_feature_check(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, "v2")
    vllm_config.compilation_config.cudagraph_mm_encoder = True

    with pytest.raises(ValueError, match="does not support cudagraph_mm_encoder"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("kernels", "has_triton"),
    [
        pytest.param("triton", False, id="triton_got_placeholders"),
        pytest.param("host", True, id="host_got_triton"),
    ],
)
def test_gate_must_match_the_kernel_source(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: VllmConfig,
    kernels: str,
    has_triton: bool,
) -> None:
    """E.g. SPYRE_MRV2_KERNELS set after `import vllm`, or a gate error at plugin load."""
    monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    monkeypatch.setenv("SPYRE_MRV2_KERNELS", kernels)
    envs.clear_env_cache()
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", has_triton)
    monkeypatch.setattr(triton_gate, "has_triton_cpu", lambda: True)

    with pytest.raises(
        ValueError, match=f"SPYRE_MRV2_KERNELS={kernels} needs HAS_TRITON={not has_triton}"
    ):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_v2_on_triton_needs_triton_cpu(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    monkeypatch.setenv("SPYRE_MRV2_KERNELS", "triton")
    envs.clear_env_cache()
    monkeypatch.setattr(triton_gate, "has_triton_cpu", lambda: False)

    with pytest.raises(ValueError, match="needs triton-cpu"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_a_conflicting_vllm_use_v2_model_runner_is_overridden(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert vllm_config.use_v2_model_runner is True


def test_a_config_that_forces_v1_trips_the_pin(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, None)
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", property(lambda self: False))

    with pytest.raises(RuntimeError, match="forces the V1 model runner"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


def test_kernels_changed_after_import_are_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    """The env cache still holds the value register() saw; the check must not trust it."""
    monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    monkeypatch.delenv("SPYRE_MRV2_KERNELS", raising=False)
    envs.clear_env_cache()
    assert envs.SPYRE_MRV2_KERNELS == "host"
    monkeypatch.setattr(triton_gate, "has_triton_cpu", lambda: True)
    monkeypatch.setenv("SPYRE_MRV2_KERNELS", "triton")

    with pytest.raises(ValueError, match="SPYRE_MRV2_KERNELS=triton needs HAS_TRITON=True"):
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


@pytest.mark.parametrize("requested", ["", "0"])
def test_register_pins_v2_before_any_config(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    """VllmConfig.__post_init__ reads use_v2_model_runner before the platform hook runs."""
    monkeypatch.setattr(triton_gate, "close", lambda: None)
    monkeypatch.setattr(triton_gate, "open", lambda: None)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", requested)

    spyre_inference.register()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
