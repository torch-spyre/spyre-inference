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

"""CPU-only tests for the CI planner's perf legs (.github/scripts/plan_test_matrix.py)."""

import importlib.util
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "vllm-benchmarks" / "benchmarks" / "spyre"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = _load("plan_test_matrix", _REPO / ".github" / "scripts" / "plan_test_matrix.py")
runner = _load("run_vllm_benchmarks", _REPO / ".github" / "scripts" / "run_vllm_benchmarks.py")


def test_perf_plan_skips_dev_configs(tmp_path):
    for name, model in [
        ("latency-tests.yaml", "ci-model"),
        ("latency-tests-dev.yaml", "dev-model"),
    ]:
        (tmp_path / name).write_text(
            yaml.safe_dump([{"test_name": "t", "parameters": {"model": model}}])
        )

    legs = plan.plan_perf(tmp_path)

    assert [leg["model"] for leg in legs] == ["ci-model"]


def test_perf_plan_sizes_each_leg_to_its_largest_tp(tmp_path):
    (tmp_path / "latency-tests.yaml").write_text(
        yaml.safe_dump(
            [
                {"test_name": "a", "parameters": {"model": "Small", "tensor_parallel_size": 1}},
                {"test_name": "b", "parameters": {"model": "Big", "tensor_parallel_size": 1}},
            ]
        )
    )
    (tmp_path / "serve-tests.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": {"server_parameters": {"tensor-parallel-size": 1}},
                "tests": [
                    {
                        "test_name": "c",
                        "server_parameters": {"model": "Big", "tensor-parallel-size": 4},
                    },
                    {
                        "test_name": "d",
                        "server_parameters": {"model": "Mid", "tensor-parallel-size": 2},
                    },
                ],
            }
        )
    )

    legs = {leg["model"]: leg["runs_on"][1] for leg in plan.plan_perf(tmp_path)}

    assert legs == {"small": "spyre_pf_x1", "big": "spyre_pf_x4", "mid": "spyre_pf_x2"}


def test_perf_plan_filters_models(tmp_path):
    (tmp_path / "latency-tests.yaml").write_text(
        yaml.safe_dump(
            [
                {"test_name": "a", "parameters": {"model": "One"}},
                {"test_name": "b", "parameters": {"model": "Two"}},
            ]
        )
    )

    assert [leg["model"] for leg in plan.plan_perf(tmp_path, {"two"})] == ["two"]


def test_perf_plan_covers_every_checked_in_model():
    """Each model in the CI configs gets a leg, so none is silently never benchmarked."""
    models = set()
    for path in _CONFIGS.glob("*-tests.yaml"):
        for config in runner._load_configs(path):
            models.add(runner._config_model(config).lower())

    assert {leg["model"] for leg in plan.plan_perf()} == models
