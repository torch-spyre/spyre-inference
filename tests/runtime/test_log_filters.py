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

import ast
import importlib
import inspect
import logging

import pytest
import torch

from spyre_inference import log_filters
from spyre_inference.log_filters import FILTERS


def _string_constants(module_name: str) -> set[str]:
    # Adjacent literals are one Constant in the AST, so a template split over lines matches.
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


@pytest.mark.parametrize("entry", FILTERS, ids=lambda e: e.template[:40])
def test_template_still_logged_by_vllm(entry):
    assert entry.template in _string_constants(entry.logger), (
        f"{entry.logger} no longer logs {entry.template!r}: vLLM reworded or removed it. "
        "Update the entry in spyre_inference/log_filters.py and its row in "
        "docs/contributing/vllm-upgrade-procedure.md (Step 2b)."
    )


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


_CAST = FILTERS[0].template
_TRITON = next(e.template for e in FILTERS if e.template.startswith("Triton"))


def _emit(name: str, msg: str, *args) -> list[str]:
    log_filters.install()
    target = logging.getLogger(name)
    handler = _Collect()
    level, disabled = target.level, target.disabled
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    target.disabled = False
    try:
        target.warning(msg, *args)
    finally:
        target.removeHandler(handler)
        target.setLevel(level)
        target.disabled = disabled
    return handler.messages


@pytest.mark.parametrize("entry", FILTERS, ids=lambda e: e.template[:40])
def test_other_messages_on_the_logger_pass(entry):
    assert _emit(entry.logger, "spyre-marker %s", 1) == ["spyre-marker 1"]


@pytest.mark.parametrize(
    "entry", [e for e in FILTERS if e.drop is None], ids=lambda e: e.template[:40]
)
def test_unconditional_entries_drop(entry):
    args = (1,) * entry.template.count("%")
    assert _emit(entry.logger, entry.template, *args) == []


@pytest.mark.parametrize(
    "weight, param, dropped",
    [
        (torch.bfloat16, torch.float16, True),
        (torch.float32, torch.float16, True),
        (torch.int8, torch.float16, False),
        (torch.float8_e4m3fn, torch.float16, False),
        (torch.float16, torch.bfloat16, False),
    ],
)
def test_weight_cast_dropped_only_for_the_platform_cast(weight, param, dropped):
    out = _emit("vllm.model_executor.models.utils", _CAST, "w", weight, param)
    assert (out == []) is dropped


@pytest.mark.parametrize("kernels, dropped", [("host", True), ("triton", False)])
def test_triton_warning_kept_when_mrv2_runs_triton(monkeypatch, kernels, dropped):
    monkeypatch.setenv("SPYRE_MRV2_KERNELS", kernels)
    assert (_emit("vllm.platforms.cpu", _TRITON) == []) is dropped
