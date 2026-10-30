# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-scoped hint selection and CPU value preservation."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from spyre_inference import config as model_config
from spyre_inference.config.kernel_hints import _row_hints, configure_kernel_hints
from spyre_inference.custom_ops.linear import SpyreTransposedWeightMethod, spyre_linear_t
from spyre_inference.custom_ops.rotary_embedding import _rotate_neox_2x2, _SpyreRotaryMixin


class _Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = SpyreTransposedWeightMethod()
        self.weight = torch.nn.Parameter(torch.empty(16, 128, dtype=torch.float16))


class _Rope(torch.nn.Module, _SpyreRotaryMixin):
    def __init__(self):
        torch.nn.Module.__init__(self)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _Linear()
        self.o_proj = _Linear()
        self.down_proj = _Linear()
        self.unrelated = _Linear()
        self.rotary_emb = _Rope()


def _config(model="ibm-granite/granite-4.1-8b", tp=4, quantization=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model, max_model_len=32768, quantization=quantization),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.mark.parametrize(
    "model_id,tp,quantization,cores,enabled",
    [
        ("ibm-granite/granite-4.1-8b", 4, None, 32, True),
        ("ibm-granite/granite-4.1-8b", 1, None, 32, False),
        ("ibm-granite/granite-3.3-8b-instruct", 4, None, 32, True),
        ("ibm-granite/granite-4.1-8b-fp8", 4, None, 32, False),
        ("another-model/same-shapes", 4, None, 32, False),
        ("/local/granite-checkpoint", 4, None, 32, False),
        ("ibm-granite/granite-4.1-8b", 4, "fp8", 32, False),
        ("ibm-granite/granite-4.1-8b", 4, None, 16, False),
    ],
)
def test_kernel_hints_are_model_tp_and_role_scoped(
    monkeypatch, model_id, tp, quantization, cores, enabled
):
    from torch_spyre._inductor import config as spyre_config

    monkeypatch.setattr(model_config, "current_platform", lambda: "x86_64")
    monkeypatch.setattr(spyre_config, "sencores", cores)
    model = _Model()
    configure_kernel_hints(model, _config(model_id, tp, quantization))
    for layer in (model.qkv_proj, model.o_proj, model.down_proj):
        assert getattr(layer, "spyre_work_div", {}) == ({512: (4, 8)} if enabled else {})
    assert getattr(model.rotary_emb, "spyre_work_div", {}) == ({512: 32} if enabled else {})
    assert not hasattr(model.unrelated, "spyre_work_div")


def test_kernel_hints_keep_quantized_layers_and_incompatible_columns_unhinted(monkeypatch):
    from torch_spyre._inductor import config as spyre_config

    monkeypatch.setattr(model_config, "current_platform", lambda: "x86_64")
    monkeypatch.setattr(spyre_config, "sencores", 32)
    model = _Model()
    model.qkv_proj.quant_method = object()
    model.o_proj.weight = torch.nn.Parameter(torch.empty(16, 130, dtype=torch.float16))
    configure_kernel_hints(model, _config())
    assert not hasattr(model.qkv_proj, "spyre_work_div")
    assert model.o_proj.spyre_work_div == {}
    assert model.down_proj.spyre_work_div == {512: (4, 8)}


@pytest.mark.parametrize(
    "table", [{0: [4, 8]}, {512: [0, 8]}, {512: [4]}, {7: [4, 8]}, {True: [1, 1]}, []]
)
def test_invalid_kernel_divisions_are_rejected(table):
    with pytest.raises(ValueError, match="[Kk]ernel hint"):
        _row_hints(table, 2, 32)


@pytest.fixture
def hint_scopes(monkeypatch):
    from torch_spyre._inductor import propagate_hints

    scopes = []

    @contextmanager
    def record(**kwargs):
        scopes.append(kwargs)
        yield

    monkeypatch.setattr(propagate_hints, "spyre_hint", record)
    return scopes


@pytest.mark.parametrize("rows", [1, 7, 16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_generic_linear_uses_only_explicit_row_hints(rows, use_bias, hint_scopes):
    x = torch.arange(rows * 16, dtype=torch.float16).reshape(rows, 16) / 1024
    weight = torch.arange(16 * 64, dtype=torch.float16).reshape(16, 64) / 2048
    bias = torch.arange(64, dtype=torch.float16) if use_bias else None
    expected = x @ weight
    if bias is not None:
        expected = expected + bias
    actual = spyre_linear_t(x, weight, bias, work_div={16: (2, 4)})
    assert torch.equal(actual, expected)
    assert len(hint_scopes) == (1 if rows == 16 else 0)
    if hint_scopes:
        assert tuple(hint_scopes[0]["work_div"].values()) == (2, 4)
    hint_scopes.clear()
    assert torch.equal(spyre_linear_t(x, weight, bias), expected)
    assert not hint_scopes


@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("head_size", [128, 256])
def test_generic_rope_uses_only_explicit_row_hints(rows, head_size, hint_scopes):
    x = torch.arange(rows * 2 * head_size, dtype=torch.float16).reshape(rows, -1) / 1024
    rotation = (
        torch.arange(rows * 4 * (head_size // 2), dtype=torch.float16).reshape(
            rows, 2, 2, head_size // 2
        )
        / 2048
    )
    expected = _rotate_neox_2x2(x, rotation, head_size)
    assert not hint_scopes
    actual = _rotate_neox_2x2(x, rotation, head_size, work_div={16: 8})
    assert torch.equal(actual, expected)
    assert len(hint_scopes) == (1 if rows == 16 else 0)
    if hint_scopes:
        assert tuple(hint_scopes[0]["work_div"].values()) == (8,)


def test_unconfigured_granite_shapes_do_not_imply_hints(hint_scopes):
    x = torch.empty(512, 4096, device="meta", dtype=torch.float16)
    weight = torch.empty(4096, 1536, device="meta", dtype=torch.float16)
    assert spyre_linear_t(x, weight, None).shape == (512, 1536)
    rope = torch.empty(512, 1024, device="meta", dtype=torch.float16)
    rotation = torch.empty(512, 2, 2, 64, device="meta", dtype=torch.float16)
    assert _rotate_neox_2x2(rope, rotation, 128).shape == rope.shape
    assert not hint_scopes


def test_linear_method_passes_layer_hints(hint_scopes):
    layer = _Linear()
    layer.weight.data.fill_(0.25)
    layer.spyre_row_padding = 0
    layer.spyre_work_div = {16: (2, 4)}
    x = torch.arange(256, dtype=torch.float16).reshape(16, 16) / 1024
    assert torch.equal(layer.quant_method.apply(layer, x), x @ layer.weight)
    assert len(hint_scopes) == 1


def test_rotary_module_passes_layer_hints(hint_scopes):
    layer = _Rope()
    layer.head_size = 128
    layer._padded_inner = 64
    layer._rotate_half = None
    layer.spyre_work_div = {16: 8}
    rotation = torch.arange(16 * 256, dtype=torch.float16).reshape(16, 256) / 4096
    layer._get_device_rotation_cache = lambda: rotation
    x = torch.arange(16 * 256, dtype=torch.float16).reshape(16, 256) / 8192
    expected = _rotate_neox_2x2(x, rotation.view(16, 2, 2, 64), 128)
    query, key = layer.forward_oot(torch.arange(16), x, x)
    assert torch.equal(query, expected) and torch.equal(key, expected)
    assert len(hint_scopes) == 2


@pytest.mark.parametrize("length", [1024, 32768])
def test_row_hints_do_not_depend_on_serving_context_length(monkeypatch, length):
    from torch_spyre._inductor import config as spyre_config

    monkeypatch.setattr(model_config, "current_platform", lambda: "x86_64")
    monkeypatch.setattr(spyre_config, "sencores", 32)
    config = _config("ibm-granite/granite-3.3-8b-instruct")
    config.model_config.max_model_len = length
    model = _Model()
    configure_kernel_hints(model, config)
    assert model.down_proj.spyre_work_div == {512: (4, 8)}


@pytest.mark.parametrize("first", [{"rope": {512: 32}}, {}])
def test_context_entries_cannot_silently_select_different_hints(monkeypatch, first):
    from spyre_inference.config import DeviceConfig, ModelEntry, ServingConfig

    entry = ModelEntry(
        "test/model",
        None,
        [
            ServingConfig(4, 1024, 1, DeviceConfig(kernel_hints=first)),
            ServingConfig(4, 32768, 1, DeviceConfig(kernel_hints={"rope": {512: 16}})),
        ],
    )
    monkeypatch.setattr(model_config, "model_registry", lambda: {"test/model": entry})
    with pytest.raises(ValueError, match="agree across context lengths"):
        configure_kernel_hints(_Model(), _config("test/model"))


def test_loader_attaches_hints_before_device_transfer():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / "spyre_inference/v1/worker/spyre_model_runner.py"
    tree = ast.parse(source.read_text())
    loader = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
    hints = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "configure_kernel_hints"
    ]
    assert len(hints) == 1
    load = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load_model"
    )
    move = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "move_model_to_spyre"
    )
    fix = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "fix_padded_model"
    )
    assert load.lineno < fix.lineno < hints[0].lineno < move.lineno


def test_kernel_hints_refuse_unsupported_host_platform(monkeypatch):
    monkeypatch.setattr(model_config, "current_platform", lambda: "unsupported-host")
    model = _Model()
    configure_kernel_hints(model, _config())
    assert not hasattr(model.down_proj, "spyre_work_div")
    assert not hasattr(model.rotary_emb, "spyre_work_div")
