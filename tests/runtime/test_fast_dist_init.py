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

"""SPYRE_FAST_DIST_INIT applies to one worker inside EngineCore only.

Everywhere else (TP > 1, the mp executor, DP > 1, a rank other than 0, the flag unset)
the worker must do what it does without the flag: the default dist backend, the seed
and the memory snapshot in init_device, and a plain load_model. No card is needed: in
those cases the module never reaches torch-spyre.
"""

import logging
from types import SimpleNamespace

import pytest
from vllm.model_executor.model_loader import base_loader

from spyre_inference import envs
from spyre_inference.v1.worker import fast_dist_init

_DEFAULT_BACKEND = "cpu:gloo,spyre:spyreccl"


def _worker(world_size=1, data_parallel_size=1, executor="uni", local_rank=0):
    parallel = SimpleNamespace(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        distributed_executor_backend=executor,
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel), local_rank=local_rank
    )


@pytest.fixture
def flag(monkeypatch):
    def set_flag(value):
        if value is None:
            monkeypatch.delenv("SPYRE_FAST_DIST_INIT", raising=False)
        else:
            monkeypatch.setenv("SPYRE_FAST_DIST_INIT", value)
        envs.clear_env_cache()

    yield set_flag
    envs.clear_env_cache()


@pytest.fixture
def no_card(monkeypatch):
    """Any start of the card fails the test."""

    def fail(self):
        raise AssertionError("the card was started")

    monkeypatch.setattr(fast_dist_init.CardStart, "start", fail)


def _assert_no_change(worker):
    card = fast_dist_init.start(worker)
    assert card is None
    assert fast_dist_init.dist_backend(card, _DEFAULT_BACKEND) == _DEFAULT_BACKEND
    assert not fast_dist_init.defers_card_work(card)
    worker._fast_card_start = card
    hook = base_loader.process_weights_after_loading
    with fast_dist_init.join_after_weight_read(worker):
        assert base_loader.process_weights_after_loading is hook
    assert base_loader.process_weights_after_loading is hook


@pytest.mark.parametrize(
    "worker",
    [
        _worker(world_size=2),
        _worker(world_size=4),
        _worker(executor="mp"),
        _worker(executor="ray"),
        _worker(data_parallel_size=2),
        _worker(local_rank=1),
    ],
    ids=["tp2", "tp4", "mp-executor", "ray-executor", "dp2", "local-rank-1"],
)
@pytest.mark.parametrize("value", ["1", "gloo"])
def test_flag_set_where_it_does_not_apply_is_a_no_op(worker, value, flag, no_card, caplog):
    flag(value)
    with caplog.at_level(logging.WARNING):
        _assert_no_change(worker)
    assert "Ignoring SPYRE_FAST_DIST_INIT" in caplog.text


@pytest.mark.parametrize("value", [None, "", "0", "off", "false"])
def test_flag_off_is_a_no_op(value, flag, no_card, caplog):
    flag(value)
    with caplog.at_level(logging.DEBUG):
        _assert_no_change(_worker())
    assert "SPYRE_FAST_DIST_INIT" not in caplog.text


def test_a_typo_raises(flag):
    flag("yes please")
    with pytest.raises(ValueError, match="SPYRE_FAST_DIST_INIT"):
        fast_dist_init.start(_worker())


def test_gloo_mode_only_changes_the_backend(flag, monkeypatch):
    flag("gloo")
    monkeypatch.setattr(fast_dist_init.importlib, "import_module", pytest.fail)
    card = fast_dist_init.start(_worker())
    assert card is not None
    assert fast_dist_init.dist_backend(card, _DEFAULT_BACKEND) == "gloo"
    assert not fast_dist_init.defers_card_work(card)


def test_the_join_runs_before_process_weights_after_loading(flag, monkeypatch):
    flag("1")
    monkeypatch.setattr(fast_dist_init.CardStart, "start", lambda self: None)
    order = []
    monkeypatch.setattr(fast_dist_init.CardStart, "join", lambda self: order.append("join"))
    hook = base_loader.process_weights_after_loading
    monkeypatch.setattr(
        base_loader, "process_weights_after_loading", lambda *a, **k: order.append("hook")
    )
    patched_hook = base_loader.process_weights_after_loading

    worker = _worker()
    card = fast_dist_init.start(worker)
    assert fast_dist_init.dist_backend(card, _DEFAULT_BACKEND) == "gloo"
    assert fast_dist_init.defers_card_work(card)
    worker._fast_card_start = card
    with fast_dist_init.join_after_weight_read(worker):
        base_loader.process_weights_after_loading()
    assert order[:2] == ["join", "hook"]
    assert base_loader.process_weights_after_loading is patched_hook
    assert hook is not None
