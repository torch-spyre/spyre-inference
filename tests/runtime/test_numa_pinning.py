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

"""Workers pin to their card's NUMA node only when nothing else has chosen their CPUs."""

import os
from types import SimpleNamespace

import pytest
import vllm.envs as vllm_envs

from spyre_inference import numa_pinning
from spyre_inference.numa_pinning import format_cpu_list, parse_cpu_list, pin_to_card_node

CARD = "0000:3b:00.0"
NODE0 = "0-35,72-107"
NODE1 = "36-71,108-143"
ALL = "0-143"
WORKER = SimpleNamespace(distributed_executor_backend="mp", world_size=1)


def _worker(backend="mp", world_size=1):
    return SimpleNamespace(distributed_executor_backend=backend, world_size=world_size)


def _sysfs(tmp_path, card_node="0", nodes=(NODE0, NODE1), online=ALL, isolated=None, smt=False):
    (tmp_path / "bus/pci/devices" / CARD).mkdir(parents=True)
    (tmp_path / "bus/pci/devices" / CARD / "numa_node").write_text(f"{card_node}\n")
    for i, cpus in enumerate(nodes):
        (tmp_path / f"devices/system/node/node{i}").mkdir(parents=True)
        (tmp_path / f"devices/system/node/node{i}/cpulist").write_text(f"{cpus}\n")
    (tmp_path / "devices/system/cpu").mkdir(parents=True)
    (tmp_path / "devices/system/cpu/online").write_text(f"{online}\n")
    if isolated is not None:
        (tmp_path / "devices/system/cpu/isolated").write_text(f"{isolated}\n")
    if smt:
        # Two threads per core: CPU n and n + 72 are siblings, on the package of their node.
        for cpu in parse_cpu_list(online):
            topology = tmp_path / f"devices/system/cpu/cpu{cpu}/topology"
            topology.mkdir(parents=True)
            (topology / "physical_package_id").write_text(f"{(cpu % 72) // 36}\n")
            (topology / "core_id").write_text(f"{cpu % 72}\n")
    return tmp_path


@pytest.fixture
def affinity(monkeypatch):
    """The process affinity, seen and set through os.sched_{get,set}affinity."""
    state = {"cpus": parse_cpu_list(ALL), "set": False}

    def setaffinity(_tid, cpus):
        state["cpus"], state["set"] = set(cpus), True

    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(state["cpus"]), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", setaffinity, raising=False)
    return state


@pytest.fixture(autouse=True)
def budget(monkeypatch):
    """The CPU budget get_cpu_count reports; one CPU unless a test sets it. A ``logical``
    count that differs is the psutil fallback, which counts physical cores by default."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    def set_budget(cpus, logical=None):
        monkeypatch.setattr(
            numa_pinning,
            "get_cpu_count",
            lambda use_logical_cpus=False: (
                logical if use_logical_cpus and logical is not None else cpus,
                "test",
            ),
        )

    set_budget(1)
    return set_budget


def test_cpu_list_round_trips():
    assert parse_cpu_list(NODE0) == set(range(36)) | set(range(72, 108))
    assert format_cpu_list(parse_cpu_list(NODE0)) == NODE0
    assert format_cpu_list(parse_cpu_list("0,2-3,7")) == "0,2-3,7"


def test_pins_to_the_cards_node(tmp_path, affinity):
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path))
    assert affinity["cpus"] == parse_cpu_list(NODE0)


def test_pins_to_a_card_on_the_second_node(tmp_path, affinity):
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path, card_node="1"))
    assert affinity["cpus"] == parse_cpu_list(NODE1)


@pytest.mark.parametrize(
    ("multiprocessing", "backend", "pins"),
    [
        (False, "uni", False),
        (True, "uni", True),
        (False, "mp", True),
        (True, "external_launcher", False),  # each torchrun rank is the user's script
    ],
)
def test_an_engine_in_the_callers_process_is_left_alone(
    monkeypatch, tmp_path, affinity, multiprocessing, backend, pins
):
    # Pinning there would confine the user's script or pytest itself, after the engine too.
    monkeypatch.setattr(vllm_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing, raising=False)
    pin_to_card_node(_worker(backend), CARD, _sysfs(tmp_path))
    assert affinity["set"] == pins


@pytest.mark.parametrize(
    ("cpus", "logical", "world_size", "pins"),
    [
        # SPYRE_NUM_CPUS or a quota: logical CPUs, against node 0's 72 CPUs.
        (64, None, 1, True),
        (72, None, 1, True),
        (73, None, 1, False),
        (144, None, 2, True),  # each of 2 workers gets 72 threads
        (146, None, 2, False),
        # psutil fallback: physical cores, against node 0's 36 cores.
        (36, 72, 1, True),
        (37, 74, 1, False),
        (72, 144, 2, True),
        (74, 148, 2, False),
    ],
)
def test_the_cards_node_must_hold_each_workers_threads(
    tmp_path, affinity, budget, cpus, logical, world_size, pins
):
    # configure_threading gives each worker ceil(budget / world_size) threads, counted in the
    # budget's own unit; a node with fewer would oversubscribe them.
    budget(cpus, logical)
    pin_to_card_node(_worker(world_size=world_size), CARD, _sysfs(tmp_path, smt=True))
    assert affinity["set"] == pins


@pytest.mark.parametrize(("omp", "pins"), [("8", True), ("73", False), ("", True), ("x", True)])
def test_omp_num_threads_is_the_thread_count_checked(
    monkeypatch, tmp_path, affinity, budget, omp, pins
):
    # The pool OMP_NUM_THREADS sized is what the node has to hold; the budget is the fallback.
    budget(16)
    monkeypatch.setenv("OMP_NUM_THREADS", omp)
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path))
    assert affinity["set"] == pins


def test_a_node_without_online_cpus_is_left_alone(tmp_path, affinity):
    # A memory-only or CXL node, or one whose CPUs are all offline.
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path, card_node="2"))
    assert not affinity["set"]


def test_a_node_whose_cpus_are_all_isolated_is_left_alone(tmp_path, affinity):
    affinity["cpus"] = parse_cpu_list(NODE1)
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path, isolated=NODE0))
    assert not affinity["set"]


@pytest.mark.parametrize(("card", "card_node"), [(CARD, "-1"), (CARD, "garbage"), ("unknown", "0")])
def test_a_card_without_a_node_is_left_alone(tmp_path, affinity, card, card_node):
    # s390x reports -1; there is no card-local node to choose.
    pin_to_card_node(WORKER, card, _sysfs(tmp_path, card_node=card_node))
    assert not affinity["set"]


def test_a_single_node_host_is_left_alone(tmp_path, affinity):
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path, nodes=(ALL,)))
    assert not affinity["set"]


@pytest.mark.parametrize("confined", ["0-7", NODE1, "0-35"])
def test_an_affinity_someone_else_chose_is_kept(tmp_path, affinity, confined):
    # taskset, --numa-bind or a cpuset, whether inside the card's node or not.
    affinity["cpus"] = parse_cpu_list(confined)
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path))
    assert not affinity["set"]
    assert affinity["cpus"] == parse_cpu_list(confined)


def test_isolated_cpus_are_not_a_users_choice(tmp_path, affinity):
    # isolcpus= narrows every task's default mask to the housekeeping CPUs.
    affinity["cpus"] = parse_cpu_list("0-99,108-143")
    pin_to_card_node(WORKER, CARD, _sysfs(tmp_path, isolated="100-107"))
    assert affinity["cpus"] == parse_cpu_list("0-35,72-99")


@pytest.mark.parametrize(
    ("devices", "local_rank", "expected"),
    [
        (None, 1, "0000:bc:00.0"),  # unset: the runtime opens the cards in rank order
        ("1,0", 0, "0000:bc:00.0"),
        ("1", 1, "unknown"),  # too short for this rank
    ],
)
def test_pcie_address_of_a_workers_card(monkeypatch, devices, local_rank, expected):
    from spyre_inference.v1.worker.spyre_worker import _get_spyre_pcie_address

    monkeypatch.setenv("AIU_WORLD_RANK_0", "0000:ab:00.0")
    monkeypatch.setenv("AIU_WORLD_RANK_1", "0000:bc:00.0")
    if devices is None:
        monkeypatch.delenv("SPYRE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("SPYRE_DEVICES", devices)
    assert _get_spyre_pcie_address(local_rank) == expected
