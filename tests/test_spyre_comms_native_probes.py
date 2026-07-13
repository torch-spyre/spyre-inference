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

"""Native libspyre_comms collective probes.

Each test attempts a *native* base-class collective on a real spyreccl
device_group at TP=2. Today every probe fails because the corresponding
SpyreCommsContext method (or torch-spyre stub) throws. They are
xfail(strict=True) so when libspyre_comms gains the native impl and a
comms RPM is rebuilt, the probe flips to passing, the strict-xfail
fails CI, and that's the signal to delete the matching manual fallback
in `spyre_inference.distributed.spyre_communicator` and (optionally)
revert `TorchSpyrePlatform.get_device_communicator_cls` to the base
class.

These tests are cheap to maintain but each spawns its own pair of
subprocesses, which is slow. They are gated on `>=2` Spyre cards so
they only run on the 2-card pods where the rest of TP=2 testing lives.

The probe bodies live in `tests/probes/tp_probe.py`; the subprocess
plumbing lives in the `run_tp_probe` fixture in `tests/conftest.py`.
"""

from __future__ import annotations

import os

import pytest


def _spyre_device_count() -> int:
    """Return the number of visible Spyre cards, or 0 if unavailable.

    Reads AIU_WORLD_SIZE (set by the Spyre runtime environment when
    cards are visible) instead of touching the Spyre runtime, so
    `uses_subprocess` tests don't import torch_spyre in the main
    pytest process.
    """
    try:
        return int(os.environ.get("AIU_WORLD_SIZE", "0"))
    except ValueError:
        return 0


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "libspyre_comms does not yet implement native allreduce; "
        "SpyreCommsContext::allreduce throws. When this flips to passing, "
        "delete SpyreCommunicator.all_reduce."
    ),
)
def test_native_all_reduce_works(run_tp_probe) -> None:
    run_tp_probe("native_all_reduce", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "torch-spyre's spyreccl backend stubs _allgather_base, so "
        "dist.all_gather_into_tensor fails even though libspyre_comms "
        "implements single-tensor allgather. When this flips to passing, "
        "the base DeviceCommunicatorBase.all_gather can replace "
        "SpyreCommunicator.all_gather."
    ),
)
def test_native_all_gather_into_tensor_works(run_tp_probe) -> None:
    run_tp_probe("native_all_gather_into_tensor", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_native_all_gather_list_works(run_tp_probe) -> None:
    run_tp_probe("native_all_gather_list", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_native_gather_works(run_tp_probe) -> None:
    run_tp_probe("native_gather", world_size=2)
