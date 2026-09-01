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

"""Spyre collective probes.

Each test attempts one collective on a real spyreccl device_group at TP=2,
covering both routes a call site can take: plain `dist.*`, and
`_c10d_functional.*` (lowered to a native device op under `torch.compile`).

Blocked probes are xfail(strict=True): when the missing piece lands they flip to
passing, the strict-xfail fails CI, and that is the signal to delete the matching
workaround in `spyre_inference.distributed.spyre_communicator`.

These tests are cheap to maintain but each spawns its own pair of
subprocesses, which is slow. They are gated on `>=2` Spyre cards so
they only run on the 2-card pods where the rest of TP=2 testing lives.

The probe bodies live in `tests/probes/tp_probe.py`; the subprocess
plumbing lives in the `run_tp_probe` fixture in
`tests/plugin/spyre_testing_plugin/pytest_plugin.py`.
"""

from __future__ import annotations

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_native_all_reduce_works(run_tp_probe) -> None:
    run_tp_probe("native_all_reduce", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
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
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_native_all_gather_list_works(run_tp_probe) -> None:
    run_tp_probe("native_all_gather_list", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_native_gather_works(run_tp_probe) -> None:
    run_tp_probe("native_gather", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_functional_all_reduce_eager_works(run_tp_probe) -> None:
    """`SpyreCommunicator.all_reduce` uses this form under `--enforce-eager` too."""
    run_tp_probe("functional_all_reduce_eager", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_compiled_all_reduce_works(run_tp_probe) -> None:
    """The row-parallel-linear reduction as it appears in the compiled graph."""
    run_tp_probe("compiled_all_reduce", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_compiled_all_reduce_multi_round_works(run_tp_probe) -> None:
    """64 sequential compiled all_reduces sharing one WSI."""
    run_tp_probe("compiled_all_reduce_multi_round", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_compiled_all_reduce_multi_block_works(run_tp_probe) -> None:
    """32 separately-compiled block fns × 2 all_reduces, mimicking STOCK_TORCH_COMPILE."""
    run_tp_probe("compiled_all_reduce_multi_block", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
def test_compiled_all_gather_works(run_tp_probe) -> None:
    """vLLM's concat-style all_gather, compiled, on a stick-aligned width."""
    run_tp_probe("compiled_all_gather_lastdim", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Eager `_c10d_functional.all_gather_into_tensor` routes to "
        "allgather_into_tensor_coalesced, which the spyreccl backend rejects. "
        "Blocker 1 of 2 for making SpyreCommunicator.all_gather functional."
    ),
)
def test_functional_all_gather_eager_works(run_tp_probe) -> None:
    run_tp_probe("functional_all_gather_eager", world_size=2)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
@pytest.mark.skipif(
    spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "`spyre::all_gather_async` reassembles by narrowing the output along "
        "dim 0, so rank r writes at storage offset r * per_rank_numel, which "
        "copy_from_d2d requires to be 64-aligned. Blocker 2 of 2, and the reason "
        "SpyreCommunicator.all_gather pads on CPU."
    ),
)
def test_compiled_all_gather_unaligned_works(run_tp_probe) -> None:
    run_tp_probe("compiled_all_gather_lastdim_unaligned", world_size=2)
