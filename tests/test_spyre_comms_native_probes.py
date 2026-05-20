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
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap

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


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Each probe runs the same setup boilerplate (env-rendezvous, vllm
# config, init_worker_distributed_environment) and then a small probe
# body specific to the collective being tested. The shared prologue is
# templated below; each probe substitutes `__PROBE_BODY__`.
_PROBE_TEMPLATE = textwrap.dedent(
    """
    import os
    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops"
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    import torch
    import torch.distributed as dist
    import torch_spyre  # registers spyreccl backend

    torch.spyre.set_device(local_rank)

    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.platforms import current_platform

    cfg = EngineArgs(
        model="facebook/opt-125m",
        tensor_parallel_size=world_size,
        dtype="float16",
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
    ).create_engine_config()

    with set_current_vllm_config(cfg):
        init_worker_distributed_environment(
            cfg, rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend=current_platform.dist_backend,
        )

        import vllm.distributed.parallel_state as ps
        device_group = ps._TP.device_group
        device = torch.device(f"spyre:{local_rank}")

        __PROBE_BODY__

    dist.destroy_process_group()
    """
)


_ALL_REDUCE_BODY = textwrap.dedent(
    """
        # Probe native dist.all_reduce on the spyreccl device_group.
        # Should fail today because SpyreCommsContext::allreduce throws.
        t = torch.full((1024,), float(rank + 1),
                       dtype=torch.float16, device=device)
        dist.all_reduce(t, group=device_group)
        expected = float(sum(range(1, world_size + 1)))
        torch.testing.assert_close(
            t.cpu(), torch.full_like(t.cpu(), expected)
        )
    """
).strip()


_ALL_GATHER_INTO_TENSOR_BODY = textwrap.dedent(
    """
        # Probe native dist.all_gather_into_tensor (single-tensor
        # allgather) on the spyreccl device_group. Should fail today
        # because torch-spyre's spyreccl backend stubs `_allgather_base`.
        t = torch.full((1024,), float(rank + 1),
                       dtype=torch.float16, device=device)
        out = torch.empty((world_size * 1024,),
                          dtype=torch.float16, device=device)
        dist.all_gather_into_tensor(out, t, group=device_group)
        out_cpu = out.cpu()
        for r in range(world_size):
            torch.testing.assert_close(
                out_cpu[r * 1024 : (r + 1) * 1024],
                torch.full((1024,), float(r + 1), dtype=torch.float16),
            )
    """
).strip()


_ALL_GATHER_LIST_BODY = textwrap.dedent(
    """
        # Probe native dist.all_gather (list form) on the spyreccl
        # device_group. Should fail today because the list-form
        # SpyreCommsContext::allgather throws.
        t = torch.full((1024,), float(rank + 1),
                       dtype=torch.float16, device=device)
        out_list = [
            torch.empty((1024,), dtype=torch.float16, device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(out_list, t, group=device_group)
        for r, o in enumerate(out_list):
            torch.testing.assert_close(
                o.cpu(), torch.full((1024,), float(r + 1), dtype=torch.float16),
            )
    """
).strip()


_GATHER_BODY = textwrap.dedent(
    """
        # Probe native dist.gather on the spyreccl device_group. Should
        # fail today because SpyreCommsContext::gather throws.
        t = torch.full((1024,), float(rank + 1),
                       dtype=torch.float16, device=device)
        if rank == 0:
            gather_list = [
                torch.empty((1024,), dtype=torch.float16, device=device)
                for _ in range(world_size)
            ]
        else:
            gather_list = None
        dist.gather(t, gather_list, dst=0, group=device_group)
        if rank == 0:
            for r, o in enumerate(gather_list):
                torch.testing.assert_close(
                    o.cpu(),
                    torch.full((1024,), float(r + 1), dtype=torch.float16),
                )
    """
).strip()


def _run_probe(probe_body: str) -> None:
    """Spawn TP=2 subprocesses running `probe_body` and assert both pass."""
    code = _PROBE_TEMPLATE.replace("__PROBE_BODY__", probe_body)
    port = _free_tcp_port()
    world_size = 2
    procs = []
    for rank in range(world_size):
        env = {
            **os.environ,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
        procs.append(
            subprocess.Popen(  # noqa: S603
                [sys.executable, "-c", code],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    results: list[tuple[int, str, str]] = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            results.append((-1, out or "", err or ""))
        else:
            results.append((p.returncode, out or "", err or ""))

    failed = [(i, rc, out, err) for i, (rc, out, err) in enumerate(results) if rc != 0]
    if failed:
        msg = "\n".join(
            f"--- rank {i} (rc={rc}) ---\n"
            f"stdout (tail):\n{out[-2000:]}\n"
            f"stderr (tail):\n{err[-2000:]}"
            for i, rc, out, err in failed
        )
        pytest.fail(f"native probe ranks failed:\n{msg}")


@pytest.mark.spyre
@pytest.mark.uses_subprocess
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
def test_native_all_reduce_works() -> None:
    _run_probe(_ALL_REDUCE_BODY)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
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
def test_native_all_gather_into_tensor_works() -> None:
    _run_probe(_ALL_GATHER_INTO_TENSOR_BODY)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "libspyre_comms does not implement list-form allgather; "
        "SpyreCommsContext::allgather(vector<>) throws. When this flips "
        "to passing, delete SpyreCommunicator.all_gather."
    ),
)
def test_native_all_gather_list_works() -> None:
    _run_probe(_ALL_GATHER_LIST_BODY)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 native-probe test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "libspyre_comms does not yet implement native gather; "
        "SpyreCommsContext::gather throws. When this flips to passing, "
        "delete SpyreCommunicator.gather."
    ),
)
def test_native_gather_works() -> None:
    _run_probe(_GATHER_BODY)
