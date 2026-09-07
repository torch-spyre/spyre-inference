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
"""TP=N probe dispatcher.

Each probe exercises one collective on a real spyreccl device_group.
The shared `main()` prologue handles env-rendezvous, vllm config, and
worker-distributed-env init, then dispatches to the requested probe.

Tests invoke this via the `run_tp_probe` fixture in
tests/plugin/spyre_testing_plugin/pytest_plugin.py, which spawns one
subprocess per rank. To run a probe directly for debugging:

    RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 LOCAL_WORLD_SIZE=2 \\
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
    python tests/probes/tp_probe.py --probe native_all_reduce

(Spawn a second shell with RANK=1 to actually complete the collective.)

This file is run as a script in a subprocess; it is never imported by
the main pytest process. That keeps `torch_spyre` out of the parent
process — same architectural rule as the rest of the spyre-touching
tests in this directory.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def probe_native_all_reduce(device, device_group, world_size, rank):
    """Raw `dist.all_reduce` on the spyreccl device_group."""
    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    dist.all_reduce(t, group=device_group)
    expected = float(sum(range(1, world_size + 1)))
    torch.testing.assert_close(t.cpu(), torch.full_like(t.cpu(), expected))


def probe_native_all_gather_into_tensor(device, device_group, world_size, rank):
    """Raw `dist.all_gather_into_tensor` on the spyreccl device_group."""
    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.empty((world_size * 1024,), dtype=torch.float16, device=device)
    dist.all_gather_into_tensor(out, t, group=device_group)
    out_cpu = out.cpu()
    for r in range(world_size):
        torch.testing.assert_close(
            out_cpu[r * 1024 : (r + 1) * 1024],
            torch.full((1024,), float(r + 1), dtype=torch.float16),
        )


def probe_native_all_gather_list(device, device_group, world_size, rank):
    """Raw `dist.all_gather` (list form) on the spyreccl device_group."""
    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    out_list = [torch.empty((1024,), dtype=torch.float16, device=device) for _ in range(world_size)]
    dist.all_gather(out_list, t, group=device_group)
    for r, o in enumerate(out_list):
        torch.testing.assert_close(o.cpu(), torch.full((1024,), float(r + 1), dtype=torch.float16))


def probe_native_gather(device, device_group, world_size, rank):
    """Raw `dist.gather` to rank 0 on the spyreccl device_group."""
    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    if rank == 0:
        gather_list = [
            torch.empty((1024,), dtype=torch.float16, device=device) for _ in range(world_size)
        ]
    else:
        gather_list = None
    dist.gather(t, gather_list, dst=0, group=device_group)
    if rank == 0:
        for r, o in enumerate(gather_list):
            torch.testing.assert_close(
                o.cpu(), torch.full((1024,), float(r + 1), dtype=torch.float16)
            )


def _group_name(device_group):
    from torch.distributed._functional_collectives import _resolve_group_name

    return _resolve_group_name(device_group)


def probe_functional_all_reduce_eager(device, device_group, world_size, rank):
    """`_c10d_functional.all_reduce` in eager, via spyreccl's native allreduce."""
    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.ops._c10d_functional.all_reduce(t, "sum", _group_name(device_group))
    out = torch.ops._c10d_functional.wait_tensor(out)
    expected = float(sum(range(1, world_size + 1)))
    torch.testing.assert_close(out.cpu(), torch.full((1024,), expected, dtype=torch.float16))


def probe_functional_all_gather_eager(device, device_group, world_size, rank):
    """`_c10d_functional.all_gather_into_tensor` in eager. Expected to fail.

    Stick-aligned shape, to isolate the rejected entry point from the alignment
    limit probed below.
    """
    t = torch.full((64, 8), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.ops._c10d_functional.all_gather_into_tensor(
        t, world_size, _group_name(device_group)
    )
    out = torch.ops._c10d_functional.wait_tensor(out)
    out_cpu = out.cpu()
    for r in range(world_size):
        torch.testing.assert_close(
            out_cpu[r * 64 : (r + 1) * 64],
            torch.full((64, 8), float(r + 1), dtype=torch.float16),
        )


def probe_compiled_all_reduce(device, device_group, world_size, rank):
    """`_c10d_functional.all_reduce` compiled, lowering to `spyre::all_reduce_async`."""
    gn = _group_name(device_group)

    def fn(x):
        y = x * 2.0
        out = torch.ops._c10d_functional.all_reduce(y, "sum", gn)
        return torch.ops._c10d_functional.wait_tensor(out)

    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.compile(fn, dynamic=False)(t)
    expected = 2.0 * float(sum(range(1, world_size + 1)))
    torch.testing.assert_close(out.cpu(), torch.full((1024,), expected, dtype=torch.float16))


def probe_compiled_all_reduce_multi_round(device, device_group, world_size, rank):
    """`_c10d_functional.all_reduce` compiled, 64 sequential rounds reusing one WSI."""
    gn = _group_name(device_group)
    num_rounds = 64

    def fn(x):
        for _ in range(num_rounds):
            out = torch.ops._c10d_functional.all_reduce(x, "sum", gn)
            x = torch.ops._c10d_functional.wait_tensor(out)
        return x

    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.compile(fn, dynamic=False)(t)
    # fp16 saturates after num_rounds sums; no exact value check, just shape
    assert out.shape == t.shape


def probe_compiled_all_reduce_multi_block(device, device_group, world_size, rank):
    """`_c10d_functional.all_reduce` compiled, 32 separately-compiled block fns × 2 rounds."""
    gn = _group_name(device_group)
    num_blocks = 32

    def make_block_fn():
        def block_fn(x):
            # two all_reduces per block, same shape (mimics attn + mlp)
            out = torch.ops._c10d_functional.all_reduce(x, "sum", gn)
            x = torch.ops._c10d_functional.wait_tensor(out)
            out = torch.ops._c10d_functional.all_reduce(x, "sum", gn)
            x = torch.ops._c10d_functional.wait_tensor(out)
            return x

        return torch.compile(block_fn, dynamic=False)

    block_fns = [make_block_fn() for _ in range(num_blocks)]

    t = torch.full((1024,), float(rank + 1), dtype=torch.float16, device=device)
    for fn in block_fns:
        t = fn(t)
    for _ in range(4):
        for fn in block_fns:
            t = fn(t)
    assert t.shape == torch.Size([1024])


def _all_gather_lastdim(x, world_size, gn):
    """vLLM's concat-style all_gather along the last dim, functional-collective form."""
    input_size = x.size()
    out = torch.ops._c10d_functional.all_gather_into_tensor(x, world_size, gn)
    out = torch.ops._c10d_functional.wait_tensor(out)
    out = out.reshape((world_size,) + tuple(input_size))
    out = out.movedim(0, x.dim() - 1)
    return out.reshape(tuple(input_size[:-1]) + (world_size * input_size[-1],))


def probe_compiled_all_gather_lastdim(device, device_group, world_size, rank):
    """Compiled all_gather along the last dim on a stick-aligned width."""
    gn = _group_name(device_group)
    width = 256
    t = torch.full((1, width), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.compile(lambda x: _all_gather_lastdim(x, world_size, gn), dynamic=False)(t)
    out_cpu = out.cpu()
    assert tuple(out_cpu.shape) == (1, width * world_size), out_cpu.shape
    for r in range(world_size):
        torch.testing.assert_close(
            out_cpu[:, r * width : (r + 1) * width],
            torch.full((1, width), float(r + 1), dtype=torch.float16),
        )


def probe_compiled_all_gather_lastdim_unaligned(device, device_group, world_size, rank):
    """Compiled all_gather on a NON-stick-aligned width. Expected to fail.

    24608 is the per-rank vocab-parallel logits width for micro-g3.3-8b at TP=2,
    the shape that forced the padding in `SpyreCommunicator.all_gather`. Reshaping
    cannot rescue it: 24608 = 32 * 769 has no factor of 64.
    """
    gn = _group_name(device_group)
    width = 24608
    t = torch.full((1, width), float(rank + 1), dtype=torch.float16, device=device)
    out = torch.compile(lambda x: _all_gather_lastdim(x, world_size, gn), dynamic=False)(t)
    out_cpu = out.cpu()
    assert tuple(out_cpu.shape) == (1, width * world_size), out_cpu.shape
    for r in range(world_size):
        torch.testing.assert_close(
            out_cpu[:, r * width : (r + 1) * width],
            torch.full((1, width), float(r + 1), dtype=torch.float16),
        )


PROBES = {
    "native_all_reduce": probe_native_all_reduce,
    "native_all_gather_into_tensor": probe_native_all_gather_into_tensor,
    "native_all_gather_list": probe_native_all_gather_list,
    "native_gather": probe_native_gather,
    "functional_all_reduce_eager": probe_functional_all_reduce_eager,
    "functional_all_gather_eager": probe_functional_all_gather_eager,
    "compiled_all_reduce": probe_compiled_all_reduce,
    "compiled_all_reduce_multi_round": probe_compiled_all_reduce_multi_round,
    "compiled_all_reduce_multi_block": probe_compiled_all_reduce_multi_block,
    "compiled_all_gather_lastdim": probe_compiled_all_gather_lastdim,
    "compiled_all_gather_lastdim_unaligned": probe_compiled_all_gather_lastdim_unaligned,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=sorted(PROBES))
    args = parser.parse_args()

    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops"
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # spyre_inference/__init__.py sets TORCH_DEVICE_BACKEND_AUTOLOAD=0 to
    # control when libspyre_comms.so is loaded (it captures RANK/WORLD_SIZE
    # at dlopen time). Trigger torch_spyre's autoload manually here, after
    # the env vars set by the parent fixture are in place.
    import torch_spyre

    torch_spyre._autoload()

    torch.spyre.set_device(local_rank)

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

    load_general_plugins()

    cfg = EngineArgs(
        model="facebook/opt-125m",
        tensor_parallel_size=world_size,
        dtype="float16",
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
    ).create_engine_config()

    with set_current_vllm_config(cfg):
        init_worker_distributed_environment(
            cfg,
            rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend=current_platform.dist_backend,
        )

        import vllm.distributed.parallel_state as ps

        device_group = ps._TP.device_group
        device = torch.device(f"spyre:{local_rank}")

        PROBES[args.probe](device, device_group, world_size, rank)

    dist.destroy_process_group()

    # Hard-exit this throwaway per-rank subprocess. Tears down all running threads
    # so it doesn't rely on correctness of dist.destroy_process_group().
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
