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

Tests invoke this via the `run_tp_probe` fixture in tests/conftest.py,
which spawns one subprocess per rank. To run a probe directly for
debugging:

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

import torch
import torch.distributed as dist


def probe_tp_all_reduce(device, device_group, world_size, rank):
    """High-level vllm `tensor_model_parallel_all_reduce` (via SpyreCommunicator).

    Verifies the manual TP fallback in SpyreCommunicator.all_reduce on
    a 1-D probe and a (seq, hidden) slab. `device_group` is unused —
    `tensor_model_parallel_all_reduce` resolves the group from
    `_TP.device_communicator` itself.
    """
    import vllm.distributed.parallel_state as ps
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

    comm_cls = type(ps._TP.device_communicator).__name__
    assert comm_cls == "SpyreCommunicator", f"got {comm_cls}"

    expected = float(sum(range(1, world_size + 1)))
    for shape in [(128,), (16, 1024)]:
        t = torch.full(shape, float(rank + 1), dtype=torch.float16, device=device)
        out = tensor_model_parallel_all_reduce(t)
        cpu = out.cpu()
        torch.testing.assert_close(cpu, torch.full_like(cpu, expected))
        dist.barrier(device_ids=[device.index])


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


def probe_vocab_parallel_embedding(device, device_group, world_size, rank):
    """TP=N SpyreVocabParallelEmbedding forward vs single-rank F.embedding.

    Constructs the OOT VocabParallelEmbedding (which OOT-swaps to
    SpyreVocabParallelEmbedding), loads each rank's shard from a
    deterministic full-vocab weight, runs forward, and asserts the
    all-reduced result matches a single-rank F.embedding over the full
    weight bit-for-bit (modulo float16 reduction noise).
    """
    import torch.nn.functional as F
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from spyre_inference.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    vocab_size = 1024
    embedding_dim = 64

    layer = VocabParallelEmbedding(
        vocab_size,
        embedding_dim,
        params_dtype=torch.float16,
    )
    assert isinstance(layer, SpyreVocabParallelEmbedding), type(layer)
    assert layer.tp_size == world_size, layer.tp_size

    # Both ranks reconstruct the same full-vocab reference, then each
    # populates its shard from the same source — keeps the assertion
    # below independent of weight initialization.
    torch.manual_seed(42)
    full_weight = torch.randn(vocab_size, embedding_dim, dtype=torch.float16) * 0.02

    start = layer.shard_indices.org_vocab_start_index
    end = layer.shard_indices.org_vocab_end_index
    layer.weight.data.zero_()
    layer.weight.data[: end - start].copy_(full_weight[start:end])
    layer.to(device)

    torch.manual_seed(7)
    input_ids = torch.randint(0, vocab_size, (16,), dtype=torch.int64)

    out = layer(input_ids.to(device)).cpu()
    expected = F.embedding(input_ids, full_weight)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-3, rtol=1e-3)
    dist.barrier(device_ids=[device.index])


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


PROBES = {
    "tp_all_reduce": probe_tp_all_reduce,
    "native_all_reduce": probe_native_all_reduce,
    "native_all_gather_into_tensor": probe_native_all_gather_into_tensor,
    "native_all_gather_list": probe_native_all_gather_list,
    "native_gather": probe_native_gather,
    "vocab_parallel_embedding": probe_vocab_parallel_embedding,
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


if __name__ == "__main__":
    main()
