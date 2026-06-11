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


def probe_merged_column_parallel_linear(device, device_group, world_size, rank):
    """TP=N SpyreMergedColumnParallelLinear forward vs single-rank F.linear.

    Models the gate_up_proj usage: output_sizes=[gate, up], so the per-rank
    weight is [gate_shard | up_shard] stacked rows-wise (each shard is the
    rank's slice of the corresponding full matrix). The expected output is
    the per-shard F.linear results concatenated along the output dim — a
    naive contiguous row-slice would mask a layout bug, so we build the
    weight shard-by-shard.
    """
    import torch.nn.functional as F
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    from spyre_inference.custom_ops.linear import SpyreMergedColumnParallelLinear
    from spyre_inference.custom_ops.utils import convert

    input_size = 512
    output_size = 1024  # per-shard size (gate, up)

    layer = MergedColumnParallelLinear(
        input_size,
        [output_size, output_size],
        bias=False,
        params_dtype=torch.float16,
    )
    assert isinstance(layer, SpyreMergedColumnParallelLinear), type(layer)

    torch.manual_seed(42)
    gate_full = torch.randn(output_size, input_size, dtype=torch.float16) * 0.02
    up_full = torch.randn(output_size, input_size, dtype=torch.float16) * 0.02

    shard = output_size // world_size
    start = rank * shard
    end = (rank + 1) * shard
    layer.weight.data.copy_(torch.cat([gate_full[start:end], up_full[start:end]], dim=0))
    layer.to(device)

    torch.manual_seed(7)
    x = torch.randn(16, input_size, dtype=torch.float16)

    out, _ = layer(convert(x, device=device))
    out = convert(out, device="cpu")
    expected = torch.cat(
        [F.linear(x, gate_full)[:, start:end], F.linear(x, up_full)[:, start:end]],
        dim=-1,
    )
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    dist.barrier(device_ids=[device.index])


def probe_qkv_parallel_linear(device, device_group, world_size, rank):
    """TP=N SpyreQKVParallelLinear forward vs single-rank F.linear.

    Uses GQA shape (total_num_heads != total_num_kv_heads) so a naive
    "slice the full weight row-wise per rank" load — which would pass for
    MHA — actually mismatches the real per-rank `[Q | K | V]` layout.
    The probe loads each shard from its corresponding full Q/K/V matrix
    using the same head/replica math QKVParallelLinear performs in
    __init__, then asserts each rank's output matches the concatenation
    of per-shard F.linear results.
    """
    import torch.nn.functional as F
    from vllm.model_executor.layers.linear import QKVParallelLinear

    from spyre_inference.custom_ops.linear import SpyreQKVParallelLinear
    from spyre_inference.custom_ops.utils import convert

    hidden_size = 512
    head_size = 64
    total_num_heads = 8
    total_num_kv_heads = 2  # GQA: kv heads < query heads

    layer = QKVParallelLinear(
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=False,
        params_dtype=torch.float16,
    )
    assert isinstance(layer, SpyreQKVParallelLinear), type(layer)

    # Mirror QKVParallelLinear.__init__ partitioning math.
    num_heads = total_num_heads // world_size
    if world_size >= total_num_kv_heads:
        num_kv_heads = 1
        num_kv_head_replicas = world_size // total_num_kv_heads
    else:
        num_kv_heads = total_num_kv_heads // world_size
        num_kv_head_replicas = 1
    assert layer.num_heads == num_heads
    assert layer.num_kv_heads == num_kv_heads
    assert layer.num_kv_head_replicas == num_kv_head_replicas

    q_full_rows = total_num_heads * head_size
    kv_full_rows = total_num_kv_heads * head_size

    torch.manual_seed(43)
    q_full = torch.randn(q_full_rows, hidden_size, dtype=torch.float16) * 0.02
    k_full = torch.randn(kv_full_rows, hidden_size, dtype=torch.float16) * 0.02
    v_full = torch.randn(kv_full_rows, hidden_size, dtype=torch.float16) * 0.02

    q_start = rank * num_heads * head_size
    q_end = q_start + num_heads * head_size
    kv_shard_idx = rank // num_kv_head_replicas
    kv_start = kv_shard_idx * num_kv_heads * head_size
    kv_end = kv_start + num_kv_heads * head_size

    q_shard = q_full[q_start:q_end]
    k_shard = k_full[kv_start:kv_end]
    v_shard = v_full[kv_start:kv_end]
    layer.weight.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))
    layer.to(device)

    torch.manual_seed(8)
    x = torch.randn(16, hidden_size, dtype=torch.float16)

    out, _ = layer(convert(x, device=device))
    expected = torch.cat(
        [
            F.linear(x, q_shard),
            F.linear(x, k_shard),
            F.linear(x, v_shard),
        ],
        dim=-1,
    )
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    dist.barrier(device_ids=[device.index])


def probe_row_parallel_linear(device, device_group, world_size, rank):
    """TP=N SpyreRowParallelLinear forward vs single-rank F.linear.

    Constructs RowParallelLinear (OOT-swapped to SpyreRowParallelLinear),
    loads each rank's input-dim shard from deterministic full weight, runs
    forward, and asserts the all-reduced result matches single-rank F.linear
    (modulo float16 all_reduce noise).
    """
    import torch.nn.functional as F
    from vllm.model_executor.layers.linear import RowParallelLinear

    from spyre_inference.custom_ops.linear import SpyreRowParallelLinear
    from spyre_inference.custom_ops.utils import convert

    input_size = 1024
    output_size = 512

    layer = RowParallelLinear(
        input_size,
        output_size,
        bias=False,
        params_dtype=torch.float16,
    )
    assert isinstance(layer, SpyreRowParallelLinear), type(layer)

    torch.manual_seed(44)
    full_weight = torch.randn(output_size, input_size, dtype=torch.float16) * 0.02

    input_size_per_partition = input_size // world_size
    start = rank * input_size_per_partition
    end = (rank + 1) * input_size_per_partition
    layer.weight.data.copy_(full_weight[:, start:end])
    layer.to(device)

    torch.manual_seed(9)
    x_full = torch.randn(16, input_size, dtype=torch.float16)
    x = x_full[:, start:end]

    out, bias = layer(convert(x, device=device))
    out = convert(out, device="cpu")
    expected = F.linear(x_full, full_weight)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    dist.barrier(device_ids=[device.index])


PROBES = {
    "tp_all_reduce": probe_tp_all_reduce,
    "native_all_reduce": probe_native_all_reduce,
    "native_all_gather_into_tensor": probe_native_all_gather_into_tensor,
    "native_all_gather_list": probe_native_all_gather_list,
    "native_gather": probe_native_gather,
    "vocab_parallel_embedding": probe_vocab_parallel_embedding,
    "merged_column_parallel_linear": probe_merged_column_parallel_linear,
    "qkv_parallel_linear": probe_qkv_parallel_linear,
    "row_parallel_linear": probe_row_parallel_linear,
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
