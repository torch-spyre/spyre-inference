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

"""TP=2 distributed tests"""

from __future__ import annotations

import gc
import os
import socket
import subprocess
import sys
import textwrap

import pytest


def _spyre_device_count() -> int:
    """Return the number of visible Spyre cards, or 0 if unavailable.

    Counts numeric entries under /dev/vfio rather than calling
    `torch.spyre.device_count()`, because `uses_subprocess` tests must
    not touch the Spyre runtime in the main pytest process.
    """
    try:
        return sum(1 for n in os.listdir("/dev/vfio") if n.isdigit())
    except OSError:
        return 0


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Inner-rank script for `test_tp2_tensor_model_parallel_all_reduce`.
# Launched once per rank as a fresh `python -c` subprocess with
# RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / MASTER_{ADDR,PORT}
# set in the environment (env://-style rendezvous, same as torchrun).
_TP_ALLREDUCE_CODE = textwrap.dedent(
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

    from vllm.platforms import PlatformEnum, current_platform
    type(current_platform)._enum = PlatformEnum.OOT

    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

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
        comm_cls = type(ps._TP.device_communicator).__name__
        assert comm_cls == "SpyreCommunicator", f"got {comm_cls}"

        from vllm.distributed.communication_op import (
            tensor_model_parallel_all_reduce,
        )

        device = torch.device(f"spyre:{local_rank}")
        for shape in [(128,), (16, 1024)]:
            t = torch.full(
                shape, float(rank + 1),
                dtype=torch.float16, device=device,
            )
            out = tensor_model_parallel_all_reduce(t)
            expected = float(sum(range(1, world_size + 1)))
            cpu = out.cpu()
            torch.testing.assert_close(cpu, torch.full_like(cpu, expected))
            dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()
    """
)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_tensor_model_parallel_all_reduce() -> None:
    """End-to-end TP=2 `tensor_model_parallel_all_reduce` on real Spyre cards.

    Spawns one subprocess per rank, each running through vllm's real
    `init_worker_distributed_environment` against a real `VllmConfig`,
    then verifies SpyreCommunicator's manual TP=2 fallback returns
    numerically correct results on a 1-D probe and a (seq, hidden) slab.
    """
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
                [sys.executable, "-c", _TP_ALLREDUCE_CODE],
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
        pytest.fail(f"TP=2 ranks failed:\n{msg}")


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "needs TP-aware Spyre custom linear layers (#134) and "
        "TP-aware vocab embedding / LM head (#135). "
        "MultiprocExecutor + spyreccl init succeed; failure is at "
        "SpyreQKVParallelLinear NotImplementedError(TP>1)."
    ),
)
def test_tp2_llm_construction() -> None:
    """Construct `vllm.LLM(tensor_parallel_size=2)` end-to-end.

    Goes through the real `MultiprocExecutor` worker-spawn path that
    `vllm serve --tensor-parallel-size 2` uses. Today fails at TP-naive
    layer construction; xfail-strict here so the test flips to passing
    automatically when #134/#135 land.
    """
    from vllm import LLM

    LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=2,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
    )


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "needs #134, #135, and a TP=2 fallback for all_gather "
        "(LM head logits gather hits unimplemented _allgather_base). "
        "When all three land, TP=1 vs TP=2 token outputs should match."
    ),
)
def test_tp2_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=2 greedy-decode golden test on opt-125m.

    Runs identical prompts at TP=1 and TP=2 with `temperature=0` and
    asserts equal output token IDs. xfail-strict so it surfaces the
    moment end-to-end TP=2 forward correctness lands.
    """
    from vllm import LLM, SamplingParams

    prompts = ["Hello, world!", "The capital of France is"]
    sp = SamplingParams(max_tokens=8, temperature=0.0)

    def run(tp: int) -> list[list[int]]:
        llm = LLM(
            model="facebook/opt-125m",
            tensor_parallel_size=tp,
            dtype="float16",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=2,
        )
        outs = llm.generate(prompts, sp)
        result = [list(o.outputs[0].token_ids) for o in outs]
        del llm
        gc.collect()
        return result

    tp1 = run(1)
    tp2 = run(2)
    assert tp1 == tp2, f"tp1 != tp2: {tp1} vs {tp2}"
