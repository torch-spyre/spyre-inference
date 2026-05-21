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


# TEMPORARY: this test exercises the low-level distributed init path
# directly because the higher-level LLM(tp=2) tests below are
# xfail-strict on #134/#135. Once those land and
# test_tp2_llm_construction / test_tp2_llm_generate_matches_tp1 pass,
# this test is redundant — delete it along with the xfail markers on
# the LLM tests.
@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_tensor_model_parallel_all_reduce(run_tp_probe) -> None:
    """End-to-end TP=2 `tensor_model_parallel_all_reduce` on real Spyre cards.

    Spawns one subprocess per rank, each running through vllm's real
    `init_worker_distributed_environment` against a real `VllmConfig`,
    then verifies SpyreCommunicator's manual TP=2 fallback returns
    numerically correct results on a 1-D probe and a (seq, hidden) slab.
    """
    run_tp_probe("tp_all_reduce", world_size=2)


# Subprocess script for `test_tp2_vocab_parallel_embedding`. Each rank:
#   1. brings up the spyreccl device_group (env-rendezvous, same as the
#      LLM(tp=2) entry path),
#   2. constructs a VocabParallelEmbedding (which OOT-swaps to
#      SpyreVocabParallelEmbedding),
#   3. loads its slice of a deterministic full-vocab weight,
#   4. runs forward(input_ids) — which masks, looks up, zeroes out-of-shard
#      rows, and all_reduces via SpyreCommunicator,
#   5. asserts the result matches a single-rank F.embedding over the full
#      weight bit-for-bit (modulo float16 reduction noise).
#
# This isolates the embedding-TP correctness check from #134 (linear TP)
# and #136-equivalent (LM head TP) so the embedding work can land
# independently. When the full LLM(tp=2) test passes, this subprocess
# test is redundant and can be deleted.
_TP_VOCAB_EMBED_CODE = textwrap.dedent("""
    import os
    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops"
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
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

    vocab_size = 1024
    embedding_dim = 64

    with set_current_vllm_config(cfg):
        init_worker_distributed_environment(
            cfg, rank,
            distributed_init_method="env://",
            local_rank=local_rank,
            backend=current_platform.dist_backend,
        )

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from spyre_inference.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )

        layer = VocabParallelEmbedding(
            vocab_size, embedding_dim, params_dtype=torch.float16,
        )
        assert isinstance(layer, SpyreVocabParallelEmbedding), type(layer)
        assert layer.tp_size == world_size, layer.tp_size

        # Both ranks reconstruct the same full-vocab reference, then each
        # populates its shard from the same source — keeps the assertion
        # below independent of weight initialization.
        torch.manual_seed(42)
        full_weight = (
            torch.randn(vocab_size, embedding_dim, dtype=torch.float16) * 0.02
        )

        start = layer.shard_indices.org_vocab_start_index
        end = layer.shard_indices.org_vocab_end_index
        layer.weight.data.zero_()
        layer.weight.data[: end - start].copy_(full_weight[start:end])
        layer.to(torch.device(f"spyre:{local_rank}"))

        torch.manual_seed(7)
        input_ids = torch.randint(0, vocab_size, (16,), dtype=torch.int64)
        device = torch.device(f"spyre:{local_rank}")

        out = layer(input_ids.to(device)).cpu()
        expected = F.embedding(input_ids, full_weight)
        torch.testing.assert_close(
            out.float(), expected.float(), atol=1e-3, rtol=1e-3
        )
        dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()
    """)


@pytest.mark.spyre
@pytest.mark.uses_subprocess
@pytest.mark.skipif(
    _spyre_device_count() < 2,
    reason="needs >=2 Spyre cards; skipping TP=2 distributed test",
)
def test_tp2_vocab_parallel_embedding() -> None:
    """End-to-end TP=2 SpyreVocabParallelEmbedding forward on real Spyre cards.

    Spawns one subprocess per rank, brings up spyreccl through vllm's
    real init path, constructs the OOT VocabParallelEmbedding, and
    asserts each rank's all-reduced output matches the full-vocab
    F.embedding reference. Independent of #134 (linear TP).
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
                [sys.executable, "-c", _TP_VOCAB_EMBED_CODE],
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
        "TP-aware LM head (still pending — embedding done in #135). "
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
# xfail-strict here can mask a TP=1 regression in the body of this test;
# accepted because the marker will be deleted when #134/#135 land.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "needs #134, the LM head half of #135, and a TP=2 fallback for "
        "all_gather (LM head logits gather hits unimplemented "
        "_allgather_base). When all three land, TP=1 vs TP=2 should "
        "match on the first few decoded tokens."
    ),
)
def test_tp2_llm_generate_matches_tp1() -> None:
    """TP=1 vs TP=2 greedy-decode prefix-match test on opt-125m.

    Runs identical prompts at TP=1 and TP=2 with `temperature=0` and
    asserts the first 2 output tokens match per prompt. Later divergence
    is expected from float16 reduction-order differences between the
    TP=1 and TP=2 paths. xfail-strict so the test flips to passing the
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
        # vllm doesn't expose an explicit LLM.shutdown(); rely on GC +
        # child-process reaping. Revisit if this flakes.
        del llm
        gc.collect()
        return result

    def _matching_prefix_len(a: list[int], b: list[int]) -> int:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))

    tp1 = run(tp=1)
    tp2 = run(tp=2)
    for i, (a, b) in enumerate(zip(tp1, tp2)):
        n = _matching_prefix_len(a, b)
        assert n >= 2, (
            f"prompt {i}: tp1 and tp2 diverged at token {n} "
            f"(expected >=2 matching tokens). tp1={a} tp2={b}"
        )
