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

"""The SPYRE_LOGITS_CPU_GATHER path of SpyreLogitsProcessor.

Each TP rank copies only its vocab shard of the logits to the host and the
shards are gathered over gloo on the CPU, instead of gathering the full vocab
on the device. The env var and ``_cpu_all_gather_logits`` had no test at all.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available


def _full_logits() -> torch.Tensor:
    # Small non-negative integers: exact in fp16, so equality is bit-exact.
    return (torch.arange(4 * 32).reshape(4, 32) % 97).half()


def _bare_processor():
    """A SpyreLogitsProcessor with its compile machinery disarmed: enough of an
    instance to call ``_gather_logits`` without a model or a live config."""
    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    processor = SpyreLogitsProcessor.__new__(SpyreLogitsProcessor)
    processor.spyre_compile_enabled = False
    processor.spyre_compiled_kernels = {}
    return processor


def test_gather_logits_dispatches_on_the_env_var(tp_group, monkeypatch):
    """SPYRE_LOGITS_CPU_GATHER switches _gather_logits onto the CPU gather."""
    from spyre_inference import envs
    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    processor = _bare_processor()
    calls = []
    monkeypatch.setattr(SpyreLogitsProcessor, "_cpu_all_gather_logits", staticmethod(calls.append))
    logits = torch.zeros(2, 8, dtype=torch.float16)

    monkeypatch.setattr(envs, "SPYRE_LOGITS_CPU_GATHER", False)
    processor._gather_logits(logits)
    assert calls == []

    monkeypatch.setattr(envs, "SPYRE_LOGITS_CPU_GATHER", True)
    processor._gather_logits(logits)
    assert calls == [logits]


def test_cpu_gather_matches_the_default_path_at_tp1(tp_group, monkeypatch):
    """At TP1 both paths must hand sampling the same CPU logits: the CPU gather
    is an escape hatch, not a numerical change."""
    from spyre_inference import envs
    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    logits = _full_logits()
    monkeypatch.setattr(envs, "SPYRE_LOGITS_CPU_GATHER", False)
    default = _bare_processor()._gather_logits(logits.clone())

    gathered = SpyreLogitsProcessor._cpu_all_gather_logits(logits.clone())
    assert gathered.device.type == "cpu"
    assert gathered.dtype == torch.float16
    assert gathered.shape == logits.shape
    torch.testing.assert_close(gathered, logits, atol=0, rtol=0)
    torch.testing.assert_close(gathered, default, atol=0, rtol=0)


# A second process holding the other rank; the parent joins both.
_RANK_SCRIPT = textwrap.dedent(
    """
    import sys

    import torch
    from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.forward_context import set_forward_context

    rank = int(sys.argv[1])
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        init_distributed_environment(2, rank, sys.argv[2], rank, "gloo")
        ensure_model_parallel_initialized(2, 1)
        from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

        full = (torch.arange(4 * 32).reshape(4, 32) % 97).half()
        shard = full[:, rank * 16 : (rank + 1) * 16].clone()
        out = SpyreLogitsProcessor._cpu_all_gather_logits(shard)
        assert out.device.type == "cpu" and out.dtype == torch.float16
        torch.testing.assert_close(out, full, atol=0, rtol=0)
        print("OK", rank, flush=True)
    """
)


def test_cpu_gather_concatenates_vocab_shards_in_rank_order():
    """Two gloo ranks: each hands over only its own vocab shard, and the shards
    concatenate along the vocab axis in rank order, like the device all-gather
    the flag replaces. Gloo-only, so no card is needed."""
    with tempfile.TemporaryDirectory() as tmp:
        store = "file://" + os.path.join(tmp, "store")
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _RANK_SCRIPT, str(rank), store],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for rank in (0, 1)
        ]
        for rank, proc in enumerate(procs):
            _, stderr = proc.communicate(timeout=300)
            assert proc.returncode == 0, f"rank {rank} failed:\n{stderr[-3000:]}"


def test_cpu_gather_brings_a_device_shard_to_the_host(tp_group):
    """The shard the ranks gather is device-resident in production, so the
    escape hatch's D2H must land bit-exact fp16 logits on the host."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops import utils
    from spyre_inference.custom_ops.logits_processor import SpyreLogitsProcessor

    utils.register()
    host = _full_logits()
    device_logits = utils.convert(host.clone(), device="spyre")
    gathered = SpyreLogitsProcessor._cpu_all_gather_logits(device_logits)
    assert gathered.device.type == "cpu"
    torch.testing.assert_close(gathered, host, atol=0, rtol=0)
