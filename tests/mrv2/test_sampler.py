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

"""MRV2's upstream Sampler and grammar bitmask, unmodified, on host logits through
triton-cpu. HAS_TRITON is fixed per process, so each check runs in a fresh V2 one."""

import os
import subprocess
import sys
import textwrap

import pytest

from spyre_inference.v1.worker.spyre import triton_gate

pytestmark = [
    pytest.mark.triton_cpu,
    pytest.mark.subprocess_isolated,
    pytest.mark.skipif(not triton_gate.has_triton_cpu(), reason="needs triton-cpu (x86)"),
]

_SAMPLER = """
import numpy as np
import torch
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.worker.spyre import device_shims

NUM_REQS, VOCAB = 4, 32
cpu = torch.device("cpu")
device_shims.install()
config = VllmConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B", max_model_len=256))


def sample(params, logits):
    req_states = RequestState(NUM_REQS, 256, 64, 0, VOCAB, cpu)
    sampler = Sampler(config, NUM_REQS, VOCAB, cpu, req_states)
    for i in range(NUM_REQS):
        req_states.add_request(f"r{i}", 3, [1, 2, 3], 0, 16)
        sampler.add_request(req_states.req_id_to_index[f"r{i}"], params)
    req_states.apply_staged_writes()
    sampler.apply_staged_writes()
    batch = InputBatch.make_dummy(NUM_REQS, NUM_REQS, InputBuffers(NUM_REQS, 64, cpu))
    idx = np.array([req_states.req_id_to_index[f"r{i}"] for i in range(NUM_REQS)])
    batch.idx_mapping_np = idx
    batch.idx_mapping = batch.expanded_idx_mapping = torch.from_numpy(idx).to(torch.int64)
    return sampler(logits.clone(), batch)


with set_current_vllm_config(config):
    # A CPU generator: torch.manual_seed also seeds Spyre, which the parent may hold.
    logits = torch.randn(NUM_REQS, VOCAB, generator=torch.Generator().manual_seed(0))
    print("RESULT has_triton", HAS_TRITON)

    greedy = sample(SamplingParams(temperature=0.0), logits).sampled_token_ids.view(-1)
    print("RESULT greedy_matches_v1", greedy.tolist() == greedy_sample(logits).view(-1).tolist())

    picked = sample(SamplingParams(temperature=1.0, top_k=2), logits).sampled_token_ids.view(-1)
    top2 = logits.topk(2, dim=-1).indices
    in_window = all(int(picked[i]) in top2[i].tolist() for i in range(NUM_REQS))
    print("RESULT top_k_in_window", in_window)

    kept = logits.softmax(-1).sort(dim=-1, descending=True)
    nucleus = [set(kept.indices[i][: int((kept.values[i].cumsum(0) < 0.5).sum()) + 1].tolist())
               for i in range(NUM_REQS)]
    picked = sample(SamplingParams(temperature=1.0, top_p=0.5), logits).sampled_token_ids.view(-1)
    print("RESULT top_p_in_nucleus", all(int(picked[i]) in nucleus[i] for i in range(NUM_REQS)))

    lp = sample(SamplingParams(temperature=0.0, logprobs=3), logits).logprobs_tensors
    want = torch.log_softmax(logits, -1).topk(3, dim=-1)
    print("RESULT logprobs_match",
          torch.allclose(lp.logprobs[:, 1:], want.values, atol=1e-5)
          and lp.logprob_token_ids[:, 1:].tolist() == want.indices.tolist()
          and lp.logprob_token_ids[:, 0].tolist() == greedy.tolist())

    draws = [sample(SamplingParams(temperature=1.0, seed=11), logits).sampled_token_ids.view(-1)
             .tolist() for _ in range(2)]
    print("RESULT seeded_reproducible", draws[0] == draws[1])
"""

_GRAMMAR = """
import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

from spyre_inference.v1.worker.spyre import device_shims

NUM_REQS, VOCAB = 4, 64
cpu = torch.device("cpu")
device_shims.install()
worker = StructuredOutputsWorker(
    max_num_logits=NUM_REQS, vocab_size=VOCAB, device=cpu, mask_stride=1, num_bonus_tokens=1
)
batch = InputBatch.make_dummy(NUM_REQS, NUM_REQS, InputBuffers(NUM_REQS, 64, cpu))
logits = torch.randn(NUM_REQS, VOCAB, generator=torch.Generator().manual_seed(0))
want = logits.clone()
rng = np.random.default_rng(0)
bitmask = np.zeros((2, VOCAB // 32), dtype=np.int32)
for row, i in enumerate((1, 3)):
    allowed = rng.random(VOCAB) < 0.25
    bitmask[row] = np.packbits(allowed.astype(np.uint8), bitorder="little").view(np.int32)
    want[i][~torch.from_numpy(allowed)] = float("-inf")

worker.apply_grammar_bitmask(logits, batch, [batch.req_ids[1], batch.req_ids[3]], bitmask)
print("RESULT grammar_mask_matches", torch.equal(logits, want))
"""


def _run(script: str) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "SPYRE_MODEL_RUNNER": "v2"},
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    results = [line.split(" ", 2) for line in result.stdout.splitlines()]
    return {r[1]: r[2] for r in results if r[0] == "RESULT"}


def test_sampler_runs_unmodified_on_host_logits() -> None:
    got = _run(_SAMPLER)
    assert got == {
        "has_triton": "True",
        "greedy_matches_v1": "True",
        "top_k_in_window": "True",
        "top_p_in_nucleus": "True",
        "logprobs_match": "True",
        "seeded_reproducible": "True",
    }


def test_grammar_bitmask_masks_only_constrained_rows() -> None:
    assert _run(_GRAMMAR) == {"grammar_mask_matches": "True"}
