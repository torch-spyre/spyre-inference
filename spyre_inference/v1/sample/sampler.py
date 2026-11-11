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

from typing import Any

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    if not has_sampling_kernels():
        return Sampler.greedy_sample(logits)
    return torch.ops._spyre_C.greedy_argmax(logits)  # ty: ignore[invalid-argument-type]


def install_rank0_sampling(sampler: Sampler) -> None:
    """Sample on TP rank 0 and share the result with the other ranks.

    Each rank otherwise draws its own token. Only rank 0's is kept, so a
    diverging draw feeds the next forward a different token per rank.
    """
    sampler.forward = _sample_on_one_tp_rank.__get__(  # ty: ignore[invalid-assignment]
        sampler, Sampler
    )


def _sample_on_one_tp_rank(
    self: Sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override: Any = None,
) -> SamplerOutput:
    # Imported here: parallel_state pulls in the distributed runtime, and this
    # module is imported from paths that never start a process group.
    from vllm.distributed.parallel_state import _TP, get_tp_group

    if logits.device.type != "cpu" or _TP is None or get_tp_group().world_size == 1:
        return Sampler.forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    tp_group = get_tp_group()
    if tp_group.is_first_rank:
        output = Sampler.forward(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )
    else:
        output = SamplerOutput(
            sampled_token_ids=torch.empty(
                (logits.shape[0], 1), dtype=torch.int32, device=logits.device
            ),
            logprobs_tensors=None,
        )

    # GroupCoordinator.broadcast uses device_group, which is spyreccl.
    # SpyreCCL's broadcast reads a device address (prepare_tensor), so a CPU
    # int32 token id cannot ride it. cpu_group is the gloo group the platform
    # already uses for host coordination.
    token_ids = output.sampled_token_ids
    if not token_ids.is_contiguous():
        token_ids = token_ids.contiguous()
        output.sampled_token_ids = token_ids
    torch.distributed.broadcast(token_ids, src=tp_group.ranks[0], group=tp_group.cpu_group)
    output.logprobs_tensors = tp_group.broadcast_object(output.logprobs_tensors, src=0)
    return output
