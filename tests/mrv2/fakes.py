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

"""Tiny fakes shared across tests/mrv2. Frozen after #1236: a later package
adds fakes it needs to its own test module rather than editing this one.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


class FakeModel(nn.Module):
    """A 2-block toy model standing in for a real model's repeated blocks."""

    def __init__(self, hidden_size: int = 8, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(num_blocks))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def make_fake_scheduler_output(num_scheduled_tokens: dict[str, int]) -> SchedulerOutput:
    """A SchedulerOutput with only the fields a dummy run reads."""
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = dict(num_scheduled_tokens)
    output.total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    return output


def make_fake_kv_cache_config(
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 64,
    num_kv_heads: int = 1,
    head_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> KVCacheConfig:
    """A KVCacheConfig for num_layers independent full-attention layers,
    each with its own num_blocks-block group."""
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
            for name in layer_names
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name in layer_names
        ],
    )


@dataclass
class FakeAttentionGroup:
    """A duck-typed stand-in for vllm.v1.worker.utils.AttentionGroup, with no
    real backend or metadata builder attached."""

    layer_names: list[str]
    kv_cache_spec: object
    kv_cache_group_id: int = 0
    backend: object = None
    metadata_builders: list[object] = field(default_factory=list)


def make_fake_attn_groups(
    num_windows: int = 1, layers_per_window: int = 2
) -> list[FakeAttentionGroup]:
    """One fake attention group per sliding-window size, mirroring the shape
    attn_groups.init() is expected to split into."""
    groups = []
    for window in range(num_windows):
        layer_names = [f"layer.{window}.{i}" for i in range(layers_per_window)]
        groups.append(FakeAttentionGroup(layer_names=layer_names, kv_cache_spec=None))
    return groups


def make_fake_input_batch(num_reqs: int = 1, num_tokens: int = 1) -> SimpleNamespace:
    """A duck-typed stand-in for InputBatch, with only the attributes model
    states and warmup commonly read."""
    quot, rem = divmod(num_tokens, num_reqs)
    return SimpleNamespace(
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_scheduled_tokens=torch.tensor(
            [quot + (i < rem) for i in range(num_reqs)], dtype=torch.int32
        ),
    )
