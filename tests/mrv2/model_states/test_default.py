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

"""The decoder state's prepare_attn hands SpyreAttentionMetadataBuilder.build
every field it reads, with the values V1 gave it."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState

from .state_fakes import make_config, make_input_batch


def _fields_read(*functions) -> set[str]:
    """Every ``common_attn_metadata.<field>`` the functions' source reads."""
    fields: set[str] = set()
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "common_attn_metadata"
            ):
                fields.add(node.attr)
    return fields


# build() also hands the metadata to split_decodes_and_prefills.
_BUILDER_FIELDS = _fields_read(SpyreAttentionMetadataBuilder.build, split_decodes_and_prefills)


class _Recorder:
    supports_update_block_table = False  # read by build_attn_metadata

    def __init__(self):
        self.seen: list[CommonAttentionMetadata] = []

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.seen.append(common_attn_metadata)
        return SimpleNamespace()


def _prepare_attn(batch):
    state = SpyreDefaultModelState(
        make_config(),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    recorder = _Recorder()
    group = SimpleNamespace(
        layer_names=["layer.0"], kv_cache_spec=None, get_metadata_builder=lambda _: recorder
    )
    block_tables = (torch.arange(batch.num_reqs * 4, dtype=torch.int32).view(batch.num_reqs, 4),)
    slot_mappings = torch.full((1, batch.num_tokens_after_padding), -1, dtype=torch.int64)
    attn_metadata = state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        [[group]],  # ty: ignore[invalid-argument-type]
        # build_attn_metadata only counts the groups; KVCacheTensor differs across vLLM pins.
        SimpleNamespace(kv_cache_groups=[None]),  # ty: ignore[invalid-argument-type]
    )
    assert list(attn_metadata) == ["layer.0"]
    (common,) = recorder.seen
    return common, block_tables


def test_the_field_list_is_not_vacuous():
    assert {"seq_lens", "query_start_loc", "slot_mapping", "is_prefilling"} <= _BUILDER_FIELDS


@pytest.mark.parametrize("field", sorted(_BUILDER_FIELDS))
def test_every_field_the_builder_reads_is_set(field):
    common, _ = _prepare_attn(make_input_batch([1, 1, 5], num_computed=[10, 20, 0]))
    assert getattr(common, field) is not None


def test_values_match_the_step():
    """Decode rows first, then a prefill, the body bucketed from 7 to 8 tokens."""
    batch = make_input_batch([1, 1, 5], num_computed=[10, 20, 0], num_tokens_after_padding=8)
    common, block_tables = _prepare_attn(batch)

    assert common.num_reqs == 3
    # Real tokens only; the bucket pad past them is not a request.
    assert common.num_actual_tokens == 7
    # Padded, so the traced KV write keeps one shape per bucket.
    assert common.slot_mapping.shape == (8,)
    assert common.query_start_loc.tolist() == [0, 1, 2, 7]
    assert common.query_start_loc_cpu.tolist() == [0, 1, 2, 7]
    assert common.seq_lens.tolist() == [11, 21, 5]
    assert (common.max_query_len, common.max_seq_len) == (5, 21)
    assert common.block_table_tensor is block_tables[0]
    assert common.causal is True
