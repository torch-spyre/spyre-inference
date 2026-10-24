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

"""CPU semantics of complete per-head cache staging from fused projection views."""

import pytest
import torch

from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)


def _snapshot_reference(key, value, k_rows, v_rows, indices):
    key, value = key.clone(), value.clone()
    for h, index in enumerate(indices):
        k_rows.index_copy_(0, index, key[:, h])
        v_rows.index_copy_(0, index, value[:, h])


@pytest.mark.parametrize("rows", [1, 17, 512])
def test_strided_qkv_views_preserve_cache_and_source(rows):
    torch.manual_seed(121)
    qkv = torch.randn(rows, 1536, dtype=torch.float16)
    key = qkv[:, 1024:1280].view(rows, 2, 128)
    value = qkv[:, 1280:].view(rows, 2, 128)
    source = qkv.clone()
    key_cache = torch.randn(rows * 2 + 9, 128, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    expected_key, expected_value = key_cache.clone(), value_cache.clone()
    indices = [torch.arange(rows - 1, -1, -1) + 3, torch.arange(rows) + rows + 3]
    _snapshot_reference(key, value, expected_key, expected_value, indices)

    reshape_and_cache_head_major_kernel(key, value, key_cache, value_cache, indices)

    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(qkv, source, rtol=0, atol=0)


@pytest.mark.parametrize("shared_cache", [False, True])
def test_all_heads_are_snapshotted_before_cache_mutation(shared_cache):
    rows, heads, width = 3, 2, 128
    key_cache = torch.arange(rows * heads * width, dtype=torch.float32).view(-1, width)
    value_cache = key_cache if shared_cache else key_cache.neg().clone()
    # Head zero overwrites rows that head one will read. Staging inside the
    # mutation loop would corrupt head one's source, including cross-cache aliases.
    key = key_cache.view(heads, rows, width).transpose(0, 1)
    value = value_cache.view(heads, rows, width).transpose(0, 1)
    expected_key = key_cache.clone()
    expected_value = expected_key if shared_cache else value_cache.clone()
    indices = [torch.arange(rows) + rows, torch.arange(rows)]
    _snapshot_reference(key, value, expected_key, expected_value, indices)

    reshape_and_cache_head_major_kernel(key, value, key_cache, value_cache, indices)

    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)


def test_complete_snapshots_precede_every_cache_write():
    from torch.fx.experimental.proxy_tensor import make_fx

    qkv = torch.randn(17, 1536, dtype=torch.float16)
    key = qkv[:, 1024:1280].view(17, 2, 128)
    value = qkv[:, 1280:].view(17, 2, 128)
    cache = torch.zeros(64, 128, dtype=torch.float16)
    graph = make_fx(
        lambda key, value, k_rows, v_rows, row_index: reshape_and_cache_head_major_kernel(
            key, value, k_rows, v_rows, row_index
        )
    )(key, value, cache.clone(), cache.clone(), [torch.arange(17), torch.arange(17) + 32]).graph
    nodes = list(graph.nodes)
    scatters = [node for node in nodes if node.target == torch.ops.aten.index_copy_.default]
    assert len(scatters) == 4
    sources = [node.args[3] for node in scatters]
    assert len(set(sources)) == 4
    assert all(source.target == torch.ops.aten.select.int for source in sources)
    assert all(tuple(source.meta["val"].shape) == (17, 128) for source in sources)

    snapshots = [node for node in nodes if node.target == torch.ops.aten.mul.Tensor]
    assert len(snapshots) == 2
    inputs = [node for node in nodes if node.op == "placeholder"]
    assert [snapshot.args for snapshot in snapshots] == [
        (inputs[0], 1.0),
        (inputs[1], 1.0),
    ]
    assert all(tuple(snapshot.meta["val"].shape) == (17, 2, 128) for snapshot in snapshots)
    assert [source.args[0] for source in sources] == snapshots * 2
    assert [source.args[1:] for source in sources] == [(1, 0), (1, 0), (1, 1), (1, 1)]
    assert max(nodes.index(snapshot) for snapshot in snapshots) < min(
        nodes.index(node) for node in scatters
    )
