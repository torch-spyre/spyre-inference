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

"""``SpyreMambaMixer2`` at a TP size that is a multiple of ``n_groups``, on CPU.

Upstream refuses such a split (Nemotron-3 has 8 groups, Ultra needs TP16). The reference is
the unsharded layer: a TP shard must hold exactly the rows of the heads it owns and of the
one group those heads read, and the gated norm, split over ranks, must equal the norm of
the whole intermediate.
"""

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.mamba_mixer2 import group_replicated_loader, replicates_groups

H, P, G, N = 16, 4, 4, 8
INNER = H * P


@pytest.mark.parametrize(
    ("n_groups", "tp", "expected"),
    [(8, 16, True), (4, 8, True), (8, 8, False), (8, 4, False), (8, 1, False), (1, 8, False)],
)
def test_only_splits_upstream_refuses_are_replicated(n_groups, tp, expected):
    """Every split upstream shards itself keeps its code path."""
    assert replicates_groups(n_groups, tp) is expected


def _checkpoint_in_proj(gen):
    """``in_proj`` rows as a checkpoint stores them: gate, x, B, C, dt."""
    return {
        "gate": torch.randn(INNER, 3, generator=gen),
        "x": torch.randn(INNER, 3, generator=gen),
        "B": torch.randn(G * N, 3, generator=gen),
        "C": torch.randn(G * N, 3, generator=gen),
        "dt": torch.randn(H, 3, generator=gen),
    }


@pytest.mark.parametrize("tp", [8, 16])
def test_in_proj_shard_holds_its_heads_and_their_group(tp):
    """Rank r owns heads [r*H/tp, (r+1)*H/tp), which all read group r*G//tp."""
    parts = _checkpoint_in_proj(torch.Generator().manual_seed(tp))
    full = torch.cat([parts[k] for k in ("gate", "x", "B", "C", "dt")])
    heads, groups = (INNER, tp), (G * N, G)
    for rank in range(tp):
        loader = group_replicated_loader([heads, heads, groups, groups, (H, tp)], tp, rank)
        shard = torch.empty(2 * INNER // tp + 2 * N + H // tp, 3)
        loader(shard, full)
        rows, group = INNER // tp, rank * G // tp
        expected = torch.cat(
            [
                parts["gate"][rank * rows : (rank + 1) * rows],
                parts["x"][rank * rows : (rank + 1) * rows],
                parts["B"][group * N : (group + 1) * N],
                parts["C"][group * N : (group + 1) * N],
                parts["dt"][rank * H // tp : (rank + 1) * H // tp],
            ]
        )
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)


def test_heads_a_rank_owns_read_the_group_it_loads():
    """Upstream's head-to-group map (h // (H // G)) agrees with the loader's group pick."""
    for tp in (8, 16):
        for rank in range(tp):
            owned = range(rank * H // tp, (rank + 1) * H // tp)
            assert {h // (H // G) for h in owned} == {rank * G // tp}


def test_conv1d_bias_shard_matches_the_unsharded_layout():
    """conv1d is [x | B | C]: the same loader, without the gate and dt segments."""
    gen = torch.Generator().manual_seed(0)
    x, b, c = (
        torch.randn(INNER, generator=gen),
        torch.randn(G * N, generator=gen),
        torch.randn(G * N, generator=gen),
    )
    tp = 8
    for rank in range(tp):
        loader = group_replicated_loader([(INNER, tp), (G * N, G), (G * N, G)], tp, rank)
        shard = torch.empty(INNER // tp + 2 * N)
        loader(shard, torch.cat([x, b, c]))
        group, rows = rank * G // tp, INNER // tp
        expected = torch.cat(
            [
                x[rank * rows : (rank + 1) * rows],
                b[group * N : (group + 1) * N],
                c[group * N : (group + 1) * N],
            ]
        )
        torch.testing.assert_close(shard, expected, rtol=0, atol=0)


def test_gated_norm_split_over_ranks_equals_the_unsharded_norm(monkeypatch):
    """Each rank's partial sum of squares, all-reduced in its group's column, gives every
    rank the variance of its whole group."""
    from spyre_inference.custom_ops import mamba_mixer2 as m

    tp, eps = 8, 1e-5
    gen = torch.Generator().manual_seed(1)
    x, gate, weight = (torch.randn(5, INNER, generator=gen) for _ in range(3))
    y = x * F.silu(gate)
    grouped = y.view(5, G, INNER // G)
    want = (grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)).view(5, INNER)
    want = want * weight[0]

    norms = []
    for rank in range(tp):
        monkeypatch.setattr(
            "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
            lambda: tp,
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_rank",
            lambda rank=rank: rank,
        )
        norm = object.__new__(m.SpyreMixer2RMSNormGated)
        torch.nn.Module.__init__(norm)
        m.SpyreMixer2RMSNormGated.__init__(norm, INNER, G, True, eps=eps)
        with torch.no_grad():
            norm.weight.copy_(weight[0][rank * INNER // tp : (rank + 1) * INNER // tp])
        norms.append(norm)

    partials = []
    for rank, norm in enumerate(norms):
        cols = slice(rank * INNER // tp, (rank + 1) * INNER // tp)
        yr = y[:, cols]
        partials.append(yr.pow(2).sum(-1, keepdim=True) * norm._group_column)
    reduced = torch.stack(partials).sum(0)
    for rank, norm in enumerate(norms):
        cols = slice(rank * INNER // tp, (rank + 1) * INNER // tp)
        monkeypatch.setattr(m, "tensor_model_parallel_all_reduce", lambda _t: reduced)
        got = norm.forward_native(x[:, cols], gate[:, cols])
        torch.testing.assert_close(got, want[:, cols], rtol=1e-5, atol=1e-5)


def test_norm_keeps_upstream_path_when_tp_divides_the_groups(monkeypatch):
    from spyre_inference.custom_ops import mamba_mixer2 as m

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_rank", lambda: 0
    )
    norm = object.__new__(m.SpyreMixer2RMSNormGated)
    torch.nn.Module.__init__(norm)
    m.SpyreMixer2RMSNormGated.__init__(norm, INNER, G, True, eps=1e-5)
    assert norm._group_column is None
