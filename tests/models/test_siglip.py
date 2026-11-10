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

"""Tests for `spyre_inference/models/siglip.py`.

`pad_siglip_heads` zero-extends each head of a SigLIP tower's Q/K/V/O projections from the
checkpoint's 72 to the 128-wide stick, on a stand-in tower that carries what it reads. The tests
cover every layer being widened, a stick-aligned head_dim and layers after an aligned one,
idempotency, the original values staying in place, and output equivalence with the unpadded
attention.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("vllm.model_executor.models.siglip")

pytestmark = pytest.mark.siglip

SIG_ORIG, SIG_PAD, SIG_HEADS = 72, 128, 16  # so400m: head_dim 72, padded to the 64-wide stick
SIG_HIDDEN = SIG_HEADS * SIG_ORIG


class _TransposedLinear(nn.Module):
    """Stand-in for a Spyre vLLM linear after loading: weight stored as Wᵀ, forward returns
    ``(output, None)``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features, dtype=torch.float16))
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None
        )

    def forward(self, x: torch.Tensor):
        out = x @ self.weight
        return (out if self.bias is None else out + self.bias), None


def _siglip_tower(
    num_layers: int = 2, num_heads: int = SIG_HEADS, head_dim: int = SIG_ORIG, bias=False
):
    """A tower with ``encoder.layers[*].self_attn`` carrying what ``pad_siglip_heads`` reads."""
    hidden = num_heads * head_dim
    rng = torch.Generator(device="cpu").manual_seed(42)
    layers = nn.ModuleList()
    for _ in range(num_layers):
        attn = nn.Module()
        attn.head_dim = head_dim
        attn.num_heads_per_partition = num_heads
        attn.qkv_proj = _TransposedLinear(hidden, 3 * hidden, bias=bias)
        attn.out_proj = _TransposedLinear(hidden, hidden)
        for p in (*attn.qkv_proj.parameters(), *attn.out_proj.parameters()):
            p.data.normal_(std=0.02, generator=rng)
        # What MMEncoderAttention holds: the scale is fixed at construction, from the real dim.
        attn.attn = SimpleNamespace(head_size=head_dim, scale=head_dim**-0.5)
        layer = nn.Module()
        layer.self_attn = attn
        layers.append(layer)
    tower = nn.Module()
    tower.encoder = nn.Module()
    tower.encoder.layers = layers
    return tower


def test_pad_siglip_heads_widens_every_layer():
    from spyre_inference.models.siglip import pad_siglip_heads

    tower = _siglip_tower(num_layers=3)
    pad_siglip_heads(tower)

    for layer in tower.encoder.layers:
        attn = layer.self_attn
        assert attn.qkv_proj.weight.shape == (SIG_HIDDEN, 3 * SIG_HEADS * SIG_PAD)
        assert attn.out_proj.weight.shape == (SIG_HEADS * SIG_PAD, SIG_HIDDEN)
        assert attn.attn.head_size == SIG_PAD
        assert attn.attn.scale == SIG_ORIG**-0.5, "the scale must stay at the real head dim"


def test_pad_siglip_heads_leaves_a_stick_aligned_head_dim_alone():
    from spyre_inference.models.siglip import pad_siglip_heads

    tower = _siglip_tower(num_heads=4, head_dim=64)
    attn = tower.encoder.layers[0].self_attn
    qkv_shape, out_shape = attn.qkv_proj.weight.shape, attn.out_proj.weight.shape

    pad_siglip_heads(tower)

    assert attn.qkv_proj.weight.shape == qkv_shape
    assert attn.out_proj.weight.shape == out_shape
    assert attn.attn.head_size == 64


def test_pad_siglip_heads_pads_layers_after_an_aligned_one():
    """An aligned layer is skipped on its own; the layers after it are still padded."""
    from spyre_inference.models.siglip import pad_siglip_heads

    tower = _siglip_tower(num_layers=2)
    aligned = _siglip_tower(num_layers=1, num_heads=4, head_dim=64).encoder.layers[0]
    tower.encoder.layers[0] = aligned
    aligned_shape = aligned.self_attn.qkv_proj.weight.shape

    pad_siglip_heads(tower)

    assert aligned.self_attn.qkv_proj.weight.shape == aligned_shape
    assert aligned.self_attn.attn.head_size == 64
    padded = tower.encoder.layers[1].self_attn
    assert padded.qkv_proj.weight.shape == (SIG_HIDDEN, 3 * SIG_HEADS * SIG_PAD)
    assert padded.attn.head_size == SIG_PAD


def test_pad_siglip_heads_is_idempotent():
    from spyre_inference.models.siglip import pad_siglip_heads

    tower = _siglip_tower()
    pad_siglip_heads(tower)
    padded = tower.encoder.layers[0].self_attn.qkv_proj.weight.detach().clone()
    pad_siglip_heads(tower)  # would widen again if the first pass were not remembered

    assert tower.encoder.layers[0].self_attn.qkv_proj.weight.shape == padded.shape
    assert all(layer.self_attn._spyre_head_dim_padded for layer in tower.encoder.layers)


@pytest.mark.parametrize("with_bias", [False, True])
def test_pad_siglip_heads_keeps_the_original_values_in_place(with_bias):
    """Each head block of Q, K and V keeps its 72 values at its padded offset, followed by
    zeros, in the weight and the bias; the out weight pads its rows the same way."""
    from spyre_inference.models.siglip import pad_siglip_heads

    tower = _siglip_tower(bias=with_bias)
    attn = tower.encoder.layers[0].self_attn
    if with_bias:
        attn.qkv_proj.bias.data.normal_(std=0.02)
    qkv_w = attn.qkv_proj.weight.data.clone()
    qkv_b = attn.qkv_proj.bias.data.clone() if with_bias else None
    out_w = attn.out_proj.weight.data.clone()

    pad_siglip_heads(tower)

    n, d, pd = SIG_HEADS, SIG_ORIG, SIG_PAD
    for block in range(3):
        for h in range(n):
            src, dst = block * n * d + h * d, block * n * pd + h * pd
            assert torch.equal(attn.qkv_proj.weight[:, dst : dst + d], qkv_w[:, src : src + d])
            assert not attn.qkv_proj.weight[:, dst + d : dst + pd].any()
            if with_bias:
                assert torch.equal(attn.qkv_proj.bias[dst : dst + d], qkv_b[src : src + d])
                assert not attn.qkv_proj.bias[dst + d : dst + pd].any()
    for h in range(n):
        src, dst = h * d, h * pd
        assert torch.equal(attn.out_proj.weight[dst : dst + d], out_w[src : src + d])
        assert not attn.out_proj.weight[dst + d : dst + pd].any()


def test_padded_siglip_attention_matches_unpadded():
    """Padded projections through ``padded_sdpa`` at the real scale are the unpadded attention:
    the zero columns add nothing to the scores, and the zero rows of the out weight drop the
    padded value lanes."""
    from spyre_inference.models.siglip import pad_siglip_heads
    from spyre_inference.multimodal.utils import padded_sdpa

    tower = _siglip_tower(num_layers=1)
    attn = tower.encoder.layers[0].self_attn
    bsz, seq = 1, 64
    rng = torch.Generator(device="cpu").manual_seed(6)
    x = torch.randn(bsz, seq, SIG_HIDDEN, dtype=torch.float16, generator=rng)

    def run(dim: int) -> torch.Tensor:
        q, k, v = (
            t.view(bsz, seq, SIG_HEADS, dim).transpose(1, 2)
            for t in (x @ attn.qkv_proj.weight).chunk(3, dim=-1)
        )
        out = padded_sdpa(q, k, v, None, scale=SIG_ORIG**-0.5)
        return out.transpose(1, 2).reshape(bsz, seq, -1) @ attn.out_proj.weight

    q, k, v = (
        t.view(bsz, seq, SIG_HEADS, SIG_ORIG).transpose(1, 2)
        for t in (x @ attn.qkv_proj.weight).chunk(3, dim=-1)
    )
    want = F.scaled_dot_product_attention(q, k, v, scale=SIG_ORIG**-0.5)
    want = want.transpose(1, 2).reshape(bsz, seq, -1) @ attn.out_proj.weight

    pad_siglip_heads(tower)

    torch.testing.assert_close(run(SIG_PAD).float(), want.float(), atol=1e-3, rtol=1e-3)
