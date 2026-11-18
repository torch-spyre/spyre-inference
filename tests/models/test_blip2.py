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

"""Tests for `spyre_inference/models/blip2.py`.

`SpyreBlip2QFormerMultiHeadAttention` is an instance-level subclass of upstream's
`Blip2QFormerMultiHeadAttention`: it calls `F.scaled_dot_product_attention` with an explicit
`.contiguous()` after each head-shape transpose, so it runs fully on-device on Spyre.
`use_spyre_qformer_attention` swaps it onto the modules under a root, leaving the upstream class
alone. The tests cover the staleness tripwire, the swap's scope, output equivalence with the
stock forward (self-attention and cross-attention), and on-card execution.
"""

import sys

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

blip2 = pytest.importorskip("vllm.model_executor.models.blip2")

pytestmark = pytest.mark.blip2

# HEAD_DIM must be stick-aligned (a multiple of 64) on Spyre, as Granite Vision's Q-Former is.
HEAD_DIM, NUM_HEADS = 64, 4
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM

# The swap never changes the upstream class, so this is the stock forward whenever it is read.
_STOCK_FORWARD = blip2.Blip2QFormerMultiHeadAttention.forward


# --- Helpers ---


def _finish_weight_loading(module: nn.Module) -> None:
    """Run ``process_weights_after_loading`` on every vLLM linear in ``module``, so the Spyre
    linears have their transposed weight and ``spyre_row_padding`` set before any forward."""
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _qformer_config(cross: bool = False):
    kwargs = {"encoder_hidden_size": HIDDEN_SIZE} if cross else {}
    return blip2.Blip2QFormerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_probs_dropout_prob=0.0,
        **kwargs,
    )


def _qformer_attention(cross: bool = False, seed: int = 0) -> nn.Module:
    """A real upstream ``Blip2QFormerMultiHeadAttention`` with deterministic weights."""
    attn = blip2.Blip2QFormerMultiHeadAttention(
        _qformer_config(cross), quant_config=None, cache_config=None, is_cross_attention=cross
    ).to(torch.float16)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for p in attn.parameters():
        p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
    _finish_weight_loading(attn)
    return attn


def _swapped(attn: nn.Module) -> nn.Module:
    """``attn`` as a model holds it after construction: a child module, swapped."""
    from spyre_inference.models.blip2 import use_spyre_qformer_attention

    parent = nn.Module()
    parent.attn = attn
    use_spyre_qformer_attention(parent)
    return parent.attn


# --- 1. Staleness tripwire ---


def test_swap_target_still_exists():
    """The class the swap looks for must still exist upstream; otherwise the swap silently
    matches nothing."""
    assert getattr(blip2, "Blip2QFormerMultiHeadAttention", None) is not None, (
        "vllm.model_executor.models.blip2.Blip2QFormerMultiHeadAttention is gone, so "
        "use_spyre_qformer_attention swaps nothing and must be updated"
    )


# --- 2. The swap ---


def test_swap_replaces_only_the_given_modules():
    """The Spyre class goes onto the modules under ``root``, not onto the upstream class other
    models share. Parameters and their names stay put, so the checkpoint maps unchanged."""
    from spyre_inference.models.blip2 import (
        SpyreBlip2QFormerMultiHeadAttention,
        use_spyre_qformer_attention,
    )

    mine, other = nn.Module(), nn.Module()
    mine.attn, other.attn = _qformer_attention(), _qformer_attention()
    names = list(mine.state_dict())

    assert use_spyre_qformer_attention(mine) == 1
    assert use_spyre_qformer_attention(mine) == 0  # per-load re-entrancy

    assert type(mine.attn) is SpyreBlip2QFormerMultiHeadAttention
    assert type(other.attn) is blip2.Blip2QFormerMultiHeadAttention
    assert blip2.Blip2QFormerMultiHeadAttention.forward is _STOCK_FORWARD
    assert list(mine.state_dict()) == names


def test_swap_reaches_self_and_cross_attention():
    """A Q-Former layer holds one self-attention and one cross-attention; both are swapped."""
    from spyre_inference.models.blip2 import (
        SpyreBlip2QFormerMultiHeadAttention,
        use_spyre_qformer_attention,
    )

    layer = blip2.Blip2QFormerLayer(
        _qformer_config(cross=True), quant_config=None, cache_config=None, layer_idx=0
    )
    assert use_spyre_qformer_attention(layer) == 2
    assert type(layer.attention.attention) is SpyreBlip2QFormerMultiHeadAttention
    assert type(layer.crossattention.attention) is SpyreBlip2QFormerMultiHeadAttention


# --- 3. Numeric equivalence on CPU ---


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_attention_matches_stock(tp_group, bsz):
    rng = torch.Generator(device="cpu").manual_seed(10 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    expected = _STOCK_FORWARD(_qformer_attention(), hidden_states)
    actual = _swapped(_qformer_attention())(hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_cross_attention_matches_stock(tp_group, bsz):
    rng = torch.Generator(device="cpu").manual_seed(20 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    expected = _STOCK_FORWARD(
        _qformer_attention(cross=True, seed=2), hidden_states, encoder_hidden_states
    )
    actual = _swapped(_qformer_attention(cross=True, seed=2))(hidden_states, encoder_hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


# --- 4. On-card: forward on device (skipped without Spyre) ---


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_attention_matches_cpu_on_spyre(tp_group, bsz):
    if not spyre_available():
        pytest.skip("Spyre device not available")

    rng = torch.Generator(device="cpu").manual_seed(5 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    expected = _STOCK_FORWARD(_qformer_attention(), hidden_states)

    device = torch.device("spyre")
    actual = _swapped(_qformer_attention()).to(device)(hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("bsz", [1, 2])
def test_swapped_cross_attention_matches_cpu_on_spyre(tp_group, bsz):
    if not spyre_available():
        pytest.skip("Spyre device not available")

    rng = torch.Generator(device="cpu").manual_seed(7 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    expected = _STOCK_FORWARD(
        _qformer_attention(cross=True, seed=2), hidden_states, encoder_hidden_states
    )

    device = torch.device("spyre")
    attn = _swapped(_qformer_attention(cross=True, seed=2)).to(device)
    actual = attn(hidden_states.to(device), encoder_hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
