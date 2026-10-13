# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `spyre_inference/multimodal/blip2.py`.

`patch_blip2_qformer_attention` is a class-level patch with a `_spyre_patched`
flag. The patched forward replaces `Blip2QFormerMultiHeadAttention.forward` to
call `F.scaled_dot_product_attention` with explicit `.contiguous()` materialization
after each transpose so it runs fully on-device on Spyre. The tests cover the
staleness tripwires, idempotency, apply() entry point, output equivalence with
stock forward (self-attention and cross-attention), and on-card execution.
"""

import sys

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

blip2 = pytest.importorskip("vllm.model_executor.models.blip2")

# Minimal dimensions that exercise the attention path without a full model load.
# HEAD_DIM must be stick-aligned (a multiple of 64) on Spyre, matching Granite Vision's
# Q-Former head_dim=64 configuration.
HEAD_DIM = 64
NUM_HEADS = 4
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 256

# Capture the unpatched forward at import time, before any test can trigger
# the process-wide class patch via patch_blip2_qformer_attention().
_STOCK_FORWARD = blip2.Blip2QFormerMultiHeadAttention.forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finish_weight_loading(module: nn.Module) -> None:
    """Run `process_weights_after_loading` on every linear in `module`.

    Required so the Spyre OOT linears have their transposed weight and
    `spyre_row_padding` set before any `forward` call.
    """
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _make_qformer_attention(tp_group) -> nn.Module:
    """Instantiate a real Blip2QFormerMultiHeadAttention with deterministic weights."""
    from vllm.model_executor.models.blip2 import Blip2QFormerConfig

    config = Blip2QFormerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_probs_dropout_prob=0.0,
    )
    attn = blip2.Blip2QFormerMultiHeadAttention(
        config, quant_config=None, cache_config=None, is_cross_attention=False
    ).to(torch.float16)
    rng = torch.Generator(device="cpu").manual_seed(0)
    for p in attn.parameters():
        p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
    _finish_weight_loading(attn)
    return attn


# ---------------------------------------------------------------------------
# 1. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.blip2
@pytest.mark.parametrize(
    "symbol",
    [
        "Blip2QFormerMultiHeadAttention",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the patch reaches for must still exist in the vLLM module.
    The `try/except ImportError` path returns silently on missing symbols,
    so this is the only place a rename or removal is caught."""
    assert getattr(blip2, symbol, None) is not None, (
        f"vllm.model_executor.models.blip2.{symbol} is gone — the corresponding "
        "Spyre patch in multimodal/blip2.py is now a silent no-op and must be updated"
    )


# ---------------------------------------------------------------------------
# 2. Patch application and idempotency
# ---------------------------------------------------------------------------


@pytest.mark.blip2
def test_patch_is_applied_and_idempotent():
    """`patch_blip2_qformer_attention` must mark the forward with `_spyre_patched`
    and a second call must leave the same function in place."""
    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()
    patched_forward = blip2.Blip2QFormerMultiHeadAttention.forward
    assert getattr(patched_forward, "_spyre_patched", False) is True

    patch_blip2_qformer_attention()
    assert blip2.Blip2QFormerMultiHeadAttention.forward is patched_forward, (
        "second call must be a no-op — forward must not be double-wrapped"
    )


# ---------------------------------------------------------------------------
# 3. Numeric equivalence on CPU
# ---------------------------------------------------------------------------


@pytest.mark.blip2
@pytest.mark.parametrize("bsz", [1, 2])
def test_patched_forward_output_matches_stock(tp_group, bsz):
    """The SDPA patched forward must produce the same output as the stock forward."""
    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    seq_len = 8
    rng = torch.Generator(device="cpu").manual_seed(10 + bsz)
    hidden_states = torch.randn(bsz, seq_len, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    # Use the forward captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    attn_stock = _make_qformer_attention(tp_group)
    expected = _STOCK_FORWARD(attn_stock, hidden_states)

    patch_blip2_qformer_attention()
    attn_patched = _make_qformer_attention(tp_group)
    actual = blip2.Blip2QFormerMultiHeadAttention.forward(attn_patched, hidden_states)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.blip2
@pytest.mark.parametrize("bsz", [1, 2])
def test_patched_forward_with_cross_attention_matches_stock(tp_group, bsz):
    """Cross-attention variant (encoder_hidden_states is not None) must also
    produce the same output as the stock forward."""
    from vllm.model_executor.models.blip2 import Blip2QFormerConfig

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    # encoder_hidden_size must match hidden_size so the cross-attention key/value
    # projections (in_features=encoder_hidden_size) accept our test tensors.
    config = Blip2QFormerConfig(
        hidden_size=HIDDEN_SIZE,
        encoder_hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_probs_dropout_prob=0.0,
    )

    def _make_cross_attn():
        attn = blip2.Blip2QFormerMultiHeadAttention(
            config, quant_config=None, cache_config=None, is_cross_attention=True
        ).to(torch.float16)
        rng = torch.Generator(device="cpu").manual_seed(2)
        for p in attn.parameters():
            p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
        _finish_weight_loading(attn)
        return attn

    rng = torch.Generator(device="cpu").manual_seed(20 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    # Use the forward captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    expected = _STOCK_FORWARD(_make_cross_attn(), hidden_states, encoder_hidden_states)

    patch_blip2_qformer_attention()
    actual = blip2.Blip2QFormerMultiHeadAttention.forward(
        _make_cross_attn(), hidden_states, encoder_hidden_states
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.blip2
def test_apply_invokes_patch():
    """`apply(model, device)` must invoke `patch_blip2_qformer_attention` and be idempotent."""
    from spyre_inference.multimodal import blip2 as blip2_patch

    dummy_model = nn.Module()
    blip2_patch.apply(dummy_model, torch.device("cpu"))
    assert getattr(blip2.Blip2QFormerMultiHeadAttention.forward, "_spyre_patched", False) is True

    # Calling apply again must succeed without error
    blip2_patch.apply(dummy_model, torch.device("cpu"))
    assert getattr(blip2.Blip2QFormerMultiHeadAttention.forward, "_spyre_patched", False) is True


# ---------------------------------------------------------------------------
# 4. On-card: forward on device (skipped without Spyre)
# ---------------------------------------------------------------------------


@pytest.mark.blip2
@pytest.mark.parametrize("bsz", [1, 2])
def test_patched_forward_output_matches_cpu_on_spyre(tp_group, bsz):
    """The patched forward on Spyre must match the stock forward on CPU."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    rng = torch.Generator(device="cpu").manual_seed(5 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    # Use the forward captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    attn_cpu = _make_qformer_attention(tp_group)
    expected = _STOCK_FORWARD(attn_cpu, hidden_states)

    device = torch.device("spyre")
    attn_dev = _make_qformer_attention(tp_group).to(device)
    actual = blip2.Blip2QFormerMultiHeadAttention.forward(attn_dev, hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.blip2
@pytest.mark.parametrize("bsz", [1, 2])
def test_patched_forward_cross_attention_output_matches_cpu_on_spyre(tp_group, bsz):
    """Patched cross-attention on Spyre must match the stock forward on CPU."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.model_executor.models.blip2 import Blip2QFormerConfig

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    config = Blip2QFormerConfig(
        hidden_size=HIDDEN_SIZE,
        encoder_hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_probs_dropout_prob=0.0,
    )

    def _make_cross_attn():
        attn = blip2.Blip2QFormerMultiHeadAttention(
            config, quant_config=None, cache_config=None, is_cross_attention=True
        ).to(torch.float16)
        rng = torch.Generator(device="cpu").manual_seed(2)
        for p in attn.parameters():
            p.data.copy_(torch.empty_like(p.data, device="cpu").normal_(std=0.02, generator=rng))
        _finish_weight_loading(attn)
        return attn

    rng = torch.Generator(device="cpu").manual_seed(7 + bsz)
    hidden_states = torch.randn(bsz, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)
    encoder_hidden_states = torch.randn(bsz, 16, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    # Use the forward captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    attn_cpu = _make_cross_attn()
    expected = _STOCK_FORWARD(attn_cpu, hidden_states, encoder_hidden_states)

    device = torch.device("spyre")
    attn_dev = _make_cross_attn().to(device)
    actual = blip2.Blip2QFormerMultiHeadAttention.forward(
        attn_dev, hidden_states.to(device), encoder_hidden_states.to(device)
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
