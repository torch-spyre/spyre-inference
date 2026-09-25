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
flag. The patched forward offloads the entire `Blip2QFormerMultiHeadAttention`
to CPU for the duration of the call and moves the result back to the original
device. The tests cover the staleness tripwire, idempotency, device restoration,
and output equivalence.

Section 4 repeats the numeric check on the card and skips without a device.
"""

import sys

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

blip2 = pytest.importorskip("vllm.model_executor.models.blip2")

# Minimal dimensions that exercise the attention path without a full model load.
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

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
# 3. CPU offload contract
# ---------------------------------------------------------------------------


@pytest.mark.blip2
def test_patched_forward_output_matches_stock(tp_group):
    """The CPU-offloaded forward must produce the same output as the stock
    forward on CPU — moving to CPU and back must not change values."""
    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    seq_len = 8
    hidden_states = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float16)

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
def test_patched_forward_with_cross_attention_matches_stock(tp_group):
    """Cross-attention variant (encoder_hidden_states is not None) must also
    produce the same output after the CPU offload."""
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

    hidden_states = torch.randn(1, 8, HIDDEN_SIZE, dtype=torch.float16)
    encoder_hidden_states = torch.randn(1, 16, HIDDEN_SIZE, dtype=torch.float16)

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
def test_patched_forward_restores_module_device_when_inputs_on_cpu(tp_group):
    """target_device is read from module parameters, not the input tensor.

    Staleness guard: asserts that next(self.parameters()).device is used to
    determine the restore target.  Uses a monkeypatched sentinel so the test
    is meaningful on CPU: if the implementation reverts to hidden_states.device
    the sentinel would never be consulted and the assertion fires.
    """
    import types

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    attn = _make_qformer_attention(tp_group)  # on CPU

    # Replace next(self.parameters()) with a sentinel that records whether it
    # was called.  The patched forward calls next(self.parameters()).device, so
    # if it regresses to hidden_states.device the sentinel is never hit.
    parameters_called = []
    real_parameters = attn.parameters

    def _spy_parameters(self):
        parameters_called.append(True)
        return real_parameters()

    attn.parameters = types.MethodType(_spy_parameters, attn)

    hidden_states = torch.randn(1, 8, HIDDEN_SIZE, dtype=torch.float16)
    blip2.Blip2QFormerMultiHeadAttention.forward(attn, hidden_states)

    assert parameters_called, (
        "patched forward never called self.parameters() — "
        "target_device is not being derived from module parameters"
    )
    for name, param in attn.named_parameters():
        assert param.device.type == "cpu", (
            f"parameter {name!r} is on {param.device} after patched forward — "
            "module was not restored to its original device"
        )


# ---------------------------------------------------------------------------
# 4. On-card: module restored to device after call (skipped without Spyre)
# ---------------------------------------------------------------------------


@pytest.mark.blip2
def test_patched_forward_restores_module_to_spyre_with_cpu_inputs(tp_group):
    """Module on Spyre, inputs on CPU: the module must be restored to Spyre.

    This is the exact regression scenario Kevin's fix addresses.  The old code
    read target_device from hidden_states.device (CPU), so the finally block
    called self.to("cpu") and permanently stranded the module.  The fix reads
    next(self.parameters()).device (Spyre) instead.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    device = torch.device("spyre")
    attn = _make_qformer_attention(tp_group).to(device)

    # Inputs deliberately on CPU — this is what triggers the bug in old code.
    hidden_states = torch.randn(1, 8, HIDDEN_SIZE, dtype=torch.float16)
    blip2.Blip2QFormerMultiHeadAttention.forward(attn, hidden_states)

    for name, param in attn.named_parameters():
        assert param.device.type == "spyre", (
            f"parameter {name!r} is on {param.device} after patched forward with "
            "CPU inputs — module was stranded off Spyre (target_device regression)"
        )


@pytest.mark.blip2
def test_patched_forward_restores_module_to_spyre(tp_group):
    """The patched forward moves `self` to CPU for the call and must restore
    it to Spyre afterward — a failure here would silently strand the module
    on CPU, causing every subsequent layer to get a CPU input on Spyre."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    device = torch.device("spyre")
    attn = _make_qformer_attention(tp_group).to(device)

    hidden_states = torch.randn(1, 8, HIDDEN_SIZE, dtype=torch.float16).to(device)
    blip2.Blip2QFormerMultiHeadAttention.forward(attn, hidden_states)

    # All parameters must be back on Spyre after the call.
    for name, param in attn.named_parameters():
        assert param.device.type == "spyre", (
            f"parameter {name!r} is on {param.device} after patched forward — "
            "module was not restored to Spyre"
        )


@pytest.mark.blip2
def test_patched_forward_output_matches_cpu_on_spyre(tp_group):
    """The patched forward on-card must equal the same forward on CPU."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.multimodal.blip2 import patch_blip2_qformer_attention

    patch_blip2_qformer_attention()

    rng = torch.Generator(device="cpu").manual_seed(5)
    hidden_states = torch.randn(1, 8, HIDDEN_SIZE, dtype=torch.float16, generator=rng)

    # Use the forward captured at import time — guards against earlier tests
    # having already applied the process-wide class patch.
    attn_cpu = _make_qformer_attention(tp_group)
    expected = _STOCK_FORWARD(attn_cpu, hidden_states)

    device = torch.device("spyre")
    attn_dev = _make_qformer_attention(tp_group).to(device)
    actual = blip2.Blip2QFormerMultiHeadAttention.forward(attn_dev, hidden_states.to(device))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
