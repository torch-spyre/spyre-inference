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

"""Tests for the Mistral/Llama-4 attention-temperature-scaling detour.

The scale needs an int64->float convert and torch-spyre's typecast table has no
int64 entry, so it runs on CPU in an opaque op reusing upstream's formula. These
tests pin that reuse, the per-step cache, and the per-instance patch.
"""

import sys

import pytest
import torch
import torch.nn as nn

BETA = 8.0
ORIG_MAX = 8192


def _fake_mistral_attention(do_scaling: bool = True):
    """A real `MistralAttention` carrying only the attributes the patch reads.

    Built without `__init__` (which needs a full `VllmConfig`), but a genuine
    instance so the patch's `isinstance` traversal finds it."""
    from vllm.model_executor.models.mistral import MistralAttention

    attn = object.__new__(MistralAttention)
    nn.Module.__init__(attn)
    attn.do_llama_4_scaling = do_scaling
    if do_scaling:
        attn.llama_4_scaling_beta = BETA
        attn.llama_4_scaling_original_max_position_embeddings = ORIG_MAX
    return attn


def _model_with(*attns):
    model = nn.Module()
    for i, attn in enumerate(attns):
        model.add_module(f"layer_{i}", attn)
    return model


class _RunnerStub:
    """`_patch_llama4_attn_scale` only touches `self.model`."""

    def __init__(self, model):
        self.model = model


@pytest.fixture(autouse=True)
def reset_scale_cache(monkeypatch):
    """The scale cache is module-global and holds a `positions` reference; clear it
    so one test cannot serve another's result."""
    from spyre_inference.v1.worker import spyre_model_runner

    monkeypatch.setattr(spyre_model_runner, "_llama4_scale_cache", None)
    yield


# ---------------------------------------------------------------------------
# The op body reuses upstream's formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 5, 64])
def test_op_matches_upstream_formula(num_tokens):
    """Equals upstream's `_get_llama_4_attn_scale`, so a formula change fails here
    rather than drifting silently."""
    from spyre_inference.v1.worker.spyre_model_runner import _llama4_attn_scale_op

    positions = torch.arange(num_tokens, dtype=torch.int64) * 4096
    expected = _fake_mistral_attention()._get_llama_4_attn_scale(positions)

    actual = _llama4_attn_scale_op(positions, BETA, ORIG_MAX)

    assert actual.dtype == torch.float16, "the decoder consumes fp16 activations"
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_op_shape_matches_fake_impl():
    """The fake (meta) impl must agree with the real one or torch.compile traces
    a wrong-shaped graph."""
    from spyre_inference.v1.worker.spyre_model_runner import (
        _llama4_attn_scale_fake,
        _llama4_attn_scale_op,
    )

    positions = torch.arange(7, dtype=torch.int64)
    real = _llama4_attn_scale_op(positions, BETA, ORIG_MAX)
    fake = _llama4_attn_scale_fake(positions, BETA, ORIG_MAX)

    assert real.shape == fake.shape == (7, 1)
    assert real.dtype == fake.dtype


def test_scale_is_one_below_the_original_context_length():
    """Sanity anchor on the formula itself: within the original context the
    temperature scale is exactly 1 (log(1 + 0))."""
    from spyre_inference.v1.worker.spyre_model_runner import _llama4_attn_scale_op

    positions = torch.arange(0, ORIG_MAX, 1024, dtype=torch.int64)
    actual = _llama4_attn_scale_op(positions, BETA, ORIG_MAX)

    torch.testing.assert_close(actual.float(), torch.ones_like(actual.float()))


# ---------------------------------------------------------------------------
# The per-step cache
# ---------------------------------------------------------------------------


def test_scale_is_cached_across_decoder_layers():
    """All 40 decoder layers get the same `positions`, so the D2H/H2D round trip
    must happen once per step, not once per layer."""
    from spyre_inference.v1.worker.spyre_model_runner import _llama4_attn_scale_op

    positions = torch.arange(16, dtype=torch.int64) * 4096

    first = _llama4_attn_scale_op(positions, BETA, ORIG_MAX)
    assert all(_llama4_attn_scale_op(positions, BETA, ORIG_MAX) is first for _ in range(39))


def test_scale_cache_misses_on_new_positions():
    """The next step brings new positions; the cache must not serve the old scale."""
    from spyre_inference.v1.worker.spyre_model_runner import _llama4_attn_scale_op

    first = _llama4_attn_scale_op(torch.arange(4, dtype=torch.int64) * 4096, BETA, ORIG_MAX)
    second = _llama4_attn_scale_op(
        torch.arange(4, dtype=torch.int64) * 4096 + ORIG_MAX * 8, BETA, ORIG_MAX
    )

    assert second is not first
    assert not torch.equal(second, first)


def test_scale_cache_misses_on_a_buffer_mutated_in_place():
    """`positions` is freshly allocated per step today, so identity alone would do. The
    endpoints are in the key so a persistent buffer written in place still misses."""
    from spyre_inference.v1.worker.spyre_model_runner import _llama4_attn_scale_op

    buffer = torch.arange(4, dtype=torch.int64) * 4096
    first = _llama4_attn_scale_op(buffer, BETA, ORIG_MAX).clone()

    # Same storage, same shape, new values: a data_ptr-only key would hit.
    buffer.copy_(torch.arange(4, dtype=torch.int64) * 4096 + ORIG_MAX * 8)
    second = _llama4_attn_scale_op(buffer, BETA, ORIG_MAX)

    assert not torch.equal(second, first)


# ---------------------------------------------------------------------------
# The per-instance patch
# ---------------------------------------------------------------------------


def test_patch_rebinds_only_scaling_modules():
    """Modules with `do_llama_4_scaling=False` keep upstream's bound method."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    scaling = _fake_mistral_attention(do_scaling=True)
    plain = _fake_mistral_attention(do_scaling=False)
    stub = _RunnerStub(_model_with(scaling, plain))

    TorchSpyreModelRunner._patch_llama4_attn_scale(stub)

    assert "_get_llama_4_attn_scale" in scaling.__dict__, "scaling module not patched"
    assert "_get_llama_4_attn_scale" not in plain.__dict__, "non-scaling module patched"


def test_patch_raises_when_upstream_attribute_disappears(monkeypatch):
    """Staleness tripwire: if upstream renames `do_llama_4_scaling`, the patch must
    fail loudly instead of silently skipping every layer."""
    from vllm.model_executor.models.mistral import MistralAttention

    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    attn = _fake_mistral_attention(do_scaling=True)
    monkeypatch.delattr(attn, "do_llama_4_scaling")
    # Guard against the attribute being inherited from the class instead.
    if hasattr(MistralAttention, "do_llama_4_scaling"):
        monkeypatch.delattr(MistralAttention, "do_llama_4_scaling")

    stub = _RunnerStub(_model_with(attn))

    with pytest.raises(RuntimeError, match="do_llama_4_scaling"):
        TorchSpyreModelRunner._patch_llama4_attn_scale(stub)


def test_patch_is_a_noop_without_mistral_modules():
    """Non-Mistral models must be untouched (and must not trip the guard)."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    stub = _RunnerStub(nn.Sequential(nn.Linear(4, 4)))
    TorchSpyreModelRunner._patch_llama4_attn_scale(stub)  # must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
