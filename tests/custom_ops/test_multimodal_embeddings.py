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

"""Tests for the Spyre ``_merge_multimodal_embeddings`` replacement.

The CPU-placement + ``torch.where`` merge exists because the upstream boolean-mask
``index_put`` lowers to ``aten::_index_put_impl_``, which has no Spyre kernel. The
on-card cases below are the ones that matter: they prove the merge lands the image
tokens in the right rows on device and leaves the text rows untouched, which the
CPU reference can only assert against.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.models import utils as vllm_utils

from spyre_inference.custom_ops import multimodal_embeddings as me


def _cpu_reference(inputs_embeds, multimodal_embeddings, is_multimodal):
    """Upstream placement on CPU: the oracle the Spyre path reproduces."""
    out = inputs_embeds.clone()
    out[is_multimodal] = vllm_utils._flatten_embeddings(multimodal_embeddings)
    return out


@pytest.mark.multimodal
@pytest.mark.parametrize("hidden_size", [64, 128])
def test_merge_matches_cpu_reference_on_spyre(default_vllm_config, hidden_size):
    """A single image tensor is scattered into its placeholder rows on card."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(0)
    dtype = torch.float16
    is_multimodal = torch.tensor([False, True, True, False, True, False])
    num_mm = int(is_multimodal.sum())

    inputs_embeds = torch.randn(len(is_multimodal), hidden_size, dtype=dtype)
    mm_embeds = torch.randn(num_mm, hidden_size, dtype=dtype)

    expected = _cpu_reference(inputs_embeds, (mm_embeds,), is_multimodal)

    device = torch.device("spyre")
    merged = me._spyre_merge_multimodal_embeddings(
        inputs_embeds.to(device), (mm_embeds.to(device),), is_multimodal.to(device)
    )

    assert merged.device.type == "spyre"
    torch.testing.assert_close(merged.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.multimodal
def test_merge_handles_nested_embeddings_on_spyre(default_vllm_config):
    """A multi-image step arrives as a list of tensors; each is flattened and
    concatenated in order before scattering."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(1)
    dtype = torch.float16
    hidden_size = 64
    is_multimodal = torch.tensor([False, True, True, False, True, True, False, True])
    mm_embeds = [
        torch.randn(2, hidden_size, dtype=dtype),
        torch.randn(3, hidden_size, dtype=dtype),
    ]
    inputs_embeds = torch.randn(len(is_multimodal), hidden_size, dtype=dtype)

    expected = _cpu_reference(inputs_embeds, mm_embeds, is_multimodal)

    device = torch.device("spyre")
    mm_on_device = [t.to(device) for t in mm_embeds]
    merged = me._spyre_merge_multimodal_embeddings(
        inputs_embeds.to(device), mm_on_device, is_multimodal.to(device)
    )

    assert merged.device.type == "spyre"
    torch.testing.assert_close(merged.cpu().float(), expected.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.multimodal
def test_merge_leaves_text_rows_untouched_on_spyre(default_vllm_config):
    """The ``torch.where`` select must carry every non-placeholder row through from
    ``inputs_embeds``, not from the zero-filled scatter buffer: a dropped select
    predicate would zero the text rows, which the row-wise check below would catch
    well outside fp16 round-off."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(2)
    dtype = torch.float16
    hidden_size = 64
    is_multimodal = torch.tensor([True, False, False, True, False])
    num_mm = int(is_multimodal.sum())

    inputs_embeds = torch.randn(len(is_multimodal), hidden_size, dtype=dtype)
    mm_embeds = torch.randn(num_mm, hidden_size, dtype=dtype)

    merged = me._spyre_merge_multimodal_embeddings(
        inputs_embeds.to(torch.device("spyre")),
        (mm_embeds.to(torch.device("spyre")),),
        is_multimodal.to(torch.device("spyre")),
    ).cpu()

    text_rows = ~is_multimodal
    torch.testing.assert_close(
        merged[text_rows].float(), inputs_embeds[text_rows].float(), atol=1e-2, rtol=1e-3
    )


@pytest.mark.multimodal
def test_empty_embeddings_return_inputs_unchanged(default_vllm_config):
    """No multimodal input is the common text-only step: the function returns the
    original tensor object without any device round-trip."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    inputs_embeds = torch.randn(4, 64, dtype=torch.float16).to(torch.device("spyre"))
    is_multimodal = torch.zeros(4, dtype=torch.bool).to(torch.device("spyre"))

    merged = me._spyre_merge_multimodal_embeddings(inputs_embeds, [], is_multimodal)

    assert merged is inputs_embeds


@pytest.mark.multimodal
def test_token_count_mismatch_raises_value_error(default_vllm_config):
    """Fewer image tokens than placeholders is a wiring bug upstream turns into a
    ValueError; the Spyre path must reach the same diagnostic through its CPU
    scatter rather than a bare index error."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    dtype = torch.float16
    hidden_size = 64
    is_multimodal = torch.tensor([False, True, True, False, True, False])  # 3 placeholders
    mm_embeds = torch.randn(2, hidden_size, dtype=dtype)  # only 2 image tokens
    inputs_embeds = torch.randn(len(is_multimodal), hidden_size, dtype=dtype)

    device = torch.device("spyre")
    with pytest.raises(ValueError, match="2 multimodal tokens to 3 placeholders"):
        me._spyre_merge_multimodal_embeddings(
            inputs_embeds.to(device), (mm_embeds.to(device),), is_multimodal.to(device)
        )


@pytest.mark.multimodal
def test_non_spyre_input_delegates_to_upstream(default_vllm_config):
    """A CPU tensor must bypass the Spyre detour entirely and defer to the captured
    upstream implementation."""
    dtype = torch.float16
    hidden_size = 64
    is_multimodal = torch.tensor([False, True, True, False])
    mm_embeds = torch.randn(2, hidden_size, dtype=dtype)
    inputs_embeds = torch.randn(len(is_multimodal), hidden_size, dtype=dtype)

    merged = me._spyre_merge_multimodal_embeddings(
        inputs_embeds, (mm_embeds,), is_multimodal
    )

    assert merged.device.type == "cpu"
    expected = _cpu_reference(inputs_embeds, (mm_embeds,), is_multimodal)
    torch.testing.assert_close(merged, expected)


@pytest.mark.multimodal
def test_register_patches_upstream_merge(default_vllm_config):
    """``register`` swaps vLLM's merge for the Spyre one while keeping a handle on
    the original for the CPU-delegation path above."""
    assert vllm_utils._merge_multimodal_embeddings is me._spyre_merge_multimodal_embeddings
    assert me._orig_merge_multimodal_embeddings is not me._spyre_merge_multimodal_embeddings
