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

"""CPU tests for CLIP's batching workarounds in ``multimodal/clip.py``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.multimodal.clip import (
    has_text_tokens,
    merge_text_and_vision,
    patch_mixed_batches,
    select_class_rows,
)


def test_text_only_step_runs_text_tower():
    assert has_text_tokens(None, None)
    assert has_text_tokens([], torch.zeros(4, dtype=torch.bool))


def test_image_only_step_skips_text_tower():
    assert not has_text_tokens([torch.ones(2, 8)], torch.ones(2, dtype=torch.bool))


def test_mixed_step_runs_text_tower():
    # The case vLLM 0.28 got wrong: any image made the whole step image-only.
    mask = torch.tensor([True, False, False, False])
    assert has_text_tokens([torch.ones(1, 8)], mask)


def test_merge_keeps_vision_on_image_rows_and_text_elsewhere():
    text = torch.zeros(4, 2)
    vision = torch.ones(4, 2)
    out = merge_text_and_vision(text, vision, torch.tensor([False, True, False, False]))
    assert torch.equal(out, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def test_merge_pads_a_short_mask_with_text_rows():
    out = merge_text_and_vision(torch.zeros(4, 2), torch.ones(4, 2), torch.tensor([True]))
    assert torch.equal(out[0], torch.ones(2))
    assert torch.equal(out[1:], torch.zeros(3, 2))


def test_merge_without_image_rows_returns_text():
    text = torch.randn(3, 2)
    assert merge_text_and_vision(text, torch.ones(3, 2), torch.zeros(3, dtype=torch.bool)) is text


def test_merge_output_keeps_the_standard_device_layout():
    """A device-side [N, 1] mask view gave torch.where's output a layout that index_select
    faults the card on past row 63 (torch-spyre#5230).

    Checks the layout only, never gathers, so it cannot fault the card.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")
    text = torch.randn(512, 512, dtype=torch.float16).to("spyre")
    vision = torch.randn(512, 512, dtype=torch.float16).to("spyre")
    mask = torch.zeros(512, dtype=torch.bool)
    mask[[0, 100]] = True
    out = merge_text_and_vision(text, vision, mask)
    assert out.device_tensor_layout().stride_map == text.device_tensor_layout().stride_map


@pytest.mark.parametrize("batch", [1, 3])
def test_select_class_rows_matches_the_class_token_slice(batch):
    feats = torch.randn(batch, 50, 8)
    assert torch.equal(select_class_rows(feats), feats[:, :1, :])


def _align(mask: torch.Tensor, num_tokens: int, grid=None) -> SimpleNamespace:
    from vllm.model_executor.models.clip import CLIPEmbeddingModel

    patch_mixed_batches()
    model = SimpleNamespace(_spyre_mm_token_mask=mask, _is_text_input=True)
    CLIPEmbeddingModel.spyre_align_token_mask(model, num_tokens, grid)
    return model


def test_padding_rows_do_not_count_as_text():
    # Two image rows, then bucket padding: the step is image-only.
    model = _align(torch.tensor([True, True, False, False]), num_tokens=2)
    assert model._is_text_input is False
    assert torch.equal(model._spyre_mm_token_mask, torch.tensor([True, True]))


def test_a_real_text_row_keeps_the_text_tower():
    model = _align(torch.tensor([True, False, False, False]), num_tokens=3)
    assert model._is_text_input is True


def test_mask_is_laid_out_like_the_grid():
    # Packed: text (2 rows), then image (1 row). Grid extent 4: request i starts at 4*i,
    # so the image row moves from packed row 2 to grid row 4.
    model = _align(torch.tensor([False, False, True, False]), 3, grid=(4, 2, [2, 1]))
    expected = torch.tensor([False, False, False, False, True, False, False, False])
    assert torch.equal(model._spyre_mm_token_mask, expected)
