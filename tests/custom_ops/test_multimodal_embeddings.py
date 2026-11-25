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

"""The Spyre-device branch of the multimodal-embedding merge.

``tests/models/test_vision_encoder_graph.py`` covers the merge through the model
wrapper on CPU, where ``_spyre_merge_multimodal_embeddings`` delegates to
upstream. The device branch -- scatter on the host, ``torch.where`` the rows back
in, and the count-mismatch errors -- is the reason the monkeypatch exists
(upstream's boolean-mask ``index_put`` has no Spyre kernel at all), so it is
pinned here on card.
"""

from __future__ import annotations

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.multimodal_embeddings import (
    _spyre_merge_multimodal_embeddings,
)
from spyre_inference.custom_ops.utils import convert


@pytest.fixture()
def spyre_device():
    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


def _mask(num_tokens: int, rows: list[int]) -> torch.Tensor:
    is_multimodal = torch.zeros(num_tokens, dtype=torch.bool)
    is_multimodal[rows] = True
    return is_multimodal


def test_register_all_installs_the_merge_patch():
    from vllm.model_executor.models import utils as vllm_utils

    from spyre_inference.custom_ops import register_all

    register_all()
    assert vllm_utils._merge_multimodal_embeddings is _spyre_merge_multimodal_embeddings


def test_merge_on_spyre_places_each_images_rows_where_upstream_does(spyre_device):
    """Two images, placeholder rows out of order in the prompt: the device merge
    must land them where upstream's CPU merge puts them."""
    num_tokens, hidden = 12, 64
    text = torch.randn(num_tokens, hidden, dtype=torch.float16)
    images = [torch.randn(3, hidden).half(), torch.randn(2, hidden).half()]
    is_multimodal = _mask(num_tokens, [1, 2, 3, 7, 8])

    merged = _spyre_merge_multimodal_embeddings(
        convert(text, device=spyre_device), images, is_multimodal
    )

    assert merged.device.type == "spyre"
    want = text.clone()
    want[is_multimodal] = torch.cat(images)
    torch.testing.assert_close(convert(merged, device="cpu"), want, atol=1e-2, rtol=1e-2)


def test_merge_without_image_tokens_returns_the_text_tensor(spyre_device):
    text = convert(torch.randn(4, 64, dtype=torch.float16), device=spyre_device)
    out = _spyre_merge_multimodal_embeddings(text, [], torch.zeros(4, dtype=torch.bool))
    assert out is text


def test_merge_on_spyre_reports_a_count_mismatch_like_upstream(spyre_device):
    text = convert(torch.randn(6, 64, dtype=torch.float16), device=spyre_device)
    is_multimodal = _mask(6, [1, 2, 3])
    with pytest.raises(ValueError, match="2 multimodal tokens to 3 placeholders"):
        _spyre_merge_multimodal_embeddings(text, [torch.ones(2, 64).half()], is_multimodal)
