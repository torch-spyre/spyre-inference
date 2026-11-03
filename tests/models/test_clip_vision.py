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

"""``SpyreCLIPEncoderLayer`` against upstream's ``CLIPEncoderLayer`` with the same weights,
on CPU: packed images under the same-image mask each match their own unpacked run."""

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.models.clip import CLIPEncoderLayer

from spyre_inference.models._vision_encoder import (
    SEGMENT_ITEMS,
    one_hot_segments,
    segment_mask,
)
from spyre_inference.models.clip import SpyreCLIPAttention, SpyreCLIPEncoderLayer

# Rows per image: 50 is ViT-B/32@224's grid**2 + 1; packed with 30 they fill 80 of 128 rows.
SIZES = (50, 30)
ROWS = 128


def _config() -> CLIPVisionConfig:
    return CLIPVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,  # head_dim 64
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
    )


def _finish_weight_loading(module: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase

    for m in module.modules():
        if isinstance(m, LinearBase):
            m.quant_method.process_weights_after_loading(m)


def _layers() -> tuple[CLIPEncoderLayer, SpyreCLIPEncoderLayer]:
    """Upstream's layer and the Spyre one, with the same seeded weights."""
    config = _config()
    torch.manual_seed(0)
    upstream = CLIPEncoderLayer(config, prefix="layer", attn_cls=MMEncoderAttention)
    for p in upstream.parameters():
        p.data.normal_(std=0.05)
    for name, p in upstream.named_parameters():
        if "layer_norm" in name and name.endswith("weight"):
            p.data.add_(1.0)
    spyre = SpyreCLIPEncoderLayer(config, prefix="layer", attn_cls=MMEncoderAttention)
    # Copied before loading finishes: the Spyre linears transpose `weight` at that step.
    spyre.load_state_dict(upstream.state_dict())
    _finish_weight_loading(upstream)
    _finish_weight_loading(spyre)
    return upstream.float(), spyre.float()


def _packed(images: list[torch.Tensor], pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[1, ROWS, D]`` rows of ``images`` back to back, ``pad`` in the tail, and their mask."""
    tail = ROWS - sum(img.shape[0] for img in images)
    ids = torch.cat(
        [torch.full((img.shape[0],), i) for i, img in enumerate(images)]
        + [torch.full((tail,), SEGMENT_ITEMS)]
    )
    x = torch.cat([*images, pad[:tail]])
    return x[None], segment_mask(one_hot_segments(ids), torch.float32)


def _images(hidden: int) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(n, hidden) for n in SIZES]


def test_swaps_in_the_spyre_attention_with_upstream_weight_names(tp_group):
    upstream, spyre = _layers()
    assert type(spyre.self_attn) is SpyreCLIPAttention
    assert spyre.state_dict().keys() == upstream.state_dict().keys()


def test_one_image_matches_upstream(tp_group):
    upstream, spyre = _layers()
    torch.manual_seed(1)
    x = torch.randn(1, 64, _config().hidden_size)
    mask = segment_mask(one_hot_segments(torch.zeros(64, dtype=torch.int64)), torch.float32)
    assert not mask.any()

    with torch.inference_mode():
        torch.testing.assert_close(spyre(x, mask), upstream(x), atol=1e-4, rtol=1e-4)


def test_packed_images_each_match_their_own_run(tp_group):
    upstream, spyre = _layers()
    images = _images(_config().hidden_size)
    x, mask = _packed(images, torch.zeros(ROWS, _config().hidden_size))

    with torch.inference_mode():
        out = spyre(x, mask)[0]
        start = 0
        for img in images:
            want = upstream(img[None])[0]
            got = out[start : start + img.shape[0]]
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
            start += img.shape[0]


def test_padding_rows_never_change_real_rows(tp_group):
    _, spyre = _layers()
    hidden = _config().hidden_size
    images = _images(hidden)
    real = sum(SIZES)
    torch.manual_seed(2)

    with torch.inference_mode():
        zeros = spyre(*_packed(images, torch.zeros(ROWS, hidden)))[0, :real]
        noise = spyre(*_packed(images, 100 * torch.randn(ROWS, hidden)))[0, :real]
    torch.testing.assert_close(zeros, noise, atol=0, rtol=0)
