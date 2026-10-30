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

"""Final normalization of an online softmax, with optional attention sinks."""

import torch


def normalize(
    tile_max: torch.Tensor,
    tile_sum: torch.Tensor,
    tile_output: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """``tile_output / tile_sum``, adding each head's sink logit to the denominator.

    A sink takes softmax mass and contributes no value (gpt-oss). It enters after the
    page walk, so the loop body and both walks are unchanged. The max is lifted to the
    sink first, as upstream seeds it, so no exponent exceeds 0 in fp16.

    Args:
        tile_max: running max of the scores, broadcastable against ``tile_sum``.
        tile_sum: softmax denominator relative to ``tile_max``.
        tile_output: probability-weighted values relative to ``tile_max``.
        sinks: per-head sink logits broadcastable against ``tile_max``, or None.
    """
    if sinks is None:
        return tile_output / tile_sum
    new_max = torch.maximum(tile_max, sinks)
    rescale = torch.exp(tile_max - new_max)
    return tile_output * rescale / (tile_sum * rescale + torch.exp(sinks - new_max))
