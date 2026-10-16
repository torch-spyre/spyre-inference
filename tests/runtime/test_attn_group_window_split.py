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

"""Device-free tests for splitting attention groups by per-layer sliding window.

A hybrid decoder whose layers share a head shape reaches the builders as one group
carrying one window (``FullAttentionSpec.merge``), which would clamp its full-attention
layers. Nothing in the e2e suite exercises gemma-3 past its 512-token window, so the
clamp is invisible there and this is where a regression would be caught.
"""

from __future__ import annotations

import types

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

# gemma-3-1b-it: 26 layers, window 512, every 6th layer full attention.
_NUM_LAYERS = 26
_WINDOW = 512
_FULL_IDX = (5, 11, 17, 23)
_SHAPE = dict(block_size=64, num_kv_heads=1, head_size=256, dtype=torch.float16)


def _names(indices) -> list[str]:
    return [f"model.layers.{i}.self_attn.attn" for i in indices]


def _index(name: str) -> int:
    return int(name.split(".")[2])


def _group(names, spec) -> AttentionGroup:
    return AttentionGroup(object, list(names), spec, 0)


@pytest.fixture
def split(monkeypatch):
    """Run the split over ``groups``, with each layer's window stubbed.

    The windows come off the ``Attention`` modules, reached through
    ``get_layers_from_vllm_config``; stubbing it keeps this device- and model-free.
    """

    def _split(windows_by_layer: dict[str, int | None], groups):
        import vllm.config

        layers = {
            name: types.SimpleNamespace(sliding_window=window)
            for name, window in windows_by_layer.items()
        }
        monkeypatch.setattr(
            vllm.config, "get_layers_from_vllm_config", lambda *args, **kwargs: layers
        )
        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
        runner.vllm_config = types.SimpleNamespace()
        runner.attn_groups = [list(groups)]
        runner._split_attn_groups_by_layer_window()
        return runner.attn_groups[0]

    return _split


def test_splits_one_group_per_distinct_window(split) -> None:
    """Each layer ends up under the window its own module declares, exactly once."""
    names = _names(range(_NUM_LAYERS))
    windows = {n: (None if _index(n) in _FULL_IDX else _WINDOW) for n in names}
    merged = FullAttentionSpec(sliding_window=_WINDOW, **_SHAPE)

    groups = split(windows, [_group(names, merged)])

    assert len(groups) == 2
    by_window = {g.kv_cache_spec.sliding_window: g for g in groups}
    assert sorted(_index(n) for n in by_window[None].layer_names) == list(_FULL_IDX)
    assert sorted(n for g in groups for n in g.layer_names) == sorted(names)
    for group in groups:
        for name in group.layer_names:
            assert group.kv_cache_spec.sliding_window == windows[name]
        # Only the window may change: allocation and grouping must survive intact.
        assert (group.kv_cache_group_id, group.backend) == (0, object)
        for field, value in _SHAPE.items():
            assert getattr(group.kv_cache_spec, field) == value


@pytest.mark.parametrize(
    "windows,groups",
    [
        pytest.param(
            {n: _WINDOW for n in _names(range(4))},
            [_group(_names(range(4)), FullAttentionSpec(sliding_window=_WINDOW, **_SHAPE))],
            id="one_window_shared_by_every_layer",
        ),
        pytest.param(
            {n: _WINDOW for n in _names(range(4))},
            [_group(_names(range(4)), SlidingWindowSpec(sliding_window=_WINDOW, **_SHAPE))],
            id="pure_swa_keeps_its_sliding_window_spec",
        ),
        pytest.param(
            {**{n: 1024 for n in _names(range(5))}, **{n: None for n in _names(range(5, 8))}},
            [
                _group(
                    _names(range(5)),
                    FullAttentionSpec(
                        block_size=64,
                        num_kv_heads=8,
                        head_size=256,
                        dtype=torch.float16,
                        sliding_window=1024,
                    ),
                ),
                _group(
                    _names(range(5, 8)),
                    FullAttentionSpec(
                        block_size=64, num_kv_heads=2, head_size=512, dtype=torch.float16
                    ),
                ),
            ],
            id="gemma4_style_differing_head_shapes",
        ),
    ],
)
def test_single_window_groups_are_left_alone(split, windows, groups) -> None:
    """A group that already carries one window is returned untouched.

    Covers gemma-4, whose differing head shapes stop the merge, so vLLM hands the
    builders per-layer specs already: re-splitting those would be wrong.
    """
    assert split(windows, groups) == groups
