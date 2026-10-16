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

"""`walk_tiles` presents the same body contract as `for_each_tile` does."""

import pytest
import torch

from spyre_inference.v1.attention.ops import tile_loop
from spyre_inference.v1.attention.ops.tile_loop import walk_tiles

EXTENT, WIDTH = 8, 3
INIT = (torch.zeros(1),)


@pytest.fixture(autouse=True)
def _use_python_walk(monkeypatch):
    """These contract tests inspect calls, so use the observable fallback walk."""
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)


def _recording_body(log: list):
    """Body that records what it was handed and accumulates the tiled operand."""

    def body(carry, tiles):
        log.append((carry, tiles))
        total = tiles[0].sum().reshape(1)
        return (total if carry is None else carry[0] + total,), None

    return body


def test_first_trip_receives_none_not_init():
    """The loop path builds the carry from the first tile; `init` is tiled-path only.

    A materialized init constant and a computed tile stickify differently, so the
    body must be able to start from the tile alone.
    """
    log: list = []
    walk_tiles(
        _recording_body(log),
        (torch.arange(EXTENT, dtype=torch.float32),),
        dims=(0,),
        init=INIT,
    )

    assert log[0][0] is None
    assert all(entry[0] is not None for entry in log[1:])


def test_tiles_keep_the_tiled_axis_at_tile_size():
    """narrow, not select: a tile keeps its leading axis so one body serves both paths."""
    log: list = []
    operand = torch.arange(EXTENT * WIDTH, dtype=torch.float32).reshape(EXTENT, WIDTH)
    walk_tiles(_recording_body(log), (operand,), dims=(0,), tile_size=2, init=INIT)

    assert [entry[1][0].shape for entry in log] == [(2, WIDTH)] * (EXTENT // 2)


def test_untiled_operands_pass_through_whole():
    log: list = []
    tiled = torch.arange(EXTENT, dtype=torch.float32)
    invariant = torch.ones(WIDTH)
    walk_tiles(_recording_body(log), (tiled, invariant), dims=(0, None), init=INIT)

    assert all(entry[1][1] is invariant for entry in log)


def test_trip_count_is_derived_from_the_operands():
    """Both paths iterate exactly the extent `for_each_tile` would have tiled."""
    log: list = []
    (total,), _ = walk_tiles(
        _recording_body(log),
        (torch.arange(EXTENT, dtype=torch.float32),),
        dims=(0,),
        tile_size=2,
        init=INIT,
    )

    assert len(log) == EXTENT // 2
    assert total.item() == pytest.approx(float(sum(range(EXTENT))))


def test_dims_must_cover_every_operand():
    with pytest.raises(ValueError, match="dims has 1 entries for 2 operands"):
        walk_tiles(
            _recording_body([]),
            (torch.zeros(EXTENT), torch.zeros(EXTENT)),
            dims=(0,),
            init=INIT,
        )


def test_tiled_operands_must_agree_on_their_extent():
    with pytest.raises(ValueError, match=r"disagree on their extent: \[4, 8\]"):
        walk_tiles(
            _recording_body([]),
            (torch.zeros(EXTENT), torch.zeros(EXTENT // 2)),
            dims=(0, 0),
            init=INIT,
        )


def test_extent_must_be_a_whole_number_of_tiles():
    with pytest.raises(ValueError, match="extent 8 is not a whole number of 3-wide tiles"):
        walk_tiles(
            _recording_body([]),
            (torch.zeros(EXTENT),),
            dims=(0,),
            tile_size=WIDTH,
            init=INIT,
        )
