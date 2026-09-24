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

"""Drive a tiled loop body either with `for_each_tile` or with a Python loop."""

from collections.abc import Callable, Sequence
from typing import Any

import torch

# Read at module scope, unlike the kernels here: a process-wide path choice, not
# per-call configuration. Threading it through the kernel signatures would put the
# switch into the kernel ABI, where `dynamic=False` would specialize on it anyway.
from spyre_inference import envs

USE_FOR_EACH_TILE = envs.SPYRE_ATTN_FOR_EACH_TILE


def walk_tiles(
    body: Callable[[tuple | None, tuple], tuple[tuple, Any]],
    operands: Sequence[torch.Tensor],
    *,
    dims: Sequence[int | None],
    tile_size: int = 1,
    init: tuple,
) -> tuple[tuple, Any]:
    """Run `body` once per tile of `operands`, threading the carry through.

    Same contract as `torch_spyre`'s `for_each_tile`, which is what runs with
    SPYRE_ATTN_FOR_EACH_TILE set. Unset, the identical body runs under a Python
    `for` loop, as the kernels did before `for_each_tile` existed, so a regression
    in the tiled op or in the pin is isolated by unsetting one variable.

    The loop path passes `carry=None` on the first trip rather than `init`: a
    materialized init and a computed tile can stickify differently for the same
    logical shape, leaving the pointwise that combines them no legal device layout
    (`no mechanism to resolve stick incompatibility`). So `init` describes the tiled
    path only, and `body` builds its carry from the first tile.

    Args:
        body: ``body(carry, tiles) -> (carry, per_tile_output)``, where `tiles` holds
            the tile for each tiled operand and the whole tensor for the rest, and
            ``carry=None`` means "this is the first tile".
        operands: the tensors to walk.
        dims: per operand, the axis to tile, or None to pass it through whole.
        tile_size: elements of the tiled axis per trip.
        init: the initial carry, used by the tiled path only.

    Returns:
        ``(carry, per_tile_outputs)``, the tiled op's own return shape. The loop path
        stacks nothing and returns the last `per_tile_output`; the paged-attention
        kernels return None there and read only the carry.

    Raises:
        ValueError: if the tiled operands disagree on their extent, or the extent is
            not a whole number of tiles.
    """
    if len(dims) != len(operands):
        raise ValueError(f"dims has {len(dims)} entries for {len(operands)} operands")

    if USE_FOR_EACH_TILE:
        from torch_spyre._inductor.wsr import for_each_tile

        return for_each_tile(
            body,
            tuple(operands),
            dims=tuple(dims),
            tile_size=tile_size,
            init=init,
        )

    # Derived, not passed in, so both paths iterate the extent `for_each_tile` would.
    extents = {
        operand.shape[dim] for operand, dim in zip(operands, dims, strict=True) if dim is not None
    }
    if len(extents) != 1:
        raise ValueError(f"tiled operands disagree on their extent: {sorted(extents)}")
    extent = extents.pop()
    if extent % tile_size:
        raise ValueError(f"extent {extent} is not a whole number of {tile_size}-wide tiles")

    # None, not `init`: the body builds the carry from the first tile. See above.
    carry: tuple | None = None
    out = None
    for t in range(extent // tile_size):
        # narrow, not select: the tile keeps its leading axis, so `body` indexes it
        # the same way on both paths.
        tiles = tuple(
            operand if dim is None else operand.narrow(dim, t * tile_size, tile_size)
            for operand, dim in zip(operands, dims, strict=True)
        )
        carry, out = body(carry, tiles)
    assert carry is not None
    return carry, out
