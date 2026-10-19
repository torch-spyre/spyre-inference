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

"""MRV2: the per-step encoder plan, moved off the runner onto the model state.

V1 keeps this on TorchSpyreModelRunner (_encoder_grid, _forced_encoder_rect,
...); MRV2 expects per-step state on the model state instead. Set by
encoder_only.py, read by pooling and warmup through this module only.
"""

from dataclasses import dataclass, field


@dataclass
class EncoderPlan:
    """The current step's encoder dispatch: a dense rectangle or ragged rows."""

    is_rectangular: bool
    # (extent, width, query_lens) on the rectangular path; empty on the ragged one.
    grid: tuple[int, int, list[int]] | None
    # Row ranges actually holding real (non-padding) tokens.
    rows: list[tuple[int, int]] = field(default_factory=list)
    # Warmup-only forcing; see warmup.py.
    forced_rect: tuple[int, int] | None = None
    force_ragged: bool = False


def record_dispatch(plan: EncoderPlan) -> None:
    """Update the dispatch counters (rect/ragged steps, real tokens, rows, seqs)."""
    raise NotImplementedError
