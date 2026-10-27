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

V1 keeps this on TorchSpyreModelRunner (_encoder_grid); MRV2 expects per-step
state on the model state instead. Built by encoder_only.py, which publishes it as
``SpyreEncoderOnlyModelState.encoder_plan`` for pooling to read. Warmup forces a
plan through ``SpyreEncoderOnlyModelState.forced_plan()``, not here.
"""

from dataclasses import dataclass

from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    EncoderGroupPlan,
    EncoderRectPlan,
)


# eq=False: the generated __eq__/__hash__ would compare the plan's tensors and lists.
@dataclass(frozen=True, eq=False)
class EncoderPlan:
    """The current step's encoder dispatch: a dense rectangle or ragged rows."""

    attn_plan: EncoderRectPlan | list[EncoderGroupPlan]

    @property
    def is_rectangular(self) -> bool:
        return isinstance(self.attn_plan, EncoderRectPlan)

    @property
    def grid(self) -> tuple[int, int, list[int]] | None:
        """``(extent, width, query_lens)`` on the rectangular path; None on the ragged one.

        Real rows are ``encoder_dense_row_indices(query_lens, extent)``, the layout V1
        unpads with too; the ragged body is already packed.
        """
        plan = self.attn_plan
        if not isinstance(plan, EncoderRectPlan):
            return None
        return plan.extent, plan.width, plan.query_lens
