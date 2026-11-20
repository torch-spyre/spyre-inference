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

"""spec_decode: the Gemma-4 MTP speculator's drafter boundary and its dispatch."""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.worker.spyre import spec_decode


class _Drafter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, torch.Tensor | None] = {}

    def forward(self, *, input_ids, positions, hidden_states, inputs_embeds):
        self.seen = dict(input_ids=input_ids, positions=positions, hidden_states=hidden_states)
        return hidden_states * 2, hidden_states + 1


def _wrap(drafter: torch.nn.Module, buckets: list[int]) -> spec_decode.SpyreDraftModelWrapper:
    bucketer = SimpleNamespace(find_bucket=lambda n: next((b for b in buckets if b >= n), None))
    return spec_decode.SpyreDraftModelWrapper(
        drafter,
        torch.device("cpu"),
        shape_bucketer=bucketer,  # ty: ignore[invalid-argument-type]
        model_dtype=torch.float16,
    )


def test_a_draft_forward_pads_onto_the_body_bucket_and_trims_back() -> None:
    drafter = _Drafter()
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    draft, backbone = _wrap(drafter, [1, 4, 8])(
        input_ids=torch.tensor([5, 6, 7], dtype=torch.int32),
        positions=torch.tensor([10, 11, 12]),
        hidden_states=hidden,
    )

    assert drafter.seen["input_ids"].tolist() == [5, 6, 7, 0]
    assert drafter.seen["input_ids"].dtype == torch.int64
    assert drafter.seen["hidden_states"].shape == (4, 2)
    assert drafter.seen["hidden_states"].dtype == torch.float16
    assert torch.equal(draft, (hidden * 2).half())
    assert torch.equal(backbone, (hidden + 1).half())


def test_a_bucket_sized_draft_forward_is_not_padded() -> None:
    drafter = _Drafter()
    _wrap(drafter, [4])(
        input_ids=torch.zeros(4, dtype=torch.int64),
        positions=torch.zeros(4, dtype=torch.int64),
        hidden_states=torch.zeros(4, 2),
    )
    assert drafter.seen["positions"].shape == (4,)


@pytest.mark.parametrize(
    ("gemma4_mtp", "cls"), [(True, "SpyreGemma4Speculator"), (False, "SpyreDFlashSpeculator")]
)
def test_init_speculator_dispatches_on_the_method(monkeypatch, gemma4_mtp, cls) -> None:
    for name in ("SpyreGemma4Speculator", "SpyreDFlashSpeculator"):
        monkeypatch.setattr(spec_decode, name, lambda *args, name=name: name)
    config = SimpleNamespace(speculative_config=SimpleNamespace(use_gemma4_mtp=lambda: gemma4_mtp))
    device = torch.device("cpu")
    assert spec_decode.init_speculator(config, device, device) == cls  # ty: ignore[invalid-argument-type]
