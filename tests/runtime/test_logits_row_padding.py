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

"""``compute_logits`` pads its sampled rows onto the warmed row buckets."""

import torch
import torch.nn as nn

from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper
from spyre_inference.v1.worker.spyre_shape_bucketer import logits_row_buckets

HIDDEN, VOCAB = 8, 5


class _Model(nn.Module):
    """Records the row count each projection sees."""

    def __init__(self):
        super().__init__()
        self.weight = torch.randn(HIDDEN, VOCAB, dtype=torch.float16)
        self.seen_rows: list[int] = []

    def compute_logits(self, hidden_states):
        self.seen_rows.append(hidden_states.shape[0])
        return hidden_states @ self.weight


def _wrapper(buckets):
    model = _Model()
    return model, _SpyreModelWrapper(
        model,
        torch.device("cpu"),
        logits_row_buckets=buckets,
    )


def test_pads_up_to_the_next_warmed_width():
    model, wrapper = _wrapper([1, 2, 4, 8])
    logits = wrapper.compute_logits(torch.randn(5, HIDDEN, dtype=torch.float16))

    assert model.seen_rows == [8]
    assert logits.shape == (5, VOCAB)


def test_pad_rows_do_not_change_the_real_logits():
    model, wrapper = _wrapper([1, 2, 4, 8])
    hidden = torch.randn(3, HIDDEN, dtype=torch.float16)

    padded = wrapper.compute_logits(hidden)

    torch.testing.assert_close(padded, hidden @ model.weight)


def test_an_exact_width_is_not_padded():
    model, wrapper = _wrapper([1, 2, 4, 8])
    wrapper.compute_logits(torch.randn(4, HIDDEN, dtype=torch.float16))

    assert model.seen_rows == [4]


def test_a_width_above_the_buckets_is_left_alone():
    """Nothing warmed covers it, so padding would only add a second unwarmed shape."""
    model, wrapper = _wrapper([1, 2, 4])
    wrapper.compute_logits(torch.randn(7, HIDDEN, dtype=torch.float16))

    assert model.seen_rows == [7]


def test_every_reachable_width_lands_on_a_warmed_shape():
    """The projection must never see a width warmup did not compile."""
    buckets = logits_row_buckets([1, 2, 4, 8, 512], max_num_reqs=8)
    model, wrapper = _wrapper(buckets)

    for rows in range(1, 9):
        wrapper.compute_logits(torch.randn(rows, HIDDEN, dtype=torch.float16))

    assert set(model.seen_rows) <= set(buckets)


def test_no_buckets_keeps_the_old_behaviour():
    model, wrapper = _wrapper([])
    wrapper.compute_logits(torch.randn(5, HIDDEN, dtype=torch.float16))

    assert model.seen_rows == [5]
