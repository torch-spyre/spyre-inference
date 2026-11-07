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

"""Device greedy argmax, checked on CPU against ``torch.argmax``.

The card path is the same function. It matches once torch-spyre #5845 runs
``argmax`` on the device with PyTorch tie and NaN rules.
"""

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from spyre_inference.v1.sample import device_greedy as dg

VOCAB = 128
PADDED = 192


def _meta(rows: int, *, greedy: bool) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.zeros(rows) if greedy else torch.full((rows,), 0.8),
        all_greedy=greedy,
        all_random=not greedy,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_greedy_batch_is_eligible() -> None:
    assert dg.batch_is_device_greedy(_meta(4, greedy=True), has_structured_output=False)
    assert not dg.batch_is_device_greedy(_meta(4, greedy=False), has_structured_output=False)
    assert not dg.batch_is_device_greedy(_meta(4, greedy=True), has_structured_output=True)


def test_penalties_logprobs_and_masks_stay_on_the_host() -> None:
    meta = _meta(2, greedy=True)
    meta.no_penalties = False
    assert not dg.batch_is_device_greedy(meta, False)
    meta = _meta(2, greedy=True)
    meta.max_num_logprobs = 0
    assert not dg.batch_is_device_greedy(meta, False)
    meta = _meta(2, greedy=True)
    meta.allowed_token_ids_mask = torch.ones(2, VOCAB, dtype=torch.bool)
    assert not dg.batch_is_device_greedy(meta, False)
    meta = _meta(2, greedy=True)
    meta.bad_words_token_ids = {0: [[1, 2]]}
    assert not dg.batch_is_device_greedy(meta, False)
    meta = _meta(2, greedy=True)
    meta.spec_token_ids = [[7]]
    assert not dg.batch_is_device_greedy(meta, False)


def test_argmax_matches_cpu_including_ties_and_nan() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, VOCAB).half()
    mask = dg.padding_mask(VOCAB, VOCAB, logits.device)
    assert torch.equal(dg.greedy_token_ids(logits, mask), logits.argmax(dim=-1))

    logits = logits.clone()
    logits[0, 3] = logits[0].max() + 1
    logits[0, 90] = logits[0, 3]
    assert dg.greedy_token_ids(logits, mask)[0] == 3

    logits = torch.randn(2, VOCAB).half()
    logits[1, 11] = float("nan")
    assert dg.greedy_token_ids(logits, mask)[1] == 11


def test_padding_columns_cannot_win() -> None:
    logits = torch.randn(2, PADDED).half()
    logits[0, VOCAB + 4] = 100
    mask = dg.padding_mask(PADDED, VOCAB, logits.device)
    got = dg.greedy_token_ids(logits, mask)
    assert got[0] == logits[0, :VOCAB].argmax()
    assert int(got.max()) < VOCAB


def test_sampler_output_is_one_int32_id_per_row() -> None:
    out = dg.sampler_output(torch.tensor([3, 11, 0]))
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (3, 1)
    assert out.logprobs_tensors is None
    assert torch.equal(out.sampled_token_ids.flatten(), torch.tensor([3, 11, 0], dtype=torch.int32))


def test_sampling_context_restores() -> None:
    assert not dg.sampling_active()
    with dg.activated():
        assert dg.sampling_active()
    assert not dg.sampling_active()
