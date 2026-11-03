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

"""Device inverse-CDF sampling, checked on CPU.

The card runs the same function once cumsum, topk and the fp16 upcast do.
The token is not the host Gumbel token. It is the inverse CDF of the masked
softmax, so the same uniform draws the same id.
"""

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from spyre_inference.v1.sample import device_random as dr

VOCAB = 64
PADDED = 96


def _meta(rows: int, *, random: bool) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.full((rows,), 0.8) if random else torch.zeros(rows),
        all_greedy=not random,
        all_random=random,
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


def _reference(logits, pad_mask, temperature, uniform, top_k, top_p):
    """Sort-based top-k and top-p, then a token-order inverse CDF."""
    values = logits.masked_fill(pad_mask, float("-inf")).float()
    values = values / temperature.unsqueeze(1)
    if top_k is not None:
        ascending = torch.argsort(values, dim=-1, descending=False)
        sorted_values = values.gather(-1, ascending)
        kth_index = (values.shape[-1] - top_k).unsqueeze(1).long()
        kth = sorted_values.gather(1, kth_index)
        sorted_values = sorted_values.masked_fill(sorted_values < kth, float("-inf"))
        values = torch.empty_like(values).scatter_(-1, ascending, sorted_values)
    if top_p is not None:
        ascending = torch.argsort(values, dim=-1, descending=False)
        sorted_values = values.gather(-1, ascending)
        cdf = torch.cumsum(torch.softmax(sorted_values, dim=-1), dim=-1)
        drop_rank = cdf <= (1 - top_p.unsqueeze(1))
        drop_rank = drop_rank.clone()
        drop_rank[:, -1] = False
        rank = torch.argsort(ascending, dim=-1)
        values = values.masked_fill(drop_rank.gather(-1, rank), float("-inf"))
    cdf = torch.cumsum(torch.softmax(values, dim=-1), dim=-1)
    total = torch.amax(cdf, dim=-1, keepdim=True)
    return torch.sum(cdf < uniform.unsqueeze(1) * total, dim=-1)


def _draw(logits, uniform, temperature, top_k=None, top_p=None, org=VOCAB):
    mask = dr.padding_mask(logits.shape[-1], org, logits.device)
    k_max = 0 if top_k is None else int(top_k.max())
    got = dr.random_token_ids(logits, mask, temperature, uniform, top_k, k_max, top_p)
    expected = _reference(logits, mask, temperature, uniform, top_k, top_p)
    return got, expected


def test_random_batch_is_eligible() -> None:
    assert dr.batch_is_device_random(_meta(4, random=True), has_structured_output=False)
    assert not dr.batch_is_device_random(_meta(4, random=False), has_structured_output=False)
    assert not dr.batch_is_device_random(_meta(4, random=True), has_structured_output=True)


def test_penalties_logprobs_and_masks_stay_on_the_host() -> None:
    meta = _meta(2, random=True)
    meta.no_penalties = False
    assert not dr.batch_is_device_random(meta, False)
    meta = _meta(2, random=True)
    meta.max_num_logprobs = 0
    assert not dr.batch_is_device_random(meta, False)
    meta = _meta(2, random=True)
    meta.allowed_token_ids_mask = torch.ones(2, VOCAB, dtype=torch.bool)
    assert not dr.batch_is_device_random(meta, False)
    meta = _meta(2, random=True)
    meta.bad_words_token_ids = {0: [[1, 2]]}
    assert not dr.batch_is_device_random(meta, False)
    meta = _meta(2, random=True)
    meta.spec_token_ids = [[7]]
    assert not dr.batch_is_device_random(meta, False)
    meta = _meta(2, random=True)
    meta.logitsprocs.argmax_invariant.append(object())  # type: ignore[arg-type]
    assert not dr.batch_is_device_random(meta, False)


def test_inverse_cdf_matches_the_masked_softmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, VOCAB).half()
    temperature = torch.tensor([0.7, 1.0, 1.3, 0.5])
    uniform = torch.tensor([0.01, 0.25, 0.5, 0.99])
    got, expected = _draw(logits, uniform, temperature)
    assert torch.equal(got, expected)


def test_same_uniform_is_the_same_token_in_any_row() -> None:
    torch.manual_seed(1)
    row = torch.randn(1, VOCAB).half()
    uniform = torch.tensor([0.4])
    temperature = torch.ones(1)
    alone, _ = _draw(row, uniform, temperature)
    batch = row.expand(3, VOCAB).contiguous()
    got, expected = _draw(
        batch, torch.tensor([0.1, 0.4, 0.9]), torch.ones(3)
    )
    assert torch.equal(got, expected)
    assert got[1] == alone[0]


def test_top_k_and_top_p_match_the_sort_mask() -> None:
    torch.manual_seed(2)
    logits = torch.randn(3, VOCAB).half()
    logits[0] = -1
    logits[0, 4] = 5
    logits[0, 20] = 5
    temperature = torch.ones(3)
    uniform = torch.tensor([0.2, 0.6, 0.95])
    top_k = torch.tensor([1, 8, VOCAB])
    got, expected = _draw(logits, uniform, temperature, top_k=top_k)
    assert torch.equal(got, expected)
    # u = 0.2 falls in the first tied token. k = 1 keeps both equals.
    assert got[0] == 4

    top_p = torch.tensor([0.2, 0.9, 1.0])
    got, expected = _draw(logits, uniform, temperature, top_k=top_k, top_p=top_p)
    assert torch.equal(got, expected)
    got, expected = _draw(logits, uniform, temperature, top_p=top_p)
    assert torch.equal(got, expected)


def test_padding_columns_take_no_mass() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, PADDED).half()
    logits[0, VOCAB + 4] = 100
    temperature = torch.ones(2)
    uniform = torch.tensor([0.5, 0.999])
    got, expected = _draw(logits, uniform, temperature, org=VOCAB)
    assert torch.equal(got, expected)
    assert int(got.max()) < VOCAB


def test_seeded_row_does_not_depend_on_its_position() -> None:
    alone = dr.row_uniforms(1, {0: torch.Generator().manual_seed(7)})
    batch = dr.row_uniforms(
        4,
        {0: torch.Generator().manual_seed(1), 3: torch.Generator().manual_seed(7)},
    )
    assert torch.equal(batch[3], alone[0])
    again = dr.row_uniforms(1, {0: torch.Generator().manual_seed(7)})
    assert torch.equal(again, alone)


def test_sampler_output_is_one_int32_id_per_row() -> None:
    out = dr.sampler_output(torch.tensor([3, 11, 0]))
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (3, 1)
    assert out.logprobs_tensors is None
    assert torch.equal(
        out.sampled_token_ids.flatten(), torch.tensor([3, 11, 0], dtype=torch.int32)
    )


def test_sampling_context_restores() -> None:
    assert not dr.sampling_active()
    with dr.activated():
        assert dr.sampling_active()
    assert not dr.sampling_active()
