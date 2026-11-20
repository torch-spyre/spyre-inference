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

"""Parity cases for the spec-decode host kernels: the rejection sampler XPress
verifies with and DFlash's input prep. Registered into parity_cases.CASES."""

import types

import numpy as np
import torch

from .parity_cases import _logits, _m, case

_RS = "gpu.spec_decode.rejection_sampler_utils"


def _spec_batch(g, num_reqs: int, steps: int, vocab: int, greedy_mix: bool, invalid: bool):
    """Per request 1 + n_drafts logits rows (n_drafts in [0, steps]), request-major."""
    max_reqs = num_reqs + 3
    n_drafts = torch.randint(0, steps + 1, (num_reqs,), generator=g)
    n_drafts[0] = steps
    rows = (n_drafts + 1).to(torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), rows.cumsum(0).to(torch.int32)])
    num_logits = int(cu[-1])
    idx = torch.randperm(max_reqs, generator=g)[:num_reqs].to(torch.int32)
    expanded = torch.repeat_interleave(idx, rows)
    local_pos = torch.cat([torch.arange(int(r)) for r in rows]).to(torch.int32)
    target = _logits(g, num_logits, vocab)
    # Make the draft agree with the target argmax often, so acceptance runs vary.
    argmax = target.argmax(-1)
    draft_sampled = torch.randint(0, vocab, (num_logits,), generator=g)
    agree = torch.rand(num_logits, generator=g) < 0.6
    shifted = torch.cat([argmax[:1], argmax[:-1]])
    draft_sampled = torch.where(agree, shifted, draft_sampled)
    if invalid:
        draft_sampled[int(cu[0]) + 2] = -1
    temps = torch.rand(max_reqs, generator=g) * 1.5 + 0.2
    if greedy_mix:
        temps[idx[::2].long()] = 0.0
    return dict(
        num_reqs=num_reqs,
        max_reqs=max_reqs,
        steps=steps,
        target=target,
        draft_sampled=draft_sampled,
        cu=cu,
        pos=torch.randint(0, 3000, (num_logits,), generator=g),
        idx=idx,
        expanded=expanded,
        local_pos=local_pos,
        temps=temps,
        seeds=torch.randint(-(2**62), 2**62, (max_reqs,), generator=g),
    )


def _rejection(
    g, *, draft_logits: bool, greedy_mix: bool, fp64=False, invalid=False, all_greedy=False
):
    rs = _m("spec_decode.rejection_sampler_utils")
    b = _spec_batch(g, 4, 5, 9000, greedy_mix, invalid)
    if all_greedy:
        b["temps"].zero_()
    dl = None
    if draft_logits:
        dl = torch.randn(b["max_reqs"], b["steps"], 9000, generator=g) * 3

    def run(s):
        sampled, num_sampled = rs.rejection_sample(
            b["target"],
            dl,
            b["draft_sampled"],
            b["cu"],
            b["pos"],
            b["idx"],
            b["expanded"],
            b["local_pos"],
            b["temps"],
            b["seeds"],
            b["steps"],
            use_fp64=fp64,
        )
        # Slots past num_sampled are torch.empty that neither side writes.
        keep = torch.arange(sampled.shape[1]) < num_sampled[:, None]
        return sampled.where(keep, 0), num_sampled

    return run, {}


@case(
    f"{_RS}._compute_local_logits_stats_kernel",
    f"{_RS}._rejection_kernel",
    f"{_RS}._resample_kernel",
    f"{_RS}._insert_resampled_kernel",
)
def rejection_sample_greedy(g):
    return _rejection(g, draft_logits=False, greedy_mix=False, all_greedy=True, invalid=True)


@case(
    f"{_RS}._compute_local_logits_stats_kernel",
    f"{_RS}._rejection_kernel",
    f"{_RS}._resample_kernel",
    f"{_RS}._insert_resampled_kernel",
)
def rejection_sample_mixed_one_hot(g):
    return _rejection(g, draft_logits=False, greedy_mix=True, invalid=True)


@case(
    f"{_RS}._compute_local_logits_stats_kernel",
    f"{_RS}._rejection_kernel",
    f"{_RS}._resample_kernel",
    f"{_RS}._insert_resampled_kernel",
)
def rejection_sample_draft_logits(g):
    return _rejection(g, draft_logits=True, greedy_mix=True)


@case(
    f"{_RS}._compute_local_logits_stats_kernel",
    f"{_RS}._rejection_kernel",
    f"{_RS}._resample_kernel",
    f"{_RS}._insert_resampled_kernel",
)
def rejection_sample_fp64(g):
    return _rejection(g, draft_logits=True, greedy_mix=False, fp64=True)


@case("gpu.spec_decode.rejection_sampler._gather_draft_sampled_kernel")
def gather_draft_sampled(g):
    rsm = _m("spec_decode.rejection_sampler")
    num_tokens, num_logits, max_reqs = 40, 12, 6
    expanded = torch.tensor([0, 0, 0, 3, 3, 1, 1, 1, 1, 5, 2, 2], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 0, 1], dtype=torch.int32)
    return lambda s: rsm.gather_draft_sampled(
        torch.randint(0, 900, (num_tokens,), generator=g, dtype=torch.int32),
        torch.randint(0, 60, (num_tokens,), generator=g),
        torch.randperm(num_tokens, generator=g)[:num_logits],
        expanded,
        local_pos,
        torch.randint(0, 50, (max_reqs,), generator=g, dtype=torch.int32),
    ), {}


@case("gpu.spec_decode.rejection_sampler._flatten_sampled_kernel")
def flatten_sampled(g):
    rsm = _m("spec_decode.rejection_sampler")
    num_reqs, steps = 4, 3
    cu = torch.tensor([0, 4, 6, 10, 11], dtype=torch.int32)
    num_sampled = torch.tensor([3, 2, 4, 1], dtype=torch.int32)
    sampled = torch.randint(0, 300, (num_reqs, steps + 1), generator=g)
    logits = _logits(g, 11, 300)
    sampler = types.SimpleNamespace(logprobs_mode="raw_logprobs")
    self = types.SimpleNamespace(sampler=sampler, enable_adaptive_verification=False)
    return lambda s: rsm.RejectionSampler._get_logprobs_tensors(
        self, sampled, num_sampled, logits, cu, cu.numpy(), 2
    ), {}


def _dflash_batch(g, num_reqs: int, chunked: bool):
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    max_reqs, max_tokens, steps, block_size = 6, 96, 4, 16
    qlens = torch.randint(1, 9, (num_reqs,), generator=g)
    qsl = torch.cat([torch.zeros(1, dtype=torch.int64), qlens.cumsum(0)]).to(torch.int32)
    num_tokens = int(qsl[-1])
    positions = torch.cat(
        [
            torch.arange(int(s), int(s) + int(n))
            for s, n in zip(torch.randint(0, 80, (num_reqs,), generator=g), qlens)
        ]
    )
    buffers = InputBuffers(
        max_num_reqs=max_reqs, max_num_tokens=max_tokens, device=torch.device("cpu")
    )
    buffers.positions[:num_tokens] = positions
    buffers.query_start_loc[: num_reqs + 1] = qsl
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=qlens.numpy().astype(np.int32),
        positions=buffers.positions[:num_tokens],
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        idx_mapping=torch.randperm(max_reqs, generator=g)[:num_reqs].to(torch.int64),
    )
    draft_buffers = InputBuffers(
        max_num_reqs=max_reqs, max_num_tokens=max_tokens, device=torch.device("cpu")
    )
    num_sampled = torch.ones(num_reqs, dtype=torch.int32)
    if chunked:
        num_sampled[0] = 0
    table = torch.randint(0, 40, (max_reqs, 8), generator=g, dtype=torch.int32)
    table[0, 0] = 0  # a null block
    state = dict(
        input_ids=draft_buffers.input_ids,
        positions=draft_buffers.positions,
        is_padding=draft_buffers.is_padding,
        qsl=draft_buffers.query_start_loc,
        seq_lens=draft_buffers.seq_lens,
        query_slots=torch.full((max_tokens,), -9, dtype=torch.int64),
        context_positions=torch.full((max_tokens,), -9, dtype=torch.int64),
        context_slots=torch.full((max_tokens,), -9, dtype=torch.int64),
        sample_indices=torch.full((max_reqs * steps,), -9, dtype=torch.int64),
        sample_pos=torch.full((max_reqs * steps,), -9, dtype=torch.int64),
        sample_idx=torch.full((max_reqs * steps,), -9, dtype=torch.int64),
        temperature=torch.zeros(max_reqs),
        seeds=torch.zeros(max_reqs, dtype=torch.int64),
    )
    args = dict(
        num_sampled=num_sampled,
        num_rejected=torch.minimum(torch.randint(0, 3, (num_reqs,), generator=g), qlens - 1).to(
            torch.int32
        ),
        last_sampled=torch.randint(0, 900, (max_reqs,), generator=g),
        next_prefill_tokens=torch.randint(0, 900, (max_reqs,), generator=g),
        input_temperature=torch.rand(max_reqs, generator=g),
        input_seeds=torch.randint(0, 2**40, (max_reqs,), generator=g),
        block_table=table,
        block_size=block_size,
        steps=steps,
        max_reqs=max_reqs,
        max_tokens=max_tokens,
    )
    return batch, draft_buffers, state, args


def _prepare_dflash(g, num_reqs: int, chunked: bool):
    ds = _m("spec_decode.dflash.speculator")
    batch, buffers, state, a = _dflash_batch(g, num_reqs, chunked)

    def run(s):
        ds.prepare_dflash_inputs(
            buffers,
            s["query_slots"],
            s["context_positions"],
            s["context_slots"],
            s["sample_indices"],
            s["sample_pos"],
            s["sample_idx"],
            s["temperature"],
            s["seeds"],
            batch,
            a["num_sampled"],
            a["num_rejected"],
            a["last_sampled"],
            a["next_prefill_tokens"],
            a["input_temperature"],
            a["input_seeds"],
            a["block_table"],
            a["block_size"],
            0,
            1,
            1,
            77,
            a["steps"] + 1,
            a["steps"],
            a["max_reqs"],
            a["max_tokens"],
            120,
        )

    return run, state


@case("gpu.spec_decode.dflash.speculator._prepare_dflash_inputs_kernel")
def prepare_dflash_inputs(g):
    return _prepare_dflash(g, 4, chunked=False)


@case("gpu.spec_decode.dflash.speculator._prepare_dflash_inputs_kernel")
def prepare_dflash_inputs_chunked_prefill(g):
    return _prepare_dflash(g, 3, chunked=True)


_TDAR = "gpu.spec_decode.target_dependent_ar.speculator"


def _draft_buffers(max_reqs: int, max_tokens: int):
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    return InputBuffers(
        max_num_reqs=max_reqs, max_num_tokens=max_tokens, device=torch.device("cpu")
    )


def _draft_prefill(g, num_reqs: int, chunked: bool):
    sp = _m("spec_decode.target_dependent_ar.speculator")
    max_reqs, max_tokens = 6, 64
    qlens = torch.randint(1, 6, (num_reqs,), generator=g)
    qlens[0] = 4  # room for rejections
    qsl = torch.cat([torch.zeros(1, dtype=torch.int64), qlens.cumsum(0)]).to(torch.int32)
    num_tokens = int(qsl[-1])
    batch = types.SimpleNamespace(
        num_reqs=num_reqs,
        input_ids=torch.randint(0, 900, (num_tokens,), generator=g, dtype=torch.int32),
        positions=torch.randint(0, 2000, (num_tokens,), generator=g),
        idx_mapping=_perm_reqs(g, max_reqs, num_reqs),
        query_start_loc=qsl,
        seq_lens=torch.randint(10, 3000, (num_reqs,), generator=g, dtype=torch.int32),
    )
    buffers = _draft_buffers(max_reqs, max_tokens)
    num_sampled = torch.ones(num_reqs, dtype=torch.int32)
    if chunked:
        num_sampled[-1] = 0
    num_rejected = torch.minimum(torch.randint(0, 3, (num_reqs,), generator=g), qlens - 1).to(
        torch.int32
    )
    last_sampled = torch.randint(0, 2**33, (max_reqs,), generator=g)  # truncated to int32
    next_prefill = torch.randint(0, 900, (max_reqs,), generator=g, dtype=torch.int32)
    state = dict(
        last_token_indices=torch.full((max_reqs,), -9, dtype=torch.int64),
        current_draft_step=torch.tensor(5, dtype=torch.int64),
        input_ids=buffers.input_ids,
        positions=buffers.positions,
        query_start_loc=buffers.query_start_loc,
        seq_lens=buffers.seq_lens,
    )

    def run(s):
        sp.prepare_prefill_inputs(
            s["last_token_indices"],
            s["current_draft_step"],
            buffers,
            batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill,
            max_reqs,
        )

    return run, state


def _perm_reqs(g, max_reqs: int, num_reqs: int) -> torch.Tensor:
    return torch.randperm(max_reqs, generator=g)[:num_reqs].to(torch.int32)


@case(f"{_TDAR}._prepare_prefill_inputs_kernel")
def draft_prepare_prefill_inputs(g):
    return _draft_prefill(g, 4, chunked=False)


@case(f"{_TDAR}._prepare_prefill_inputs_kernel")
def draft_prepare_prefill_inputs_chunked_prefill(g):
    return _draft_prefill(g, 3, chunked=True)


def _draft_decode(g, advance: bool):
    sp = _m("spec_decode.target_dependent_ar.speculator")
    max_reqs, max_tokens, steps, num_reqs, max_model_len = 6, 64, 3, 4, 100
    buffers = _draft_buffers(max_reqs, max_tokens)
    buffers.positions[:max_reqs] = torch.randint(90, 100, (max_reqs,), generator=g)
    draft_tokens = torch.randint(0, 900, (max_reqs, steps), generator=g)
    state = dict(
        input_ids=buffers.input_ids,
        positions=buffers.positions,
        query_start_loc=buffers.query_start_loc,
        seq_lens=buffers.seq_lens,
        sample_src_positions=torch.randint(0, 99, (max_reqs,), generator=g),
    )
    target_seq_lens = torch.randint(95, 101, (num_reqs,), generator=g, dtype=torch.int32)
    num_rejected = torch.randint(0, 3, (num_reqs,), generator=g, dtype=torch.int32)

    def run(s):
        sp.prepare_decode_inputs(
            draft_tokens[:num_reqs, 0],
            target_seq_lens,
            num_rejected,
            buffers,
            s["sample_src_positions"],
            max_model_len,
            max_reqs,
            advance_draft_positions=advance,
        )

    return run, state


@case(f"{_TDAR}._prepare_decode_inputs_kernel")
def draft_prepare_decode_inputs(g):
    return _draft_decode(g, advance=True)


@case(f"{_TDAR}._prepare_decode_inputs_kernel")
def draft_prepare_decode_inputs_fixed_positions(g):
    """Gemma-4 MTP: Q-only drafts keep positions and seq_lens."""
    return _draft_decode(g, advance=False)


def _draft_update(g, step: int, advance: bool):
    sp = _m("spec_decode.target_dependent_ar.speculator")
    max_reqs, max_tokens, steps, num_reqs, hidden, max_model_len = 6, 64, 3, 4, 40, 100
    buffers = _draft_buffers(max_reqs, max_tokens)
    buffers.positions[:max_reqs] = torch.randint(95, 100, (max_reqs,), generator=g)
    buffers.seq_lens[:max_reqs] = torch.randint(96, 101, (max_reqs,), generator=g)
    state = dict(
        input_ids=buffers.input_ids,
        positions=buffers.positions,
        seq_lens=buffers.seq_lens,
        output_draft_tokens=torch.full((max_reqs, steps), -1, dtype=torch.int64),
        next_input_hidden_states=torch.zeros(max_tokens, hidden, dtype=torch.float16),
        sample_src_positions=torch.randint(0, 99, (max_reqs,), generator=g),
    )
    draft_tokens = torch.randint(0, 900, (num_reqs,), generator=g)
    # Wider than the buffer's rows, as a draft model's output slice can be.
    hidden_states = torch.randn(num_reqs + 2, hidden, generator=g).half()[:num_reqs]

    def run(s):
        sp.update_draft_inputs(
            draft_tokens,
            torch.tensor(step, dtype=torch.int64),
            hidden_states,
            s["output_draft_tokens"],
            s["next_input_hidden_states"],
            buffers,
            s["sample_src_positions"],
            num_reqs,
            max_model_len,
            steps,
            advance_draft_positions=advance,
        )

    return run, state


@case(f"{_TDAR}._update_draft_inputs_kernel")
def update_draft_inputs(g):
    return _draft_update(g, step=1, advance=True)


@case(f"{_TDAR}._update_draft_inputs_kernel")
def update_draft_inputs_fixed_positions(g):
    return _draft_update(g, step=1, advance=False)


@case(f"{_TDAR}._update_draft_inputs_kernel")
def update_draft_inputs_final_step(g):
    """The last step records its draft and leaves the next step's inputs alone."""
    return _draft_update(g, step=2, advance=True)
