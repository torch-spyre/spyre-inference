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

"""Parity cases: each runs one upstream MRV2 wrapper on randomized inputs, once
through triton-cpu and once with kernels.bind()'s host op in its place, and
compares every output and every buffer the kernel may write.

Run as a script under SPYRE_MODEL_RUNNER=v2 SPYRE_MRV2_KERNELS=triton (the
gate must open real Triton before vLLM imports), so test_parity.py spawns it.
"""

import importlib
import sys
from collections.abc import Callable
from typing import Any

import torch

from spyre_inference.v1.worker.spyre import kernels

CASES: dict[str, Callable[[torch.Generator], tuple[Callable[[], Any], dict[str, Any]]]] = {}


def case(*kernel_names: str):
    """Register a case covering ``kernel_names`` (keys of kernels.HOST_KERNELS)."""

    def register(fn):
        CASES[fn.__name__] = fn
        fn.kernels = kernel_names
        return fn

    return register


def _m(name: str):
    return importlib.import_module(f"vllm.v1.worker.gpu.{name}")


def _ragged(g: torch.Generator, num_reqs: int, max_len: int) -> torch.Tensor:
    lens = torch.randint(0, max_len + 1, (num_reqs,), generator=g)
    lens[0] = max(int(lens[0]), 1)
    return torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(torch.int32)


def _perm(g: torch.Generator, n: int, k: int) -> torch.Tensor:
    return torch.randperm(n, generator=g)[:k].to(torch.int32)


@case("gpu.input_batch._prepare_prefill_inputs_kernel")
def prepare_prefill_inputs(g):
    ib = _m("input_batch")
    max_reqs, num_reqs, max_len = 12, 9, 64
    qsl = _ragged(g, num_reqs, 20)
    num_tokens = int(qsl[-1])
    prefill_len = torch.randint(1, max_len - 22, (max_reqs,), generator=g, dtype=torch.int32)
    computed = (torch.rand(max_reqs, generator=g) * prefill_len * 1.3).to(torch.int32)
    state = dict(
        input_ids=torch.full((num_tokens + 5,), -7, dtype=torch.int32),
        next_prefill_tokens=torch.full((3, max_reqs), -7, dtype=torch.int32),
        idx_mapping=_perm(g, max_reqs, num_reqs),
        query_start_loc=qsl,
        all_token_ids=torch.randint(0, 1000, (max_reqs, max_len), generator=g, dtype=torch.int32),
        prefill_len=prefill_len,
        num_computed_tokens=computed,
    )
    return lambda s: ib.prepare_prefill_inputs(**s), state


@case("gpu.input_batch._prepare_pos_seq_lens_kernel")
def prepare_pos_seq_lens(g):
    ib = _m("input_batch")
    max_reqs, num_reqs = 16, 7
    qsl = _ragged(g, num_reqs, 30)
    state = dict(
        idx_mapping=_perm(g, max_reqs, num_reqs),
        query_start_loc=qsl,
        num_computed_tokens=torch.randint(0, 500, (max_reqs,), generator=g, dtype=torch.int32),
        pos=torch.full((int(qsl[-1]) + 3,), -7, dtype=torch.int64),
        seq_lens=torch.full((max_reqs,), -7, dtype=torch.int32),
    )
    return lambda s: ib.prepare_pos_seq_lens(**s), state


@case("gpu.input_batch._combine_sampled_and_draft_tokens_kernel")
def combine_sampled_and_draft_tokens(g):
    ib = _m("input_batch")
    max_reqs, num_reqs = 10, 8
    qsl = _ragged(g, num_reqs, 6)
    num_logits_per = torch.ones(num_reqs, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), num_logits_per.cumsum(0).to(torch.int32)])
    idx = _perm(g, max_reqs, num_reqs)
    seq_lens = torch.randint(1, 40, (num_reqs,), generator=g, dtype=torch.int32)
    prefill_len = torch.randint(1, 40, (max_reqs,), generator=g, dtype=torch.int32)
    state = dict(
        input_ids=torch.randint(0, 100, (int(qsl[-1]) + 2,), generator=g, dtype=torch.int32),
        idx_mapping=idx,
        last_sampled_tokens=torch.randint(0, 1000, (max_reqs, 1), generator=g),
        query_start_loc=qsl,
        seq_lens=seq_lens,
        prefill_len=prefill_len,
        draft_tokens=torch.zeros(max_reqs, 0, dtype=torch.int64),
        cu_num_logits=cu,
        num_logits=num_reqs,
    )
    return lambda s: ib.combine_sampled_and_draft_tokens(**s), state


@case("gpu.input_batch._get_num_sampled_and_rejected_kernel")
def get_num_sampled_and_rejected(g):
    ib = _m("input_batch")
    max_reqs, num_reqs = 10, 8
    lens = torch.randint(1, 4, (num_reqs,), generator=g)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(torch.int32)
    state = dict(
        num_sampled=torch.randint(0, 3, (num_reqs,), generator=g, dtype=torch.int32),
        seq_lens=torch.randint(1, 40, (num_reqs,), generator=g, dtype=torch.int32),
        cu_num_logits=cu,
        idx_mapping=_perm(g, max_reqs, num_reqs),
        prefill_len=torch.randint(1, 40, (max_reqs,), generator=g, dtype=torch.int32),
    )
    return lambda s: ib.get_num_sampled_and_rejected(**s), state


@case("gpu.input_batch._post_update_kernel")
def post_update(g):
    ib = _m("input_batch")
    max_reqs, num_reqs, vocab, max_len, spec = 10, 8, 300, 64, 2
    idx = _perm(g, max_reqs, num_reqs)
    idx[3] = -1
    state = dict(
        idx_mapping=idx,
        num_computed_tokens=torch.randint(0, 20, (max_reqs,), generator=g, dtype=torch.int32),
        last_sampled_tokens=torch.zeros(max_reqs, 1, dtype=torch.int64),
        output_bin_counts=torch.randint(0, 3, (max_reqs, vocab), generator=g, dtype=torch.int32),
        sampled_tokens=torch.randint(0, vocab, (num_reqs, spec + 1), generator=g),
        num_sampled=torch.randint(0, spec + 2, (num_reqs,), generator=g, dtype=torch.int32),
        num_rejected=torch.randint(0, 2, (num_reqs,), generator=g, dtype=torch.int32),
        query_start_loc=_ragged(g, num_reqs, 5),
        all_token_ids=torch.zeros(max_reqs, max_len, dtype=torch.int32),
        total_len=torch.randint(0, 30, (max_reqs,), generator=g, dtype=torch.int32),
    )
    return lambda s: ib.post_update(**s), state


@case("gpu.input_batch._post_update_kernel")
def post_update_with_draft_tokens(g):
    run, state = post_update(g)
    num_reqs, max_reqs = state["sampled_tokens"].shape[0], state["num_computed_tokens"].shape[0]
    state["draft_tokens"] = torch.randint(0, 300, (num_reqs, 3), generator=g)
    state["draft_tokens_out"] = torch.full((max_reqs, 3), -7, dtype=torch.int64)
    return run, state


@case("gpu.input_batch._post_update_kernel")
def post_update_without_counts_or_qsl(g):
    run, state = post_update(g)
    state["output_bin_counts"] = None
    state["query_start_loc"] = None
    return run, state


@case("gpu.input_batch._post_update_num_computed_tokens_kernel")
def post_update_num_computed_tokens(g):
    ib = _m("input_batch")
    max_reqs, num_reqs = 10, 6
    state = dict(
        idx_mapping=_perm(g, max_reqs, num_reqs),
        num_computed_tokens=torch.randint(0, 20, (max_reqs,), generator=g, dtype=torch.int32),
        query_start_loc=_ragged(g, num_reqs, 9),
    )
    return lambda s: ib.post_update_num_computed_tokens(**s), state


@case("gpu.input_batch._expand_idx_mapping_kernel")
def expand_idx_mapping(g):
    ib = _m("input_batch")
    num_reqs = 7
    lens = torch.randint(1, 4, (num_reqs,), generator=g)
    cu = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(torch.int32)
    state = dict(
        idx_mapping=_perm(g, 12, num_reqs),
        total_num_logits=int(cu[-1]),
        cu_num_logits=cu,
        max_expand_len=int(lens.max()),
    )
    return lambda s: ib.expand_idx_mapping(**s), state


def _block_tables(g, num_groups: int, sliding_window_reach: list[int] | None = None):
    from vllm.v1.worker.gpu.block_table import BlockTables

    bt = BlockTables(
        block_sizes=[16, 32][:num_groups],
        max_num_reqs=8,
        max_num_batched_tokens=96,
        max_num_blocks_per_group=[12, 6][:num_groups],
        device=torch.device("cpu"),
        sliding_window_reach=sliding_window_reach,
    )
    for req in range(8):
        for gi in range(num_groups):
            cap = bt.block_tables[gi].gpu.shape[1]
            n = int(torch.randint(0, cap + 1, (1,), generator=g))
            bt.block_tables[gi].stage_write(
                req, 0, torch.randint(1, 500, (n,), generator=g).tolist()
            )
            bt.num_blocks.np[gi, req] = n
    return bt


def _block_table_state(bt) -> dict[str, Any]:
    return {
        **{f"table{i}": t.gpu for i, t in enumerate(bt.block_tables)},
        **{f"input{i}": t for i, t in enumerate(bt.input_block_tables)},
        "slot_mappings": bt.slot_mappings,
    }


@case("gpu.buffer_utils._apply_write_kernel")
def staged_writes_single_group(g):
    bt = _block_tables(g, 1)
    return lambda s: bt.apply_staged_writes(), _block_table_state(bt)


@case("gpu.buffer_utils._apply_write_kernel")
def staged_writes_multi_group(g):
    bt = _block_tables(g, 2)
    return lambda s: bt.apply_staged_writes(), _block_table_state(bt)


@case("gpu.buffer_utils._apply_write_kernel")
def staged_writes_float(g):
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    t = StagedWriteTensor((6, 40), dtype=torch.float32, device=torch.device("cpu"))
    # Disjoint writes: Triton runs overlapping ones in an undefined order.
    for row, start in ((4, 0), (1, 3), (4, 25)):
        t.stage_write(
            row,
            start,
            torch.randn(int(torch.randint(1, 16, (1,), generator=g)), generator=g).tolist(),
        )
    return lambda s: t.apply_write(), {"t": t.gpu}


@case(
    "gpu.block_table._gather_block_tables_kernel",
    "gpu.block_table._compute_slot_mappings_kernel",
)
def block_tables(g):
    bt = _block_tables(g, 2)
    bt.apply_staged_writes()
    num_reqs = 5
    idx = _perm(g, 8, num_reqs)
    idx[2] = -1
    qsl = _ragged(g, num_reqs, 14)
    positions = torch.randint(0, 180, (int(qsl[-1]),), generator=g)
    for i in range(int(qsl[-1])):
        positions[i] = min(int(positions[i]), 175)

    def run(s):
        bt.gather_block_tables(idx.clamp(min=0), num_reqs_padded=7)
        bt.compute_slot_mappings(idx, qsl, positions, num_tokens_padded=int(qsl[-1]) + 4)

    return run, _block_table_state(bt)


@case("gpu.block_table._gather_block_tables_kernel")
def block_tables_sliding_window(g):
    # Group 0 full attention (reach 0), group 1 a 40-token window over 32-token blocks.
    bt = _block_tables(g, 2, sliding_window_reach=[0, 40])
    bt.apply_staged_writes()
    idx = _perm(g, 8, 6)
    computed = torch.randint(0, 200, (8,), generator=g, dtype=torch.int32)

    def run(s):
        bt.gather_block_tables(idx, num_reqs_padded=7, num_computed_tokens=computed)

    return run, _block_table_state(bt)


def _logits(g, num_tokens: int, vocab: int, dtype=torch.float32) -> torch.Tensor:
    x = torch.randn(num_tokens, vocab, generator=g) * 3
    x[0, :7] = float("-inf")
    return x.to(dtype)


@case("gpu.sample.gumbel._temperature_kernel")
def temperature(g):
    gb = _m("sample.gumbel")
    temps = torch.tensor([0.0, 1.0, 0.7, 1.3, 0.05], dtype=torch.float32)
    state = dict(
        logits=_logits(g, 6, 9000),
        expanded_idx_mapping=torch.tensor([4, 2, 0, 1, 3, 2], dtype=torch.int32),
        temperature=temps,
    )
    return lambda s: gb.apply_temperature(**s), state


def _gumbel(g, use_fp64: bool, apply_temperature: bool, is_drafting: bool):
    gb = _m("sample.gumbel")
    num_tokens, vocab = 9, 5000
    logits = _logits(g, num_tokens, vocab)
    logits[3] = float("-inf")
    logits[4, 100] = 50.0
    logits[5, 10:20] = 30.0
    state = dict(
        logits=logits,
        expanded_idx_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 2, 6, 1], dtype=torch.int32),
        temperature=torch.tensor([0.0, 1.0, 0.7, 1.3, 0.05, 1.0, 2.5], dtype=torch.float32),
        seed=torch.randint(-(2**62), 2**62, (7,), generator=g),
        pos=torch.randint(0, 4000, (num_tokens,), generator=g),
        apply_temperature=apply_temperature,
        is_drafting=is_drafting,
        use_fp64=use_fp64,
    )
    return lambda s: gb.gumbel_sample(**s), state


@case("gpu.sample.gumbel._gumbel_sample_kernel")
def gumbel_sample(g):
    return _gumbel(g, use_fp64=False, apply_temperature=False, is_drafting=False)


@case("gpu.sample.gumbel._gumbel_sample_kernel")
def gumbel_sample_temperature_drafting(g):
    return _gumbel(g, use_fp64=False, apply_temperature=True, is_drafting=True)


@case("gpu.sample.gumbel._gumbel_sample_kernel")
def gumbel_sample_fp64(g):
    return _gumbel(g, use_fp64=True, apply_temperature=True, is_drafting=False)


def _gumbel_logits_cache(g, per_token_col: bool, source: bool):
    run, state = _gumbel(g, use_fp64=False, apply_temperature=True, is_drafting=True)
    num_tokens, vocab = state["logits"].shape
    # Distinct requests: two tokens writing one (req, col) race in Triton's grid.
    state["expanded_idx_mapping"] = torch.tensor(
        [0, 1, 2, 3, -1, 5, 4, 6, -1][:num_tokens], dtype=torch.int32
    )
    state["logits_cache"] = torch.full((7, 3, vocab + 8), -7.0)
    state["logits_cache_col"] = (
        torch.randint(0, 3, (num_tokens,), generator=g) if per_token_col else torch.tensor(2)
    )
    if source:
        state["logits_cache_source"] = torch.randn(num_tokens, vocab, generator=g)
    return run, state


@case("gpu.sample.gumbel._gumbel_sample_kernel")
def gumbel_sample_logits_cache(g):
    return _gumbel_logits_cache(g, per_token_col=False, source=False)


@case("gpu.sample.gumbel._gumbel_sample_kernel")
def gumbel_sample_logits_cache_per_token_source(g):
    return _gumbel_logits_cache(g, per_token_col=True, source=True)


@case("gpu.sample.min_p._min_p_kernel")
def min_p(g):
    mp = _m("sample.min_p")
    state = dict(
        logits=_logits(g, 5, 3000),
        expanded_idx_mapping=torch.tensor([0, 1, 2, 2, 3], dtype=torch.int32),
        min_p=torch.tensor([0.0, 0.1, 0.5, 0.01], dtype=torch.float32),
    )
    return lambda s: mp.apply_min_p(**s), state


@case("gpu.sample.penalties._penalties_kernel", "gpu.sample.penalties._bincount_kernel")
def penalties(g):
    pen = _m("sample.penalties")
    max_reqs, vocab, max_len = 6, 9000, 128
    all_token_ids = torch.randint(0, vocab, (max_reqs, max_len), generator=g, dtype=torch.int32)
    prompt_len = torch.randint(1, 60, (max_reqs,), generator=g, dtype=torch.int32)
    prefill_len = prompt_len + torch.randint(0, 40, (max_reqs,), generator=g, dtype=torch.int32)
    prompt_bin_mask = torch.zeros(max_reqs, (vocab + 31) // 32, dtype=torch.int32)
    output_bin_counts = torch.zeros(max_reqs, vocab, dtype=torch.int32)
    reqs = torch.tensor([0, 2, 3, 5], dtype=torch.int32)
    # Requests 0 and 3 run two logits each: the second also counts the draft.
    expanded = torch.tensor([0, 0, 2, 3, 3, 5], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int32)
    state = dict(
        logits=_logits(g, 6, vocab),
        prompt_bin_mask=prompt_bin_mask,
        output_bin_counts=output_bin_counts,
    )

    def run(s):
        pen.bincount(
            reqs,
            all_token_ids,
            prompt_len,
            prefill_len,
            s["prompt_bin_mask"],
            s["output_bin_counts"],
            int(prefill_len.max()),
        )
        pen.apply_penalties(
            s["logits"],
            expanded,
            torch.randint(0, vocab, (6,), generator=torch.Generator().manual_seed(1)),
            local_pos,
            torch.tensor([1.3, 1.0, 0.8, 1.0, 1.0, 1.1], dtype=torch.float32),
            torch.tensor([0.0, 0.0, 0.5, 0.2, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.7], dtype=torch.float32),
            s["prompt_bin_mask"],
            s["output_bin_counts"],
        )

    return run, state


def _logit_bias_state(g, restore: bool):
    vocab, max_reqs = 4000, 5
    n_allowed = torch.tensor([0, 30, 0, 5, 0], dtype=torch.int32)
    n_bias = torch.tensor([3, 0, 0, 7, 2], dtype=torch.int32)
    n_stop = torch.tensor([2, 0, 4, 1, 3], dtype=torch.int32)
    logits = _logits(g, 6, vocab)
    if restore:
        logits[5] = float("-inf")
        logits[5, 11] = 2.0
    stop_ids = torch.randint(0, vocab, (max_reqs, 16), generator=g, dtype=torch.int32)
    stop_ids[4, :3] = torch.tensor([11, 12, 13])
    return dict(
        logits=logits,
        expanded_idx_mapping=torch.tensor([0, 1, 2, 3, 4, 4], dtype=torch.int32),
        pos=torch.tensor([3, 9, 1, 0, 2, 2], dtype=torch.int64),
        num_allowed_token_ids=n_allowed,
        allowed_token_ids=_perm(g, vocab, max_reqs * 32).view(max_reqs, 32),
        num_logit_bias=n_bias,
        logit_bias_token_ids=_perm(g, vocab, max_reqs * 16).view(max_reqs, 16),
        logit_bias=torch.randn(max_reqs, 16, generator=g),
        min_lens=torch.tensor([10, 0, 5, 0, 8], dtype=torch.int32),
        num_stop_token_ids=n_stop,
        restore_when_all_masked=torch.tensor([0, 0, 0, 0, int(restore)], dtype=torch.int32),
        stop_token_ids=stop_ids,
        check_all_masked_rows=restore,
    )


@case("gpu.sample.logit_bias._bias_kernel")
def logit_bias(g):
    lb = _m("sample.logit_bias")
    return lambda s: lb.apply_logit_bias(**s), _logit_bias_state(g, restore=False)


@case("gpu.sample.logit_bias._bias_kernel")
def logit_bias_restore_all_masked(g):
    lb = _m("sample.logit_bias")
    return lambda s: lb.apply_logit_bias(**s), _logit_bias_state(g, restore=True)


@case("gpu.sample.bad_words._bad_words_kernel")
def bad_words(g):
    bw = _m("sample.bad_words")
    vocab, max_reqs, max_len = 500, 4, 64
    all_ids = torch.randint(0, 6, (max_reqs, max_len), generator=g, dtype=torch.int32)
    tokens = torch.zeros(max_reqs, 32, dtype=torch.int32)
    offsets = torch.zeros(max_reqs, 6, dtype=torch.int32)
    num = torch.tensor([3, 0, 2, 4], dtype=torch.int32)
    for r in range(max_reqs):
        cursor = 0
        for w in range(int(num[r])):
            n = int(torch.randint(1, 4, (1,), generator=g))
            tokens[r, cursor : cursor + n] = torch.randint(0, 6, (n,), generator=g)
            cursor += n
            offsets[r, w + 1] = cursor
    state = dict(
        logits=_logits(g, 5, vocab),
        expanded_idx_mapping=torch.tensor([0, 2, 2, 3, 1], dtype=torch.int32),
        bad_word_token_ids=tokens,
        bad_word_offsets=offsets,
        num_bad_words=num,
        all_token_ids=all_ids,
        prompt_len=torch.tensor([5, 3, 10, 2], dtype=torch.int32),
        total_len=torch.tensor([20, 9, 30, 11], dtype=torch.int32),
        input_ids=torch.randint(0, 6, (5,), generator=g, dtype=torch.int32),
        expanded_local_pos=torch.tensor([0, 0, 1, 0, 0], dtype=torch.int32),
        max_num_bad_words=4,
    )
    return lambda s: bw.apply_bad_words(**s), state


@case(
    "gpu.sample.logprob._topk_log_softmax_kernel",
    "gpu.sample.logprob._ranks_kernel",
)
def topk_scores(g):
    lp = _m("sample.logprob")
    logits = _logits(g, 5, 9000)
    sampled = torch.randint(0, 9000, (5,), generator=g)
    return lambda s: lp.compute_topk_scores(logits, 4, sampled), {}


@case(
    "gpu.sample.logprob._topk_log_softmax_kernel",
    "gpu.sample.logprob._ranks_kernel",
    "gpu.sample.logprob._fill_logprob_token_ids_kernel",
)
def topk_scores_custom_token_ids(g):
    lp = _m("sample.logprob")
    st = lp.LogprobTokenIdsState(6, torch.device("cpu"))
    st.num_token_ids.np[:] = [0, 3, 0, 5, 0, 0]
    st.token_ids.stage_write(1, 0, [1, 2, 3])
    st.token_ids.stage_write(3, 0, [40, 41, 42, 43, 44])
    st.apply_staged_writes()
    logits = _logits(g, 4, 3000)
    sampled = torch.randint(0, 3000, (4,), generator=g)
    expanded = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    return lambda s: lp.compute_topk_scores(
        logits,
        2,
        sampled,
        logprob_token_ids_state=st,
        expanded_idx_mapping=expanded,
        max_per_req_token_ids=5,
    ), {}


@case("gpu.sample.prompt_logprob._prompt_logprobs_token_ids_kernel")
def prompt_logprobs_token_ids(g):
    pl = _m("sample.prompt_logprob")
    qsl = _ragged(g, 4, 12)
    state = dict(
        num_tokens=int(qsl[-1]),
        query_start_loc=qsl,
        idx_mapping=_perm(g, 6, 4),
        num_computed_tokens=torch.randint(0, 30, (6,), generator=g, dtype=torch.int32),
        all_token_ids=torch.randint(0, 999, (6, 64), generator=g, dtype=torch.int32),
    )
    return lambda s: pl.get_prompt_logprobs_token_ids(**s), state


@case("gpu.sample.output._compact_sampling_mask_kernel")
def compact_sampling_mask(g):
    out = _m("sample.output")
    logits = _logits(g, 4, 10000)
    logits[1, ::3] = float("-inf")
    logits[2] = float("-inf")
    logits[2, [5, 9000]] = 1.0
    num_sampled = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    cu = torch.arange(5, dtype=torch.int32)
    return lambda s: _written_mask(
        out.SamplingMaskTensors.from_logits(logits, cu, num_sampled, 50), num_sampled, 1
    ), {}


def _written_mask(mask, num_sampled: torch.Tensor, rows_per_request: int):
    """The parts of SamplingMaskTensors the kernel writes; the rest is torch.empty."""
    token_ids, packed, counts = mask.token_ids.clone(), mask.packed_mask.clone(), mask.counts
    for row in range(token_ids.shape[0]):
        if row % rows_per_request >= int(num_sampled[row // rows_per_request]):
            token_ids[row] = 0
            packed[row] = 0
        else:
            token_ids[row, int(counts[row]) :] = 0
    return token_ids, packed, counts


@case("gpu.sample.output._compact_sampling_mask_kernel")
def compact_sampling_mask_rows_per_request(g):
    out = _m("sample.output")
    logits = _logits(g, 7, 4000)
    logits[1, ::2] = float("-inf")
    logits[4] = float("-inf")
    logits[4, 3] = 0.0
    # Three requests with up to three rows each, sampling 2, 3 and 1 of them.
    num_sampled = torch.tensor([2, 3, 1], dtype=torch.int32)
    cu = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    return lambda s: _written_mask(
        out.SamplingMaskTensors.from_logits(logits, cu, num_sampled, 30, rows_per_request=3),
        num_sampled,
        3,
    ), {}


@case("gpu.sample.trace_replay._trace_replay_kernel")
def trace_replay(g):
    tr = _m("sample.trace_replay")
    idx = torch.tensor([2, -1, 0, 3], dtype=torch.int32)
    state = dict(
        sampled=torch.randint(0, 999, (4,), generator=g),
        idx_mapping=idx,
        trace_token_ids=torch.randint(0, 999, (5, 20), generator=g, dtype=torch.int32),
        trace_len=torch.tensor([10, 0, 4, 0, 3], dtype=torch.int32),
        total_len=torch.tensor([12, 5, 9, 8, 7], dtype=torch.int32),
        prompt_len=torch.tensor([5, 5, 4, 8, 7], dtype=torch.int32),
    )
    return lambda s: tr.apply_trace_tokens(**s), state


def _thinking(g, cold: bool):
    tb = _m("sample.thinking_budget")
    max_reqs, max_len = 5, 200
    start, natural_end, end = [7, 8], [9], [9, 10, 11]
    ids = torch.randint(0, 6, (max_reqs, max_len), generator=g, dtype=torch.int32)
    ids[0, 10:12] = torch.tensor(start)
    ids[1, 20:22] = torch.tensor(start)
    ids[1, 40] = 9
    ids[2, 50:52] = torch.tensor(start)
    ids[2, 98:100] = torch.tensor([9, 10])
    ids[3, 150:152] = torch.tensor(start)
    total_len = torch.tensor([60, 80, 100, 170, 30], dtype=torch.int32)
    scan = (
        torch.zeros(max_reqs, dtype=torch.int32)
        if cold
        else torch.tensor([5, 30, 40, 120, 31], dtype=torch.int32)
    )
    last = torch.full((max_reqs,), -1, dtype=torch.int32)
    state = dict(
        logits=_logits(g, 6, 300),
        req_ids=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 1, 2, 3, 3, 4], dtype=torch.int32),
        thinking_token_budget=torch.tensor([20, 5, 3, 10, -1], dtype=torch.int32),
        all_token_ids=ids,
        total_len=total_len,
        input_ids=torch.tensor([1, 2, 9, 7, 8, 3], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.int32),
        cached_last_start=last.clone(),
        cached_last_end=last.clone(),
        cached_scan_pos=scan,
        reasoning_start_token_ids=torch.tensor(start, dtype=torch.int32),
        natural_reasoning_end_token_ids=torch.tensor(natural_end, dtype=torch.int32),
        reasoning_end_token_ids=torch.tensor(end, dtype=torch.int32),
    )
    return lambda s: tb.apply_thinking_budget(**s), state


@case(
    "gpu.sample.thinking_budget._update_committed_marker_cache_kernel",
    "gpu.sample.thinking_budget._thinking_budget_kernel",
)
def thinking_budget_cold_scan(g):
    return _thinking(g, cold=True)


@case(
    "gpu.sample.thinking_budget._update_committed_marker_cache_kernel",
    "gpu.sample.thinking_budget._thinking_budget_kernel",
)
def thinking_budget_incremental(g):
    return _thinking(g, cold=False)


@case("gpu.sample.watermark._philox_gumbel_kernel")
def philox_gumbel(g):
    wm = _m("sample.watermark")
    logits = _logits(g, 4, 3001)
    logits[2, 17] = float("nan")
    contexts = torch.randint(0, 3000, (4, 5), generator=g)
    return lambda s: wm.philox_gumbel_sample(logits, contexts, 0x123456789ABCDEF), {}


@case("gpu.sample.watermark._philox_gumbel_kernel")
def philox_gumbel_with_skip_rows(g):
    wm = _m("sample.watermark")
    logits = _logits(g, 4, 3001)
    contexts = torch.randint(0, 3000, (4, 3), generator=g)
    return lambda s: wm.philox_gumbel_sample(
        logits,
        contexts,
        99,
        skip_mask=torch.tensor([True, False, True, False]),
        expanded_idx_mapping=torch.tensor([0, 1, 2, 1], dtype=torch.int32),
        temperatures=torch.tensor([0.0, 1.0, 0.9], dtype=torch.float32),
        seeds=torch.tensor([5, 6, 7]),
        positions=torch.tensor([1, 2, 3, 4]),
    ), {}


def _draft_philox_gumbel(g, deduplicate: bool):
    wm = _m("sample.watermark")
    num_reqs, num_steps, vocab, width, max_reqs, max_len = 4, 3, 2100, 2, 6, 40
    logits = _logits(g, num_reqs * num_steps, vocab)
    # A tiny vocabulary of history tokens makes repeated contexts likely.
    all_token_ids = torch.randint(0, 4, (max_reqs, max_len), generator=g)
    prompt_lens = torch.randint(0, 10, (max_reqs,), generator=g, dtype=torch.int32)
    total_lens = prompt_lens + torch.randint(3, 25, (max_reqs,), generator=g, dtype=torch.int32)
    idx = torch.tensor([3, 0, 5, 1], dtype=torch.int32).repeat_interleave(num_steps)
    idx[-num_steps:] = -1
    temperatures = torch.tensor([0.8, 0.0, 1.0, 1.4, 0.5, 2.0], dtype=torch.float32)
    seeds = torch.randint(0, 2**40, (max_reqs,), generator=g)
    positions = torch.randint(0, 30, (num_reqs * num_steps,), generator=g)
    state = dict(
        contexts=torch.randint(0, 4, (num_reqs, width), generator=g),
        prior_contexts=torch.full((num_reqs, num_steps, width), -3, dtype=torch.int64),
        logits_cache=torch.full((max_reqs, num_steps, vocab), -7.0),
    )
    return lambda s: wm.draft_philox_gumbel_sample(
        logits,
        s["contexts"],
        0xFEEDFACE12345,
        num_steps=num_steps,
        expanded_idx_mapping=idx,
        temperatures=temperatures,
        seeds=seeds,
        positions=positions,
        enabled=torch.tensor([True, True, False, True]),
        logits_cache=s["logits_cache"],
        logits_cache_col=torch.arange(num_steps).repeat(num_reqs),
        use_fp64=False,
        prior_contexts=s["prior_contexts"],
        all_token_ids=all_token_ids,
        prompt_lens=prompt_lens,
        total_lens=total_lens,
        deduplicate=deduplicate,
        max_history=5 if deduplicate else None,
        include_prompt=deduplicate,
    ), state


@case("gpu.sample.watermark._philox_gumbel_kernel", "gpu.sample.watermark._draft_step_kernel")
def draft_philox_gumbel(g):
    return _draft_philox_gumbel(g, deduplicate=False)


@case("gpu.sample.watermark._philox_gumbel_kernel", "gpu.sample.watermark._draft_step_kernel")
def draft_philox_gumbel_deduplicate(g):
    return _draft_philox_gumbel(g, deduplicate=True)


@case("gpu.structured_outputs._apply_grammar_bitmask_kernel")
def grammar_bitmask(g):
    so = _m("structured_outputs")
    vocab, mask_stride = 10000, 3
    logits = _logits(g, 6, vocab)
    words = (vocab + 31) // 32 - 2  # narrower than the vocab: padded tail is masked
    bitmask = torch.randint(-(2**31), 2**31 - 1, (4, words), generator=g, dtype=torch.int32)
    mapping = torch.tensor(
        [0 * mask_stride, 2 * mask_stride + 1, 3 * mask_stride, 1 * mask_stride + 2],
        dtype=torch.int32,
    )
    cu = torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32)
    grid = (4, (vocab + 8191) // 8192)

    def run(s):
        so._apply_grammar_bitmask_kernel[grid](
            s["logits"],
            s["logits"].stride(0),
            mapping,
            cu,
            bitmask,
            bitmask.stride(0),
            vocab,
            MASK_STRIDE=mask_stride,
            BLOCK_SIZE=8192,
        )

    return run, {"logits": logits}


@case("gpu.metrics.logits._num_nans_kernel")
def num_nans(g):
    ml = _m("metrics.logits")
    logits = _logits(g, 3, 9000)
    logits[1, ::7] = float("nan")
    return lambda s: ml.get_num_nans(logits), {}


@case("gpu.mm.rope._prepare_rope_positions_kernel")
def rope_positions(g):
    rope = _m("mm.rope")
    st = rope.RopeState(3, 6, 64, 80, torch.device("cpu"))
    for r in range(6):
        for d in range(3):
            st.prefill_positions.stage_write(
                3 * r + d, 0, torch.randint(0, 99, (80,), generator=g).tolist()
            )
    st.prefill_delta.np[:] = torch.randint(-5, 5, (6,), generator=g).numpy()
    st.apply_staged_writes()
    qsl = _ragged(g, 4, 12)
    idx = _perm(g, 6, 4)
    prefill = torch.randint(5, 40, (6,), generator=g, dtype=torch.int32)
    computed = torch.randint(0, 50, (6,), generator=g, dtype=torch.int32)
    return lambda s: st.prepare_positions(idx, qsl, prefill, computed), {"positions": st.positions}


@case("gpu.model_states.prompt_embeds._apply_prompt_embeds_kernel")
def prompt_embeds(g):
    pe = _m("model_states.prompt_embeds")
    hidden, num_reqs = 96, 3
    embeds = [torch.randn(n, hidden, generator=g, dtype=torch.float16) for n in (20, 0, 9)]
    masks = [torch.randint(0, 2, (20,), generator=g, dtype=torch.uint8), None, None]
    ptrs = torch.tensor([e.data_ptr() for e in embeds], dtype=torch.uint64)
    mask_ptrs = torch.tensor([0 if m is None else m.data_ptr() for m in masks], dtype=torch.uint64)
    lens = torch.tensor([20, 0, 9], dtype=torch.int32)
    qsl = torch.tensor([0, 8, 10, 22], dtype=torch.int32)
    computed = torch.tensor([14, 0, 2], dtype=torch.int32)
    out = torch.zeros(24, hidden, dtype=torch.float16)
    grid = (num_reqs, 3)
    keep = (embeds, masks)

    def run(s):
        assert keep
        pe._apply_prompt_embeds_kernel[grid](
            s["out"],
            s["out"].stride(0),
            ptrs,
            mask_ptrs,
            lens,
            torch.arange(num_reqs, dtype=torch.int32),
            qsl,
            computed,
            hidden,
            TOKEN_BLOCK=4,
            BLOCK_SIZE=1024,
        )

    return run, {"out": out}


@case(
    "gpu.model_states.mamba_hybrid._scatter_num_accepted_kernel",
    "gpu.model_states.mamba_hybrid._fill_num_accepted_kernel",
)
def num_accepted(g):
    mh = _m("model_states.mamba_hybrid")
    idx = torch.tensor([3, -1, 0, 5], dtype=torch.int32)
    sampled = torch.tensor([0, 2, 3, 1], dtype=torch.int32)

    def run(s):
        mh._scatter_num_accepted_kernel[(4,)](idx, sampled, s["a"])
        mh._fill_num_accepted_kernel[(4,)](idx, s["b"], 2)

    return run, {"a": torch.zeros(6, dtype=torch.int32), "b": torch.zeros(6, dtype=torch.int32)}


@case("gpu.cp_utils._dcp_local_seq_lens_kernel")
def dcp_local_seq_lens(g):
    cp = _m("cp_utils")
    seq_lens = torch.randint(0, 300, (9,), generator=g, dtype=torch.int32)
    out = torch.full((12,), -1, dtype=torch.int32)
    return lambda s: cp.prepare_dcp_local_seq_lens(
        s["out"], seq_lens, 9, 4, 2, 16, num_reqs_padded=10
    ), {"out": out}


@case("utils._zero_kv_blocks_kernel")
def zero_kv_blocks(g):
    import vllm.v1.worker.utils as wu

    kv = torch.randn(6, 2, 16, 32, generator=g, dtype=torch.float16)
    seg_addrs = torch.tensor([kv.data_ptr(), kv[0, 1].data_ptr()], dtype=torch.uint64)
    strides = torch.tensor([kv.stride(0) // 2, kv.stride(0) // 2], dtype=torch.int64)
    pages = torch.tensor([kv.stride(1) // 2, kv.stride(1) // 2], dtype=torch.int64)
    blocks = torch.tensor([4, 1], dtype=torch.int64)

    def run(s):
        wu._zero_kv_blocks_kernel[(2, 2, 1)](seg_addrs, strides, pages, blocks, BLOCK_SIZE=256)

    return run, {"kv": kv}


def _snapshot(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.clone()
    if hasattr(x, "_asdict"):
        return _snapshot(x._asdict())
    if isinstance(x, (tuple, list)):
        return type(x)(_snapshot(v) for v in x)
    if isinstance(x, dict):
        return {k: _snapshot(v) for k, v in x.items()}
    if hasattr(x, "__dataclass_fields__"):
        return {k: _snapshot(getattr(x, k)) for k in x.__dataclass_fields__}
    return x


def _diff(path: str, a: Any, b: Any) -> list[str]:
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return [f"{path}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"]
        if a.is_floating_point():
            same = torch.isclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)
            same |= a == b
        else:
            same = a == b
        if not bool(same.all()):
            where = (~same).nonzero()[:3].tolist()
            return [
                f"{path}: {int((~same).sum())} differ, e.g. at {where}: "
                f"{[a[tuple(i)].item() for i in where]} vs {[b[tuple(i)].item() for i in where]}"
            ]
        return []
    if isinstance(a, (tuple, list)):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(f"{path}[{i}]", x, y)]
    if isinstance(a, dict):
        return [d for k in a for d in _diff(f"{path}.{k}", a[k], b[k])]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def run_case(name: str, seed: int) -> list[str]:
    """Run ``name`` through triton-cpu and the host op; return the differences."""
    fn = CASES[name]
    results = []
    for host in (False, True):
        g = torch.Generator().manual_seed(seed)
        run, state = fn(g)
        if host:
            kernels.bind(force_host=True)
        try:
            out = _snapshot(run(state))
        finally:
            kernels.unbind()
        results.append((out, _snapshot(state)))
    (ref_out, ref_state), (out, state) = results
    return _diff("out", ref_out, out) + _diff("state", ref_state, state)


if __name__ == "__main__":
    # Run as a script this module is __main__; take the importable copy's cases too.
    from tests.mrv2.kernels import parity_cases as _importable
    from tests.mrv2.kernels import spec_decode_cases  # noqa: F401

    CASES.update(_importable.CASES)
    failures = 0
    for name in sys.argv[1:] or CASES:
        for seed in range(3):
            diffs = run_case(name, seed)
            print(f"{'FAIL' if diffs else 'ok'} {name} seed={seed}")
            for d in diffs:
                print(f"    {d}")
            failures += bool(diffs)
    sys.exit(1 if failures else 0)
