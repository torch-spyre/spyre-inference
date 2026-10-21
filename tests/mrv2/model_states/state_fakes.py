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

"""Fakes for the model-state tests: a config namespace upstream's ModelState
constructors accept, and a real InputBatch built from query lengths."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CompilationMode
from vllm.v1.worker.gpu.input_batch import InputBatch


def make_config(
    *,
    max_model_len: int = 128,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 512,
    hf_config: Any = None,
    compiled: bool = True,
) -> SimpleNamespace:
    """The fields ModelState, DefaultModelState and EncoderOnlyModelState read,
    plus what the Spyre states and the encoder shape tables read."""
    hidden = 64
    model_config = SimpleNamespace(
        max_model_len=max_model_len,
        dtype=torch.float16,
        hf_config=hf_config,
        uses_mrope=False,
        uses_xdrope_dim=0,
        is_mm_prefix_lm=False,
        is_hybrid=False,
        is_attention_free=False,
        enable_prompt_embeds=False,
        multimodal_config=None,
        get_inputs_embeds_size=lambda: hidden,
        get_sliding_window=lambda: None,
    )
    return SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.STOCK_TORCH_COMPILE if compiled else CompilationMode.NONE,
            static_forward_context={},
        ),
    )


def make_input_batch(
    query_lens: list[int],
    *,
    num_tokens_after_padding: int | None = None,
    num_computed: list[int] | None = None,
    input_ids: list[int] | None = None,
) -> InputBatch:
    """A real InputBatch for ``query_lens``, the body padded to
    ``num_tokens_after_padding`` the way MRV2's prepare_inputs pads it."""
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    padded = num_tokens if num_tokens_after_padding is None else num_tokens_after_padding
    computed = np.array(num_computed or [0] * num_reqs, dtype=np.int32)
    scheduled = np.array(query_lens, dtype=np.int32)

    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(scheduled, out=query_start_loc_np[1:])
    seq_lens_np = computed + scheduled

    ids = torch.zeros(padded, dtype=torch.int32)
    ids[:num_tokens] = torch.tensor(input_ids or list(range(1, num_tokens + 1)), dtype=torch.int32)
    positions = torch.zeros(padded, dtype=torch.int64)
    positions[:num_tokens] = torch.cat(
        [
            torch.arange(c, c + n, dtype=torch.int64)
            for c, n in zip(computed, query_lens, strict=True)
        ]
    )
    is_padding = torch.zeros(padded, dtype=torch.bool)
    is_padding[num_tokens:] = True
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64)
    query_start_loc = torch.from_numpy(query_start_loc_np.copy())

    fields: dict[str, Any] = dict(
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping.numpy(),
        expanded_idx_mapping=idx_mapping,
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
        num_scheduled_tokens=scheduled,
        num_tokens=num_tokens,
        num_tokens_after_padding=padded,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.from_numpy(seq_lens_np.copy()),
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np.copy()),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=computed,
        prefill_len_np=seq_lens_np,
        num_computed_prefill_tokens_np=computed,
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        has_prefill=False,
        decode_graph_eligible=False,
        max_seq_len_np=None,
        input_ids=ids,
        positions=positions,
        is_padding=is_padding,
        logits_indices=query_start_loc[1:] - 1,
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32),
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )
    return InputBatch(**fields)
