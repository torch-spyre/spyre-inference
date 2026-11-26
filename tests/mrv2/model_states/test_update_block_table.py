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

"""update_block_table(build(table A), table B) equals build(table B), field by field; only
builders whose metadata vLLM can key on spec and builder type opt into it; and prepare_attn
shares metadata across same-spec groups as vLLM's runner does once it gates on that flag."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    reuse_metadata_across_groups,
)
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState

from .state_fakes import make_config, make_input_batch

BS = 128


def _builder(window, max_model_len=2048):
    cfg = get_current_vllm_config()
    cfg.model_config.get_num_attention_heads = Mock(return_value=4)  # ty: ignore[invalid-assignment]
    cfg.model_config.get_num_kv_heads = Mock(return_value=1)  # ty: ignore[invalid-assignment]
    cfg.cache_config.block_size = BS
    cfg.model_config.max_model_len = max_model_len
    if window is None:
        spec = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.float16)
    else:
        spec = SlidingWindowSpec(
            block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.float16, sliding_window=window
        )
    return SpyreAttentionMetadataBuilder(spec, ["layers.0.self_attn"], cfg, torch.device("cpu"))


def _common(query_lens, seq_lens, bt, first_slot=0):
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
    n = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(seq_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=bt,
        slot_mapping=torch.arange(first_slot, first_slot + n, dtype=torch.int64),
        causal=True,
        is_prefilling=torch.tensor([q > 1 for q in query_lens]),
    )


def _eq(a, b, path):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b), path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]")
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            if f.name in ("metadata", "rows_dev"):
                continue
            _eq(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    else:
        assert a == b, (path, a, b)


@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize(
    "query_lens,seq_lens,max_model_len",
    [
        ([1] * 8, [1030 + 7 * i for i in range(8)], 2048),
        # A one-page sequence below the page range's minimum: padded with its last page.
        ([1] * 4, [100, 700, 1500, 2000], 2048),
        ([1, 1, 300], [900, 1200, 812], 2048),
        ([512], [1536], 2048),
        # Without a window, jagged decode groups.
        ([1] * 3, [129, 129, 8191], 8192),
    ],
)
def test_update_block_table_matches_build(
    default_vllm_config, monkeypatch, window, query_lens, seq_lens, max_model_len
):
    b0 = _builder(window, max_model_len)
    b1 = _builder(window, max_model_len)
    g = torch.Generator().manual_seed(0)
    width = max_model_len // BS
    bt_a = torch.randint(1, 5000, (len(seq_lens), width), dtype=torch.int32, generator=g)
    bt_b = torch.randint(1, 5000, (len(seq_lens), width), dtype=torch.int32, generator=g)
    md_a = b0.build(0, _common(query_lens, seq_lens, bt_a))
    cb = _common(query_lens, seq_lens, bt_b, first_slot=sum(query_lens))
    published = []
    monkeypatch.setattr(b1, "publish_slot_mapping", published.append)
    upd = b1.update_block_table(md_a, bt_b, cb.slot_mapping)
    assert len(published) == 1 and published[0] is cb.slot_mapping
    ref = b1.build(0, cb)
    for f in dataclasses.fields(ref):
        _eq(getattr(upd, f.name), getattr(ref, f.name), f.name)


def test_shared_build_needs_scope_and_spec(default_vllm_config):
    full, full2, swa, draft = _builder(None), _builder(None), _builder(512), _builder(None)
    draft.causal_override = True
    c = _common([1, 1], [1100, 1300], torch.randint(1, 5000, (2, 16), dtype=torch.int32))
    with reuse_metadata_across_groups():
        md = full.build(0, c)
        assert full2.build(0, c).attention_mask_stacks is md.attention_mask_stacks
        # Another window is another spec.
        assert swa.build(0, c).attention_mask_stacks is not md.attention_mask_stacks
        # A draft's builder opts out.
        assert draft.build(0, c).attention_mask_stacks is not md.attention_mask_stacks
    assert full2.build(0, c).attention_mask_stacks is not md.attention_mask_stacks


def _prepare_attn(builders, batch, block_tables):
    state = SpyreDefaultModelState(
        make_config(),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    groups = [
        [
            AttentionGroup(
                backend=None,  # ty: ignore[invalid-argument-type]
                layer_names=[f"layer.{i}"],
                kv_cache_spec=b.kv_cache_spec,
                kv_cache_group_id=i,
                metadata_builders=[b],
            )
        ]
        for i, b in enumerate(builders)
    ]
    slot_mappings = torch.arange(len(builders) * batch.num_tokens_after_padding).view(
        len(builders), -1
    )
    return state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        groups,
        SimpleNamespace(kv_cache_groups=[None] * len(builders)),
    )


_CASES = [([1] * 4, [129, 699, 1499, 1999]), ([1, 1, 300], [899, 1199, 512]), ([512], [1024])]


def _check_prepare_attn_shares(window, query_lens, num_computed):
    g = torch.Generator().manual_seed(2)
    block_tables = tuple(
        torch.randint(1, 5000, (len(query_lens), 16), dtype=torch.int32, generator=g)
        for _ in range(3)
    )
    batch = make_input_batch(query_lens, num_computed=num_computed)
    # Groups 0 and 2 share a spec; group 1's window differs from theirs.
    windows = [window, 256 if window is not None else 1024, window]
    md = _prepare_attn([_builder(w) for w in windows], batch, block_tables)
    assert md["layer.2"].attention_mask_stacks is md["layer.0"].attention_mask_stacks
    assert md["layer.1"].attention_mask_stacks is not md["layer.0"].attention_mask_stacks
    seq_lens = [c + q for c, q in zip(num_computed, query_lens)]
    for i, layer in enumerate(md):
        ref = _builder(windows[i]).build(0, _common(query_lens, seq_lens, block_tables[i]))
        for f in dataclasses.fields(ref):
            # The runner's slot mappings are its own; update_block_table's are checked above.
            if f.name != "slot_mapping":
                _eq(getattr(md[layer], f.name), getattr(ref, f.name), f"{layer}.{f.name}")


@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize("query_lens,num_computed", _CASES)
def test_runner_shares_metadata_across_same_spec_groups(
    default_vllm_config, monkeypatch, window, query_lens, num_computed
):
    # A runner that gates the reuse on supports_update_block_table alone; v0.33's
    # also requires a Mamba spec.
    monkeypatch.setattr(attn_utils, "MambaSpec", object, raising=False)
    monkeypatch.delattr(SpyreDefaultModelState, "prepare_attn", raising=False)
    _check_prepare_attn_shares(window, query_lens, num_computed)


@pytest.mark.parametrize("window", [None, 512])
@pytest.mark.parametrize("query_lens,num_computed", _CASES)
def test_prepare_attn_shares_as_the_runner_would(
    default_vllm_config, window, query_lens, num_computed
):
    """SpyreDefaultModelState.prepare_attn's scope on v0.33's runner."""
    _check_prepare_attn_shares(window, query_lens, num_computed)
