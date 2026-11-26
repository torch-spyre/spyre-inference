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

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.models.gemma4 import SpyreGemma4ForCausalLM
from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3Model
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend
from spyre_inference.v1.worker.kv_cache_setup import split_attn_groups_by_window


@pytest.mark.parametrize("draft_window", [1024, 2048])
def test_gemma4_draft_kv_group_preserves_windows_and_head_shapes(draft_window):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    common = dict(block_size=128, dtype=torch.float16)
    specs = {
        "target.sliding": SlidingWindowSpec(
            num_kv_heads=8, head_size=256, sliding_window=1024, **common
        ),
        "target.full": FullAttentionSpec(num_kv_heads=2, head_size=512, **common),
        "draft.sliding": SlidingWindowSpec(
            num_kv_heads=8, head_size=128, sliding_window=draft_window, is_draft=True, **common
        ),
        "draft.full": FullAttentionSpec(num_kv_heads=8, head_size=128, is_draft=True, **common),
    }
    original = dict(specs)
    groups = SpyreGemma4ForCausalLM.get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert set(groups[0].layer_names) == set(specs)
    grouped = groups[0].kv_cache_spec
    assert isinstance(grouped, UniformTypeKVCacheSpecs)
    assert specs == original
    for name, spec in grouped.kv_cache_specs.items():
        before = original[name]
        assert isinstance(spec, FullAttentionSpec)
        assert spec.sliding_window == getattr(before, "sliding_window", None)
        assert (spec.num_kv_heads, spec.head_size, spec.is_draft) == (
            before.num_kv_heads,
            before.head_size,
            before.is_draft,
        )
        assert spec.max_memory_usage_bytes(config) == 32 * spec.page_size_bytes
    if draft_window == 1024:
        assert groups == get_kv_cache_groups(config, dict(original))


def test_qwen3_mixed_draft_groups_preserve_target_full_attention():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    common = dict(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.float16)
    specs = {
        "target.full": FullAttentionSpec(**common),
        "draft.sliding": SlidingWindowSpec(sliding_window=2048, is_draft=True, **common),
        "draft.full": FullAttentionSpec(is_draft=True, **common),
    }
    groups = get_kv_cache_groups(config, dict(specs))
    assert len(groups) == 1
    merged = groups[0].kv_cache_spec
    assert isinstance(merged, FullAttentionSpec)
    assert merged.sliding_window == 2048
    layers = {
        name: SimpleNamespace(sliding_window=getattr(spec, "sliding_window", None))
        for name, spec in specs.items()
    }
    # The target runner sees all layers; the proposer rebuilds only the draft groups.
    for names in (list(specs), [name for name in specs if name.startswith("draft.")]):
        attn_groups = [[AttentionGroup(SpyreAttentionBackend, names, merged, 0)]]
        split_attn_groups_by_window(attn_groups, layers)
        assert len(attn_groups[0]) == 2
        for group in attn_groups[0]:
            assert group.kv_cache_group_id == 0
            for name in group.layer_names:
                assert group.kv_cache_spec.sliding_window == layers[name].sliding_window
                assert group.kv_cache_spec.num_kv_heads == specs[name].num_kv_heads
                assert group.kv_cache_spec.head_size == specs[name].head_size


@pytest.mark.parametrize("rows", [1, 16, 64])
def test_context_projection_packs_live_transposed_weights_without_query_columns(monkeypatch, rows):
    torch.manual_seed(93)
    attentions, references = [], []
    for _ in range(2):
        query = torch.randn(256, 192)
        key, value = torch.randn(2, 128, 192)
        attention = torch.nn.Module()
        attention.qkv_proj = torch.nn.Module()
        attention.qkv_proj.weight = torch.nn.Parameter(
            torch.cat((query, key, value)).t().contiguous()
        )
        attention.q_size, attention.kv_size = 256, 128
        attention.num_kv_heads, attention.head_dim = 2, 64
        attention.k_norm = torch.nn.Identity()
        attention.rotary_emb = lambda positions, query, key: (query, key)
        attentions.append(attention)
        references.append((key, value))
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attention) for attention in attentions],
        hidden_norm=torch.nn.Identity(),
    )
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    SpyreDFlashQwen3Model.prepare_for_spyre(model)
    context = torch.randn(rows, 192)
    for attention, reference in zip(attentions, references):
        assert attention._context_kv_weight_t.shape == (192, 256)
        assert attention._context_kv_weight_t.is_contiguous()
        assert "_context_kv_weight_t" not in attention.state_dict()
        actual = model._context_project(attention, context, torch.arange(rows))
        for result, weight in zip(actual, reference):
            expected = torch.nn.functional.linear(context, weight).view(rows, 2, 64)
            torch.testing.assert_close(result, expected)
