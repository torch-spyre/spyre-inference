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
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config
from vllm.config import get_current_vllm_config
from vllm.model_executor.models import qwen3_dflash
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.models.gemma4 import SpyreGemma4ForCausalLM
from spyre_inference.models.qwen3_dflash import (
    SpyreDFlashQwen3DecoderLayer,
    SpyreDFlashQwen3Model,
)
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionBackend
from spyre_inference.v1.spec_decode.dflash import SpyreDFlashProposer
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


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("causal", [None, False, True])
@pytest.mark.parametrize("override", [{}, {"use_swa": True}, {"causal": True}, {"causal": False}])
@pytest.mark.parametrize(
    "layer_types",
    [
        ["full_attention"] * 3,
        ["sliding_attention"] * 3,
        ["sliding_attention", "full_attention", "sliding_attention"],
    ],
)
def test_layer_attention_matches_upstream_v2(monkeypatch, wrapped, causal, override, layer_types):
    config = Qwen3Config(
        num_hidden_layers=3,
        layer_types=layer_types,
        sliding_window=192,
        use_sliding_window=True,
        architectures=["Qwen3XPressModel"],
        dflash_config=override,
        is_causal=causal,
    )
    if wrapped:
        config = EAGLEConfig(config, method="dflash")
    before = config.to_dict()
    current = SimpleNamespace(use_v2_model_runner=True)
    monkeypatch.setattr(qwen3_dflash, "get_current_vllm_config", lambda: current)
    expected = [qwen3_dflash._resolve_layer_attention(config, i) for i in range(3)]
    current.use_v2_model_runner = False
    actual = []
    original_config = config

    def capture(self, *args, config, layer_idx, **kwargs):
        assert config is original_config
        actual.append(qwen3_dflash._resolve_layer_attention(config, layer_idx))

    monkeypatch.setattr(qwen3_dflash.DFlashQwen3DecoderLayer, "__init__", capture)
    assert SpyreDFlashQwen3Model.decoder_layer_cls is SpyreDFlashQwen3DecoderLayer
    for index in range(3):
        SpyreDFlashQwen3DecoderLayer(None, config=config, layer_idx=index)
    assert actual == expected
    assert config.to_dict() == before
    assert qwen3_dflash.get_current_vllm_config() is current
    assert current.use_v2_model_runner is False


def test_layer_attention_gate_is_restored_after_initialization_error(monkeypatch):
    current = SimpleNamespace(use_v2_model_runner=False)
    monkeypatch.setattr(qwen3_dflash, "get_current_vllm_config", lambda: current)

    def fail(self, *args, **kwargs):
        assert qwen3_dflash.get_current_vllm_config().use_v2_model_runner
        raise ValueError("invalid draft layer")

    monkeypatch.setattr(qwen3_dflash.DFlashQwen3DecoderLayer, "__init__", fail)
    with pytest.raises(ValueError, match="invalid draft layer"):
        SpyreDFlashQwen3DecoderLayer(None)
    assert qwen3_dflash.get_current_vllm_config() is current
    assert current.use_v2_model_runner is False


@pytest.mark.parametrize("causal", [False, True])
def test_draft_backend_causality_is_checked_at_initialization(monkeypatch, causal):
    group = SimpleNamespace(
        layer_names=["draft.attn"],
        kv_cache_spec=None,
        backend=SimpleNamespace(
            supports_non_causal=lambda: False, get_name=lambda: "causal-only backend"
        ),
        create_metadata_builders=Mock(),
    )

    def initialize(self, kv_cache_config, kernel_block_sizes):
        self.kv_cache_gid, self.block_size = 0, 64
        self.draft_attn_groups = [group]

    monkeypatch.setattr(DFlashProposer, "initialize_attn_backend", initialize)
    drafter = object.__new__(SpyreDFlashProposer)
    drafter.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    drafter.device = torch.device("cpu")
    drafter.model = SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: group.layer_names,
        get_draft_attn_causal=lambda: [causal],
    )
    cache = SimpleNamespace(kv_cache_groups=[None])
    if causal:
        drafter.initialize_attn_backend(cache)
        group.create_metadata_builders.assert_called_once()
    else:
        with pytest.raises(ValueError, match="causal-only backend.*non-causal DFlash"):
            drafter.initialize_attn_backend(cache)
        group.create_metadata_builders.assert_not_called()


@pytest.mark.parametrize("causal_override", [None, False, True])
@pytest.mark.parametrize(
    "context,tokens,window",
    [(63, 1, 64), (64, 1, 64), (128, 1, 64), (96, 16, 64), (96, 16, 7), (6, 250, 128)],
)
def test_draft_groups_keep_windows_and_causality(
    monkeypatch, default_vllm_config, causal_override, context, tokens, window
):
    config = get_current_vllm_config()
    monkeypatch.setattr(config.cache_config, "block_size", 64)
    monkeypatch.setattr(config.model_config, "max_model_len", 256)
    names = [f"draft.layers.{i}.self_attn.attn" for i in range(3)]
    windows = [window, None, window]
    causal = [True, False, True] if causal_override is None else [causal_override] * 3
    for name, window in zip(names, windows, strict=True):
        monkeypatch.setitem(
            config.compilation_config.static_forward_context,
            name,
            SimpleNamespace(
                attn_type=AttentionType.DECODER,
                impl=SimpleNamespace(),
                sliding_window=window,
            ),
        )
    spec = FullAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=64, dtype=torch.float16, sliding_window=window
    )

    def initialize(self, kv_cache_config, kernel_block_sizes):
        self.kv_cache_gid, self.block_size = 1, 64
        self.draft_attn_groups = [AttentionGroup(SpyreAttentionBackend, names, spec, 1)]

    monkeypatch.setattr(DFlashProposer, "initialize_attn_backend", initialize)
    drafter = object.__new__(SpyreDFlashProposer)
    drafter.vllm_config = config
    drafter.draft_model_config = SimpleNamespace(max_model_len=128)
    drafter.device = torch.device("cpu")
    drafter.model = SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: names,
        get_draft_attn_causal=lambda: causal,
    )
    drafter.initialize_attn_backend(SimpleNamespace(kv_cache_groups=[None, None]), [64, 64])
    assert len(drafter.draft_attn_groups) == 2
    assert drafter.draft_model_config.max_model_len == 128
    for group in drafter.draft_attn_groups:
        assert group.metadata_builders[0].attn_bucketer.find_kv_bucket(256) == 256
    kv_len = context + tokens
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, tokens], dtype=torch.int32),
        seq_lens=torch.tensor([kv_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=tokens,
        max_query_len=tokens,
        max_seq_len=kv_len,
        block_table_tensor=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        slot_mapping=torch.arange(context, kv_len) + 64,
        causal=False,
    )
    per_group, per_layer = drafter.build_per_group_and_layer_attn_metadata(common)
    assert len(per_group) == 2
    assert set(per_layer) == set(names)
    assert common.causal is False
    for group in drafter.draft_attn_groups:
        assert group.kv_cache_group_id == 1
        for name in group.layer_names:
            index = names.index(name)
            metadata = per_layer[name]
            assert metadata.causal is causal[index]
            assert group.metadata_builders[0].sliding_window == windows[index]
            tiles = metadata.attention_mask_stacks[0]
            active = metadata.active_block_indices
            blocks = active[0] if active is not None else range(tiles.shape[0])
            padded_kv = max((kv_len + 63) // 64, max(blocks) + 1) * 64
            mask = torch.zeros(tokens, padded_kv, dtype=torch.bool)
            for block, tile in zip(blocks, tiles):
                mask[:, block * 64 : (block + 1) * 64] |= tile[:tokens] == 0
            queries = torch.arange(context, kv_len)[:, None]
            keys = torch.arange(padded_kv)[None, :]
            expected = (keys < kv_len).expand(tokens, -1).clone()
            if causal[index]:
                expected &= keys <= queries
            if windows[index] is not None:
                expected &= keys >= queries - windows[index] + 1
                if not causal[index]:
                    expected &= keys < queries + windows[index]
            assert torch.equal(mask, expected)
