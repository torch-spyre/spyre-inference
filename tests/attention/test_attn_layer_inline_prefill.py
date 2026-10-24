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

"""CPU checks for the fresh-prompt prefill attention that attn_layer traces in."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.attn_helpers import _build_metadata

from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends import spyre_attn
from spyre_inference.v1.attention.backends.spyre_encoder_attn import _encoder_rect_kernel

pytestmark = pytest.mark.attention


@pytest.mark.parametrize("query_lens,num_tokens", [([64], 64), ([5, 40, 3], 64), ([7], 16)])
def test_block_causal_mask_matches_per_sequence_causal_attention(query_lens, num_tokens):
    torch.manual_seed(0)
    heads, kv_heads, d = 8, 2, 64
    q = torch.randn(num_tokens, heads, d)
    k = torch.randn(num_tokens, kv_heads, d)
    v = torch.randn(num_tokens, kv_heads, d)
    mask = attn_layer.prefill_block_causal_mask(query_lens, num_tokens, torch.float32)
    got = _encoder_rect_kernel(q, k, v, mask, d**-0.5, 1, num_tokens, heads, kv_heads, d)

    want, start = [], 0
    for n in query_lens:
        qs, ks, vs = (t[start : start + n].transpose(0, 1) for t in (q, k, v))
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
        want.append(out.transpose(0, 1))
        start += n
    torch.testing.assert_close(got[:start], torch.cat(want), rtol=1e-5, atol=1e-5)
    # A padded row attends only to itself: finite, and its own value row.
    torch.testing.assert_close(got[start:], v[start:].repeat_interleave(heads // kv_heads, 1))


def _holder() -> attn_layer.SlotMapping:
    impl = Mock(spec=["kv_slot_views", "kv_write_index", "model_dtype"])
    impl.model_dtype = torch.float16
    impl.kv_write_index.side_effect = lambda slots, device: slots
    layer = Mock(spec=["impl", "kv_cache"])
    layer.impl, layer.kv_cache = impl, [torch.zeros(1)]
    holder = attn_layer.SlotMapping([layer])
    holder.inline_prefill = True
    return holder


def test_only_fresh_prompt_steps_publish_the_mask(default_vllm_config, monkeypatch):
    holder = _holder()
    monkeypatch.setattr(spyre_attn.attn_layer, "install", lambda layers: holder)
    pages = torch.arange(8, dtype=torch.int32).view(2, 4)

    def step(query_lens, seq_lens):
        qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
        _build_metadata(
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            query_start_loc=qsl,
            block_table=pages[: len(seq_lens)],
            slot_mapping=torch.arange(sum(query_lens), dtype=torch.int64),
        )
        return holder.prefill_mask

    fresh = step([40, 24], [40, 24])
    assert fresh is not None and fresh.shape == (1, 1, 64, 64)
    assert step([1, 1], [41, 25]) is None  # pure decode
    assert step([1, 30], [41, 30]) is None  # mixed: a decode row reads its pages
    assert step([32], [96]) is None  # chunked continuation reads earlier chunks
    assert step([40, 24], [40, 24]) is fresh  # same layout: no second build

    # Dummy runs clear it, unless warmup asks for the one-prompt variant.
    holder.publish_null(64)
    assert holder.prefill_mask is None
    holder.publish_null(64, inline_prefill=True)
    assert holder.prefill_mask is not None and holder.prefill_mask.shape[-1] == 64
    holder.publish_null(attn_layer.INLINE_PREFILL_MAX_TOKENS * 2, inline_prefill=True)
    assert holder.prefill_mask is None


def test_inline_forward_still_writes_the_cache_and_skips_the_opaque_op(monkeypatch):
    heads, kv_heads, d, t = 8, 2, 64, 16
    layer = Mock()
    layer.num_heads, layer.num_kv_heads, layer.head_size, layer.head_size_v = heads, kv_heads, d, d
    layer.layer_name = "layers.0.self_attn"
    layer.impl.scale = d**-0.5
    layer.impl.staging_buffers = None
    layer.spyre_slots.slots = torch.arange(t)
    layer.spyre_slots.prefill_mask = attn_layer.prefill_block_causal_mask([t], t, torch.float32)
    opaque = Mock()
    monkeypatch.setattr(torch.ops.vllm, "unified_attention_with_output", opaque)
    q, k, v = torch.randn(t, heads * d), torch.randn(t, kv_heads * d), torch.randn(t, kv_heads * d)

    out = attn_layer._spyre_attention_forward(layer, q, k, v)

    layer.impl.do_kv_cache_update.assert_called_once()
    opaque.assert_not_called()
    want = _encoder_rect_kernel(
        q.view(t, heads, d),
        k.view(t, kv_heads, d),
        v.view(t, kv_heads, d),
        layer.spyre_slots.prefill_mask,
        d**-0.5,
        1,
        t,
        heads,
        kv_heads,
        d,
    )
    torch.testing.assert_close(out, want.reshape(t, heads * d))

    # A caller asking for another output shape takes the opaque op.
    layer.impl.do_kv_cache_update.reset_mock()
    attn_layer._spyre_attention_forward(layer, q, k, v, output_shape=torch.Size((t, heads * d)))
    opaque.assert_called_once()


def test_fp8_cache_keeps_prefill_opaque(monkeypatch):
    heads, kv_heads, width, tokens = 8, 2, 128, 16
    layer = Mock()
    layer.attn_type = attn_layer.AttentionType.DECODER
    layer.kv_sharing_target_layer_name = None
    layer.query_quant = None
    layer.sliding_window = None
    layer.num_heads, layer.num_kv_heads = heads, kv_heads
    layer.head_size = layer.head_size_v = width
    layer.layer_name = "layers.0.self_attn"
    layer.kv_cache = [torch.zeros(1)]
    layer.impl._compile_attn = True
    layer.impl.kv_cache_fp8 = True
    layer.impl.alibi_slopes = None
    layer.impl.logits_soft_cap = 0.0
    layer.impl.staging_buffers = None
    layer.impl.do_kv_cache_update.return_value = torch.ones(1)

    holder = attn_layer.install([layer])
    assert not holder.inline_prefill
    holder.publish_prefill([tokens], tokens)
    assert holder.prefill_mask is None
    holder.slots = torch.arange(tokens)

    opaque = Mock(side_effect=lambda q, k, v, out, *args, **kwargs: out.zero_())
    monkeypatch.setattr(torch.ops.vllm, "unified_attention_with_output", opaque)
    query = torch.randn(tokens, heads * width)
    key = torch.randn(tokens, kv_heads * width)
    value = torch.randn_like(key)
    result = layer.forward(query, key, value)

    layer.impl.do_kv_cache_update.assert_called_once()
    opaque.assert_called_once()
    assert result.shape == query.shape
    assert torch.count_nonzero(result) == 0


def test_inline_prefill_requires_a_full_logical_rectangle():
    holder = _holder()
    holder.publish_prefill([512], 512)
    full_prompt_mask = holder.prefill_mask
    assert full_prompt_mask is not None
    assert full_prompt_mask.shape == (1, 1, 512, 512)

    holder.publish_prefill([64], 512)
    assert holder.prefill_mask is None
    holder.publish_prefill([1], 1)
    assert holder.prefill_mask is None

    holder.publish_prefill([512], 512)
    assert holder.prefill_mask is full_prompt_mask


def _compact_holder(kv_buckets=(128, 256, 512)):
    from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

    holder = _holder()
    bucketer = object.__new__(SpyreAttnBucketer)
    bucketer._kv_buckets = list(kv_buckets)
    holder.prefill_bucketer = bucketer
    return holder


@pytest.mark.parametrize("logical,kv_tokens", [(64, 128), (128, 128), (129, 256)])
def test_compact_single_prompt_mask_matches_causal_gqa(logical, kv_tokens):
    holder = _compact_holder()
    holder.publish_prefill([logical], 512)
    assert holder.prefill_mask is not None
    assert holder.prefill_mask.shape == (1, 1, 512, kv_tokens)
    mask = attn_layer.prefill_block_causal_mask([logical], 512, torch.float32, kv_tokens)
    assert torch.all(mask[..., logical:] < 0)
    assert torch.all(mask[..., logical:, 0] == 0)
    assert torch.all(mask[..., logical:, 1:] < 0)
    assert torch.all((mask == 0).any(-1))
    torch.manual_seed(121)
    q = torch.randn(512, 8, 128)
    k, v = torch.randn(512, 2, 128), torch.randn(512, 2, 128)
    got = _encoder_rect_kernel(
        q,
        k[:kv_tokens],
        v[:kv_tokens],
        mask,
        128**-0.5,
        1,
        512,
        8,
        2,
        128,
        kv_extent=kv_tokens,
    )
    want = F.scaled_dot_product_attention(
        q[:logical].transpose(0, 1),
        k[:logical].transpose(0, 1),
        v[:logical].transpose(0, 1),
        is_causal=True,
        enable_gqa=True,
    ).transpose(0, 1)
    torch.testing.assert_close(got[:logical], want, rtol=1e-5, atol=1e-5)
    full_mask = attn_layer.prefill_block_causal_mask([logical], 512, torch.float32)
    full = _encoder_rect_kernel(q, k, v, full_mask, 128**-0.5, 1, 512, 8, 2, 128)
    torch.testing.assert_close(got[:logical], full[:logical], rtol=1e-5, atol=1e-5)
    expected_padding = v[0].repeat_interleave(4, 0).expand(512 - logical, -1, -1)
    torch.testing.assert_close(got[logical:], expected_padding, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(got).all()
    assert got.shape == q.shape


def test_compact_prefill_keeps_unsupported_layouts_opaque():
    holder = _compact_holder()
    for lengths, rows in [([32, 32], 512), ([1], 512), ([300], 512), ([513], 512)]:
        holder.publish_prefill(lengths, rows)
        assert holder.prefill_mask is None
    holder = _compact_holder((256, 512))
    holder.publish_prefill([64], 512)
    assert holder.prefill_mask.shape == (1, 1, 512, 256)
    holder = _compact_holder((512,))
    holder.publish_prefill([64], 512)
    assert holder.prefill_mask is None
    holder.publish_prefill([512], 512)
    assert holder.prefill_mask.shape == (1, 1, 512, 512)


def test_compact_forward_writes_full_cache_before_bounding_operands(monkeypatch):
    import spyre_inference.v1.attention.backends.spyre_encoder_attn as encoder

    holder = _compact_holder()
    holder.publish_prefill([64], 512)
    holder.slots = torch.arange(512)
    layer = Mock()
    layer.num_heads, layer.num_kv_heads = 8, 2
    layer.head_size = layer.head_size_v = 128
    layer.layer_name = "compact.test"
    layer.impl.scale = 128**-0.5
    layer.impl.staging_buffers = None
    layer.spyre_slots = holder
    events = []

    def update(_layer, key, value, _cache, slots):
        assert key.shape == value.shape == (512, 2, 128)
        assert slots.shape == (512,)
        events.append("cache")
        return torch.ones(1)

    def rectangle(query, key, value, mask, *args, kv_extent=None):
        assert events == ["cache"]
        assert query.shape == (512, 8, 128)
        assert key.shape == value.shape == (128, 2, 128)
        assert mask.shape == (1, 1, 512, 128)
        assert kv_extent == 128
        assert key.data_ptr() == original_key.data_ptr()
        assert value.data_ptr() == original_value.data_ptr()
        events.append("attention")
        return torch.zeros_like(query)

    layer.impl.do_kv_cache_update.side_effect = update
    monkeypatch.setattr(encoder, "_encoder_rect_kernel", rectangle)
    opaque = Mock()
    monkeypatch.setattr(torch.ops.vllm, "unified_attention_with_output", opaque)
    query = torch.randn(512, 1024)
    original_key, original_value = torch.randn(512, 256), torch.randn(512, 256)
    saved = [x.clone() for x in (query, original_key, original_value)]
    result = attn_layer._spyre_attention_forward(layer, query, original_key, original_value)
    assert events == ["cache", "attention"]
    assert result.shape == query.shape
    assert all(torch.equal(x, y) for x, y in zip((query, original_key, original_value), saved))
    opaque.assert_not_called()


def test_compact_warmup_covers_group_bucket_pairs_and_clears_on_error():
    holders = [_compact_holder((128, 256, 512)), _compact_holder((256, 512))]
    previous = list(attn_layer._holders)
    attn_layer._holders.clear()
    attn_layer._holders.update(holders)
    try:
        variants = attn_layer.inline_prefill_warmup_tokens(512)
        assert variants[0] == 512
        reachable = {
            tuple(h._prefill_kv_extent(n, 512) for h in holders) for n in range(2, 513)
        } - {(None, None)}
        warmed = {tuple(h._prefill_kv_extent(n, 512) for h in holders) for n in variants}
        assert warmed == reachable
        with (
            pytest.raises(RuntimeError, match="warmup failure"),
            attn_layer.inline_prefill_warmup(2),
        ):
            for holder in holders:
                holder.publish_null(512, inline_prefill=True)
                assert holder.prefill_mask.shape[-1] in (128, 256)
            raise RuntimeError("warmup failure")
        assert all(h._prefill_warmup_tokens is None and h.prefill_mask is None for h in holders)
    finally:
        attn_layer._holders.clear()
        attn_layer._holders.update(previous)
