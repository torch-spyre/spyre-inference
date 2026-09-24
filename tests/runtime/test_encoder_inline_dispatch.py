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

"""Which encoder layers take the traced path, and that the traced path agrees with the
opaque one.

CPU-only: the question here is dispatch and eligibility, not device numerics, so these
run anywhere. ``test_spyre_encoder_attn.py`` covers the kernels themselves.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig
from vllm.v1.attention.backend import AttentionType

from spyre_inference.v1.attention.backends import spyre_encoder_attn as enc
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_LEN_ALIGNMENT,
    EncoderGroupPlan,
    EncoderRectPlan,
    SpyreEncoderAttentionImpl,
    _encoder_rect_kernel,
    encoder_grid,
    encoder_inline_active,
    encoder_key_pad_mask,
    install_encoder,
    publish_encoder_grid,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_rectangles

pytestmark = pytest.mark.runtime


@pytest.fixture(autouse=True)
def clean_inline_state():
    """The holder and the bound-layer set are module state, so reset around each test."""
    enc._inline_layers.clear()
    encoder_grid().publish(None)
    yield
    enc._inline_layers.clear()
    encoder_grid().publish(None)


def _config(max_model_len=512, max_num_seqs=4, max_num_batched_tokens=2048) -> VllmConfig:
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return config


def _impl(config, *, head_size=64, num_heads=4, num_kv_heads=1, compiled=True):
    with set_current_vllm_config(config):
        impl = SpyreEncoderAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size**-0.5,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )
    impl._compile_attn = compiled
    return impl


class _FakeLayer:
    """The attributes ``install_encoder`` and the traced forward read off ``Attention``."""

    def __init__(self, impl, attn_type=AttentionType.ENCODER_ONLY):
        self.impl = impl
        self.head_size = impl.head_size
        self.head_size_v = impl.head_size
        self.num_heads = impl.num_heads
        self.num_kv_heads = impl.num_kv_heads
        self.attn_type = attn_type


def _layer(**kwargs):
    attn_type = kwargs.pop("attn_type", AttentionType.ENCODER_ONLY)
    return _FakeLayer(_impl(_config(), **kwargs), attn_type=attn_type)


def _bound(layer) -> bool:
    return "forward" in vars(layer)


class TestEligibility:
    def test_binds_a_stick_aligned_compiled_encoder_layer(self):
        layer = _layer(head_size=64)
        install_encoder([layer])
        assert _bound(layer)
        assert encoder_inline_active()

    def test_refuses_a_sub_stick_head_size(self):
        """D=32 needs ``_widen_head_dim``'s host round trip, which cannot be traced."""
        layer = _layer(head_size=32)
        assert layer.head_size % ENCODER_LEN_ALIGNMENT
        install_encoder([layer])
        assert not _bound(layer)
        assert not encoder_inline_active()

    def test_refuses_eager_attention(self):
        layer = _layer(compiled=False)
        install_encoder([layer])
        assert not _bound(layer)

    def test_refuses_mismatched_head_size_v(self):
        layer = _layer()
        layer.head_size_v = layer.head_size // 2
        install_encoder([layer])
        assert not _bound(layer)

    def test_ignores_decoder_layers(self):
        layer = _layer(attn_type=AttentionType.DECODER)
        install_encoder([layer])
        assert not _bound(layer)
        assert not encoder_inline_active()

    def test_one_ineligible_layer_refuses_the_whole_stack(self):
        """Every layer in a stack shares the block graph, so they take one path together."""
        good, bad = _layer(head_size=64), _layer(head_size=32)
        install_encoder([good, bad])
        assert not _bound(good)
        assert not _bound(bad)


class TestGridPublication:
    def test_the_mask_shape_carries_the_grid(self):
        """``(width, extent)`` is read out of ``mask.shape``, for every declared rectangle."""
        config = _config(max_model_len=512, max_num_seqs=32)
        rectangles = encoder_rectangles(config)
        assert rectangles
        for extent, width in rectangles:
            mask = encoder_key_pad_mask(extent, [extent] * width, torch.float16)
            assert tuple(mask.shape) == (width, 1, 1, extent)

    def test_a_rectangular_plan_publishes_its_mask(self):
        mask = encoder_key_pad_mask(128, [128] * 16, torch.float16)
        publish_encoder_grid(
            EncoderRectPlan(extent=128, width=16, mask=mask, query_lens=[128] * 16)
        )
        assert encoder_grid().mask is mask

    def test_a_ragged_plan_publishes_nothing(self):
        """The ragged path must never be traced in: its dispatch count varies per step."""
        publish_encoder_grid(
            [
                EncoderGroupPlan(
                    starts=[0],
                    query_lens=[64],
                    extent=64,
                    row_table=torch.arange(64, dtype=torch.int64),
                    mask=encoder_key_pad_mask(64, [64], torch.float16),
                )
            ]
        )
        assert encoder_grid().mask is None

    def test_no_plan_publishes_nothing(self):
        publish_encoder_grid(None)
        assert encoder_grid().mask is None


class TestTracedForward:
    def _inputs(self, layer, rows, seed=0):
        torch.manual_seed(seed)
        heads, kv_heads, dim = layer.num_heads, layer.num_kv_heads, layer.head_size
        return (
            torch.randn(rows, heads * dim, dtype=torch.float16),
            torch.randn(rows, kv_heads * dim, dtype=torch.float16),
            torch.randn(rows, kv_heads * dim, dtype=torch.float16),
        )

    def test_falls_through_to_upstream_without_a_grid(self, monkeypatch):
        layer = _layer()
        install_encoder([layer])
        called = []

        def fake_upstream(*args, **kwargs):
            called.append(args)
            return "upstream"

        monkeypatch.setattr(enc, "_ORIG_ATTENTION_FORWARD", fake_upstream)
        assert encoder_grid().mask is None
        assert layer.forward(*self._inputs(layer, 64)) == "upstream"
        assert len(called) == 1

    def test_falls_through_on_a_foreign_output_dtype(self, monkeypatch):
        layer = _layer()
        install_encoder([layer])
        publish_encoder_grid(
            EncoderRectPlan(
                extent=64,
                width=4,
                mask=encoder_key_pad_mask(64, [64] * 4, torch.float16),
                query_lens=[64] * 4,
            )
        )
        monkeypatch.setattr(enc, "_ORIG_ATTENTION_FORWARD", lambda *a, **k: "upstream")
        query, key, value = self._inputs(layer, 256)
        assert layer.forward(query, key, value, output_dtype=torch.float32) == "upstream"

    def test_falls_through_on_a_requested_output_shape(self, monkeypatch):
        """The inline path returns [tokens, heads * head_size] and ignores output_shape.

        Even a shape equal to what it would return takes the fallthrough, so the contract
        is "no output_shape" rather than "a compatible one".
        """
        layer = _layer()
        install_encoder([layer])
        monkeypatch.setattr(enc, "_ORIG_ATTENTION_FORWARD", lambda *a, **k: "upstream")
        extent, width = 64, 4
        publish_encoder_grid(
            EncoderRectPlan(
                extent=extent,
                width=width,
                mask=encoder_key_pad_mask(extent, [extent] * width, torch.float16),
                query_lens=[extent] * width,
            )
        )
        query, key, value = self._inputs(layer, extent * width)
        wanted = torch.Size((extent * width, layer.num_heads * layer.head_size))
        assert layer.forward(query, key, value, output_shape=wanted) == "upstream"

    def test_inlined_result_matches_the_kernel_the_impl_runs(self):
        """Parity: the traced path and the opaque path must be the same computation."""
        layer = _layer()
        install_encoder([layer])
        extent, width = 64, 4
        rows = extent * width
        mask = encoder_key_pad_mask(extent, [extent] * width, torch.float16)
        publish_encoder_grid(
            EncoderRectPlan(extent=extent, width=width, mask=mask, query_lens=[extent] * width)
        )
        query, key, value = self._inputs(layer, rows)
        heads, kv_heads, dim = layer.num_heads, layer.num_kv_heads, layer.head_size

        traced = layer.forward(query, key, value)
        reference = _encoder_rect_kernel(
            query.view(-1, heads, dim),
            key.view(-1, kv_heads, dim),
            value.view(-1, kv_heads, dim),
            mask,
            layer.impl.scale,
            width,
            extent,
            heads,
            kv_heads,
            dim,
        ).reshape(-1, heads * dim)

        assert traced.shape == (rows, heads * dim)
        assert torch.equal(traced, reference)

    def test_inlined_result_matches_the_impl_forward(self):
        """End to end against the opaque path, which routes through ``impl.forward``."""
        layer = _layer()
        install_encoder([layer])
        extent, width = 128, 2
        rows = extent * width
        query_lens = [extent, extent // 2]
        kv_lens = query_lens + [1] * (width - len(query_lens))
        mask = encoder_key_pad_mask(extent, kv_lens, torch.float16)
        plan = EncoderRectPlan(extent=extent, width=width, mask=mask, query_lens=query_lens)
        publish_encoder_grid(plan)
        query, key, value = self._inputs(layer, rows, seed=3)
        heads, kv_heads, dim = layer.num_heads, layer.num_kv_heads, layer.head_size

        traced = layer.forward(query, key, value)

        output = torch.zeros((rows, heads * dim), dtype=torch.float16).view(-1, heads, dim)
        metadata = SimpleNamespace(encoder_plan=plan)
        opaque = layer.impl.forward(
            layer,
            query.view(-1, heads, dim),
            key.view(-1, kv_heads, dim),
            value.view(-1, kv_heads, dim),
            None,
            metadata,
            output,
        )
        assert torch.equal(traced, opaque.reshape(-1, heads * dim))


class TestForcedWarmupPlan:
    """The plan warmup declares per rectangle, which the dummy batch cannot produce itself.

    Upstream's ``_dummy_run`` splits its tokens evenly over ``max_num_seqs``, so it only
    ever lands on the shortest rectangle. Forcing the rest is what gets each one's block
    graph compiled before serving.
    """

    @staticmethod
    def _plan(rect):
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = SimpleNamespace(
            _forced_encoder_rect=rect,
            _model_dtype=lambda: torch.float16,
            _spyre_device=torch.device("cpu"),
        )
        return TorchSpyreModelRunner._forced_encoder_plan(runner)

    def test_no_rect_forced_means_no_plan(self):
        assert self._plan(None) is None

    @pytest.mark.parametrize(("extent", "width"), encoder_rectangles(_config(512, 32, 2048)))
    def test_fills_the_body_exactly(self, extent, width):
        """``sum(query_lens) == width * extent`` keeps ``_preprocess``'s expansion an
        identity and lets ``_unpad_encoder_hidden`` short-circuit, so the dummy's buffers
        stay self-consistent."""
        plan = self._plan((extent, width))
        assert isinstance(plan, EncoderRectPlan)
        assert (plan.extent, plan.width) == (extent, width)
        assert sum(plan.query_lens) == width * extent
        assert tuple(plan.mask.shape) == (width, 1, 1, extent)

    def test_every_key_is_real(self):
        """A forced plan is all real tokens, so no lane needs the batch-pad workaround."""
        plan = self._plan((64, 4))
        assert torch.equal(plan.mask, torch.zeros_like(plan.mask))
