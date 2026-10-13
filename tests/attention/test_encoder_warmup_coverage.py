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

"""Encoder attention warmup coverage: every shape a batch can reach is declared, and
``warm_kernels`` traces exactly the declared set -- so nothing compiles mid-request.

CPU-only, and not about numerics -- ``test_spyre_encoder_attn.py`` covers those.
``warm_kernels`` runs on the first attention call of warmup's single body-shape dummy
run, against that call's own tensors, because a Spyre tensor's device layout is part of
its cache key.
"""

import pytest
import torch
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig

from spyre_inference.v1.attention.backends import spyre_encoder_attn
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_LEN_ALIGNMENT,
    EncoderRectPlan,
    SpyreEncoderAttentionImpl,
    build_encoder_plan,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_group_shapes,
    encoder_group_width_caps,
    encoder_rectangles,
    encoder_shape_tables,
)

# (max_model_len, max_num_seqs, max_num_batched_tokens). The first is the plan's
# worked example; the second is the narrow config, where no batch can miss the rectangular
# path and the group family is therefore empty.
_CONFIGS = [
    (512, 32, 2048),
    (512, 4, 2048),
    (256, 16, 1024),
]


def _config(max_model_len, max_num_seqs, max_num_batched_tokens) -> VllmConfig:
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return config


def _make_impl(config, num_heads=4, num_kv_heads=1, head_size=64):
    with set_current_vllm_config(config):
        return SpyreEncoderAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size**-0.5,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )


def _buffers(impl, rows, dtype=torch.float16):
    query = torch.zeros((rows, impl.num_heads, impl.head_size), dtype=dtype)
    key = torch.zeros((rows, impl.num_kv_heads, impl.head_size), dtype=dtype)
    value = torch.zeros((rows, impl.num_kv_heads, impl.head_size), dtype=dtype)
    # Built the way vLLM builds it: a view of a 2-D allocation.
    output = torch.zeros((rows, impl.num_heads * impl.head_size), dtype=dtype).view(
        -1, impl.num_heads, impl.head_size
    )
    return query, key, value, output


def _fake_metadata(query_lens):
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    return type(
        "FakeMetadata",
        (),
        {
            "query_start_loc": torch.tensor(starts, dtype=torch.int32),
            "seq_lens": torch.tensor(query_lens, dtype=torch.int32),
            "num_actual_tokens": starts[-1],
            "num_seqs": len(query_lens),
            "encoder_plan": None,
        },
    )()


class TestWarmKernelsTracesTheDeclaredSet:
    @pytest.mark.parametrize(("max_model_len", "max_num_seqs", "budget"), _CONFIGS)
    def test_every_declared_shape_and_nothing_else(
        self, monkeypatch, max_model_len, max_num_seqs, budget
    ):
        config = _config(max_model_len, max_num_seqs, budget)
        impl = _make_impl(config)
        rows = encoder_shape_tables(config).budget

        seen_rects: list[tuple[int, int]] = []
        seen_groups: list[tuple[int, int]] = []
        real_rect, real_fused = impl._run_rect, impl._run_fused

        def record_rect(out, q, k, v, mask, width, extent, *args, **kwargs):
            seen_rects.append((extent, width))
            return real_rect(out, q, k, v, mask, width, extent, *args, **kwargs)

        def record_fused(out, row_index, q, k, v, mask, group, *args, **kwargs):
            seen_groups.append((group, row_index.shape[0] // group))
            return real_fused(out, row_index, q, k, v, mask, group, *args, **kwargs)

        monkeypatch.setattr(impl, "_run_rect", record_rect)
        monkeypatch.setattr(impl, "_run_fused", record_fused)
        traced = impl.warm_kernels(*_buffers(impl, rows), impl.num_heads, impl.num_kv_heads, 64)

        assert seen_rects == encoder_rectangles(config)
        assert seen_groups == encoder_group_shapes(config)
        assert traced == len(seen_rects) + len(seen_groups)

    @pytest.mark.parametrize(("max_model_len", "max_num_seqs", "budget"), _CONFIGS)
    def test_rectangles_are_left_to_the_block_graph_when_inlined(
        self, monkeypatch, max_model_len, max_num_seqs, budget
    ):
        """Traced in, a rectangle's shape is a block graph the runner's dummy runs compile.

        The group shapes are still this method's: the ragged path stays behind the opaque op.
        """
        monkeypatch.setattr(spyre_encoder_attn, "encoder_inline_active", lambda: True)
        config = _config(max_model_len, max_num_seqs, budget)
        impl = _make_impl(config)

        seen_rects: list[tuple[int, int]] = []
        seen_groups: list[tuple[int, int]] = []
        monkeypatch.setattr(
            impl, "_run_rect", lambda out, q, k, v, m, w, e, *a: seen_rects.append((e, w))
        )
        monkeypatch.setattr(
            impl,
            "_run_fused",
            lambda out, rows, q, k, v, m, g, *a: seen_groups.append((g, rows.shape[0] // g)),
        )
        traced = impl.warm_kernels(
            *_buffers(impl, encoder_shape_tables(config).budget),
            impl.num_heads,
            impl.num_kv_heads,
            64,
        )

        assert seen_rects == []
        assert seen_groups == encoder_group_shapes(config)
        assert traced == len(seen_groups)

    def test_warmed_against_the_callers_own_output_tensor(self, monkeypatch):
        """Regression: the store's destination layout is part of its cache key.

        vLLM hands ``forward`` an output built as ``torch.empty(rows, H*D).view(-1, H,
        D)``. Warming against a fresh ``torch.zeros((rows, H, D))`` instead is the same
        shape with a different Spyre device layout, so the graph warmup compiled was
        not the one real traffic could reuse.
        """
        config = _config(512, 32, 2048)
        impl = _make_impl(config)
        query, key, value, output = _buffers(impl, encoder_shape_tables(config).budget)

        seen_out = []
        real_rect, real_fused = impl._run_rect, impl._run_fused

        def record_rect(out, *args, **kwargs):
            seen_out.append(out)
            return real_rect(out, *args, **kwargs)

        def record_fused(out, *args, **kwargs):
            seen_out.append(out)
            return real_fused(out, *args, **kwargs)

        monkeypatch.setattr(impl, "_run_rect", record_rect)
        monkeypatch.setattr(impl, "_run_fused", record_fused)
        impl.warm_kernels(query, key, value, output, impl.num_heads, impl.num_kv_heads, 64)

        assert seen_out, "warmup must exercise the store"
        assert all(o is output for o in seen_out), (
            "store must be warmed against the caller's output, not a substitute"
        )

    def test_warmed_against_the_callers_own_query_layout(self, monkeypatch):
        """Regression: a fused-QKV projection (``qkv.split(...)``) hands out a strided
        view, and that stride is part of the compiled kernel's cache key under
        ``dynamic=False``."""
        config = _config(512, 32, 2048)
        impl = _make_impl(config)
        rows = encoder_shape_tables(config).budget
        fused = torch.zeros((rows, 4 * 64 + 64 + 64), dtype=torch.float16)
        q_flat, k_flat, v_flat = fused.split([4 * 64, 64, 64], dim=-1)
        query = q_flat.view(rows, 4, 64)
        assert not query.is_contiguous(), "test setup must exercise a genuinely strided view"

        seen_strides: list[tuple] = []
        real_rect, real_fused = impl._run_rect, impl._run_fused

        def record_rect(out, q, *a, **k):
            seen_strides.append(q.stride())
            return real_rect(out, q, *a, **k)

        def record_fused(out, row_index, q, *a, **k):
            seen_strides.append(q.stride())
            return real_fused(out, row_index, q, *a, **k)

        monkeypatch.setattr(impl, "_run_rect", record_rect)
        monkeypatch.setattr(impl, "_run_fused", record_fused)
        impl.warm_kernels(
            query,
            k_flat.view(rows, 1, 64),
            v_flat.view(rows, 1, 64),
            torch.zeros((rows, 4 * 64), dtype=torch.float16).view(-1, 4, 64),
            impl.num_heads,
            impl.num_kv_heads,
            64,
        )

        assert seen_strides and all(s == query.stride() for s in seen_strides)

    def test_is_idempotent(self, monkeypatch):
        """Every layer calls it; only the first may trace."""
        config = _config(512, 32, 2048)
        impl = _make_impl(config)
        buffers = _buffers(impl, encoder_shape_tables(config).budget)
        first = impl.warm_kernels(*buffers, impl.num_heads, impl.num_kv_heads, 64)
        assert first > 0
        assert impl.warm_kernels(*buffers, impl.num_heads, impl.num_kv_heads, 64) == 0


class TestEveryReachableBatchLandsOnADeclaredShape:
    """The constraint that matters: the scheduler is upstream's, so for *any* batch it
    can form, every dispatch the backend makes must be a shape warmup traced."""

    @pytest.mark.parametrize(("max_model_len", "max_num_seqs", "budget"), _CONFIGS)
    def test_across_the_reachable_batch_space(self, max_model_len, max_num_seqs, budget):
        config = _config(max_model_len, max_num_seqs, budget)
        rectangles = encoder_rectangles(config)
        declared_rects = set(rectangles)
        declared_groups = set(encoder_group_shapes(config))
        caps = encoder_group_width_caps(config)

        checked = 0
        for num_seqs in range(1, max_num_seqs + 1):
            for length in (1, 63, 64, 65, 200, max_model_len - 1, max_model_len):
                if length < 1:
                    continue
                # Ragged as well as uniform: a step with several extents is what makes
                # the ragged path's group space multi-dimensional.
                for lens in (
                    [length] * num_seqs,
                    [max(1, length - i * 37) for i in range(num_seqs)],
                ):
                    if sum(lens) > budget:
                        continue
                    plan = build_encoder_plan(
                        _fake_metadata(lens),
                        rectangles=rectangles,
                        width_cap_for=caps,
                        device=torch.device("cpu"),
                        dtype=torch.float16,
                        batched=True,
                    )
                    if isinstance(plan, EncoderRectPlan):
                        assert (plan.extent, plan.width) in declared_rects
                    else:
                        for group in plan:
                            assert (group.group, group.extent) in declared_groups, (
                                f"undeclared group {(group.group, group.extent)} for lens={lens}"
                            )
                    checked += 1
        assert checked > 50, "too few batches checked -- test is near-vacuous"

    def test_a_group_wider_than_the_cap_is_chunked_onto_declared_widths(self):
        """17 same-extent requests at a cap of 16 must split 16 + 1, not pad to 32."""
        config = _config(512, 32, 2048)
        lens = [64] * 17
        plan = build_encoder_plan(
            _fake_metadata(lens),
            # No rectangles, so the ragged path is forced regardless of the batch.
            rectangles=(),
            width_cap_for={ENCODER_LEN_ALIGNMENT: 16},
            device=torch.device("cpu"),
            dtype=torch.float16,
            batched=True,
        )
        assert [p.group for p in plan] == [16, 1]
        assert set(encoder_group_shapes(config)) >= {(16, 64), (1, 64)}
