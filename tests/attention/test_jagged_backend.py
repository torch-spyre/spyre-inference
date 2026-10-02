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

import warnings
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from test_jagged_page_attn import dense_reference, jagged_cases, jagged_device_inputs
from vllm.config import CompilationMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference import envs
from spyre_inference.v1.attention.backends import spyre_attn, spyre_head_major_attn
from spyre_inference.v1.attention.jagged_plan import jagged_plan_variants
from spyre_inference.v1.attention.ops.jagged_decode_attn import jagged_decode_attn_kernel
from spyre_inference.v1.attention.ops.jagged_tile_attn import jagged_tile_attn_kernel

pytestmark = pytest.mark.attention


def _cpu_kernel(function, q, k, v, *args, **kwargs):
    tables, scale = args[:-1], args[-1]
    return function(
        q,
        k,
        v,
        *(t.long() if t.dtype == torch.int32 else t for t in tables),
        scale,
        **kwargs,
    )


@pytest.fixture
def backend_config(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            max_model_len=384,
            dtype=torch.float16,
            model_arch_config=SimpleNamespace(per_layer_overrides=None),
            get_num_attention_heads=Mock(return_value=4),
            get_num_kv_heads=Mock(return_value=2),
        ),
        cache_config=SimpleNamespace(block_size=64),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=192, max_num_seqs=8),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.STOCK_TORCH_COMPILE, static_forward_context={}
        ),
    )
    monkeypatch.setattr(envs, "SPYRE_JAGGED_ATTENTION", True)
    monkeypatch.setattr(spyre_attn, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(spyre_head_major_attn, "get_current_vllm_config", lambda: config)
    return config


@pytest.mark.parametrize("device_type", ["cpu", "spyre"])
@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize("pre_staged", [False, True])
def test_jagged_builder_and_packed_dispatch(
    monkeypatch, backend_config, head_major, pre_staged, device_type
):
    from torch_spyre.ops.fallbacks import FallbackWarning

    if device_type == "spyre":
        from spyre_testing_plugin.pytest_plugin import spyre_available

        if not spyre_available():
            pytest.skip("Spyre device not available")
    torch._dynamo.reset()
    device = torch.device(device_type)
    config = backend_config

    kernels = []
    for name, function in (
        ("_jagged_decode_compiled", jagged_decode_attn_kernel),
        ("_jagged_tile_compiled", jagged_tile_attn_kernel),
    ):
        kernel = Mock(
            side_effect=(
                torch.compile(function, fullgraph=True, dynamic=False)
                if device_type == "spyre"
                else partial(_cpu_kernel, function)
            )
        )
        kernels.append(kernel)
        monkeypatch.setattr(spyre_attn, name, kernel)
    builder = spyre_attn.SpyreAttentionMetadataBuilder(
        AttentionSpec(block_size=64, num_kv_heads=2, head_size=64, dtype=torch.float16),
        [],
        config,
        device,
    )
    publish = Mock(wraps=builder._slot_mapping.publish)
    monkeypatch.setattr(builder._slot_mapping, "publish", publish)
    impl_cls = (
        spyre_head_major_attn.SpyreHeadMajorAttentionImpl
        if head_major
        else spyre_attn.SpyreAttentionImpl
    )
    impl = impl_cls(num_heads=4, num_kv_heads=2, head_size=64, scale=64**-0.5)
    with (
        patch("torch.accelerator.is_available", return_value=False),
        torch.inference_mode(),
        warnings.catch_warnings(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for q, k, v, starts, lengths, pages in jagged_cases(torch.float16):
            tokens = int(starts[-1])
            slots = torch.arange(tokens, dtype=torch.int64)
            metadata = builder.build(
                0,
                CommonAttentionMetadata(
                    query_start_loc=starts,
                    query_start_loc_cpu=starts,
                    seq_lens=lengths,
                    num_reqs=3,
                    num_actual_tokens=tokens,
                    max_query_len=int((starts[1:] - starts[:-1]).max()),
                    max_seq_len=int(lengths.max()),
                    block_table_tensor=pages,
                    slot_mapping=slots,
                ),
            )
            publish.assert_called_with(slots)
            assert metadata.jagged_plan is not None
            assert metadata.attention_mask_stacks is None
            assert metadata.page_index_tables_cpu is None
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=True, window=None, cap=0.0
            )
            if device_type == "spyre":
                q, k, v, *_ = jagged_device_inputs(q, k, v, metadata.jagged_plan, head_major)
            elif head_major:
                k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            if pre_staged:
                query, output = impl.staging_buffers(device)
                query[:tokens].copy_(q[:tokens])
            else:
                query, output = q[:tokens], torch.empty_like(q[:tokens])
            cache = spyre_attn.SpyrePagedKVCache(k, v)
            assert impl.forward(None, query, None, None, cache, metadata, output) is output
            torch.testing.assert_close(
                output[:tokens].cpu().double(), expected[:tokens], atol=0.002, rtol=0.02
            )
            tables = metadata.jagged_tables_device
            impl.forward(None, query, None, None, cache, metadata, output)
            assert metadata.jagged_tables_device is tables
            assert all(kernel.call_args.kwargs["head_major"] == head_major for kernel in kernels)
    assert all(kernel.call_count == 4 for kernel in kernels)


def test_jagged_warmup_covers_scheduled_group_shapes(monkeypatch, backend_config):
    config = backend_config
    builder = spyre_attn.SpyreAttentionMetadataBuilder(
        AttentionSpec(block_size=64, num_kv_heads=2, head_size=64, dtype=torch.float16),
        [],
        config,
        torch.device("cpu"),
    )
    impl = spyre_attn.SpyreAttentionImpl(num_heads=4, num_kv_heads=2, head_size=64, scale=64**-0.5)
    recorded = set()

    def record(query, k, v, *args, **kwargs):
        recorded.add(tuple(tuple(t.shape) for t in args[:-1]))
        return kwargs["out"]

    monkeypatch.setattr(spyre_attn, "_jagged_decode_compiled", record)
    monkeypatch.setattr(spyre_attn, "_jagged_tile_compiled", record)
    cache = spyre_attn.SpyrePagedKVCache(torch.zeros(65, 64, 2, 64), torch.zeros(65, 64, 2, 64))
    with torch.inference_mode():
        count = impl.record_graphs(None, cache, builder)
        assert count == len(recorded) == len(jagged_plan_variants(192, 8, 384, 64, 193))
        generator = torch.Generator().manual_seed(72)
        for _ in range(80):
            requests = int(torch.randint(1, 9, (), generator=generator))
            q_lens = torch.randint(1, 130, (requests,), generator=generator)
            while q_lens.sum() > 192:
                q_lens = (q_lens // 2).clamp_min(1)
            starts = torch.cat((torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0).int()))
            lengths = q_lens + torch.randint(0, 255, (requests,), generator=generator)
            metadata = builder.build(
                0,
                CommonAttentionMetadata(
                    query_start_loc=starts,
                    query_start_loc_cpu=starts,
                    seq_lens=lengths,
                    num_reqs=requests,
                    num_actual_tokens=int(starts[-1]),
                    max_query_len=int(q_lens.max()),
                    max_seq_len=int(lengths.max()),
                    block_table_tensor=torch.zeros(requests, 6, dtype=torch.int32),
                    slot_mapping=torch.zeros(int(starts[-1]), dtype=torch.int64),
                ),
            )
            for group in metadata.jagged_plan.groups:
                assert tuple(tuple(t.shape) for t in group.tensors) in recorded


@pytest.mark.parametrize("device_type", ["cpu", "spyre"])
def test_jagged_resident_tables_update_between_steps(backend_config, monkeypatch, device_type):
    from torch_spyre.ops.fallbacks import FallbackWarning

    if device_type == "spyre":
        from spyre_testing_plugin.pytest_plugin import spyre_available

        if not spyre_available():
            pytest.skip("Spyre device not available")
    else:
        monkeypatch.setattr(
            spyre_attn, "_jagged_decode_compiled", partial(_cpu_kernel, jagged_decode_attn_kernel)
        )
        monkeypatch.setattr(
            spyre_attn, "_jagged_tile_compiled", partial(_cpu_kernel, jagged_tile_attn_kernel)
        )
    device = torch.device(device_type)
    builder = spyre_attn.SpyreAttentionMetadataBuilder(
        AttentionSpec(block_size=64, num_kv_heads=2, head_size=64, dtype=torch.float16),
        [],
        backend_config,
        device,
    )
    impl = spyre_head_major_attn.SpyreHeadMajorAttentionImpl(
        num_heads=4, num_kv_heads=2, head_size=64, scale=64**-0.5
    )
    q, k, v, starts, lengths, pages = jagged_cases(torch.float16)[0]
    pointers = None
    cache = None
    with (
        torch.inference_mode(),
        warnings.catch_warnings(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
        patch("torch.accelerator.is_available", return_value=False),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for step in range(3):
            step_lengths = lengths + step
            step_pages = pages.roll(step, dims=1)
            metadata = builder.build(
                0,
                CommonAttentionMetadata(
                    query_start_loc=starts,
                    query_start_loc_cpu=starts,
                    seq_lens=step_lengths,
                    num_reqs=3,
                    num_actual_tokens=69,
                    max_query_len=65,
                    max_seq_len=int(step_lengths.max()),
                    block_table_tensor=step_pages,
                    slot_mapping=torch.zeros(69, dtype=torch.int64),
                ),
            )
            if cache is None:
                if device_type == "spyre":
                    q_dev, k_dev, v_dev, *_ = jagged_device_inputs(
                        q, k, v, metadata.jagged_plan, True
                    )
                else:
                    q_dev, k_dev, v_dev = (
                        q,
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous(),
                    )
                cache = spyre_attn.SpyrePagedKVCache(k_dev, v_dev)
                query, output = impl.staging_buffers(device)
                query.copy_(q_dev)
            impl.forward(None, query, None, None, cache, metadata, output)
            current = tuple(t.data_ptr() for t in metadata.jagged_tables_device)
            if pointers is not None:
                assert current == pointers
            pointers = current
            expected = dense_reference(
                q, k, v, starts, step_lengths, step_pages, causal=True, window=None, cap=0.0
            )
            torch.testing.assert_close(
                output[:69].cpu().double(), expected[:69], atol=0.002, rtol=0.02
            )
            del metadata
