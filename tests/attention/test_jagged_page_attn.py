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
from unittest.mock import patch

import pytest
import torch

from spyre_inference.v1.attention.jagged_plan import (
    build_jagged_decode_plan,
    build_jagged_mixed_plan,
    build_jagged_plan,
    build_jagged_tile_plan,
    partition_jagged_decode_plan,
)
from spyre_inference.v1.attention.ops.jagged_decode_attn import jagged_decode_attn_kernel
from spyre_inference.v1.attention.ops.jagged_mixed_attn import jagged_join_outputs
from spyre_inference.v1.attention.ops.jagged_page_attn import jagged_page_attn_kernel
from spyre_inference.v1.attention.ops.jagged_split_decode import (
    jagged_decode_merge_kernel,
    jagged_decode_partials_kernel,
)
from spyre_inference.v1.attention.ops.jagged_tile_attn import jagged_tile_attn_kernel

pytestmark = pytest.mark.attention

PLAN_BUILDERS = {
    "flat": build_jagged_plan,
    "nested": build_jagged_tile_plan,
    "page_parallel": build_jagged_decode_plan,
    "page_parallel_64": partial(build_jagged_decode_plan, max_parallel_entries=64),
}
KERNELS = {
    "flat": jagged_page_attn_kernel,
    "nested": jagged_tile_attn_kernel,
    "page_parallel": jagged_decode_attn_kernel,
    "page_parallel_64": jagged_decode_attn_kernel,
}


def dense_reference(query, keys, values, starts, lengths, pages, *, causal, window, cap):
    """Materialize each logical sequence independently of the work schedule."""
    output = torch.zeros_like(query, dtype=torch.float64)
    block_size, num_kv_heads, head_size = keys.shape[1:]
    groups = query.shape[1] // num_kv_heads
    for seq, length in enumerate(lengths):
        start, end = int(starts[seq]), int(starts[seq + 1])
        if start == end:
            continue
        page_ids = pages[seq, : (length + block_size - 1) // block_size].long()
        k = keys[page_ids].flatten(0, 1)[:length].double().repeat_interleave(groups, dim=1)
        v = values[page_ids].flatten(0, 1)[:length].double().repeat_interleave(groups, dim=1)
        q = query[start:end].double().transpose(0, 1)
        scores = q @ k.transpose(0, 1).transpose(-2, -1) / head_size**0.5
        if cap:
            scores = torch.tanh(scores / cap) * cap
        q_pos = (length - (end - start) + torch.arange(end - start)).unsqueeze(1)
        k_pos = torch.arange(length).unsqueeze(0)
        visible = torch.ones((end - start, length), dtype=torch.bool)
        if causal:
            visible &= k_pos <= q_pos
        if window is not None:
            visible &= k_pos > q_pos - window
        scores = scores.masked_fill(~visible, float("-inf"))
        output[start:end] = (scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1)
    return output


def jagged_cases(dtype=torch.float32, *, num_heads=4, num_kv_heads=2, block_size=64, head_size=64):
    generator = torch.Generator().manual_seed(930)
    keys = (
        torch.randn(20, block_size, num_kv_heads, head_size, generator=generator, dtype=dtype)
        * 0.25
    )
    values = torch.randn(keys.shape, generator=generator, dtype=dtype) * 0.25
    cases = []
    for q_lens, lengths in (([1, 65, 3], [193, 150, 65]), ([9, 1, 42], [130, 257, 80])):
        query = torch.randn(193, num_heads, head_size, generator=generator, dtype=dtype) * 0.25
        starts = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
        pages = torch.randperm(20, generator=generator)[:18].reshape(3, 6).to(torch.int32)
        pages[1, 0] = pages[0, 0]  # Shared prefix; physical IDs do not encode causality.
        cases.append((query, keys, values, starts, torch.tensor(lengths), pages))
    return cases


def long_context_case(dtype=torch.float32, *, num_heads=4, num_kv_heads=1):
    generator = torch.Generator().manual_seed(931)
    keys = torch.randn(130, 64, num_kv_heads, 64, generator=generator, dtype=dtype) * 0.25
    values = torch.randn(keys.shape, generator=generator, dtype=dtype) * 0.25
    query = torch.randn(65, num_heads, 64, generator=generator, dtype=dtype) * 0.25
    starts = torch.tensor([0, 3, 4], dtype=torch.int32)
    lengths = torch.tensor([4097, 4099], dtype=torch.int32)
    pages = torch.randperm(130, generator=generator).reshape(2, 65).to(torch.int32)
    # Allocated cache pages need not have initialized storage past seq_lens.
    for seq, length in enumerate(lengths.tolist()):
        tail_page = int(pages[seq, length // 64])
        keys[tail_page, length % 64 :] = float("nan")
        values[tail_page, length % 64 :] = float("nan")
    return query, keys, values, starts, lengths, pages


def large_decode_group_cases(dtype=torch.float32):
    generator = torch.Generator().manual_seed(934)
    q = torch.randn(193, 4, 64, dtype=dtype, generator=generator) * 0.25
    k = torch.randn(256, 128, 2, 64, dtype=dtype, generator=generator) * 0.25
    v = torch.randn(k.shape, dtype=dtype, generator=generator) * 0.25
    starts = torch.arange(66, dtype=torch.int32)
    return [
        (
            q,
            k,
            v,
            starts,
            torch.randint(4097, 8193, (65,), generator=generator, dtype=torch.int32),
            torch.randint(0, 256, (65, 64), generator=generator, dtype=torch.int32),
        )
        for _ in range(2)
    ]


@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 1), (2, 2)])
@pytest.mark.parametrize("window", [3, 67])
def test_jagged_long_positions_and_uninitialized_cache_tail(
    head_major, num_heads, num_kv_heads, window
):
    q, k, v, starts, lengths, pages = long_context_case(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    plan = build_jagged_plan(
        starts,
        lengths,
        pages,
        64,
        query_capacity=q.shape[0],
        work_capacity=8,
        sliding_window=window,
    )
    expected = dense_reference(q, k, v, starts, lengths, pages, causal=True, window=window, cap=0.0)
    if head_major:
        k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    with patch("torch.accelerator.is_available", return_value=False), torch.inference_mode():
        actual = jagged_page_attn_kernel(
            q,
            k,
            v,
            *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors),
            64**-0.5,
            head_major=head_major,
        )[: q.shape[0]]
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize(
    "causal,window,cap", [(True, None, 0.0), (True, 67, 0.0), (False, None, 5.0)]
)
def test_jagged_packed_attention_reuses_graph(head_major, causal, window, cap):
    graphs = []

    def capture(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    kernel = partial(
        jagged_page_attn_kernel,
        scale=64**-0.5,
        head_major=head_major,
        logits_soft_cap=cap,
    )
    with (
        patch("torch.accelerator.is_available", return_value=False),
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        compiled = torch.compile(kernel, backend=capture, fullgraph=True, dynamic=False)
        for q, k, v, starts, lengths, pages in jagged_cases():
            plan = build_jagged_plan(
                starts,
                lengths,
                pages,
                64,
                query_capacity=q.shape[0],
                work_capacity=16,
                causal=causal,
                sliding_window=window,
            )
            expected = dense_reference(
                q,
                k,
                v,
                starts,
                lengths,
                pages,
                causal=causal,
                window=window,
                cap=cap,
            )
            if head_major:
                k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            # CPU index_copy requires int64; Spyre consumes the int32 tables.
            tables = tuple(t.long() if t.dtype == torch.int32 else t for t in plan.tensors)
            result = compiled(q, k, v, *tables)[: q.shape[0]]
            torch.testing.assert_close(result.double(), expected, atol=2e-6, rtol=2e-5)
    assert len(graphs) == 1
    assert (
        sum(n.op == "call_function" and "scan" in str(n.target) for n in graphs[0].graph.nodes) == 1
    )


def test_jagged_plan_final_writes_cover_packed_rows_once():
    q, _, _, starts, lengths, pages = jagged_cases()[0]
    plan = build_jagged_plan(
        starts, lengths, pages, 64, query_capacity=q.shape[0], work_capacity=16
    )
    real_writes = plan.output_indices[plan.output_indices < q.shape[0]]
    torch.testing.assert_close(
        real_writes.sort().values, torch.arange(int(starts[-1]), dtype=torch.int32)
    )
    for row in plan.output_indices:
        assert row.unique().numel() == row.numel()
    assert plan.num_work_items < 16
    assert torch.all(plan.query_bounds[plan.num_work_items :] == -1)
    assert torch.all(plan.key_offsets[plan.num_work_items :] == -1)
    assert all(t.is_contiguous() for t in plan.tensors)
    assert all(t.dtype == torch.int32 for t in plan.tensors[:3])
    assert all(t.dtype == torch.float16 for t in plan.tensors[3:])


def test_jagged_empty_plan_is_one_inactive_visit():
    plan = build_jagged_plan(
        torch.tensor([0, 0, 0]),
        torch.tensor([128, 0]),
        torch.zeros((2, 2), dtype=torch.int32),
        64,
        query_capacity=65,
    )
    assert plan.num_work_items == 0
    assert plan.query_indices.shape == (1, 64)
    assert torch.all(plan.output_indices >= 65)
    assert torch.all(plan.query_bounds == -1)


@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize(
    "schedule,query_tile_size",
    [("nested", 1), ("nested", 64), ("nested", 128), ("page_parallel", 1)],
)
@pytest.mark.parametrize(
    "causal,window,cap", [(True, None, 0.0), (True, 67, 0.0), (False, None, 5.0)]
)
def test_tiled_packed_attention_reuses_graph(
    head_major, schedule, query_tile_size, causal, window, cap
):
    graphs = []

    def capture(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    kernel = partial(
        KERNELS[schedule],
        scale=64**-0.5,
        head_major=head_major,
        logits_soft_cap=cap,
    )
    with (
        patch("torch.accelerator.is_available", return_value=False),
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        compiled = torch.compile(kernel, backend=capture, fullgraph=True, dynamic=False)
        for q, k, v, starts, lengths, pages in jagged_cases():
            plan = PLAN_BUILDERS[schedule](
                starts,
                lengths,
                pages,
                64,
                query_capacity=q.shape[0],
                query_tile_size=query_tile_size,
                tile_capacity=128 if query_tile_size == 1 else 4,
                page_capacity=8,
                causal=causal,
                sliding_window=window,
            )
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=causal, window=window, cap=cap
            )
            if head_major:
                k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            tables = tuple(t.long() if t.dtype == torch.int32 else t for t in plan.tensors)
            result = compiled(q, k, v, *tables)[: q.shape[0]]
            torch.testing.assert_close(result.double(), expected, atol=2e-6, rtol=2e-5)
    assert len(graphs) == 1


@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize("window", [3, 67])
def test_tiled_long_positions_and_uninitialized_cache_tail(head_major, window):
    q, k, v, starts, lengths, pages = long_context_case()
    plan = build_jagged_tile_plan(
        starts, lengths, pages, 64, query_capacity=q.shape[0], sliding_window=window
    )
    expected = dense_reference(q, k, v, starts, lengths, pages, causal=True, window=window, cap=0.0)
    if head_major:
        k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    with patch("torch.accelerator.is_available", return_value=False), torch.inference_mode():
        actual = jagged_tile_attn_kernel(
            q,
            k,
            v,
            *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors),
            64**-0.5,
            head_major=head_major,
        )[: q.shape[0]]
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


def test_jagged_tile_plan_covers_output_once_and_rejects_small_capacity():
    q, _, _, starts, lengths, pages = jagged_cases()[0]
    plan = build_jagged_tile_plan(
        starts, lengths, pages, 64, query_capacity=q.shape[0], tile_capacity=8, page_capacity=8
    )
    real_writes = plan.output_indices[plan.output_indices < q.shape[0]]
    torch.testing.assert_close(
        real_writes.sort().values, torch.arange(int(starts[-1]), dtype=torch.int32)
    )
    assert plan.num_query_tiles == 4
    assert torch.all(plan.query_bounds[4:] == -1)
    assert torch.all(plan.key_offsets[4:] == -1)
    with pytest.raises(ValueError, match="capacities"):
        build_jagged_tile_plan(
            starts, lengths, pages, 64, query_capacity=q.shape[0], page_capacity=1
        )


def large_position_case(dtype=torch.float32):
    length = (1 << 24) + 3
    pages = torch.zeros((1, (length + 63) // 64), dtype=torch.int32)
    pages[0, -1] = 1
    q = torch.zeros(65, 1, 64, dtype=dtype)
    k = torch.zeros(2, 64, 1, 64, dtype=dtype)
    v = torch.zeros_like(k)
    v[0, -2] = 1
    v[0, -1] = 2
    v[1, 0] = 3
    v[1, 1] = 4
    v[1, 2] = 5
    k[1, 3:] = float("nan")
    v[1, 3:] = float("nan")
    return q, k, v, torch.tensor([0, 3]), torch.tensor([length]), pages


def test_jagged_bounds_above_fp32_integer_precision():
    q, k, v, starts, lengths, pages = large_position_case()
    plan = build_jagged_plan(
        starts,
        lengths,
        pages,
        64,
        query_capacity=65,
        work_capacity=4,
        sliding_window=3,
    )
    with patch("torch.accelerator.is_available", return_value=False), torch.inference_mode():
        result = jagged_page_attn_kernel(
            q,
            k,
            v,
            *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors),
            64**-0.5,
        )
    # Zero logits make each output the mean of its last three values.
    expected = torch.zeros_like(q)
    expected[:3] = torch.tensor([2.0, 3.0, 4.0]).reshape(3, 1, 1)
    torch.testing.assert_close(result[: q.shape[0]], expected)


def max_context_case(dtype=torch.float32):
    generator = torch.Generator().manual_seed(932)
    block_size, max_length = 128, 128 * 1024
    keys = torch.randn(2048, block_size, 2, 64, generator=generator, dtype=dtype) * 0.25
    values = torch.randn(keys.shape, generator=generator, dtype=dtype) * 0.25 + 0.5
    query = torch.randn(65, 4, 64, generator=generator, dtype=dtype) * 0.25
    starts = torch.tensor([0, 1, 4], dtype=torch.int32)
    lengths = torch.tensor([max_length, max_length - 125], dtype=torch.int32)
    pages = torch.randperm(2048, generator=generator).reshape(2, 1024).to(torch.int32)
    tail_page = int(pages[1, -1])
    keys[tail_page, 3:] = float("nan")
    values[tail_page, 3:] = float("nan")
    return query, keys, values, starts, lengths, pages


@pytest.mark.parametrize("value", [0.3, 0.5])
@pytest.mark.parametrize("schedule", ["flat", "nested", "page_parallel", "page_parallel_64"])
def test_jagged_fp16_128k_uniform_attention(value, schedule):
    # Reuse one physical page to exercise all 1024 visits with small CPU inputs.
    q = torch.zeros((65, 1, 1), dtype=torch.float16)
    k = torch.zeros((2, 128, 1, 1), dtype=torch.float16)
    v = torch.full_like(k, value)
    builder = PLAN_BUILDERS[schedule]
    kernel = KERNELS[schedule]
    plan = builder(
        torch.tensor([0, 1]),
        torch.tensor([128 * 1024]),
        torch.zeros((1, 1024), dtype=torch.int32),
        128,
        query_capacity=65,
    )
    with patch("torch.accelerator.is_available", return_value=False), torch.inference_mode():
        actual = kernel(
            q, k, v, *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors), 1.0
        )[:65]
    expected = torch.zeros_like(q)
    expected[0] = value
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def jagged_device_inputs(q, k, v, plan, head_major):
    from spyre_inference.custom_ops.utils import convert, row_outermost_layout
    from spyre_inference.v1.attention.ops.layout import (
        head_major_kv_layout,
        slot_major_kv_layout,
    )

    num_pages, block_size, kv_heads, head_size = k.shape
    if head_major:
        k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        cache_layout = head_major_kv_layout(num_pages * kv_heads, block_size, head_size, k.dtype)
    else:
        cache_layout = slot_major_kv_layout(num_pages * block_size, kv_heads, head_size, k.dtype)
    device = torch.device("spyre")
    return (
        convert(q, device, device_layout=row_outermost_layout(q.shape, q.dtype)),
        convert(k, device, device_layout=cache_layout),
        convert(v, device, device_layout=cache_layout),
        *(
            convert(t, device, device_layout=row_outermost_layout(t.shape, t.dtype))
            for t in plan.tensors
        ),
    )


@pytest.mark.parametrize("head_major", [False, True], ids=["token", "head"])
@pytest.mark.parametrize("schedule", ["flat", "nested", "page_parallel", "page_parallel_64"])
@pytest.mark.parametrize(
    "case",
    [
        "decode",
        "many_decode",
        "mixed",
        "window",
        "softcap",
        "long_mqa",
        "long_mha",
        "wide",
        "single_head",
        "max_context",
        "max_context_single_query",
        "max_context_window",
    ],
)
def test_jagged_attention_on_spyre(head_major, schedule, case):
    from spyre_testing_plugin.pytest_plugin import spyre_available
    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    if not spyre_available():
        pytest.skip("Spyre device not available")
    torch._dynamo.reset()
    window = {"window": 67, "long_mqa": 3, "long_mha": 67, "max_context_window": 131}.get(case)
    causal = case != "softcap"
    cap = 5.0 if case == "softcap" else 0.0
    query_tile_size = (
        1
        if schedule != "flat"
        and (
            schedule.startswith("page_parallel")
            or case in ("decode", "many_decode", "max_context_single_query")
        )
        else 64
    )
    capacity = 16
    if case.startswith("long_"):
        heads, kv_heads = (4, 1) if case == "long_mqa" else (2, 2)
        cases = [long_context_case(torch.float16, num_heads=heads, num_kv_heads=kv_heads)]
        capacity = 8
    elif case.startswith("max_context"):
        cases = [max_context_case(torch.float16)]
        capacity = 8 if window else (4096 if query_tile_size == 1 else 2048)
    elif case == "single_head":
        cases = jagged_cases(torch.float16, num_heads=1, num_kv_heads=1, block_size=128)
    elif case == "wide":
        cases = jagged_cases(torch.float16, num_heads=8, block_size=128, head_size=128)
    else:
        cases = jagged_cases(torch.float16)
    if case == "decode":
        cases = [
            (q, k, v, torch.tensor([0, 1, 2, 3], dtype=torch.int32), lengths, pages)
            for q, k, v, _, lengths, pages in cases
        ]
    elif case == "max_context_single_query":
        cases = [
            (q, k, v, torch.tensor([0, 1], dtype=torch.int32), lengths[:1], pages[:1])
            for q, k, v, _, lengths, pages in cases
        ]
    elif case == "many_decode":
        generator = torch.Generator().manual_seed(933)
        cases = [
            (
                q,
                k,
                v,
                torch.arange(66, dtype=torch.int32),
                torch.randint(1, 257, (65,), generator=generator, dtype=torch.int32),
                torch.randint(0, 20, (65, 4), generator=generator, dtype=torch.int32),
            )
            for q, k, v, _, _, _ in cases
        ]
        capacity = 512
    kernel = KERNELS[schedule]
    compiled = torch.compile(
        partial(
            kernel,
            scale=cases[0][0].shape[-1] ** -0.5,
            head_major=head_major,
            logits_soft_cap=cap,
        ),
        fullgraph=True,
        dynamic=False,
    )
    with (
        warnings.catch_warnings(),
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for number, (q, k, v, starts, lengths, pages) in enumerate(cases):
            builder = PLAN_BUILDERS[schedule]
            capacities = (
                {"work_capacity": capacity}
                if schedule == "flat"
                else {
                    "tile_capacity": (128 if query_tile_size == 1 and case != "decode" else 4)
                    if len(cases) > 1
                    else None,
                    "page_capacity": 1024 if case.startswith("max_context") and not window else 8,
                }
            )
            plan = builder(
                starts,
                lengths,
                pages,
                k.shape[1],
                query_capacity=q.shape[0],
                query_tile_size=query_tile_size,
                **capacities,
                causal=causal,
                sliding_window=window,
            )
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=causal, window=window, cap=cap
            )
            inputs = jagged_device_inputs(q, k, v, plan, head_major)
            before = counters["stats"]["unique_graphs"]
            actual = compiled(*inputs)[: q.shape[0]].cpu().double()
            if number:
                assert counters["stats"]["unique_graphs"] == before
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


@pytest.mark.parametrize("head_major", [False, True], ids=["token", "head"])
@pytest.mark.parametrize("entry_budget", [32, 64], ids=["entry32", "entry64"])
def test_jagged_page_parallel_large_groups_on_spyre(head_major, entry_budget):
    from spyre_testing_plugin.pytest_plugin import spyre_available
    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    if not spyre_available():
        pytest.skip("Spyre device not available")
    torch._dynamo.reset()
    compiled = torch.compile(
        partial(jagged_decode_attn_kernel, scale=64**-0.5, head_major=head_major),
        dynamic=False,
        fullgraph=True,
    )
    with (
        torch.inference_mode(),
        warnings.catch_warnings(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for step, (q, k, v, starts, lengths, pages) in enumerate(
            large_decode_group_cases(torch.float16)
        ):
            plan = build_jagged_decode_plan(
                starts,
                lengths,
                pages,
                128,
                query_capacity=q.shape[0],
                tile_capacity=128,
                page_capacity=64,
                max_parallel_entries=entry_budget,
            )
            # The short-context 65-request fixture uses only eight queries per group.
            assert plan.page_indices.shape == (128 // entry_budget, 64, entry_budget, 32)
            assert plan.output_indices.shape == (128 // entry_budget, entry_budget)
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=True, window=None, cap=0.0
            )
            inputs = jagged_device_inputs(q, k, v, plan, head_major)
            before = counters["stats"]["unique_graphs"]
            actual = compiled(*inputs)[: q.shape[0]].cpu().double()
            if step:
                assert counters["stats"]["unique_graphs"] == before
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def test_jagged_plan_rejects_overflow_and_invalid_lengths():
    q, _, _, starts, lengths, pages = jagged_cases()[0]
    with pytest.raises(ValueError, match="cannot hold"):
        build_jagged_plan(starts, lengths, pages, 64, query_capacity=q.shape[0], work_capacity=1)
    with pytest.raises(ValueError, match="query length"):
        build_jagged_plan(starts, torch.zeros_like(lengths), pages, 64, query_capacity=q.shape[0])
    with pytest.raises(ValueError, match="too short"):
        build_jagged_plan(starts, lengths, pages[:, :1], 64, query_capacity=q.shape[0])
    with pytest.raises(ValueError, match="at most 1024"):
        build_jagged_plan(
            starts,
            lengths,
            pages,
            2048,
            query_capacity=q.shape[0],
        )


@pytest.mark.parametrize("target", ["cpu", "spyre"])
@pytest.mark.parametrize("head_major", [False, True], ids=["token", "head"])
@pytest.mark.parametrize(
    "case", ["mixed", "window", "softcap", "tail", "max_context", "large_groups"]
)
@pytest.mark.parametrize("pages_per_partition", [1, 8], ids=["page", "partition"])
def test_jagged_split_decode(target, head_major, case, pages_per_partition):
    from contextlib import nullcontext

    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    if target == "spyre":
        from spyre_testing_plugin.pytest_plugin import spyre_available

        if not spyre_available():
            pytest.skip("Spyre device not available")
    torch._dynamo.reset()
    dtype = torch.float16 if target == "spyre" else torch.float32
    window = 67 if case == "window" else None
    cap = 5.0 if case == "softcap" else 0.0
    causal = case != "softcap"
    cases = (
        large_decode_group_cases(dtype)
        if case == "large_groups"
        else [max_context_case(dtype)]
        if case == "max_context"
        else [long_context_case(dtype)]
        if case == "tail"
        else jagged_cases(dtype)
    )
    backend = "inductor" if target == "spyre" else lambda gm, _: gm.forward
    partials_fn = torch.compile(
        partial(
            jagged_decode_partials_kernel,
            head_major=head_major,
            logits_soft_cap=cap,
            reduce_pages=pages_per_partition > 1,
        ),
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    merge_fn = torch.compile(
        jagged_decode_merge_kernel, backend=backend, fullgraph=True, dynamic=False
    )
    with (
        nullcontext()
        if target == "spyre"
        else patch("torch.accelerator.is_available", return_value=False),
        warnings.catch_warnings(),
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for step, (q, k, v, starts, lengths, pages) in enumerate(cases):
            plan = build_jagged_decode_plan(
                starts,
                lengths,
                pages,
                k.shape[1],
                query_capacity=q.shape[0],
                tile_capacity=128 if len(cases) > 1 else None,
                page_capacity=64 if case == "large_groups" else 8 if len(cases) > 1 else None,
                causal=causal,
                sliding_window=window,
            )
            plan = partition_jagged_decode_plan(plan, pages_per_partition)
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=causal, window=window, cap=cap
            )
            if target == "spyre":
                inputs = jagged_device_inputs(q, k, v, plan, head_major)
            else:
                if head_major:
                    k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
                inputs = (
                    q,
                    k,
                    v,
                    *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors),
                )
            q_dev, k_dev, v_dev, q_ids, out_ids, p_ids, bounds, offsets = inputs
            before = counters["stats"]["unique_graphs"]
            partials = partials_fn(
                q_dev, k_dev, v_dev, q_ids, p_ids, bounds, offsets, q.shape[-1] ** -0.5
            )
            actual = (
                merge_fn(*partials, out_ids, q.shape[0], offsets.shape[-1])[: q.shape[0]]
                .cpu()
                .double()
            )
            if step:
                assert counters["stats"]["unique_graphs"] == before
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.002 if target == "spyre" else 2e-6,
                rtol=0.02 if target == "spyre" else 2e-5,
            )


@pytest.mark.parametrize("pages_per_partition", [1, 8])
@pytest.mark.parametrize("value", [0.3, 0.5])
def test_jagged_split_fp16_128k_uniform_attention(pages_per_partition, value):
    q = torch.zeros((65, 1, 1), dtype=torch.float16)
    k = torch.zeros((2, 128, 1, 1), dtype=torch.float16)
    v = torch.full_like(k, value)
    plan = partition_jagged_decode_plan(
        build_jagged_decode_plan(
            torch.tensor([0, 1]),
            torch.tensor([128 * 1024]),
            torch.zeros((1, 1024), dtype=torch.int32),
            128,
            query_capacity=65,
        ),
        pages_per_partition,
    )
    q_ids, out_ids, p_ids, bounds, offsets = (
        t.long() if t.dtype == torch.int32 else t for t in plan.tensors
    )
    with patch("torch.accelerator.is_available", return_value=False), torch.inference_mode():
        partials = jagged_decode_partials_kernel(
            q, k, v, q_ids, p_ids, bounds, offsets, 1.0, reduce_pages=pages_per_partition > 1
        )
        actual = jagged_decode_merge_kernel(*partials, out_ids, q.shape[0], 128)[:65]
    expected = torch.zeros_like(q)
    expected[0] = value
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("target", ["cpu", "spyre"])
@pytest.mark.parametrize("head_major", [False, True], ids=["token", "head"])
@pytest.mark.parametrize("window,cap", [(None, 0.0), (67, 0.0), (None, 5.0)])
@pytest.mark.parametrize("prefill_queries", [65, 129])
@pytest.mark.parametrize("direct_out", [False, True], ids=["join", "direct"])
def test_jagged_mixed_schedule(target, head_major, window, cap, prefill_queries, direct_out):
    from contextlib import nullcontext

    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    if target == "spyre":
        from spyre_testing_plugin.pytest_plugin import spyre_available

        if not spyre_available():
            pytest.skip("Spyre device not available")
    torch._dynamo.reset()
    dtype = torch.float16 if target == "spyre" else torch.float32
    q, k, v, starts, lengths, pages = jagged_cases(dtype)[0]
    starts = torch.tensor([0, 1, prefill_queries + 1, prefill_queries + 4], dtype=torch.int32)
    backend = "inductor" if target == "spyre" else lambda gm, _: gm.forward
    kernels = [
        torch.compile(
            partial(kernel, scale=64**-0.5, head_major=head_major, logits_soft_cap=cap),
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        for kernel in (jagged_decode_attn_kernel, jagged_tile_attn_kernel)
    ]
    join = torch.compile(jagged_join_outputs, backend=backend, fullgraph=True, dynamic=False)
    with (
        nullcontext()
        if target == "spyre"
        else patch("torch.accelerator.is_available", return_value=False),
        warnings.catch_warnings(),
        torch.inference_mode(),
        torch._dynamo.config.patch(capture_scalar_outputs=True),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for step in range(2):
            if step:
                starts = torch.tensor(
                    [0, 3, prefill_queries + 4, prefill_queries + 5], dtype=torch.int32
                )
                lengths = torch.tensor([191, 145, 64])
                pages = pages.roll(1, dims=1)
            plan = build_jagged_mixed_plan(
                starts,
                lengths,
                pages,
                64,
                query_capacity=q.shape[0],
                causal=cap == 0.0,
                sliding_window=window,
            )
            assert len(plan.groups) == 2
            assert plan.groups[0].page_indices.ndim == 4
            assert plan.groups[1].query_indices.shape == (1 if prefill_queries == 65 else 2, 128)
            real_writes = torch.cat(
                [group.output_indices[group.output_indices < q.shape[0]] for group in plan.groups]
            )
            torch.testing.assert_close(
                real_writes.sort().values, torch.arange(int(starts[-1]), dtype=torch.int32)
            )
            expected = dense_reference(
                q, k, v, starts, lengths, pages, causal=cap == 0.0, window=window, cap=cap
            )
            destination = None
            if direct_out:
                from spyre_inference.custom_ops.utils import convert, row_outermost_layout

                shape = (q.shape[0] + 128, *q.shape[1:])
                destination = torch.full(shape, 7.0, dtype=q.dtype)
                if target == "spyre":
                    destination = convert(
                        destination,
                        torch.device("spyre"),
                        device_layout=row_outermost_layout(shape, q.dtype),
                    )
            before = counters["stats"]["unique_graphs"]
            outputs = []
            for kernel, group in zip(kernels, plan.groups, strict=True):
                if target == "spyre":
                    inputs = jagged_device_inputs(q, k, v, group, head_major)
                else:
                    keys, values = (
                        (k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
                        if head_major
                        else (k, v)
                    )
                    inputs = (
                        q,
                        keys,
                        values,
                        *(t.long() if t.dtype == torch.int32 else t for t in group.tensors),
                    )
                if direct_out:
                    assert kernel(*inputs, out=destination) is destination
                else:
                    outputs.append(kernel(*inputs))
            actual = (
                (destination[: q.shape[0]] if direct_out else join(tuple(outputs), q.shape[0]))
                .cpu()
                .double()
            )
            if step and window is None:
                assert counters["stats"]["unique_graphs"] == before
            if direct_out:
                # Every real row must be written by its group; other groups
                # and the staging tail retain their previous contents.
                assert torch.all(actual[int(starts[-1]) :] == 7.0)
                actual = actual[: int(starts[-1])]
                expected = expected[: int(starts[-1])]
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.002 if target == "spyre" else 2e-6,
                rtol=0.02 if target == "spyre" else 2e-5,
            )
