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

"""Tests for the parallel attention precompile in warmup (no device needed)."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch_spyre._inductor import parallel_precompile

from spyre_inference.v1.attention import warmup
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

pytestmark = pytest.mark.attention


def _run(monkeypatch, cold):
    """Two layers that share their graphs; returns what the precompile did."""
    monkeypatch.setenv("SPYRE_PARALLEL_ATTN_PRECOMPILE", "4")
    monkeypatch.setattr(
        parallel_precompile, "fork_inner_compiles", lambda n: contextlib.nullcontext()
    )
    resets = []
    monkeypatch.setattr(torch._dynamo, "reset_code", resets.append)
    calls = []

    def compiled():
        pass

    def kernel():
        pass

    kernel._torchdynamo_orig_callable = compiled

    def record(kind, i):
        def run(bucket, *args, **kwargs):
            calls.append((kind, i, bucket.num_blocks))
            if cold:
                raise parallel_precompile.CompileOnlyAbort()

        return run

    static_ctx = {}
    for i in range(2):
        impl = object.__new__(SpyreAttentionImpl)
        impl._attn_compiled = lambda device: True
        impl.graph_key = lambda: "same"
        impl.kernel = kernel
        impl._record_one = record("one", i)
        impl._record_batched_one = record("batched", i)
        impl._batched_decode_supported = lambda device: True
        static_ctx[f"l{i}"] = SimpleNamespace(impl=impl)
    builder = SimpleNamespace(
        attn_bucketer=SimpleNamespace(
            # 9 blocks do not fit the 4 KV pages, so that variant is skipped.
            variants=lambda: [SimpleNamespace(num_blocks=n) for n in (1, 2, 9)],
            batched_decode_variants=lambda: [SimpleNamespace(num_blocks=1)],
        )
    )
    warmup._precompile_in_workers(
        static_ctx,
        {name: [torch.empty(4)] for name in static_ctx},
        dict.fromkeys(static_ctx, builder),
        dict.fromkeys(static_ctx, True),
    )
    return calls, resets, compiled


def test_cold_start_traces_every_variant_once_for_shared_layers(monkeypatch):
    calls, resets, compiled = _run(monkeypatch, cold=True)
    assert calls == [("one", 0, 1), ("one", 0, 2), ("batched", 0, 1)]
    assert compiled.__code__ in resets  # the stand-in graphs are forgotten


def test_restart_with_warm_caches_skips_the_precompile(monkeypatch):
    calls, resets, compiled = _run(monkeypatch, cold=False)
    assert calls == [("one", 0, 1)]
    assert compiled.__code__ in resets
