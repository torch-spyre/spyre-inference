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
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.attention import warmup
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl

pytestmark = pytest.mark.attention


def _layers(num_seqs=(1,)):
    """Two layers that share their graphs, and their builders."""
    static_ctx = {}
    for i in range(2):
        impl = object.__new__(SpyreAttentionImpl)
        impl.graph_key = lambda: "same"
        impl._batched_decode_supported = lambda: True
        static_ctx[f"l{i}"] = SimpleNamespace(impl=impl)
    builder = SimpleNamespace(
        attn_bucketer=SimpleNamespace(
            # 9 blocks do not fit the 4 KV pages, so that variant is skipped.
            variants=lambda: [SimpleNamespace(num_blocks=n) for n in (1, 2, 9)],
            batched_decode_variants=lambda: [
                SimpleNamespace(num_blocks=1, num_seqs=n) for n in num_seqs
            ],
        )
    )
    kv_caches = {name: [torch.empty(4)] for name in static_ctx}
    return static_ctx, kv_caches, dict.fromkeys(static_ctx, builder)


def _run(monkeypatch, cold, marker=None):
    """Returns what the precompile did for the layers of _layers()."""
    monkeypatch.setenv("SPYRE_PARALLEL_ATTN_PRECOMPILE", "4")

    class CompileOnlyAbort(Exception):
        pass

    # A stand-in for torch-spyre's helper: the compiles run inline, and the test does not
    # need a torch-spyre that has it.
    parallel_precompile = SimpleNamespace(
        CompileOnlyAbort=CompileOnlyAbort,
        SecondPassCheck=lambda expected: SimpleNamespace(expected=expected),
        fork_inner_compiles=lambda n: contextlib.nullcontext(SimpleNamespace(ok=3)),
    )
    monkeypatch.setitem(
        sys.modules, "torch_spyre._inductor.parallel_precompile", parallel_precompile
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

    static_ctx, kv_caches, builders = _layers()
    for i, layer in enumerate(static_ctx.values()):
        layer.impl.kernel = kernel
        layer.impl._record_one = record("one", i)
        layer.impl._record_batched_one = record("batched", i)
    check = warmup._precompile_in_workers(
        static_ctx, kv_caches, builders, dict.fromkeys(static_ctx, True), marker=marker
    )
    return calls, resets, compiled, check


def test_cold_start_traces_every_variant_once_for_shared_layers(monkeypatch):
    calls, resets, compiled, check = _run(monkeypatch, cold=True)
    assert calls == [("one", 0, 1), ("one", 0, 2), ("batched", 0, 1)]
    assert compiled.__code__ in resets  # the stand-in graphs are forgotten
    assert check.expected == 3  # the recording is checked against the workers' compiles


def test_restart_with_warm_caches_skips_the_precompile(monkeypatch, tmp_path):
    marker = tmp_path / "recorded"
    marker.touch()
    calls, resets, compiled, check = _run(monkeypatch, cold=False, marker=str(marker))
    assert calls == [("one", 0, 1)]
    assert compiled.__code__ in resets
    assert check is None


def test_a_cache_hit_alone_does_not_skip_the_precompile(monkeypatch, tmp_path):
    # Another config filled the cache: its variants hit, but this set was never recorded.
    marker = str(tmp_path / "never-recorded")
    calls, _, _, check = _run(monkeypatch, cold=False, marker=marker)
    assert calls == [("one", 0, 1), ("one", 0, 2), ("batched", 0, 1)]
    assert check is None


def test_the_marker_follows_the_variant_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    causal = dict.fromkeys(["l0", "l1"], True)
    a = warmup._variant_set_marker(*_layers(num_seqs=(1, 32)), causal)
    assert a == warmup._variant_set_marker(*_layers(num_seqs=(1, 32)), causal)
    assert a != warmup._variant_set_marker(*_layers(num_seqs=(1, 256)), causal)
    assert a.startswith(str(tmp_path))


def test_the_recording_writes_the_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SPYRE_PARALLEL_ATTN_PRECOMPILE", "4")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    static_ctx, kv_caches, builders = _layers()
    causal = dict.fromkeys(static_ctx, True)
    marker = warmup._variant_set_marker(static_ctx, kv_caches, builders, causal)
    seen = []
    monkeypatch.setattr(warmup, "_precompile_in_workers", lambda *a, marker: seen.append(marker))
    for layer in static_ctx.values():
        layer.impl.record_graphs = lambda *a, **k: 1
    assert (
        warmup.record_attention_graphs(static_ctx, kv_caches, builders, causal_by_layer=causal) == 1
    )
    assert seen == [marker]
    assert os.path.exists(marker)


def test_the_reset_finds_the_attention_kernels_behind_their_wrappers():
    """spyre_attn wraps its torch.compile'd kernels in a plain function (the dynamic
    page and chunk counts); a stand-in left in one of them breaks every recording."""
    from spyre_inference.v1.attention.backends import spyre_attn

    impl = object.__new__(SpyreAttentionImpl)
    found = {fn.__name__ for fn in warmup._compiled_kernels([impl])}
    assert {"page_attn_prefill", "batched_decode"} <= found
    assert spyre_attn._batched_decode_compiled.__name__ == "batched_decode"
