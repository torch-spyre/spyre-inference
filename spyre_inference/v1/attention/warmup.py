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

"""Attention warmup, shared by both model runners."""

import functools
import sys
import time
from typing import Any

import torch
from vllm.logger import init_logger

import spyre_inference.envs as envs
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)

logger = init_logger(__name__)


def record_attention_graphs(
    static_ctx: dict[str, Any],
    kv_caches: dict[str, SpyrePagedKVCache],
    builders: dict[str, SpyreAttentionMetadataBuilder],
    *,
    causal_by_layer: dict[str, bool],
) -> int:
    """Pre-compile every Spyre attention layer's kernels; returns the graph count."""
    # The kernels are module-level torch.compile objects: a layer matching an already
    # recorded one on builder, graph_key() and causality would only replay its graphs.
    if envs.SPYRE_PARALLEL_ATTN_PRECOMPILE > 0:
        _precompile_in_workers(static_ctx, kv_caches, builders, causal_by_layer)
    t0 = time.time()
    total = 0
    num_skipped = 0
    recorded: set[tuple] = set()
    for layer_name, kv_cache in kv_caches.items():
        layer = static_ctx.get(layer_name)
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, SpyreAttentionImpl):
            continue
        builder = builders.get(layer_name)
        # A KV-cache layer on this impl is always in an attention group whose
        # backend builds SpyreAttentionMetadataBuilder, so a miss is a wiring or
        # ordering bug (recording before initialize_attn_backend), not a config.
        assert builder is not None, (
            f"Layer {layer_name} has a Spyre attention impl and a KV cache but no "
            "Spyre metadata builder; initialize_attn_backend() must run first."
        )
        causal = causal_by_layer[layer_name]
        key = (id(builder), impl.graph_key(), causal)
        if key in recorded:
            num_skipped += 1
            continue
        logger.info("Recording attention graphs for layer %s...", layer_name)
        total += impl.record_graphs(layer, kv_cache, builder, causal=causal)
        recorded.add(key)
    logger.info(
        "Attention graph recording complete: %d graphs in %.3fs; %d layers share "
        "an earlier layer's graphs.",
        total,
        time.time() - t0,
        num_skipped,
    )
    return total


def _precompile_in_workers(static_ctx, kv_caches, builders, causal_by_layer) -> None:
    """Trace every attention variant once, with its Spyre inner compile in a forked
    process (SPYRE_PARALLEL_ATTN_PRECOMPILE of them at once), so the recording that
    follows finds every compile in this start's caches. A restart already has them:
    then the first variant runs without compiling, and this stops at once.
    """
    from torch_spyre._inductor.parallel_precompile import (
        CompileOnlyAbort,
        fork_inner_compiles,
    )

    layers, seen = [], set()
    for name, kv_cache in kv_caches.items():
        layer = static_ctx.get(name)
        impl = getattr(layer, "impl", None)
        builder = builders.get(name)
        device = kv_cache[0].device
        if isinstance(impl, SpyreAttentionImpl) and builder and impl._attn_compiled(device):
            key = (id(builder), impl.graph_key(), causal_by_layer[name])
            if key not in seen:
                seen.add(key)
                layers.append((layer, impl, kv_cache, builder, causal_by_layer[name]))
    t0 = time.time()
    traced, warm = 0, False
    # At TP > 1 every rank precompiles, so stay within this rank's CPU share, which
    # vLLM's threading setup gives torch.
    workers = max(1, min(envs.SPYRE_PARALLEL_ATTN_PRECOMPILE, torch.get_num_threads()))
    with fork_inner_compiles(workers):
        for layer, impl, kv_cache, builder, causal in layers:
            num_pages = kv_cache[0].shape[0]
            runs = [
                functools.partial(
                    impl._record_one, b, layer, kv_cache, builder, set(), causal=causal
                )
                for b in builder.attn_bucketer.variants()
                if b.num_blocks <= num_pages
            ]
            if impl._batched_decode_supported(kv_cache[0].device):
                runs += [
                    functools.partial(
                        impl._record_batched_one, b, layer, kv_cache, builder, set(), num_pages
                    )
                    for b in builder.attn_bucketer.batched_decode_variants()
                ]
            for run in runs:
                try:
                    run()
                except CompileOnlyAbort:
                    traced += 1
                except Exception:  # noqa: BLE001 - the recording below compiles and reports it
                    pass
                else:
                    warm = traced == 0  # a cache hit before any compile: a restart
                    if warm:
                        break
            if warm:
                break
    # Forget the stand-in graphs, so the recording below traces these kernels again.
    for kernel in _compiled_kernels([entry[1] for entry in layers]):
        torch._dynamo.reset_code(kernel.__code__)
    logger.info(
        "Parallel attention precompile: %s in %.1fs.",
        "skipped, the caches are warm" if warm else f"{traced} variants compiled in workers",
        time.time() - t0,
    )


def _compiled_kernels(impls) -> list:
    """The torch.compile'd kernels reachable from ``impls``: their attributes and the
    module globals of their classes."""
    found = {}
    for impl in impls:
        owners = [impl] + [
            sys.modules[c.__module__] for c in type(impl).__mro__ if c.__module__ in sys.modules
        ]
        for owner in owners:
            for value in list(vars(owner).values()):
                original = getattr(value, "_torchdynamo_orig_callable", None)
                if callable(value) and original is not None:
                    found[id(original)] = original
    return list(found.values())
