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

import contextlib
import functools
import hashlib
import os
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
    check = marker = None
    if envs.SPYRE_PARALLEL_ATTN_PRECOMPILE > 0:
        marker = _variant_set_marker(static_ctx, kv_caches, builders, causal_by_layer)
        check = _precompile_in_workers(
            static_ctx, kv_caches, builders, causal_by_layer, marker=marker
        )
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
    if check is not None:
        check.finish()  # warns when the recording missed what the workers compiled
    if marker is not None:
        with contextlib.suppress(OSError):
            open(marker, "a").close()
    logger.info(
        "Attention graph recording complete: %d graphs in %.3fs; %d layers share "
        "an earlier layer's graphs.",
        total,
        time.time() - t0,
        num_skipped,
    )
    return total


def _variant_set_marker(static_ctx, kv_caches, builders, causal_by_layer) -> str | None:
    """A file in the Inductor cache dir that exists once this start's set of attention
    variants has been recorded in full into that cache."""
    entries = set()
    for name, kv_cache in kv_caches.items():
        impl = getattr(static_ctx.get(name), "impl", None)
        builder = builders.get(name)
        if not isinstance(impl, SpyreAttentionImpl) or not builder:
            continue
        num_pages = kv_cache[0].shape[0]
        bucketer = builder.attn_bucketer
        batched = bucketer.batched_decode_variants() if impl._batched_decode_supported() else []
        entries.add(
            repr(
                (
                    impl.graph_key(),
                    causal_by_layer[name],
                    [(tuple(t.shape), t.dtype) for t in kv_cache],
                    [b for b in bucketer.variants() if b.num_blocks <= num_pages],
                    batched,
                )
            )
        )
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        path = os.path.join(cache_dir(), "spyre_attn_record")
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    digest = hashlib.sha256(repr(sorted(entries)).encode()).hexdigest()[:32]
    return os.path.join(path, digest)


def _precompile_in_workers(static_ctx, kv_caches, builders, causal_by_layer, marker=None):
    """Trace every attention variant once, with its Spyre inner compile in a forked
    process (SPYRE_PARALLEL_ATTN_PRECOMPILE of them at once), so the recording that
    follows finds every compile in this start's caches. A restart already has them:
    then the first variant runs without compiling, and this stops at once. That stop
    needs ``marker`` to exist: another config (say a different max_num_seqs) shares
    some variants with this one, so one cache hit alone does not mean all are cached.

    Returns a started ``SecondPassCheck`` for the recording when the workers compiled
    something, else None.
    """
    try:
        from torch_spyre._inductor.parallel_precompile import (  # ty: ignore[unresolved-import]
            CompileOnlyAbort,
            SecondPassCheck,
            fork_inner_compiles,
        )
    except ImportError:
        logger.warning(
            "SPYRE_PARALLEL_ATTN_PRECOMPILE needs a torch-spyre with fork_inner_compiles "
            "(torch-spyre/torch-spyre#6262); compiling the attention variants one by one."
        )
        return None

    layers, seen = [], set()
    for name, kv_cache in kv_caches.items():
        layer = static_ctx.get(name)
        impl = getattr(layer, "impl", None)
        builder = builders.get(name)
        if isinstance(impl, SpyreAttentionImpl) and builder:
            key = (id(builder), impl.graph_key(), causal_by_layer[name])
            if key not in seen:
                seen.add(key)
                layers.append((layer, impl, kv_cache, builder, causal_by_layer[name]))
    t0 = time.time()
    traced, warm = 0, False
    recorded_before = marker is not None and os.path.exists(marker)
    # At TP > 1 every rank precompiles, so stay within this rank's CPU share, which
    # vLLM's threading setup gives torch.
    workers = max(1, min(envs.SPYRE_PARALLEL_ATTN_PRECOMPILE, torch.get_num_threads()))
    # The stand-in graphs must not be kept by the attention AOT cache.
    from spyre_inference.v1.attention.aot_kernel_cache import compile_only

    with fork_inner_compiles(workers) as pool, compile_only():
        for layer, impl, kv_cache, builder, causal in layers:
            num_pages = kv_cache[0].shape[0]
            runs = [
                functools.partial(
                    impl._record_one, b, layer, kv_cache, builder, set(), causal=causal
                )
                for b in builder.attn_bucketer.variants()
                if b.num_blocks <= num_pages
            ]
            if impl._batched_decode_supported():
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
                    warm = traced == 0 and recorded_before
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
    if warm or not traced:
        return None
    return SecondPassCheck(expected=pool.ok)


def _compiled_kernels(impls) -> list:
    """The torch.compile'd kernels reachable from ``impls``: their attributes and the
    module globals of their classes, also when a plain function wraps the compiled
    object (spyre_attn's ``_compile_with_dynamic_count`` keeps it in its closure)."""
    found = {}
    for impl in impls:
        owners = [impl] + [
            sys.modules[c.__module__] for c in type(impl).__mro__ if c.__module__ in sys.modules
        ]
        for owner in owners:
            for value in list(vars(owner).values()):
                if not callable(value):
                    continue
                for candidate in (value, *_closure_values(value)):
                    original = getattr(candidate, "_torchdynamo_orig_callable", None)
                    if callable(candidate) and original is not None:
                        found[id(original)] = original
    return list(found.values())


def _closure_values(fn) -> list:
    """What a function's closure holds, empty cells left out."""
    values = []
    for cell in getattr(fn, "__closure__", None) or ():
        with contextlib.suppress(ValueError):  # an empty cell
            values.append(cell.cell_contents)
    return values
