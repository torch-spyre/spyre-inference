# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A loaded model's Spyre tensors, converted, as a Spyre-native checkpoint.

``write_cache`` runs the ``model.to(spyre)`` traversal on a fully loaded host
model and saves what every tensor becomes with ``fastsafetensors.saver``:
native payloads (the device-format bytes a raw DMA sends) where the tensor can
be raw-copied, the host tensor itself where the normal path must move it, and
the tensors that remain on the host. ``restore_cache`` rebuilds
that state on a model built on the meta device: the same traversal allocates
each destination and ``ParallelLoader`` fills them shard by shard, skipping
weight loading, post-processing and conversion.

FP8 models use normal loading; their kernels move weights before this traversal
and are not supported by the host-conversion cache.

Entries are matched to the traversal by call order and to host tensors by
module path, and checked against shape, dtype, strides and layout. Compatibility
is determined by the caller's cache key and these tensor checks, not by comparing
plain model attributes. A mismatch returns False, and the caller discards the
partially restored model and loads normally.

The shards use fastsafetensors' device-native convention in standard safetensors
files. The model-specific replay recipe is JSON in ``__metadata__``; nothing
in the files is executed on load.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from fastsafetensors import ParallelLoader, SafeTensorsMetadata
from fastsafetensors.copier.spyre import (
    TransferRecorder,
    TransferRestorer,
    cache_metadata,
    check_materialized,
    native_fill,
    native_shard_metadata,
    read_cache_metadata,
)
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import NativeFormatError
from fastsafetensors.saver import (
    ParallelSaver,
    shard_paths,
    validate_shard_set,
)
from vllm.logger import init_logger

from spyre_inference.v1.worker.model_load import move_model_to_spyre

logger = init_logger(__name__)

# Buffered writes to one file serialize on its inode; separate files write in
# parallel, so the cache is sharded like a checkpoint.
_NUM_SHARDS = 16


class CacheMismatch(Exception):
    """The cache does not describe the model being restored."""


def cache_exists(base: str) -> bool:
    return all(os.path.exists(p) for p in shard_paths(base, _NUM_SHARDS))


def _fp8_layers(model: nn.Module):
    for path, layer in model.named_modules():
        owner = getattr(layer, "scheme", None) or getattr(layer, "quant_method", None)
        kernel = getattr(owner, "fp8_linear", None)
        if kernel is None:
            continue
        from spyre_inference.custom_ops.fp8_linear_kernel import SpyreFp8LinearKernel

        if isinstance(kernel, SpyreFp8LinearKernel):
            yield path, layer, kernel


def write_cache(model: nn.Module, device: torch.device, base: str, key: dict) -> bool:
    """Write ``model``'s tensors, as ``model.to(device)`` leaves them, to ``base``'s shards.

    Returns True if the cache was written: device tensors may be allocated but
    not filled, so the model must be discarded (restore from the cache instead).
    Returns False if it could not be written (e.g. no disk space or an
    unsupported tensor); the model has then been moved to ``device`` fully.
    """
    t0 = time.time()
    recorder = TransferRecorder(device)
    try:
        recorder.capture_host_state(model)
    except NativeFormatError as e:
        logger.warning("spyre_native: not caching this model (%s); moving it directly.", e)
        move_model_to_spyre(model, device, transfer=recorder)
        recorder.fill()
        return False
    move_model_to_spyre(model, device, transfer=recorder)

    try:
        if any(_fp8_layers(model)):
            raise CacheMismatch("FP8 native caching is not supported by this kernel")
        tensor_metadata, entries = recorder.checkpoint_metadata(model)
        metadata, shard0_metadata = cache_metadata(key)
    except (CacheMismatch, NativeFormatError) as e:
        logger.warning("spyre_native: not caching this model (%s); moving it directly.", e)
        recorder.fill()
        return False

    t1 = time.time()
    try:
        saver = ParallelSaver(
            num_shards=_NUM_SHARDS,
            shard_metadata=native_shard_metadata,
            fill=native_fill,
            min_free_bytes=1 << 30,
        )
        paths = saver.save_entries(
            entries,
            base,
            metadata=metadata,
            shard0_metadata={**tensor_metadata, **shard0_metadata},
        )
    except OSError as e:
        logger.warning("spyre_native: cannot write the cache (%s); moving the model directly.", e)
        recorder.fill()
        return False
    logger.info(
        "spyre_native: wrote %.2f GiB cache %s-*.safetensors in %.1fs (write %.1fs).",
        sum(os.path.getsize(p) for p in paths) / 2**30,
        base,
        time.time() - t0,
        time.time() - t1,
    )
    return True


def restore_cache(model: nn.Module, device: torch.device, base: str, key: dict) -> bool:
    """Restore ``model`` (built on meta, not loaded) from ``write_cache``'s shards.

    Returns False if the cache does not match ``key`` or the model; the model
    may be partially restored and must then be discarded.
    """
    t0 = time.time()
    paths = shard_paths(base, _NUM_SHARDS)
    try:
        framework = get_framework_op("pytorch")
        metas = [SafeTensorsMetadata.from_file(p, framework) for p in paths]
        validate_shard_set(metas)
        md = metas[0].metadata if isinstance(metas[0].metadata, dict) else {}
        read_cache_metadata(md, key)
        if any(_fp8_layers(model)):
            raise CacheMismatch("FP8 native caching is not supported by this kernel")
    except (OSError, ValueError, KeyError, CacheMismatch) as e:
        logger.warning("spyre_native: cache %s unusable (%s).", base, e)
        return False

    headers = {name: meta for meta in metas for name in meta.tensors}
    try:
        restorer = TransferRestorer.from_metadata(device, headers, md)
        restorer.restore_host_state(model)
        # Some _apply implementations derive their transfer source from CPU state.
        move_model_to_spyre(model, device, transfer=restorer)
        restorer.finish()
        check_materialized(model)
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("spyre_native: cache %s does not match the model (%s).", base, e)
        return False
    t1 = time.time()
    loader = ParallelLoader(
        None,
        paths,
        device=str(device),
        destinations=restorer.destinations,
        tensor_filter=restorer.destinations.__contains__,
        all_local=True,
        set_numa=False,
        use_tqdm_on_load=False,
        accumulate_resident=False,
    )
    try:
        for _name, _tensor in loader.iterate_weights():
            pass
    finally:
        loader.close()
    logger.info(
        "spyre_native: restored model from %s in %.1fs (prepare %.1fs, %d tensors loaded).",
        base,
        time.time() - t0,
        t1 - t0,
        len(restorer.destinations),
    )
    return True
