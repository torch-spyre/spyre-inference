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

"""SpyreModelLoader — direct-to-device weight loader for IBM Spyre."""

from __future__ import annotations

import copy
import os
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig

logger = init_logger(__name__)

_DEFAULT_LOAD_THREADS = DefaultModelLoader.DEFAULT_NUM_THREADS
# torch-spyre's get_stream_from_pool rotates over this many streams; more pool threads
# than that would put two injecting threads on one stream.
_POOL_STREAMS = 32
_KNOWN_EXTRA_CONFIG_KEYS = {"enable_multithread_load", "num_threads"}


@register_model_loader("spyre")
class SpyreModelLoader(BaseModelLoader):
    """Streams weights directly from safetensors to Spyre device memory."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            extra_config = {}
        unknown_keys = extra_config.keys() - _KNOWN_EXTRA_CONFIG_KEYS
        if unknown_keys:
            logger.warning(
                "SpyreModelLoader: ignoring unknown model_loader_extra_config keys %s",
                sorted(unknown_keys),
            )
        enable_multithread_load = extra_config.get("enable_multithread_load", True)
        if not isinstance(enable_multithread_load, bool):
            raise ValueError(
                f"enable_multithread_load must be a bool, got "
                f"{type(enable_multithread_load).__name__}"
            )
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (
            isinstance(num_threads, int) and not isinstance(num_threads, bool) and num_threads > 0
        ):
            raise ValueError(f"num_threads must be a positive integer, got {num_threads!r}")
        self._load_threads = 1
        if enable_multithread_load:
            self._load_threads = num_threads or _DEFAULT_LOAD_THREADS

    def download_model(self, model_config: ModelConfig) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            download_weights_from_hf,
        )

        download_weights_from_hf(
            model_name_or_path=model_config.model,
            cache_dir=self.load_config.download_dir,
            allow_patterns=["*.safetensors"],
            revision=model_config.revision,
        )

    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Generator[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from checkpoint shards using physical disk order.

        Tensors destined for Spyre are loaded via safe_open(device="spyre"),
        which triggers _SpyreSafeOpen to stream each tensor with optimal Spyre layout.
        CPU-fallback parameters remain on CPU via safe_open(device="cpu").

        With tensor parallelism (tp_size > 1) linear weights are sharded on read:
        only this rank's slice is fetched via ``get_slice`` and DMA'd to the device,
        because vLLM's weight loaders cannot narrow a tensor that already lives on
        the device in Spyre's stick layout.

        With ``enable_multithread_load`` the reads run ahead on a thread pool, capped
        at the process's torch thread count; the tensors are still yielded in
        checkpoint order.
        """
        from safetensors import safe_open
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        target_dtype = getattr(model_config, "dtype", torch.float16)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        cpu_fallback_names = _get_host_loaded_param_names(model)
        _, shard_files = _resolve_checkpoint_files(model_config, self.load_config)
        num_threads = min(self._load_threads, torch.get_num_threads(), _POOL_STREAMS)
        pool = None
        if num_threads > 1:
            logger.info("SpyreModelLoader: reading weights on %d threads", num_threads)
            on_card = hasattr(torch, "spyre") and torch.spyre.is_initialized()
            pool = ThreadPoolExecutor(
                num_threads,
                thread_name_prefix="spyre-load",
                initializer=_use_own_stream if on_card else None,
                initargs=(torch.spyre.current_device(),) if on_card else (),
            )
        with pool or nullcontext():
            for shard_path in shard_files:
                with (
                    safe_open(
                        shard_path,
                        framework="pt",
                        device="spyre",
                        target_dtype=target_dtype,  # ty: ignore[unknown-argument]
                    ) as f_spyre,
                    safe_open(shard_path, framework="pt", device="cpu") as f_cpu,
                ):
                    ordered_keys = f_spyre.offset_keys()
                    cpu_keys = {
                        k for k in ordered_keys if _param_name_for(model, k) in cpu_fallback_names
                    }
                    if tp_size > 1:
                        yield from _iter_tp_sharded_weights(
                            model,
                            f_spyre,
                            f_cpu,
                            ordered_keys,
                            cpu_keys,
                            tp_size,
                            tp_rank,
                            target_dtype,
                            pool,
                            num_threads,
                        )
                    else:
                        reads = (
                            (
                                name,
                                partial(f_cpu.get_tensor, name)
                                if name in cpu_keys
                                else _CardRead(f_spyre, f_cpu, name, target_dtype),
                            )
                            for name in ordered_keys
                        )
                        yield from _read_ahead(reads, pool, num_threads)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Stream checkpoint weights directly to Spyre device and CPU fallback modules."""
        if hasattr(torch, "spyre") and hasattr(torch.spyre, "is_initialized"):
            assert torch.spyre.is_initialized(), (
                "SpyreModelLoader called before torch_spyre initialization. "
                "Ensure SpyreWorkerV2.init_device() runs first."
            )

        # Delegate to model.load_weights() so fused parameters are scattered by each layer's
        # weight_loader; closing stops the pool and the files when one of them fails.
        with closing(self.get_all_weights(model_config, model)) as weights:
            loaded_weights = model.load_weights(weights)  # ty: ignore[call-non-callable,union-attr]
        logger.info("SpyreModelLoader: delegated weight loading to model.load_weights()")
        _check_all_weights_loaded(model, loaded_weights)

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        # `auto` reads *.bin when a repo has no safetensors; this loader streams
        # safetensors only, so such repos load exactly as they do on the stock path.
        if not _has_safetensors_checkpoint(model_config, self.load_config):
            logger.info(
                "SpyreModelLoader: %s has no safetensors checkpoint; using DefaultModelLoader.",
                model_config.model,
            )
            stock_config = copy.copy(self.load_config)
            stock_config.load_format = "auto"
            return DefaultModelLoader(stock_config).load_model(vllm_config, model_config, prefix)

        # Before any module is built or moved; see keep_attention_buffers_on_host.
        from spyre_inference.v1.worker.model_load import keep_attention_buffers_on_host

        keep_attention_buffers_on_host()

        with set_default_torch_dtype(model_config.dtype):  # ty: ignore[invalid-argument-type]
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )

            # Swap non-host CPU parameters for uninitialized Spyre tensors before loading.
            host_names = _get_host_loaded_param_names(model)
            for module_name, module in model.named_modules():
                for param_name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    if full_name in host_names:
                        continue
                    spyre_param = torch.Tensor._make_subclass(
                        type(param),
                        torch.empty(param.shape, dtype=param.dtype, device="spyre"),
                        param.requires_grad,
                    )
                    spyre_param.__dict__.update(param.__dict__)
                    module._parameters[param_name] = spyre_param

            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, torch.device("cpu"))

        return model.eval()


def _get_host_loaded_param_names(model: nn.Module) -> set[str]:
    """Parameters that must stay on CPU rather than be swapped for an empty Spyre placeholder.

    - VocabParallelEmbedding weights are placed by the module's own ``_apply`` after
      ``model.to(spyre)``; a direct Spyre-to-Spyre copy of a vocab-sized table exceeds
      the per-core limit, so they must arrive from the host.
    - Everything outside a multimodal model's language backbone (vision tower,
      projector). Their ``process_weights_after_loading`` hooks run eager ops on the
      weight (e.g. Pixtral pads the patch embedding) and expect a host tensor, as under
      the stock loader. They are small, so keeping them on the host costs little.

    Pooling heads are not listed: pooling runners are excluded at the platform gate
    (torch-spyre#2971).
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    from spyre_inference.custom_ops.text_backbone import non_text_parameter_names

    names = non_text_parameter_names(model)
    for module_name, module in model.named_modules():
        if isinstance(module, VocabParallelEmbedding):
            for param_name in module._parameters:
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def _hf_to_vllm_name(model: nn.Module, key: str) -> str:
    """A checkpoint key after the model's ``hf_to_vllm_mapper`` (a no-op without one).

    ``get_all_weights`` sees raw checkpoint keys, but the mapper runs inside
    ``model.load_weights``, after the loader has already chosen host or device per key.
    A key the mapper drops is returned unchanged.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return key
    mapped = mapper.apply_list([key])
    return mapped[0] if mapped else key


def _param_name_for(model: nn.Module, key: str) -> str:
    """The vLLM parameter name a checkpoint key loads into: mapper first, then the
    fused-layer name (``q_proj`` -> ``qkv_proj``) from ``packed_modules_mapping``."""
    name = _hf_to_vllm_name(model, key)
    for packed, shards in getattr(model, "packed_modules_mapping", {}).items():
        for shard in shards:
            if f".{shard}." in name:
                return name.replace(f".{shard}.", f".{packed}.", 1)
    return name


def _check_all_weights_loaded(model: nn.Module, loaded_weights: set[str] | None) -> None:
    """Fail if a parameter received no checkpoint tensor, as DefaultModelLoader does.

    ``load_model`` swaps every non-host parameter for an uninitialized
    ``torch.empty(..., device="spyre")`` placeholder, so a parameter the checkpoint never
    fills (renamed key, WeightsMapper miss, skipped prefix) would otherwise serve garbage
    silently. ``loaded_weights`` is what ``model.load_weights()`` returned; a model that
    does not track it (None) cannot be checked.
    """
    if loaded_weights is None:
        logger.warning(
            "SpyreModelLoader: %s.load_weights() did not report which weights it loaded; "
            "skipping the uninitialized-parameter check (as the stock loader does). A "
            "parameter missing from the checkpoint would keep its uninitialized placeholder.",
            type(model).__name__,
        )
        return
    weights_not_loaded = {name for name, _ in model.named_parameters()} - loaded_weights
    if weights_not_loaded:
        raise ValueError(
            f"Following weights were not initialized from checkpoint: {sorted(weights_not_loaded)}"
        )


@dataclass
class _CardRead:
    """One checkpoint tensor bound for the card, optionally only ``index`` of it."""

    f_spyre: Any
    f_cpu: Any
    name: str
    target_dtype: torch.dtype
    index: tuple[slice, ...] | None = None

    def _read(self, handle) -> torch.Tensor:
        if self.index is None:
            return handle.get_tensor(self.name)
        return handle.get_slice(self.name)[self.index]

    def __call__(self) -> torch.Tensor:
        return self._read(self.f_spyre)

    def alloc(self) -> torch.Tensor | None:
        """Allocate the card tensor the hook would, or None where only the hook knows how."""
        from safetensors.torch import _TYPES
        from torch_spyre._C import SpyreTensorLayout, spyre_empty_with_layout
        from torch_spyre._monkey_patch import _classify_safetensors_key

        source = self.f_cpu.get_slice(self.name)
        shape = list(source.get_shape())
        if self.index is not None:
            shape = [len(range(*i.indices(n))) for i, n in zip(self.index, shape, strict=True)]
        kind = _classify_safetensors_key(self.name, len(shape))
        if kind == "embedding" or not shape or 0 in shape:
            return None
        dtype = cast(torch.dtype, _TYPES[source.get_dtype()])
        if dtype.is_floating_point:
            dtype = self.target_dtype
        stride = list(torch.empty(shape, device="meta").stride())
        if kind == "linear":
            layout = SpyreTensorLayout(shape, stride, dtype, [1, 0])
        else:
            layout = SpyreTensorLayout(shape, dtype)
        # A block the loading thread freed may still be read by a copy queued on the
        # default stream; drain it before the allocator can hand that block to a pool stream.
        torch.spyre.default_stream().synchronize()
        return spyre_empty_with_layout(torch.Size(shape), tuple(stride), dtype, layout, device=None)

    def fill(self, dst: torch.Tensor) -> torch.Tensor:
        from torch_spyre._C import copy_tensor

        copy_tensor(self._read(self.f_cpu).contiguous(), dst, non_blocking=False)
        return dst


def _use_own_stream(device_index: int) -> None:
    """Pool-thread initializer: flex allows one injecting thread per stream."""
    from torch_spyre._C import get_stream_from_pool, set_current_stream, set_device

    set_device(device_index)
    set_current_stream(get_stream_from_pool(torch.device("spyre", device_index), 0))


def _read_ahead(
    reads: Iterable[tuple[Any, Callable[[], torch.Tensor]]],
    pool: ThreadPoolExecutor | None,
    num_threads: int,
) -> Generator[tuple[Any, torch.Tensor]]:
    """Yield ``(tag, read())`` in order, with up to ``num_threads`` reads running ahead."""
    if pool is None:
        for tag, read in reads:
            yield tag, read()
        return

    # Nothing orders two streams, so a pool thread (own stream) must never allocate card
    # memory: this thread allocates each destination (_CardRead.alloc), or reads it itself.
    on_card = hasattr(torch, "spyre") and torch.spyre.is_initialized()

    def submit(read) -> Future:
        if not (on_card and isinstance(read, _CardRead)):
            return pool.submit(read)
        dst = read.alloc()
        if dst is not None:
            return pool.submit(read.fill, dst)
        done: Future = Future()
        done.set_result(read())
        return done

    pending = deque()
    try:
        for tag, read in reads:
            pending.append((tag, submit(read)))
            if len(pending) >= num_threads:
                tag, future = pending.popleft()
                yield tag, future.result()
        while pending:
            tag, future = pending.popleft()
            yield tag, future.result()
    finally:
        # The pool outlives the file, so the file must not close under a running read.
        wait([future for _, future in pending if not future.cancel()])


# ---------------------------------------------------------------------------
# Tensor-parallel shard-on-read helpers (tp_size > 1)
# ---------------------------------------------------------------------------

_REPLICATED = "replicated"  # tensor is identical on every rank: load it whole


def _get_submodule(model: nn.Module, path: str) -> nn.Module | None:
    try:
        return model.get_submodule(path)
    except AttributeError:
        return None


def _resolve_module(model: nn.Module, ckpt_name: str) -> tuple[nn.Module | None, str | None]:
    """Map a checkpoint ``*.weight`` key to (module, packed shard name).

    Fused layers (q/k/v -> qkv_proj, gate/up -> gate_up_proj) are resolved through
    the model's ``packed_modules_mapping``. The second element is the original
    shard name (e.g. ``"q_proj"``) when the key belongs to a fused layer, and None when
    the key already names the layer itself, i.e. the checkpoint stores the fused tensor
    (Phi-3 maps ``qkv_proj`` to ``["qkv_proj"]``).
    """
    if not ckpt_name.endswith(".weight"):
        return None, None
    path = ckpt_name[: -len(".weight")]
    for packed, shards in getattr(model, "packed_modules_mapping", {}).items():
        for shard in shards:
            if path.endswith(f".{shard}"):
                return _get_submodule(model, path[: -len(shard)] + packed), (
                    None if shard == packed else shard
                )
    return _get_submodule(model, path), None


def _shard_plan(model: nn.Module, ckpt_name: str, shape, tp_size: int, tp_rank: int):
    """Decide how this rank should read ``ckpt_name``.

    Returns ``(param, dim, start, size)`` for a tensor that can be sharded on read,
    ``_REPLICATED`` for a tensor every rank loads whole, or ``None`` when we cannot
    pre-shard it safely (caller falls back to the host path).
    """
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    module, shard = _resolve_module(model, ckpt_name)
    if module is None:
        return None
    param = getattr(module, "weight", None)

    if isinstance(module, QKVParallelLinear):
        if shard is None:  # checkpoint already stores a fused qkv tensor
            return None
        head_size = module.head_size
        if shard.startswith("q"):
            size, index = module.num_heads * head_size, tp_rank
        else:  # k / v: kv heads may be replicated across ranks
            size = module.num_kv_heads * head_size
            index = tp_rank // module.num_kv_head_replicas
        return param, 0, index * size, size
    if isinstance(module, MergedColumnParallelLinear) and shard is None:
        # A fused [gate; up] checkpoint tensor: one contiguous shape[0] // tp_size block
        # is not this rank's share of each half. Take the host path.
        return None
    if isinstance(module, (MergedColumnParallelLinear, ColumnParallelLinear)):
        size = shape[0] // tp_size
        return param, 0, tp_rank * size, size
    if isinstance(module, RowParallelLinear):
        size = shape[1] // tp_size
        return param, 1, tp_rank * size, size
    if not hasattr(module, "tp_size") and not hasattr(module, "tp_rank"):
        return _REPLICATED  # norms etc.
    return None


def _iter_tp_sharded_weights(
    model: nn.Module,
    f_spyre,
    f_cpu,
    ordered_keys: list[str],
    cpu_keys: set[str],
    tp_size: int,
    tp_rank: int,
    target_dtype: torch.dtype,
    pool: ThreadPoolExecutor | None,
    num_threads: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield checkpoint tensors for TP>1, reading only this rank's shard.

    ``f_spyre.get_slice(name)[...]`` reads the slice on the host and DMAs just that
    shard with Spyre's optimal layout. vLLM's parameter loaders then narrow the
    tensor by ``param.tp_rank * shard_size``; we set ``param.tp_rank = 0`` for the
    duration of that single call so the narrow spans the whole (already sharded)
    tensor, the same full-range narrow that works at TP=1.

    Anything we cannot pre-shard safely is yielded from the host handle; the stock
    loader narrows it on the CPU and the copy into the Spyre param is host->device.
    """
    from vllm.model_executor.parameter import BasevLLMParameter

    def reads():
        for name in ordered_keys:
            if name in cpu_keys:
                yield (name, None), partial(f_cpu.get_tensor, name)
                continue

            shape = f_cpu.get_slice(name).get_shape()
            plan = _shard_plan(model, _hf_to_vllm_name(model, name), shape, tp_size, tp_rank)

            if plan is _REPLICATED:
                yield (name, None), _CardRead(f_spyre, f_cpu, name, target_dtype)
                continue
            if plan is None or not isinstance(plan[0], BasevLLMParameter):
                yield (name, None), partial(f_cpu.get_tensor, name)
                continue

            param, dim, start, size = plan
            index = [slice(None)] * len(shape)
            index[dim] = slice(start, start + size)
            yield (name, param), _CardRead(f_spyre, f_cpu, name, target_dtype, tuple(index))

    with closing(_read_ahead(reads(), pool, num_threads)) as loaded:
        for (name, param), tensor in loaded:
            if param is None:
                yield name, tensor
                continue

            saved_rank = param.tp_rank
            param.tp_rank = 0
            try:
                # The consumer applies weight_loader before asking for the next item.
                yield name, tensor
            finally:
                param.tp_rank = saved_rank


def _has_safetensors_checkpoint(model_config: ModelConfig, load_config) -> bool:
    """Whether the checkpoint has safetensors shards (fetched first when remote)."""
    try:
        _resolve_checkpoint_files(model_config, load_config)
    except FileNotFoundError:
        return False
    return True


def _resolve_checkpoint_files(
    model_config: ModelConfig,
    load_config,
) -> tuple[str, list[str]]:
    """Resolve and filter shard file paths for the checkpoint."""
    import glob as glob_mod

    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
    from vllm.model_executor.model_loader.weight_utils import (
        download_safetensors_index_file_from_hf,
        download_weights_from_hf,
        filter_duplicate_safetensors_files,
    )

    model_path = model_config.model
    revision = model_config.revision
    is_local = os.path.isdir(model_path)

    if not is_local:
        model_path = download_weights_from_hf(
            model_name_or_path=model_path,
            cache_dir=load_config.download_dir,
            allow_patterns=["*.safetensors"],
            revision=revision,
            ignore_patterns=load_config.ignore_patterns,
        )

    shard_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(
            f"No safetensors checkpoint shards found in {model_path}. SpyreModelLoader "
            "reads safetensors only; for .bin checkpoints pass the necessary "
            "--load-format pt."
        )
    if not is_local and len(shard_files) > 1:
        download_safetensors_index_file_from_hf(
            model_config.model,
            SAFE_WEIGHTS_INDEX_NAME,
            cache_dir=load_config.download_dir,
            revision=revision,
        )
    shard_files = sorted(
        filter_duplicate_safetensors_files(shard_files, model_path, SAFE_WEIGHTS_INDEX_NAME)
    )

    return model_path, shard_files
