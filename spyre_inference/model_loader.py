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

import os
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)


@register_model_loader("spyre")
class SpyreModelLoader(BaseModelLoader):
    """Streams weights directly from safetensors to Spyre device memory."""

    def download_model(self, model_config: ModelConfig) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            download_weights_from_hf,
        )

        download_weights_from_hf(
            model_name_or_path=model_config.model,
            cache_dir=self.load_config.download_dir,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=model_config.revision,
        )

    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from checkpoint shards using physical disk order.

        Tensors destined for Spyre are loaded via safe_open(device="spyre"),
        which triggers _SpyreSafeOpen to stream each tensor with optimal Spyre layout.
        CPU-fallback parameters remain on CPU via safe_open(device="cpu").

        With tensor parallelism (tp_size > 1) linear weights are sharded on read:
        only this rank's slice is fetched via ``get_slice`` and DMA'd to the device,
        because vLLM's weight loaders cannot narrow a tensor that already lives on
        the device in Spyre's stick layout.
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

        for shard_path in shard_files:
            if not shard_path.endswith(".safetensors"):
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                for name, tensor in state_dict.items():
                    # With TP>1 keep tensors on the host: the stock loader narrows
                    # on CPU and the copy into the Spyre param is host->device.
                    if name not in cpu_fallback_names and tp_size == 1:
                        try:
                            from torch_spyre._monkey_patch import (
                                _spyre_tensor_from_safetensors,
                            )

                            tensor = _spyre_tensor_from_safetensors(
                                tensor, name, "spyre", target_dtype=target_dtype
                            )
                        except (ImportError, AttributeError):
                            pass
                    yield name, tensor
                continue

            with safe_open(
                shard_path,
                framework="pt",
                device="spyre",
                target_dtype=target_dtype,
            ) as f_spyre:
                ordered_keys = f_spyre.offset_keys()
                cpu_keys = {k for k in ordered_keys if k in cpu_fallback_names}
                f_cpu = None
                if cpu_keys or tp_size > 1:
                    f_cpu = safe_open(shard_path, framework="pt", device="cpu")

                try:
                    if tp_size > 1:
                        yield from _iter_tp_sharded_weights(
                            model, f_spyre, f_cpu, ordered_keys, cpu_keys, tp_size, tp_rank
                        )
                    else:
                        for name in ordered_keys:
                            if name in cpu_keys and f_cpu is not None:
                                tensor = f_cpu.get_tensor(name)
                            else:
                                tensor = f_spyre.get_tensor(name)
                            yield name, tensor
                finally:
                    if f_cpu is not None:
                        f_cpu.__exit__(None, None, None)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Stream checkpoint weights directly to Spyre device and CPU fallback modules."""
        if hasattr(torch, "spyre") and hasattr(torch.spyre, "is_initialized"):
            assert torch.spyre.is_initialized(), (
                "SpyreModelLoader called before torch_spyre initialization. "
                "Ensure TorchSpyreWorker.init_device() runs first."
            )

        # Delegate to model.load_weights() so fused parameters are scattered
        # by each layer's weight_loader.
        weights = self.get_all_weights(model_config, model)
        if callable(getattr(model, "load_weights", None)):
            model.load_weights(weights)
            logger.info("SpyreModelLoader: delegated weight loading to model.load_weights()")
        else:
            bound = 0
            skipped: list[str] = []
            for name, tensor in weights:
                if _bind_weight(model, name, tensor):
                    bound += 1
                else:
                    skipped.append(name)
            logger.info(
                "SpyreModelLoader: bound %d tensors to model parameters, skipped %d",
                bound,
                len(skipped),
            )
            if skipped:
                logger.warning(
                    "SpyreModelLoader: %d checkpoint keys matched no model parameter "
                    "(first 10: %s)",
                    len(skipped),
                    skipped[:10],
                )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
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

    VocabParallelEmbedding weights are placed by the module's own ``_apply`` after
    ``model.to(spyre)``; a direct Spyre-to-Spyre copy of a vocab-sized table exceeds
    the per-core limit, so they must arrive from the host.

    NOTE: This list covers only what SpyreModelLoader currently supports — TP=1,
    unquantized, dense, generate-only models. Pooling classifier/reranker heads and
    vision-tower params are deliberately not listed here; those model classes are
    excluded from ``load_format=spyre`` at the platform gate
    (``apply_config_platform_defaults``) until their host-residency needs are mapped.
    If the gate is widened to include pooling or multimodal models, expand this list
    to cover their fp32 heads (spyre_pooler._downcast_module_to_fp16, torch-spyre#2971)
    and any vision-tower params that must not land on Spyre directly.
    """ 
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, VocabParallelEmbedding):
            for param_name in module._parameters:
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


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
    shard name (e.g. ``"q_proj"``) when the key belongs to a fused layer.
    """
    if not ckpt_name.endswith(".weight"):
        return None, None
    path = ckpt_name[: -len(".weight")]
    for packed, shards in getattr(model, "packed_modules_mapping", {}).items():
        for shard in shards:
            if path.endswith(f".{shard}"):
                return _get_submodule(model, path[: -len(shard)] + packed), shard
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

    for name in ordered_keys:
        if name in cpu_keys:
            yield name, f_cpu.get_tensor(name)
            continue

        shape = f_cpu.get_slice(name).get_shape()
        plan = _shard_plan(model, name, shape, tp_size, tp_rank)

        if plan is _REPLICATED:
            yield name, f_spyre.get_tensor(name)
            continue
        if plan is None or not isinstance(plan[0], BasevLLMParameter):
            yield name, f_cpu.get_tensor(name)
            continue

        param, dim, start, size = plan
        index = [slice(None)] * len(shape)
        index[dim] = slice(start, start + size)
        tensor = f_spyre.get_slice(name)[tuple(index)]

        saved_rank = param.tp_rank
        param.tp_rank = 0
        try:
            # The consumer applies weight_loader before asking for the next item.
            yield name, tensor
        finally:
            param.tp_rank = saved_rank


def _bind_weight(model: nn.Module, name: str, tensor: torch.Tensor) -> bool:
    """Bind loaded tensor to model parameter via weight_loader or assignment.

    Returns False when the key matches no model parameter.
    """
    parts = name.split(".")
    try:
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        param = getattr(module, parts[-1], None)
    except AttributeError:
        logger.debug("_bind_weight: key %s not found in model — skipping", name)
        return False

    if param is None:
        logger.debug("_bind_weight: key %s resolved to None — skipping", name)
        return False

    if hasattr(param, "weight_loader"):
        param.weight_loader(param, tensor)
    else:
        if isinstance(param, nn.Parameter):
            if param.device == tensor.device:
                param.data = tensor
            else:
                new_param = nn.Parameter(tensor, requires_grad=param.requires_grad)
                new_param.__dict__.update(param.__dict__)
                setattr(module, parts[-1], new_param)
        else:
            param.data.copy_(tensor)
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
        filter_files_not_needed_for_inference,
    )

    model_path = model_config.model
    revision = model_config.revision
    is_local = os.path.isdir(model_path)

    if not is_local:
        model_path = download_weights_from_hf(
            model_name_or_path=model_path,
            cache_dir=load_config.download_dir,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=revision,
            ignore_patterns=load_config.ignore_patterns,
        )

    shard_files = glob_mod.glob(os.path.join(model_path, "*.safetensors"))
    if shard_files:
        if not is_local and len(shard_files) > 1:
            download_safetensors_index_file_from_hf(
                model_config.model,
                SAFE_WEIGHTS_INDEX_NAME,
                cache_dir=load_config.download_dir,
                revision=revision,
            )
        shard_files = filter_duplicate_safetensors_files(
            shard_files, model_path, SAFE_WEIGHTS_INDEX_NAME
        )
    else:
        shard_files = glob_mod.glob(os.path.join(model_path, "*.bin"))
        shard_files = filter_files_not_needed_for_inference(shard_files)
    shard_files = sorted(shard_files)

    if not shard_files:
        raise FileNotFoundError(f"No checkpoint shards found in {model_path}")

    return model_path, shard_files
