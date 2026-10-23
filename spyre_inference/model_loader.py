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
from typing import TYPE_CHECKING, Iterable

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
        """
        from safetensors import safe_open

        target_dtype = getattr(model_config, "dtype", torch.float16)
        cpu_fallback_names = _get_host_loaded_param_names(model)
        _, shard_files = _resolve_checkpoint_files(
            model_config, self.load_config
        )

        for shard_path in shard_files:
            if not shard_path.endswith(".safetensors"):
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                for name, tensor in state_dict.items():
                    if name not in cpu_fallback_names:
                        try:
                            from torch_spyre._monkey_patch import _spyre_tensor_from_safetensors

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
                if cpu_keys:
                    f_cpu = safe_open(shard_path, framework="pt", device="cpu")

                try:
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
        raise FileNotFoundError(
            f"No checkpoint shards found in {model_path}"
        )

    return model_path, shard_files
