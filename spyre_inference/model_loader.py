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
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
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
        hf_folder, shard_files = _resolve_checkpoint_files(
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

        # Delegate to model.load_weights() so stacked/fused parameters
        # (qkv_proj, gate_up_proj, …) are scattered correctly by each layer's
        # weight_loader.  Fall back to _bind_weight for models that don't
        # implement load_weights.
        weights = self.get_all_weights(model_config, model)
        if callable(getattr(model, "load_weights", None)):
            model.load_weights(weights)
            logger.info("SpyreModelLoader: delegated weight loading to model.load_weights()")
        else:
            bound = 0
            skipped: List[str] = []
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
            # Build model on CPU. Module __init__ bodies (RoPE inv_freq, cos/sin
            # cache) run CPU-only ops (pow, arange) that Spyre doesn't implement.
            # _SpyreRotaryMixin intentionally keeps cos_sin_cache on CPU and only
            # primes the device rotation cache in _apply, so this must stay CPU.
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )

            # Prevent Attention scale buffers (_k_scale, _v_scale) from moving to Spyre
            for module in model.modules():
                if isinstance(module, Attention):
                    module._apply = lambda fn, recurse=True, _m=module: _m

            # Replace CPU parameters with uninitialized Spyre tensors of the same
            # shape and dtype.  This is a device allocation, not a copy: no weight
            # data moves host -> device.  weight_loader then copies Spyre -> Spyre.
            #
            # Skipped for "host-loaded" parameters (CPU-fallback layers and the
            # vocab-sized embedding / lm_head tables).  Those stay on CPU so the
            # runner's model.to(spyre) can place them through their own _apply
            # (row-gathered layout etc.); a plain Spyre -> Spyre copy of the vocab
            # table exceeds the per-core limit.  Buffers (RoPE cache, ...) are also
            # left for that same model.to(spyre), which runs after fix_padded_*.
            host_names = _get_host_loaded_param_names(model)
            for module_name, module in model.named_modules():
                for param_name, param in list(module._parameters.items()):
                    if param is None:
                        continue
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    if full_name in host_names:
                        continue
                    # Keep the original Parameter class: vLLM's BasevLLMParameter
                    # subclasses expose weight_loader / load_*_weight as class
                    # members, which a plain nn.Parameter lacks.  _make_subclass
                    # skips __init__, so restore the instance state via __dict__.
                    spyre_param = torch.Tensor._make_subclass(
                        type(param),
                        torch.empty(param.shape, dtype=param.dtype, device="spyre"),
                        param.requires_grad,
                    )
                    spyre_param.__dict__.update(param.__dict__)
                    module._parameters[param_name] = spyre_param

            self.load_weights(model, model_config)

            # CPU target, as DefaultModelLoader does: a non-CPU target makes
            # device_loading_context shuttle CPU-resident modules to the device.
            # Errors are not swallowed so a failing layer is visible.
            process_weights_after_loading(model, model_config, torch.device("cpu"))

        return model.eval()


def _get_host_loaded_param_names(model: nn.Module) -> Set[str]:
    """Parameters that must be loaded on CPU and placed later by the module's own
    ``_apply``: CPU-fallback params plus the vocab-sized embedding / lm_head tables."""
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    names = _get_cpu_fallback_param_names(model)
    for module_name, module in model.named_modules():
        if isinstance(module, VocabParallelEmbedding):  # includes ParallelLMHead
            for param_name in module._parameters:
                names.add(f"{module_name}.{param_name}" if module_name else param_name)
    return names


def _get_cpu_fallback_param_names(model: nn.Module) -> Set[str]:
    """Identify parameter names that must stay on CPU."""
    try:
        from spyre_inference.custom_ops import SpyreCpuFallbackMixin
    except ImportError:
        return set()

    cpu_names: Set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, SpyreCpuFallbackMixin):
            for param_name in module._parameters:
                if param_name:
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    cpu_names.add(full_name)
    return cpu_names


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
                # Parameters are allocated on CPU (platform device_type is "cpu"),
                # and Parameter.data cannot be swapped across backends, so replace
                # the Parameter object, keeping any attributes set on the original.
                new_param = nn.Parameter(tensor, requires_grad=param.requires_grad)
                new_param.__dict__.update(param.__dict__)
                setattr(module, parts[-1], new_param)
        else:
            param.data.copy_(tensor)
    return True




def _resolve_checkpoint_files(
    model_config: ModelConfig,
    load_config,
) -> Tuple[str, List[str]]:
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
        # Drop shards not listed in the index (e.g. a consolidated file next to
        # sharded ones), as the stock DefaultModelLoader does.
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
