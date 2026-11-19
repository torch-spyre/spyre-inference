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

"""Building a model on the host, ready for the move to Spyre."""

from __future__ import annotations

import time
from collections.abc import Generator
from functools import wraps
from typing import cast
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from vllm import envs as vllm_envs
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    filter_mm_encoder_only_safetensors_files,
    filter_safetensors_files_by_weight_name,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from spyre_inference import envs
from spyre_inference.v1.worker.model_load import fix_padded_model, install_pad_loaders


class _HostModelLoader(DefaultModelLoader):
    """CPU fastsafetensors loading used only while building a native cache."""

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        hf_folder, files, use_safetensors, index_file = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not use_safetensors:
            return super()._get_weights_iterator(source)
        if self._encoder_only_lm_prefixes is not None and files:
            files = filter_mm_encoder_only_safetensors_files(
                files,
                hf_folder,
                index_file,
                self._encoder_only_lm_prefixes,
                weights_mapper=self._encoder_only_weights_mapper,
            )
            if not files:
                raise RuntimeError(
                    "mm-encoder-only shard filter removed all weight files for "
                    f"`{source.model_or_path}`; check language_model prefixes "
                    f"{self._encoder_only_lm_prefixes}"
                )
        if source.is_unused_weight is not None:
            files = filter_safetensors_files_by_weight_name(files, source.is_unused_weight)
        weights = _host_fastsafetensors_weights_iterator(files, self.load_config.use_tqdm_on_load)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, tensor) for name, tensor in weights)


class _MetaHostCopies(TorchDispatchMode):
    """Keep meta-to-CPU copies symbolic until the native cache supplies their values."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func in (torch.ops.aten.to.device, torch.ops.aten.to.prim_Device)
            and args[0].is_meta
            and args[1] is not None
            and torch.device(args[1]).type == "cpu"
        ):
            args = (args[0], torch.device("meta"), *args[2:])
        if (
            func in (torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype_layout)
            and args[0].is_meta
            and kwargs.get("device") is not None
            and torch.device(kwargs["device"]).type == "cpu"
        ):
            kwargs = {**kwargs, "device": torch.device("meta")}
        return func(*args, **kwargs)


class SkeletonLoader(DefaultModelLoader):
    """Builds and post-processes the model on meta without reading any weights."""

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        attention_init = Attention.__init__

        @wraps(attention_init)
        def host_attention_init(layer, *args, **kwargs):
            # Attention stays on CPU; KV-scale finalization needs real scalar values.
            with torch.device("cpu"):
                attention_init(layer, *args, **kwargs)

        with set_default_torch_dtype(cast(torch.dtype, model_config.dtype)), _MetaHostCopies():
            with torch.device("meta"), patch.object(Attention, "__init__", host_attention_init):
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )
            self.load_weights(model, model_config)
            # Avoid upstream's offload context moving host Attention parameters to meta.
            process_weights_after_loading(model, model_config, torch.device("cpu"))
        return model.eval()


def build_model(
    load_config: LoadConfig, vllm_config: VllmConfig, model_config: ModelConfig
) -> nn.Module:
    """Build and load the model on the host, ready for the move to Spyre.

    With ``load_config.device == "meta"`` the model is only built (see
    ``SkeletonLoader``), for ``spyre_native_cache.restore_cache`` to fill.
    """
    if load_config.device == "meta":
        model_loader = SkeletonLoader(load_config)
    elif load_config.load_format == "fastsafetensors":
        model_loader = _HostModelLoader(load_config)
    else:
        model_loader = get_model_loader(load_config)

    install_pad_loaders(model_loader, model_config)

    # load_model reads the target device from vllm_config.load_config.
    saved_load_config = vllm_config.load_config
    vllm_config.load_config = load_config
    try:
        model = model_loader.load_model(vllm_config=vllm_config, model_config=model_config)
    finally:
        vllm_config.load_config = saved_load_config

    fix_padded_model(model, model_config)

    # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
    # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
    # an nn.Module, but just the attention implementation.
    Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

    return model


def _host_fastsafetensors_weights_iterator(
    hf_weights_files: list[str], use_tqdm_on_load: bool
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Read a native cache's source checkpoint onto the host, without GDS.

    By default ranks share file reads and broadcast tensors before TP slicing.
    SPYRE_NATIVE_HOST_ALL_LOCAL=1 instead makes each rank read every file locally.
    """
    from fastsafetensors import SingleGroup
    from fastsafetensors.parallel_loader import ParallelLoader
    from vllm.model_executor.model_loader.weight_utils import _natural_sort_key, enable_tqdm

    pg = torch.distributed.group.WORLD if torch.distributed.is_initialized() else SingleGroup()
    loader = ParallelLoader(
        pg=pg,
        hf_weights_files=sorted(hf_weights_files, key=_natural_sort_key),
        queue_size=vllm_envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE,
        use_tqdm_on_load=enable_tqdm(use_tqdm_on_load),
        device="cpu",
        nogds=True,
        all_local=envs.SPYRE_NATIVE_HOST_ALL_LOCAL,
    )
    try:
        yield from loader.iterate_weights()
    finally:
        loader.close()
