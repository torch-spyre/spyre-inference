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

"""``--load-format spyre_native``: load through a cache of converted Spyre weights.

Every load restores the model from its cache (``spyre_native_cache``) onto a
model built on meta. Without a cache, the model is loaded on the host (with
fastsafetensors), converted into a new cache and discarded first. If the cache cannot
be written, the host model moves to Spyre directly; if it does not match, the
model loads normally.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import time

import regex as re
import torch
import torch.nn as nn
from vllm import envs as vllm_envs
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from spyre_inference import envs
from spyre_inference.v1.worker.model_build import build_model
from spyre_inference.v1.worker.model_load import move_model_to_spyre

logger = init_logger(__name__)


@register_model_loader("spyre_native")
class SpyreNativeModelLoader(BaseModelLoader):
    """Returns the model already on Spyre, unlike loaders whose model is moved after."""

    def _host_config(self, model_config: ModelConfig) -> LoadConfig:
        """How a cache miss loads the checkpoint onto the host.

        fastsafetensors reads the whole checkpoint up front at disk speed; the
        mmap of "auto" leaves pages to fault in during post-processing, which is
        much slower from disk. Checkpoints without safetensors files use "auto".
        """
        load_config = copy.copy(self.load_config)
        load_config.device = "cpu"
        load_config.load_format = "fastsafetensors"
        try:
            DefaultModelLoader(load_config)._prepare_weights(
                model_config.model, None, model_config.revision, False, None
            )
        except (RuntimeError, ValueError):
            load_config.load_format = "auto"
        return load_config

    def download_model(self, model_config: ModelConfig) -> None:
        DefaultModelLoader(self._host_config(model_config)).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError(
            "spyre_native loads whole models only; reload weights with --load-format auto"
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
        device: torch.device | None = None,
    ) -> nn.Module:
        from fastsafetensors.copier.spyre import TransferRecorder

        from spyre_inference.v1.worker.spyre_native_cache import (
            cache_exists,
            restore_cache,
            write_cache,
        )

        device = device or torch.device("spyre")
        load_config = self._host_config(model_config)
        key = _cache_key(vllm_config, model_config, load_config)
        cache_dir = envs.SPYRE_NATIVE_CACHE_DIR or os.path.join(
            vllm_envs.VLLM_CACHE_ROOT, "spyre_native"
        )
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        slug = re.sub(r"[^A-Za-z0-9._]+", "_", model_config.model).strip("_")
        base = os.path.join(cache_dir, f"{slug}-rank{key['rank']}-{digest}")

        # Layers register themselves by name at construction, and get_rope hands
        # out one shared instance per configuration, so a model built and discarded
        # here must deregister, and must not leak its RoPE modules (holding its
        # device state) into the next build.
        compilation_config = vllm_config.compilation_config
        registered = set(compilation_config.static_forward_context)
        num_moe_layers = len(compilation_config.static_all_moe_layers)
        ropes = set(_ROPE_DICT)

        def forget_discarded_model() -> None:
            for name in set(compilation_config.static_forward_context) - registered:
                del compilation_config.static_forward_context[name]
            del compilation_config.static_all_moe_layers[num_moe_layers:]
            for rope_key in set(_ROPE_DICT) - ropes:
                del _ROPE_DICT[rope_key]
            # Frees its host weights and, after write_cache, its device buffers.
            gc.collect()

        if not cache_exists(base):
            logger.info("spyre_native: no cache at %s; converting the model.", base)
            model = build_model(load_config, vllm_config, model_config)
            if not write_cache(model, device, base, key):
                return model
            del model
            forget_discarded_model()

        meta_config = copy.copy(load_config)
        meta_config.device = "meta"
        model = build_model(meta_config, vllm_config, model_config)
        if restore_cache(model, device, base, key):
            return model
        del model
        forget_discarded_model()
        model = build_model(load_config, vllm_config, model_config)
        t0 = time.time()
        recorder = TransferRecorder(device)
        move_model_to_spyre(model, device, transfer=recorder)
        pipeline = recorder.fill()
        logger.info(
            "spyre_native: moved model to %s in %.1fs (%s)",
            device,
            time.time() - t0,
            pipeline.summary(),
        )
        return model


def _cache_key(vllm_config: VllmConfig, model_config: ModelConfig, load_config: LoadConfig) -> dict:
    """Everything that shapes the converted weights, so a stale cache never matches."""
    import fastsafetensors
    import torch_spyre
    import vllm

    import spyre_inference

    prepared = DefaultModelLoader(load_config)._prepare_weights(
        model_config.model, None, model_config.revision, True, None
    )

    weight_files = prepared[1]

    def stat(path: str) -> list:
        st = os.stat(path)
        return [path, st.st_size, st.st_mtime_ns]

    def source_digest(package) -> str:
        digest = hashlib.sha256()
        root = os.path.dirname(package.__file__)
        for dirpath, _, files in sorted(os.walk(root)):
            for name in sorted(files):
                if name.endswith(".py"):
                    with open(os.path.join(dirpath, name), "rb") as f:
                        digest.update(name.encode() + f.read())
        return digest.hexdigest()

    # Native libraries doing the conversion (torch-spyre, flex, deeptools, ...).
    with open("/proc/self/maps") as maps:
        libs = sorted(
            {
                line.split()[-1]
                for line in maps
                if ".so" in line
                and any(k in line for k in ("spyre", "flex", "deeptools", "senlib", "sentient"))
            }
        )
    return {
        "model": model_config.model,
        "revision": model_config.revision,
        "weights": [stat(f) for f in sorted(weight_files)],
        "hf_config": model_config.hf_config.to_json_string(),
        "hf_overrides": repr(model_config.hf_overrides),
        "dtype": str(model_config.dtype),
        "max_model_len": model_config.max_model_len,
        "quantization": model_config.quantization,
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
        "rank": get_tensor_model_parallel_rank(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("SPYRE_")},
        "versions": [vllm.__version__, torch.__version__, fastsafetensors.__version__],
        "sources": [
            source_digest(spyre_inference),
            source_digest(torch_spyre),
            source_digest(fastsafetensors),
        ],
        "libs": [stat(lib) for lib in libs if os.path.exists(lib)],
    }
