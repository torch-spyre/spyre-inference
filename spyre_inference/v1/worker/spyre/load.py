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

"""MRV2: SpyreModelRunnerV2.load_model (vllm/v1/worker/gpu/model_runner.py:
GPUModelRunner.load_model).

Upstream's load_model runs as-is around a wrapped model loader, which loads on the
host, applies the shared fix-ups, moves the model to Spyre, compiles per block and
returns the _SpyreModelWrapper. Everything upstream builds from ``self.model``
afterwards (model state, sampler, pooling runner) therefore holds the wrapper.
init_model_state and PoolingRunner are swapped for their Spyre counterparts for the
duration of the call, and compile.py stops upstream's own whole-model compile.
"""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.v1.worker.gpu import model_runner as upstream_model_runner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.prompt_logprob import CHUNK_SIZE as PROMPT_LOGPROBS_CHUNK_SIZE

from spyre_inference.models import check_spyre_model
from spyre_inference.multimodal import apply_multimodal_patches
from spyre_inference.v1.pool import configure_pooling_for_spyre
from spyre_inference.v1.worker.compile_granularity import compile_for_spyre
from spyre_inference.v1.worker.model_load import (
    fix_padded_model,
    install_pad_loaders,
    move_model_to_spyre,
    resolve_kv_scales,
)
from spyre_inference.v1.worker.spyre import compile
from spyre_inference.v1.worker.spyre.model_states import resolve_spyre_model_state
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _SpyreModelWrapper,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    SpyreShapeBucketer,
    encoder_len_ladder,
    logits_row_buckets,
)

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


class _SpyreModelLoader:
    """Upstream's model loader, returning the Spyre wrapper from load_model."""

    def __init__(self, loader: Any, runner: "SpyreModelRunnerV2") -> None:
        self._loader = loader
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def load_model(self, *, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        runner = self._runner
        install_pad_loaders(self._loader, model_config)
        model = self._loader.load_model(vllm_config=vllm_config, model_config=model_config)
        check_spyre_model(model, get_model_architecture(model_config)[1])
        fix_padded_model(model, model_config)
        # Before any trace: resolved lazily inside the block graph, the scales become a
        # guard on their unset value and the first real prefill recompiles the block.
        resolve_kv_scales(model, vllm_config.compilation_config.static_forward_context)
        move_model_to_spyre(model, runner.spyre_device)

        # CLS/LAST gather on Spyre. MEAN reduces each rectangle lane on Spyre and
        # falls back to the host elsewhere. FP32 linear heads stay on CPU.
        runner.pooling_on_spyre = model_config.runner_type == "pooling" and (
            configure_pooling_for_spyre(model, runner.spyre_device, encoder_len_ladder(vllm_config))
        )
        logger.info("Spyre-native layer weights moved to %s", runner.spyre_device)

        # Patches instances/classes, so it runs after load and before compile wraps modules
        # in OptimizedModule and breaks traversal.
        apply_multimodal_patches(model, runner.spyre_device)
        compiled, runner.compiled_tower_blocks = compile_for_spyre(model, vllm_config)

        bucketer = SpyreShapeBucketer.for_config(vllm_config)
        row_buckets = []
        if bucketer is not None:
            row_buckets = logits_row_buckets(bucketer.bucket_sizes, runner.max_num_reqs)
            if model_config.runner_type == "generate" and model_config.max_logprobs != 0:
                # Prompt logprobs project the prompt rows in chunks of up to CHUNK_SIZE.
                chunk = min(PROMPT_LOGPROBS_CHUNK_SIZE, runner.max_num_tokens)
                row_buckets = sorted({*row_buckets, chunk})
        dtype = model_config.dtype
        return _SpyreModelWrapper(  # ty: ignore[invalid-return-type]
            compiled,
            runner.spyre_device,
            keep_outputs_on_device=runner.pooling_on_spyre,
            logits_row_buckets=row_buckets,
            shape_bucketer=bucketer,
            model_dtype=dtype if isinstance(dtype, torch.dtype) else torch.float16,
        )


@contextlib.contextmanager
def _patched(runner: "SpyreModelRunnerV2") -> Iterator[None]:
    get_model_loader = upstream_model_runner.get_model_loader
    init_model_state = upstream_model_runner.init_model_state
    pooling_runner_cls = upstream_model_runner.PoolingRunner

    def spyre_model_loader(load_config: Any) -> _SpyreModelLoader:
        return _SpyreModelLoader(get_model_loader(load_config), runner)

    def spyre_init_model_state(vllm_config, model, encoder_cache, device):
        return resolve_spyre_model_state(
            vllm_config, model, encoder_cache, device, runner.spyre_device
        )

    upstream_model_runner.get_model_loader = spyre_model_loader  # ty: ignore[invalid-assignment]
    upstream_model_runner.init_model_state = spyre_init_model_state  # ty: ignore[invalid-assignment]
    upstream_model_runner.PoolingRunner = SpyrePoolingRunner  # ty: ignore[invalid-assignment]
    try:
        with compile.no_stock_compile():
            yield
    finally:
        upstream_model_runner.get_model_loader = get_model_loader
        upstream_model_runner.init_model_state = init_model_state
        upstream_model_runner.PoolingRunner = pooling_runner_cls


def load_and_wrap(runner: "SpyreModelRunnerV2", *args: Any, **kwargs: Any) -> None:
    """Load weights, move the model to Spyre, compile per block, and wrap.

    Postcondition: ``runner.model`` is the wrapper, and every other holder of the
    model (``model_state.model``, ``pooling_runner.model``) points at that same
    wrapper. ``runner.model_state`` comes from ``resolve_spyre_model_state``.
    """
    if kwargs.get("model") is not None:
        raise NotImplementedError("Loading a caller-built model is not supported on Spyre V2.")
    if runner.lora_config:
        raise NotImplementedError("LoRA adapters are not yet implemented and tested for Spyre.")
    if runner.speculator is not None:
        # Upstream loads a draft model through its own loader, outside the swap.
        raise NotImplementedError("Speculative decoding is not supported on the Spyre V2 runner.")

    with _patched(runner):
        GPUModelRunner.load_model(runner, *args, **kwargs)
    # The profiler measures host memory, which Spyre weights don't occupy.
    runner.model_memory_usage = 0

    model_state = runner.model_state
    pooling_runner = runner.pooling_runner
    if isinstance(pooling_runner, SpyrePoolingRunner):
        pooling_runner.pooling_on_spyre = runner.pooling_on_spyre
        pooling_runner.spyre_device = runner.spyre_device
        pooling_runner.encoder_plan_source = lambda: getattr(model_state, "encoder_plan", None)


def unwrap(model: Any) -> nn.Module:
    """The bare model under the wrapper and any whole-model OptimizedModule."""
    if isinstance(model, _SpyreModelWrapper):
        model = model._model
    return getattr(model, "_orig_mod", model)
