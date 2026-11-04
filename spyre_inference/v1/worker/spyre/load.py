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
duration of the call, and upstream's own whole-model stock compile is skipped.
"""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from vllm.config import CompilationMode, LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
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
from spyre_inference.v1.worker.spyre.model_states import resolve_spyre_model_state
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _SpyreModelWrapper,
    model_dtype,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    SpyreShapeBucketer,
    encoder_len_ladder,
    logits_row_buckets,
)

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


class _SpyreModelLoaderWrapper:
    """Wraps the BaseModelLoader that vLLM's get_model_loader picked for the load
    format, and adds the Spyre steps to its load_model, which returns the
    _SpyreModelWrapper. Holds the instance rather than subclassing, so every load
    format keeps its own loader; everything else forwards to it."""

    def __init__(self, loader: BaseModelLoader, runner: "SpyreModelRunnerV2") -> None:
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
        return _SpyreModelWrapper(  # ty: ignore[invalid-return-type]
            compiled,
            runner.spyre_device,
            keep_outputs_on_device=runner.pooling_on_spyre,
            logits_row_buckets=row_buckets,
            shape_bucketer=bucketer,
            model_dtype=model_dtype(model_config),
        )


def _required(runner: "SpyreModelRunnerV2") -> dict[str, bool]:
    """Each swapped name, in the order load_model reaches it, and whether this load
    must reach it (upstream skips the pooling runner and the stock compile otherwise).
    The keys are the whole set of swaps: _patched refuses a replacement missing here."""
    return {
        "get_model_loader": True,
        "init_model_state": True,
        "PoolingRunner": runner.is_pooling_model,
        "compile_model_with_stock_torch": (
            runner.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE
        ),
    }


@contextlib.contextmanager
def _patched(runner: "SpyreModelRunnerV2") -> Iterator[None]:
    """Every upstream name Spyre swaps during GPUModelRunner.load_model.

    Each is a module global of vllm/v1/worker/gpu/model_runner.py that load_model
    looks up at call time. If a vLLM bump imports one locally instead, the swap is a
    silent no-op, so the swaps load_model must reach have to report that they ran.
    """
    required = _required(runner)
    original = {name: getattr(upstream_model_runner, name) for name in required}
    called: set[str] = set()

    def wrapped_model_loader(load_config: LoadConfig) -> _SpyreModelLoaderWrapper:
        called.add("get_model_loader")
        return _SpyreModelLoaderWrapper(original["get_model_loader"](load_config), runner)

    def spyre_init_model_state(vllm_config, model, encoder_cache, device):
        called.add("init_model_state")
        return resolve_spyre_model_state(
            vllm_config, model, encoder_cache, device, runner.spyre_device
        )

    class _SpyrePoolingRunner(SpyrePoolingRunner):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            called.add("PoolingRunner")
            super().__init__(*args, **kwargs)

    def no_stock_compile(model: Any, vllm_config: Any) -> None:
        # Would trace the whole model over the per-block graphs, through vLLM's own
        # backend rather than Spyre's inductor; compile_for_spyre has already run.
        called.add("compile_model_with_stock_torch")

    replacements = {
        "get_model_loader": wrapped_model_loader,
        "init_model_state": spyre_init_model_state,
        "PoolingRunner": _SpyrePoolingRunner,
        "compile_model_with_stock_torch": no_stock_compile,
    }
    if replacements.keys() != required.keys():
        raise AssertionError(
            f"Spyre swaps {sorted(replacements)} but checks {sorted(required)}; "
            "a swap added to one needs its entry in the other."
        )
    for name, replacement in replacements.items():
        setattr(upstream_model_runner, name, replacement)
    try:
        yield
    except BaseException as e:
        # load_model failed partway, so the swaps after the failure were never due. One
        # missing before a swap that did run was bypassed, which is likely the cause.
        order = list(required)
        reached = max((order.index(name) + 1 for name in called), default=0)
        if bypassed := [n for n in order[:reached] if required[n] and n not in called]:
            e.add_note(_bypass_message(bypassed))
        raise
    finally:
        for name, value in original.items():
            setattr(upstream_model_runner, name, value)

    if missed := [n for n, needed in required.items() if needed and n not in called]:
        raise RuntimeError(_bypass_message(missed))


def _bypass_message(names: list[str]) -> str:
    return (
        f"Upstream GPUModelRunner.load_model bypassed Spyre's swap of {names}; it no "
        "longer reads them as module globals of vllm/v1/worker/gpu/model_runner.py."
    )


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
    if runner.parallel_config.enable_batch_sharded_sampling:
        # Upstream calls model.compute_logits_local, past the wrapper's H2D and row buckets.
        raise NotImplementedError("Batch-sharded sampling is not supported on the Spyre V2 runner.")
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
