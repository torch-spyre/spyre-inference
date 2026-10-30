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

"""MRV2: V2 warmup driver (vllm/v1/worker/gpu_worker.py:Worker.compile_or_warm_up_model,
TorchSpyreModelRunner.warming_up_model).

Decoder dummies skip attention and publish null KV slots; pooling dummies keep
attention on; a separate pass then records attention over attn_groups and runs the
sampler dummy. plan() is kept apart from run() so the dummy-call list can be asserted
without executing a model.

MRV2 only allows skip_attn together with is_profile, and is_profile skips
cudagraph_manager.dispatch() and builds the dummy batch unpadded, so plan() picks
already-bucketed token counts itself.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
    allocate_staging_buffers,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import encoder_inline_active
from spyre_inference.v1.attention.warmup import record_attention_graphs
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.pool.warmup import warm_pooler_row_widths
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _set_spyre_compilation_settings,
    _SpyreModelWrapper,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer, encoder_shape_tables

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


@dataclass(frozen=True)
class DummyCall:
    """One _dummy_run() invocation in the warmup plan."""

    num_reqs: int
    num_tokens: int
    skip_attn: bool


def plan(runner: "SpyreModelRunnerV2") -> list[DummyCall]:
    """Build the ordered list of dummy calls warmup makes, without running any."""
    max_reqs = runner.max_num_reqs
    is_pooling = runner.is_pooling_model
    bucketer = SpyreShapeBucketer.for_config(runner.vllm_config)
    if bucketer is None or (is_pooling and runner.model_config.enforce_eager):
        sizes = [min(max(16, max_reqs), runner.max_num_tokens)]
    else:
        # Largest first: Inductor's caches benefit from seeing the most complex shape first.
        sizes = sorted(bucketer.bucket_sizes, reverse=True)
    return [DummyCall(min(n, max_reqs), n, skip_attn=not is_pooling) for n in sizes]


def _dummy_run(runner: "SpyreModelRunnerV2", num_tokens: int, skip_attn: bool):
    """MRV2's _dummy_run, with a pooling model's hidden states D2H'd for upstream's host
    logits_indices gather and handed back on Spyre, where the pooler reads them."""
    if skip_attn:
        # MRV2 builds no slot mappings with skip_attn, but the block graph's KV write
        # needs a target, or warmup traces a graph serving never runs.
        attn_layer.publish_null_slots(num_tokens)
    wrapper = runner.model
    keep = wrapper._keep_outputs_on_device
    object.__setattr__(wrapper, "_keep_outputs_on_device", False)
    try:
        hidden_states, sample_hidden_states = runner._dummy_run(
            num_tokens, skip_attn=skip_attn, is_profile=True
        )
    finally:
        object.__setattr__(wrapper, "_keep_outputs_on_device", keep)
    if keep and hidden_states is not None and hidden_states.numel() > 0:
        hidden_states = convert(hidden_states, runner.spyre_device)
    return hidden_states, sample_hidden_states


def _warm_pooling(runner: "SpyreModelRunnerV2", hidden_states: torch.Tensor) -> None:
    assert runner.pooling_runner is not None
    runner.pooling_runner.dummy_pooler_run(hidden_states)
    if not runner.pooling_on_spyre:
        return
    rectangles = list(encoder_shape_tables(runner.vllm_config).rectangles)
    warm_pooler_row_widths(
        cast(VllmModelForPooling, runner.get_model()).pooler,
        hidden_states,
        rectangles,
        runner.max_num_reqs,
    )
    if rectangles:
        # pool_rows_for's re-compaction gather keeps the buffer's row count.
        select_rows(hidden_states, torch.zeros(hidden_states.shape[0], dtype=torch.int64))


def _warm_encoder_inline_paths(runner: "SpyreModelRunnerV2", bodies: set[int]) -> None:
    """One block graph per rectangle of a warmed body, plus the opaque-attention one.

    With attention traced into the block graph, it specialises on the rectangle, so a
    body-shape dummy only covers whichever rectangle its own batch landed on.
    """
    forced_plan = getattr(runner.model_state, "forced_plan", None)
    if forced_plan is None or not encoder_inline_active():
        return
    tables = encoder_shape_tables(runner.vllm_config)
    for extent, width in sorted(tables.rectangles, reverse=True):
        if extent * width in bodies:
            with forced_plan(rect=(extent, width)):
                _dummy_run(runner, extent * width, skip_attn=False)
    for body, groups in tables.tier_groups:
        if groups and body in bodies:
            with forced_plan(ragged=True):
                _dummy_run(runner, body, skip_attn=False)


def _record_attention(runner: "SpyreModelRunnerV2") -> None:
    if not envs.SPYRE_ATTN_RECORD:
        logger.info("Attention graph recording disabled (SPYRE_ATTN_RECORD=0)")
        return
    if runner.compilation_config.mode is CompilationMode.NONE:
        logger.info("Attention graph recording disabled (CompilationMode.NONE)")
        return
    builders: dict[str, SpyreAttentionMetadataBuilder] = {}
    for groups in runner.attn_groups:
        for group in groups:
            builder = group.metadata_builders[0] if group.metadata_builders else None
            if isinstance(builder, SpyreAttentionMetadataBuilder):
                builders.update(dict.fromkeys(group.layer_names, builder))
    record_attention_graphs(
        runner.compilation_config.static_forward_context,
        runner.spyre_kv_caches,
        builders,
        # Speculative decoding, whose draft layers can be non-causal, is refused at load.
        causal_by_layer=dict.fromkeys(builders, True),
    )


def run(runner: "SpyreModelRunnerV2") -> None:
    """Execute plan(runner), then record attention and the sampler dummy."""
    allocate_staging_buffers(runner.compilation_config.static_forward_context, runner.spyre_device)
    calls = plan(runner)
    logger.info("MRV2 warmup: %d dummy runs %s", len(calls), [c.num_tokens for c in calls])
    t0 = time.time()
    # Under inference_mode like MRV2's execute_model: grad mode is a guard on every
    # compiled graph, so warming outside it misses each one at serving time.
    with _set_spyre_compilation_settings(runner.vllm_config), torch.inference_mode():
        widest = None
        for call in calls:
            hidden_states, sample_hidden_states = _dummy_run(
                runner, call.num_tokens, call.skip_attn
            )
            if runner.pooling_runner is not None:
                _warm_pooling(runner, hidden_states)
            elif widest is None:
                widest = sample_hidden_states
        if runner.pooling_runner is not None and not runner.model_config.enforce_eager:
            _warm_encoder_inline_paths(runner, {c.num_tokens for c in calls})
        if widest is not None:
            # The wrapper pads compute_logits onto these widths, so the lm_head and the
            # sampler compile one graph per width here instead of on first use.
            wrapper = cast(_SpyreModelWrapper, runner.model)
            for rows in sorted(wrapper._logits_row_buckets, reverse=True):
                runner._dummy_sampler_run(widest[: min(rows, widest.shape[0])])
        if runner.spyre_kv_caches:
            _record_attention(runner)
    logger.info("MRV2 warmup done in %.3fs.", time.time() - t0)
