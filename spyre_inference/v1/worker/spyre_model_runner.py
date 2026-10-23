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

"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

Data flow in the current WIP version:
- self.device = CPU. Buffers and scatter ops stay on CPU.
- _SpyreModelWrapper converts input_ids/positions to Spyre int64 at the
  model call boundary.
- Generative: D2H hidden_states for logits/sampling. Pooling: keep on Spyre;
  pooler D2Hs only the final pooled vectors in ``_pool``.
- Embedding: Spyre int64 input → Spyre compute → float16 output on Spyre.
- Hidden states flow on Spyre between decoder layers; attention, RoPE and the
  embedding gather run on-device too.
- What stays on CPU is host-side bookkeeping: slot mapping, attention metadata
  (index tables, mask tiles) built by the metadata builder, logits indexing and
  sampling, plus the few model-specific workarounds that route one op through the
  host (see ``spyre_inference.multimodal`` and ``spyre_inference.models``).
"""

from __future__ import annotations

import bisect
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference import envs
from spyre_inference.custom_ops.bert_head_pad import install_bert_head_pad
from spyre_inference.custom_ops.conv import SpyreConv2d
from spyre_inference.custom_ops.head_pad import (
    fix_padded_attention_scale,
    fix_padded_qk_norm_eps,
    fix_padded_rope,
    install_head_pad_weight_loader,
    install_padded_head_dim,
    verify_padded_head_dim,
    verify_padded_qk_norm_weights,
)
from spyre_inference.custom_ops.mlp_pad import (
    install_mlp_pad_weight_loader,
    verify_padded_intermediate_size,
)
from spyre_inference.custom_ops.utils import convert, convert_tensor_tree
from spyre_inference.models import check_spyre_model
from spyre_inference.models._vision_encoder import SpyreVisionEncoderMixin
from spyre_inference.models.mistral import reset_llama4_scale_cache
from spyre_inference.models.roberta import offset_host_positions, roberta_position_delta
from spyre_inference.multimodal import apply_multimodal_patches
from spyre_inference.multimodal.utils import blocks_marked_compilable
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
    allocate_staging_buffers,
)
from spyre_inference.v1.pool import (
    SpyreAllPool,
    SpyreTokenPooler,
    configure_pooling_for_spyre,
    copy_pooler_output_to_cpu,
    select_rows,
)
from spyre_inference.v1.pool.encoder_grid import (
    pool_rows_for,
    unpad_encoder_grid,
)
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreClassifierLinear,
    SpyreEmbeddingPoolerHead,
    SpyreMeanPool,
    SpyreRobertaClassificationHead,
    _iter_modules,
    _mean_pool_grid_reduce,
    _mean_pool_row_mask_mul,
)
from spyre_inference.v1.sample.sampler import greedy_sample
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.encoder_grid import (
    EncoderDispatchStats,
    encoder_pad_token_id,
    expand_body_to_grid,
)
from spyre_inference.v1.worker.kv_cache_setup import (
    allocate_spyre_kv_caches,
    split_attn_groups_by_window,
)
from spyre_inference.v1.worker.spyre_encoder_graph import SpyreEncoderGraphManager
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    SpyreShapeBucketer,
    encoder_group_shapes,
    encoder_group_width_caps,
    encoder_len_ladder,
    encoder_rectangles,
    encoder_shape_tables,
    expand_packed_token_types,
    logits_row_buckets,
)

logger = init_logger(__name__)


# Pure-PyTorch replacement for torch.ops._C.compute_slot_mapping_kernel_impl
# (unavailable with VLLM_TARGET_DEVICE=empty).

_PAD_SLOT_ID = -1


def _compute_slot_mapping_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    slot_mapping: torch.Tensor,
    KV_CACHE_BLOCK_SIZE: int,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = _PAD_SLOT_ID,
    # Triton tile width; unused here, kept for call compatibility.
    BLOCK_SIZE: int = 1024,
) -> None:
    """Map each token position to its flat index in the paged KV cache.

    The upstream vLLM implementation is a Triton kernel (requires a GPU) and
    the CPU backend delegates to a C++ op in _C.so. Neither is available with
    VLLM_TARGET_DEVICE=empty, so we reimplement the logic in pure PyTorch.

    Correctness is validated indirectly by the upstream attention backend test
    (test_causal_backend_correctness) and end-to-end model generation tests.
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on Spyre."

    token_positions = positions[:num_tokens]
    block_indices = (token_positions // KV_CACHE_BLOCK_SIZE).to(torch.int64)
    block_offsets = (token_positions % KV_CACHE_BLOCK_SIZE).to(torch.int64)

    num_reqs = query_start_loc.shape[0] - 1
    req_indices = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        req_indices[start:end] = i

    flat_indices = req_indices * block_table_stride + block_indices
    block_numbers = block_table.flatten()[flat_indices].to(torch.int64)
    slot_mapping[:num_tokens] = block_numbers * KV_CACHE_BLOCK_SIZE + block_offsets
    if num_tokens < max_num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID


class _SlotMappingKernelShim:
    """Drop-in for vLLM's Triton ``ComputeSlotMappingKernel``, which Spyre cannot run:
    calls go to the pure-PyTorch impl and warmup registration is a no-op."""

    def __call__(
        self,
        num_reqs: int,
        num_tokens: int,
        max_num_tokens: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        block_table_stride: int,
        slot_mapping: torch.Tensor,
        KV_CACHE_BLOCK_SIZE: int,
        TOTAL_CP_WORLD_SIZE: int = 1,
        TOTAL_CP_RANK: int = 0,
        CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    ) -> None:
        _compute_slot_mapping_impl(
            num_tokens,
            max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table_stride,
            slot_mapping,
            KV_CACHE_BLOCK_SIZE,
            TOTAL_CP_WORLD_SIZE=TOTAL_CP_WORLD_SIZE,
            TOTAL_CP_RANK=TOTAL_CP_RANK,
            CP_KV_CACHE_INTERLEAVE_SIZE=CP_KV_CACHE_INTERLEAVE_SIZE,
        )

    def register_warmup(self, *args, **kwargs) -> None:
        pass


_compute_slot_mapping_kernel = _SlotMappingKernelShim()

# The module attribute in vllm.v1.worker.block_table that both launches the
# slot-mapping kernel and receives its warmup registration.
_SLOT_MAPPING_KERNEL_ATTR = "_COMPUTE_SLOT_MAPPING_KERNEL"


def _patch_compute_slot_mapping() -> None:
    """Swap vLLM's slot-mapping kernel instance for our shim before any BlockTable exists.

    The attribute is checked first: vLLM has moved this kernel before, and assigning a
    name nothing reads would leave the Triton placeholder in place until the first
    decode step.
    """
    from vllm.v1.worker import block_table

    if not hasattr(block_table, _SLOT_MAPPING_KERNEL_ATTR):
        raise RuntimeError(
            f"Cannot find vLLM's slot-mapping kernel to patch: "
            f"block_table.{_SLOT_MAPPING_KERNEL_ATTR} does not exist. vLLM has moved it; "
            f"update _SlotMappingKernelShim and _patch_compute_slot_mapping for the "
            f"new launch site. Leaving the patch unapplied would defer the failure to "
            f"the first decode step."
        )

    setattr(block_table, _SLOT_MAPPING_KERNEL_ATTR, _compute_slot_mapping_kernel)


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """Spyre-specific CpuGpuBuffer with Spyre-safe copies and split dtypes.
    This buffer is closely related to the CpuGpuBuffer in vllm/v1/utils.py.

    For float dtypes: .cpu on CPU, .gpu on Spyre (float16).
    For int/bool dtypes: .gpu aliased to .cpu (CPUModelRunner pattern).
    Float H2D uses ``non_blocking=True``; callers must sync via
    ``TorchSpyreModelRunner._sync_device`` (``torch.spyre.synchronize``)
    before consuming the Spyre tensors.

    Inherits from `CpuGpuBuffer` (without invoking its `__init__`) so that
    `_make_buffer` overrides remain Liskov-compatible with `GPUModelRunner`.
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        cpu_dtype: torch.dtype,
        gpu_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=cpu_dtype, device="cpu", pin_memory=pin_memory)
        if device.type == "spyre":
            self.gpu = torch.zeros(*size, dtype=gpu_dtype, device=device)
        else:
            # int/bool: alias gpu = cpu (CPUModelRunner pattern)
            self.gpu = self.cpu
        self.np: np.ndarray
        if with_numpy:
            if cpu_dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call SpyreCpuGpuBuffer with "
                    "with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.gpu if n is None else self.gpu[:n]
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        # Async H2D via torch-spyre's aten::_copy_from / copyAsync path.
        # GPUModelRunner calls _sync_device before the tensors are consumed.
        dst.copy_(src, non_blocking=True)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        # Currently only the copy_to_gpu function is invoked.
        # If the copy_to_cpu also becomes required, override it here with
        # spyre-specific aspects.
        raise NotImplementedError("SpyreCpuGpuBuffer.copy_to_cpu is not implemented")


SPYRE_COMPILE_GRANULARITIES = ("block", "model")


def _compile_granularity() -> str:
    granularity = envs.SPYRE_COMPILE_GRANULARITY
    if granularity not in SPYRE_COMPILE_GRANULARITIES:
        raise ValueError(
            f"Unsupported SPYRE_COMPILE_GRANULARITY={granularity!r}. "
            f"Expected one of {SPYRE_COMPILE_GRANULARITIES}."
        )
    return granularity


def _block_sharing_defeated_by() -> str | None:
    """Sharing needs vLLM to hoist layer names out of the graph; report when it cannot."""
    try:
        from vllm.utils.torch_utils import _USE_LAYERNAME
    except ImportError:
        return None
    if not _USE_LAYERNAME:
        return (
            "vLLM is not hoisting per-layer attention names (needs torch >= 2.11 and "
            "VLLM_USE_LAYERNAME=1), so each block compiles separately"
        )
    return None


def _is_decoder_attention_like(module: nn.Module) -> bool:
    """Return whether a module participates in decoder KV-cache attention.

    Native vLLM models own an ``Attention`` directly. The Transformers backend
    instead retains an HF dispatcher with a ``layer_idx`` and the text config's
    ``_attn_implementation`` set to ``vllm``. Those are the interface
    contract, unlike the module's class name, and do not match vision encoders.
    The backend also attaches its vLLM layer as ``.attn``, but for an MLA model that
    is an ``MLAAttention``, which is not an ``Attention``, so the check stays.
    """
    if isinstance(module, Attention):
        return True
    # Vision encoders retain their HF attention implementation (for example,
    # ``sdpa``). Only text-decoder wrappers are configured to dispatch through
    # vLLM's KV-cache attention implementation. Transformers may prefix the
    # implementation with ``paged|`` when it enables its paged-cache wrapper.
    # Vision towers compile through ``SpyreVisionEncoderMixin`` instead, per encoder
    # budget, not through ``_compile_blocks``.
    implementation = getattr(getattr(module, "config", None), "_attn_implementation", "") or ""
    return isinstance(getattr(module, "layer_idx", None), int) and "vllm" in implementation.split(
        "|"
    )


_VISION_TOWER_NAME_PARTS = frozenset(("vision_encoder", "vision_tower", "vision_model", "visual"))


def _is_vision_tower_path(qualname: str) -> bool:
    """True for ``vision_encoder.transformer.layers`` and the HF / Qwen-VL spellings."""
    return any(part in _VISION_TOWER_NAME_PARTS for part in qualname.split("."))


def _repeated_block_lists(model: nn.Module) -> list[tuple[nn.ModuleList, bool]]:
    """Each compilable block list with whether a multimodal patch marked it."""
    block_lists: list[tuple[nn.ModuleList, bool]] = []
    for qualname, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        blocks = [b for b in module if not isinstance(b, PPMissingLayer)]
        if not blocks:
            continue
        # A multimodal patch marks the blocks whose body it made traceable; reading the
        # mark off the blocks keeps an outer list that merely nests a marked tower out.
        if blocks_marked_compilable(blocks):
            block_lists.append((module, True))
            continue
        # Every other tower stays eager: its block body builds constants on the host,
        # which have no device layout. Decoder lists are never named these.
        if _is_vision_tower_path(qualname):
            continue
        # nn.Module.modules() yields the module itself, so a list of bare Attention
        # layers (Zamba2's dpa_list) would match and "compile" one opaque call per entry.
        if any(_is_decoder_attention_like(b) for b in blocks):
            continue
        # Hybrid Mamba+attention stacks (Granite 4.0, Jamba) mix classes in one list;
        # each class shares a forward code object, so compiles scale per class, not depth.
        if any(_is_decoder_attention_like(m) for b in blocks for m in b.modules()):
            block_lists.append((module, False))
    return block_lists


class _SpyreModelWrapper:
    """Transparent wrapper that converts model inputs/outputs at the boundary.

    Input conversion (CPU → Spyre):
        For example, input_ids and positions arrive as CPU tensors (int32/int64) because
        self.device=CPU in the runner and buffer scatter ops run on CPU.
        Convert them to int64 and provide them to the model.

    Output conversion (Spyre → CPU):
        The model's final hidden_states come out on Spyre. Downstream
        operations (indexing via logits_indices, sampling) run on CPU.
        The lm_head matmul runs on Spyre via SpyreParallelLMHead: ``compute_logits``
        below does the H2D of the sampled rows, SpyreLogitsProcessor the D2H of the
        logits.

    Wrapping at the model level ensures ALL call sites get the right
    device — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).
    """

    def __init__(
        self,
        model: nn.Module,
        spyre_device: torch.device,
        keep_outputs_on_device: bool = False,
        logits_row_buckets: list[int] | None = None,
        shape_bucketer: SpyreShapeBucketer | None = None,
        *,
        model_dtype: torch.dtype,
        inputs_embeds_buffer: torch.Tensor | None = None,
    ):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)
        object.__setattr__(self, "_keep_outputs_on_device", keep_outputs_on_device)
        object.__setattr__(self, "_logits_row_buckets", logits_row_buckets or [])
        object.__setattr__(self, "_shape_bucketer", shape_bucketer)
        object.__setattr__(self, "_model_dtype", model_dtype)
        object.__setattr__(self, "_inputs_embeds_buffer", inputs_embeds_buffer)

    def __call__(self, *args, **kwargs):
        # Convert integer tensor inputs to Spyre int64. Do not use int32:
        # stock torch-spyre SDSC cannot schedule integer add (warmup crash
        # ``0_add``). RoBERTa ``position_ids + padding_idx`` is applied on the
        # host in ``_preprocess``, before this copy.
        is_integer = lambda t: t.dtype in (torch.int32, torch.int64)
        args_converted = convert_tensor_tree(
            args, device=self._spyre_device, dtype=torch.int64, predicate=is_integer
        )
        kwargs_converted = convert_tensor_tree(
            kwargs, device=self._spyre_device, dtype=torch.int64, predicate=is_integer
        )

        # The Llama-4 scale cache keys on `positions` identity, blind to an in-place rewrite.
        reset_llama4_scale_cache()

        t0 = time.time()
        result = self._model(*args_converted, **kwargs_converted)

        # Pooling: keep on Spyre. Generative: D2H for sampling.
        if not self._keep_outputs_on_device:
            result = convert_tensor_tree(result, device="cpu")

        input_ids = kwargs_converted.get("input_ids")
        num_tokens = input_ids.shape[0] if input_ids is not None else -1
        logger.debug("t_token: %.2fms [num tokens %d]", (time.time() - t0) * 1000, num_tokens)

        return result

    def _to_spyre(self, t):
        return convert(t, device=self._spyre_device) if isinstance(t, torch.Tensor) else t

    def _to_cpu(self, t):
        return convert(t, device="cpu") if isinstance(t, torch.Tensor) else t

    def compute_logits(self, hidden_states, *args, **kwargs):
        """Move hidden_states onto Spyre for the lm_head custom op.

        gpu_model_runner.execute_model slices `hidden_states[logits_indices]`
        on CPU (no Spyre `aten::index.Tensor`; a device gather needs
        `select_rows`), so the tensor handed to compute_logits is on CPU;
        move it onto Spyre for the lm_head matmul. The logits are
        returned on CPU: the head's ``SpyreUnquantizedLMHeadMethod.apply`` keeps them
        on Spyre, and SpyreLogitsProcessor converts them to CPU before the vocab slice
        and scale -- in ``_apply_head`` at TP=1, right after the all_gather in
        ``_gather_logits`` at TP>1 -- so downstream sampling gets CPU logits.

        The sampled-row count is not body-bucket padded, so padding it onto the warmed
        row buckets keeps the projection on shapes warmup compiled.
        """
        num_rows = hidden_states.shape[0]
        buckets = self._logits_row_buckets
        idx = bisect.bisect_left(buckets, num_rows)
        padded_rows = buckets[idx] if idx < len(buckets) else num_rows
        if padded_rows != num_rows:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padded_rows - num_rows))

        hidden_states = convert_tensor_tree(hidden_states, device=self._spyre_device)
        logits = self._model.compute_logits(hidden_states, *args, **kwargs)

        if padded_rows != num_rows and logits is not None:
            logits = logits[:num_rows]
        return logits

    def embed_multimodal(self, **kwargs):
        """Move float multimodal inputs (e.g. ``pixel_values``) onto Spyre.

        The runner reaches this through ``__getattr__``, bypassing ``__call__``'s
        input conversion, so pixel tensors would otherwise arrive on CPU while the
        vision weights are on Spyre.
        """

        def _to_spyre_float(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return convert(t, dtype=self._model_dtype, device=self._spyre_device)
            return t

        # These prepare their inputs on the host and upload once themselves.
        if not isinstance(self._model, SpyreVisionEncoderMixin):
            kwargs = tree_map(_to_spyre_float, kwargs)
        # The parts of a vision tower outside a compiled block run eager, so each Spyre
        # op with a decomposition there reaches it through torch-spyre's lazily-compiled
        # PrivateUse1 kernel, which compiles without fullgraph. Decompositions built on
        # for_each_tile (SDPA since torch-spyre#4550) emit a scan whose while_loop
        # lowering reads the loop index with .item(); without fullgraph that needs
        # capture_scalar_outputs, or the trace dies with DataDependentOutputException.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            return self._model.embed_multimodal(**kwargs)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        """Move input_ids/is_multimodal/multimodal_embeddings onto Spyre.

        gpu_model_runner._preprocess calls this directly on `self.model`,
        bypassing __call__, so it needs its own CPU <-> Spyre boundary
        conversion: input_ids and is_multimodal are CPU buffers, while the text
        embedding table (and the merge with multimodal_embeddings) lives on
        Spyre.

        Bucket the token count: this runs on the raw scheduled count, so at TP>1
        the vocab-parallel all_reduce is `num_tokens * hidden` for every distinct
        prompt length, and some of those collective schedules fail to build. Pad
        on CPU and trim after; padding inside a compiled collective corrupts
        output. `is_multimodal` is padded the same way so it still lines up with
        `input_ids` for the merge; the padded rows are never multimodal.

        Everything comes back on device, merged or not. Upstream copies what we
        return into a row-prefix view of its persistent inputs_embeds buffer
        (`gpu_model_runner._preprocess`: `inputs_embeds.gpu[:n].copy_(...)`), and
        torch-spyre#4730 rejects an H2D copy whose destination is narrowed -- it
        requires `dev_sizes == dma_sizes`, while the view is `[n, hidden]` against
        a base allocated at the full token budget. Returning a CPU tensor makes
        that copy H2D and kills the engine on every multimodal request; keeping it
        on device makes the same copy d2d, which the check allows. It also avoids
        a D2H that upstream's H2D immediately undoes.

        The result is staged in that buffer at the bucket's shape and returned as its
        prefix view: upstream's copy is then a self-copy torch-spyre skips, instead of
        a `copy_from_d2d` compiled for every distinct `n`.

        A merged result previously came back on CPU because the merge's
        `torch.where` output layout did not survive that d2d `copy_`. If a merged
        multimodal prompt starts producing garbage rather than failing, suspect
        that layout again before anything else here.
        """
        # Generic path: model.embed_input_ids only does text embedding, or
        # boolean-mask ops are handled inside the model's own patch (e.g.
        # patch_embed_input_ids for Granite4Vision).
        num_tokens = input_ids.shape[0]
        vision = isinstance(self._model, SpyreVisionEncoderMixin)
        # An image step stages at least two rows: a one-row copy out of the merge's layout
        # fails (torch-spyre#4883).
        merges = vision and multimodal_embeddings is not None and len(multimodal_embeddings) > 0
        target = max(num_tokens, 2) if merges else num_tokens
        bucketer = self._shape_bucketer
        if bucketer is not None:
            padded_tokens = bucketer.find_bucket(target)
        else:
            padded_tokens = target if target != num_tokens else None
        if padded_tokens is not None and padded_tokens != num_tokens:
            input_ids = F.pad(input_ids, (0, padded_tokens - num_tokens))
            if is_multimodal is not None:
                is_multimodal = F.pad(is_multimodal, (0, padded_tokens - num_tokens))

        input_ids = convert(input_ids, dtype=torch.int64, device=self._spyre_device)
        # These place their image tokens on the host (``SpyreVisionEncoderMixin.embed_input_ids``),
        # where their encoder output already is, so neither goes to the card here.
        if not vision:
            is_multimodal = self._to_spyre(is_multimodal)
            multimodal_embeddings = tree_map(self._to_spyre, multimodal_embeddings)

        result = self._model.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )
        buf = self._inputs_embeds_buffer
        if (
            padded_tokens is not None
            and buf is not None
            and isinstance(result, torch.Tensor)
            and result.shape[0] == padded_tokens <= buf.shape[0]
            and result.shape[1:] == buf.shape[1:]
            and result.dtype == buf.dtype
        ):
            buf[:padded_tokens].copy_(result)
            return buf[:num_tokens]
        if padded_tokens is not None and padded_tokens != num_tokens:
            # A plain prefix slice, not select_rows: the padding above always
            # appends at the end, so the real rows are always 0..num_tokens
            # contiguously -- no gather needed, and a merged result's torch.where
            # output layout select_rows's own compiled index_select does not accept.
            result = tree_map(
                lambda t: t[:num_tokens] if isinstance(t, torch.Tensor) else t, result
            )
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        # `__init__` fills our `__dict__` via `object.__setattr__`, so a name in it is
        # ours, not the model's.
        if name in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for Spyre.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre for floats, aliased to CPU for int/bool

    Inherits from GPUModelRunner to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Store the real Spyre device before super().__init__ so that
        # _make_buffer can place .gpu tensors on Spyre directly.
        self._spyre_device = device

        # Set by load_model: whether the pooler/classifier stay on Spyre.
        self._pooling_on_spyre = False

        # Encoder shape tables, and the step state derived from them. The grid is
        # written by _build_attention_metadata and read by _preprocess and _pool,
        # which all run once per step in that order.
        is_pooling = vllm_config.model_config.runner_type == "pooling"
        self._encoder_rectangles: list[tuple[int, int]] = (
            encoder_rectangles(vllm_config) if is_pooling else []
        )
        self._encoder_width_caps: dict[int, int] = (
            encoder_group_width_caps(vllm_config) if is_pooling else {}
        )
        self._encoder_budget = encoder_shape_tables(vllm_config).budget if is_pooling else 0
        self._encoder_buffer_rows = 0
        # (extent, width, query_lens) on the rectangular path; None on the ragged one.
        self._encoder_grid: tuple[int, int, list[int]] | None = None
        # Warmup only; see _warm_encoder_inline_paths.
        self._forced_encoder_rect: tuple[int, int] | None = None
        self._force_encoder_ragged = False
        self.spyre_encoder_stats = EncoderDispatchStats()

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))
        # Patch upstream's Sampler in place, so every holder of it (e.g. the
        # RejectionSampler) takes the host-side draws.
        self.sampler.topk_topp_sampler = SpyreTopKTopPSampler(
            self.sampler.logprobs_mode, self.sampler.use_fp64_gumbel
        )
        self.sampler.greedy_sample = greedy_sample  # ty: ignore[invalid-assignment]

        # Keep self.device as CPU so buffer management (scatter, copy) stays
        # on CPU. _SpyreModelWrapper converts input_ids/positions to Spyre
        # int64 at the model boundary.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Shape bucketer for runtime dispatch (initialized after model load)
        self.spyre_shape_bucketer: SpyreShapeBucketer | None = None

        # Per-layer paged KV caches, kept so warmup can record the attention
        # kernels against real pages. Populated by initialize_kv_cache_tensors.
        self._spyre_kv_caches: dict[str, SpyrePagedKVCache] = {}

        # Replace Triton kernel with a pure-PyTorch implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # The upstream CPU backend uses a C++ kernel (torch.ops._C) as its
        # fallback, but we don't have _C.abi3.so with VLLM_TARGET_DEVICE=empty.
        _patch_compute_slot_mapping()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load weights on CPU, move Spyre layers to device, compile, and wrap."""
        logger.info("Loading model %s...", self.model_config.model)
        t0 = time.time()

        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        model_loader = get_model_loader(self.load_config)

        # Pad attention weights (q/k/v/o, and QK-norm) to the stick-aligned head_dim
        # as they stream in, when the platform overrode head_dim (e.g. head_size=64).
        # Must run before load_model builds+loads the (now 128-wide) params.
        text_config = self.model_config.hf_text_config
        install_padded_head_dim(self.model_config)
        install_bert_head_pad(self.model_config)
        install_head_pad_weight_loader(model_loader, text_config, self.model_config)
        install_mlp_pad_weight_loader(model_loader, text_config, self.model_config)

        # Load model on CPU
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        check_spyre_model(self.model, get_model_architecture(self.model_config)[1])
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        # Cases appearing in GPUModelRunner.
        # When needed, they can be implemented for Spyre.
        if self.lora_config:
            raise NotImplementedError("LoRA adapters are not yet implemented and tested for Spyre.")

        if hasattr(self, "drafter"):
            raise NotImplementedError(
                "Models with a drafter model are not yet implemented and tested for Spyre."
            )

        # Restore original RoPE frequencies and attention scale, and compensate
        # QK-norm epsilon for the padded head_dim. All passes are scoped to the
        # padded text backbone (see custom_ops.text_backbone).
        verify_padded_head_dim(self.model, text_config, self.model_config)
        verify_padded_qk_norm_weights(self.model, text_config, self.model_config)
        verify_padded_intermediate_size(self.model, text_config, self.model_config)
        fix_padded_rope(self.model, text_config, self.model_config)
        fix_padded_attention_scale(self.model, text_config, self.model_config)
        fix_padded_qk_norm_eps(self.model, text_config, self.model_config)

        # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
        # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
        # an nn.Module, but just the attention implementation.
        Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

        # Before any trace: resolved lazily inside the block graph, the scales become a
        # guard on their unset value and the first real prefill recompiles the block.
        self._resolve_kv_scales()

        # Move layer weights to Spyre device.
        self.model.to(device=self._spyre_device)
        for module in self.model.modules():
            if isinstance(module, SpyreConv2d):
                module.process_weights_after_loading()

        # CLS/LAST gather on Spyre. MEAN reduces each rectangle lane on Spyre and
        # falls back to the host elsewhere. FP32 linear heads stay on CPU.
        self._pooling_on_spyre = False
        if self.model_config.runner_type == "pooling":
            self._pooling_on_spyre = configure_pooling_for_spyre(
                self.model, self._spyre_device, encoder_len_ladder(self.vllm_config)
            )

        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)
        logger.info("Model loaded for Spyre in %.3fs.", time.time() - t0)

        # Patches instances/classes, so it runs after load and before compile wraps modules
        # in OptimizedModule and breaks traversal.
        apply_multimodal_patches(self.model, self._spyre_device)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Initialize bucket dispatcher for shape bucketing at runtime.
        self.spyre_shape_bucketer = SpyreShapeBucketer.for_config(self.vllm_config)

        # Generative: D2H model outputs. Pooling: keep hidden_states on Spyre.
        bucketer = self.spyre_shape_bucketer
        self.model = _SpyreModelWrapper(
            self.model,
            self._spyre_device,
            keep_outputs_on_device=self._pooling_on_spyre,
            logits_row_buckets=(
                []
                if bucketer is None
                else logits_row_buckets(bucketer.bucket_sizes, self.max_num_reqs)
            ),
            shape_bucketer=bucketer,
            model_dtype=self._model_dtype(),
            inputs_embeds_buffer=self.inputs_embeds.gpu,
        )

    def _resolve_kv_scales(self) -> bool:
        """Read each float8-cache layer's KV scales into its impl; True when any changed."""
        changed = False
        pairs = set()
        shared = []
        for module in self.get_model().modules():
            impl = getattr(module, "impl", None)
            if not (isinstance(module, Attention) and isinstance(impl, SpyreAttentionImpl)):
                continue
            if getattr(module, "kv_sharing_target_layer_name", None) is not None:
                shared.append(module)
                continue
            if not impl.kv_cache_fp8:
                continue
            k_scale = getattr(module, "_k_scale", None)
            if isinstance(k_scale, torch.Tensor) and k_scale.numel() > 1:
                # Upstream's per-head scheme reduces the heads to their max scale, so a head
                # with a much smaller one keeps only a few e4m3 steps of its range.
                logger.warning_once(
                    "The checkpoint carries per-head KV cache scales; the fp8 KV cache uses "
                    "one scale per layer, the largest head's, which coarsens the others."
                )
            old = impl._kv_scales
            impl.prepare_kv_scales(module)
            changed |= old is not None and impl._kv_scales != old
            pairs.add(impl._kv_scales)
        # A sharing layer reads the cache its target wrote, so it dequantizes with the
        # target's scales; its own usually default to 1.0, as it stores nothing.
        static_ctx = self.compilation_config.static_forward_context if shared else {}
        for module in shared:
            target = static_ctx[module.kv_sharing_target_layer_name]
            if module.impl.kv_cache_fp8 != target.impl.kv_cache_fp8:
                # It reads the target's pages, so it must decode them in the dtype they hold.
                raise ValueError(
                    f"KV-sharing layer {module.layer_name} and its target "
                    f"{module.kv_sharing_target_layer_name} disagree on the fp8 KV cache; "
                    "list both or neither in kv_cache_dtype_skip_layers."
                )
            if not module.impl.kv_cache_fp8:
                continue
            old = module.impl._kv_scales
            module.impl._kv_scales = target.impl._kv_scales
            changed |= old is not None and module.impl._kv_scales != old
        if len(pairs) > 1:
            # The scales are graph constants, so each distinct pair is its own compile.
            logger.warning_once(
                "The fp8 KV cache has %d distinct per-layer (k_scale, v_scale) pairs; the "
                "transformer block and every recorded attention kernel compile once per "
                "pair, which lengthens warmup.",
                len(pairs),
            )
        return changed

    def reload_weights(self, *args, **kwargs) -> None:
        super().reload_weights(*args, **kwargs)
        # Upstream's reload reprocesses the weights, which rewrites the layers' KV scales.
        # The graphs that read them guard on their values, so only those retrace; warming
        # here does that before the next request, which an armed compile guard would fail.
        if self._resolve_kv_scales():
            logger.warning(
                "KV cache scales changed on weight reload; re-warming the graphs that "
                "depend on them."
            )
            with compile_guard.allow_compile("KV cache scales changed on weight reload"):
                self.warming_up_model()

    def _compile_for_spyre(self) -> None:
        """Install torch.compile wrappers; tracing happens on the first forward.

        `dynamic=False` is mandatory: the Spyre backend rejects SymInt shapes.
        """
        mode = self.compilation_config.mode
        if mode not in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. Only "
                f"CompilationMode.NONE and CompilationMode.STOCK_TORCH_COMPILE "
                f"are supported."
            )

        # Validated before the eager short-circuit so a typo is not silently ignored.
        # Per-block is the default: identical blocks share a ``forward`` code object,
        # so the backend compiles one block rather than a program that grows with depth.
        granularity = _compile_granularity()
        self._compiled_tower_blocks = 0

        if self.vllm_config.model_config.enforce_eager or mode is CompilationMode.NONE:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        model_name = type(self.model).__name__

        if granularity == "block":
            defeated_by = _block_sharing_defeated_by()
            if defeated_by:
                logger.warning(
                    "SPYRE_COMPILE_GRANULARITY=block will not share one artifact across "
                    "blocks: %s. Expect compile time to grow with layer count; "
                    "SPYRE_COMPILE_GRANULARITY=model may warm up faster.",
                    defeated_by,
                )
            num_blocks = self._compile_blocks()
            tower = self._compiled_tower_blocks
            if num_blocks:
                logger.info(
                    "Wrapped %d transformer blocks%s of %s for per-block compile on Spyre.",
                    num_blocks,
                    f" and {tower} tower blocks" if tower else "",
                    model_name,
                )
                return
            logger.warning(
                "Found no attention-bearing block ModuleList in %s; falling back to a "
                "whole-model graph. Models whose attention is not a vLLM Attention "
                "(MLA, encoder-only vision towers) take this path.",
                model_name,
            )

        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        compile_guard.watch(self.model, f"{model_name} (whole-model graph)")
        logger.info("Wrapped %s as a single graph for Spyre.", model_name)

    def _compile_blocks(self) -> int:
        # Only decoder blocks are returned: the count decides the whole-model fallback,
        # which an opted-in tower must not pre-empt (it compiles beside either form).
        num_blocks = num_tower = 0
        # Models that re-register a slice of `layers` (e.g. gemma-4's self-/cross-decoder)
        # alias blocks across lists; recompiling one is harmless but would double the count.
        seen: set[int] = set()
        for blocks, marked in _repeated_block_lists(cast(nn.Module, self.model)):
            for block in blocks:
                if isinstance(block, PPMissingLayer) or id(block) in seen:
                    continue
                seen.add(id(block))
                # In place: rebinding blocks[i] to the returned OptimizedModule reparents
                # the block under `_orig_mod`, renaming every parameter and breaking
                # reload_weights and save_sharded_state.
                block.compile(backend="inductor", fullgraph=True, dynamic=False)
                # Identical blocks share one `forward` code object -- that is why this
                # granularity shares a single artifact -- so this collapses to one
                # registration per block *class*, which is the right granularity for
                # the message too. A marked tower is watched like the decoder:
                # `_warmup_multimodal_encoder` compiles it at every image-batch width.
                compile_guard.watch(block, f"{type(block).__name__} (transformer block)")
                if marked:
                    num_tower += 1
                else:
                    num_blocks += 1
        self._compiled_tower_blocks = num_tower
        return num_blocks

    def warming_up_model(self) -> None:
        """Warm kernels / compile.

        Decoder: dummy each 1D ``compile_sizes`` bucket (largest first), then a dummy
        logits/sampler run at each *sampled-row* width so the lm_head compiles here
        rather than mid-request. The two bucket sets differ: body buckets are packed
        token counts, rows are at most ``max_num_reqs``.
        Compiled pooling: one dummy per body shape -- and pooling has exactly one, the
        token budget. That forward traces the body and the pooler. The first attention
        call that reaches the impl runs ``SpyreEncoderAttentionImpl.warm_kernels``, which
        traces every declared ``(width, extent)`` group, plus every declared rectangle
        while the rectangular path is opaque. With the rectangular path traced into the
        block graph, ``_warm_encoder_inline_paths`` gives each rectangle its own dummy and
        forces one onto the ragged path, which is what reaches the impl.
        Eager pooling: one short dummy, then ``mark_warmed_up()``. It does not run the
        vision tower, whose eager ops compile per op on first use anyway.
        A compiled pooling model's vision tower, and any model's compiled one, also runs
        ``embed_multimodal`` at every image-batch width up to ``mm_max_items_per_batch``;
        see ``_warmup_multimodal_encoder``.
        Upstream dummy skips encoder attention unless ``force_attention=True``.
        """
        is_pooling = self.model_config.runner_type == "pooling"
        # Before the first trace: see allocate_staging_buffers.
        allocate_staging_buffers(self.compilation_config.static_forward_context, self._spyre_device)

        if is_pooling and not self.vllm_config.model_config.enforce_eager:
            logger.info(
                "Warming up model: body [%d, hidden], %d encoder rectangle(s), "
                "%d ragged-path group shape(s).",
                self._encoder_budget,
                len(self._encoder_rectangles),
                len(encoder_group_shapes(self.vllm_config)),
            )
            t0 = time.time()
            with _set_spyre_compilation_settings(self.vllm_config):
                if self.spyre_shape_bucketer is not None:
                    for size in sorted(self.spyre_shape_bucketer.bucket_sizes, reverse=True):
                        hidden_states, _ = self._dummy_run(size, force_attention=True)
                        self._warmup_input_embedding(size)
                        self._dummy_pooler_run(hidden_states)
                        self._warm_pooler_row_widths(hidden_states)
                        self._warm_encoder_unpack(hidden_states)
                    self._warm_encoder_inline_paths()
                    self._warm_encoder_embed_gather()
                    self.spyre_shape_bucketer.mark_warmed_up()
                if self._spyre_kv_caches:
                    # A decoder-type text tower (e.g. CLIP's) has a real KV cache;
                    # record its (num_blocks, query_len) variants directly, sidestepping
                    # the dummy-batch seq_lens bug. No-op for encoder-only pooling
                    # models (BERT/RoBERTa), which never get a KV cache.
                    self._record_attention_graphs()
                self._warmup_multimodal_encoder()
            logger.info("Warmup done in %.3fs.", time.time() - t0)
            return

        if is_pooling or self.spyre_shape_bucketer is None:
            logger.info("Running single warmup pass (graph manager Disabled)...")
            t0 = time.time()
            num_tokens = min(
                max(16, self.max_num_reqs),
                self.scheduler_config.max_num_batched_tokens,
            )
            with _set_spyre_compilation_settings(self.vllm_config):
                self._dummy_run(num_tokens)
                self._warmup_input_embedding(num_tokens)
                # Only a compiled tower is warmed here: enforce_eager reaches this
                # branch, and an eager tower compiles per op on first use regardless.
                if self._compiled_tower_blocks:
                    self._warmup_multimodal_encoder()
            if is_pooling and self.spyre_shape_bucketer is not None:
                self.spyre_shape_bucketer.mark_warmed_up()
            logger.info("Warmup done in %.3fs.", time.time() - t0)
            return

        bucket_sizes = self.spyre_shape_bucketer.bucket_sizes
        logger.info(
            "Warming up model with %d bucket sizes [%d..%d]...",
            len(bucket_sizes),
            bucket_sizes[0] if bucket_sizes else 0,
            bucket_sizes[-1] if bucket_sizes else 0,
        )
        row_widths = logits_row_buckets(bucket_sizes, self.max_num_reqs)
        t0 = time.time()
        with _set_spyre_compilation_settings(self.vllm_config):
            # Compile largest bucket first: Inductor's internal caches benefit
            # from seeing the most complex shape first, so subsequent smaller
            # shapes compile faster via partial cache hits.
            widest_hidden_states = None
            for size in sorted(bucket_sizes, reverse=True):
                _, last_hidden_states = self._dummy_run(size)
                self._warmup_input_embedding(size)
                if widest_hidden_states is None:
                    widest_hidden_states = last_hidden_states
            # Row buckets, not one run per body bucket: the prefill bucket's token count
            # exceeds any reachable row count, so it would compile an unreachable width.
            if widest_hidden_states is not None:
                for rows in sorted(row_widths, reverse=True):
                    self._dummy_sampler_run(widest_hidden_states[:rows])
            self._warmup_multimodal_encoder()
        self.spyre_shape_bucketer.mark_warmed_up()
        logger.info(
            "Warmup complete in %.3fs for %d buckets.",
            time.time() - t0,
            len(bucket_sizes),
        )
        self._record_attention_graphs()
        self._record_vision_graphs()

    @torch.inference_mode()
    def _record_vision_graphs(self) -> None:
        """Compile the vision encoder at every budget; serving then runs through the manager."""
        model = self._compiled_vision_model()
        if model is None:
            return
        with _set_spyre_compilation_settings(self.vllm_config):
            model.compile_vision_graphs()
            self._maybe_init_encoder_cudagraph_manager()
            assert self.encoder_cudagraph_manager is not None
            self.encoder_cudagraph_manager.capture(graph_pool=None)
            self._warmup_vision_merge(model)

    def _warmup_vision_merge(self, model: SpyreVisionEncoderMixin) -> None:
        """Compile ``vision_merge`` at every token bucket an image step can land on."""
        bucketer = self.spyre_shape_bucketer
        if bucketer is None:
            return
        wrapper = cast(_SpyreModelWrapper, self.model)
        image_rows = [torch.zeros(1, model.out_hidden_size, dtype=self._model_dtype())]
        # Image steps stage at least two rows (see _SpyreModelWrapper.embed_input_ids), so a
        # bucket below two is never reached.
        for size in bucketer.bucket_sizes:
            if size < 2:
                continue
            is_multimodal = torch.zeros(size, dtype=torch.bool)
            is_multimodal[0] = True
            wrapper.embed_input_ids(
                torch.zeros(size, dtype=torch.int32), image_rows, is_multimodal=is_multimodal
            )
        logger.info("Vision merge compiled at every token bucket.")

    def _compiled_vision_model(self) -> SpyreVisionEncoderMixin | None:
        if not self.supports_mm_inputs or self.compilation_config.mode is CompilationMode.NONE:
            return None
        model = getattr(self.model, "_model", self.model)
        return model if isinstance(model, SpyreVisionEncoderMixin) else None

    def _create_encoder_cudagraph_manager(self) -> SpyreEncoderGraphManager | None:
        # Not keyed on `cudagraph_mm_encoder`, which also gates GPU-only memory profiling.
        model = self._compiled_vision_model()
        if model is None:
            return None
        # The processor's largest item, as upstream computes it for the scheduler's encoder
        # budget; a model whose encoder tokens differ overrides its budget range.
        assert self.mm_budget is not None
        model.max_tokens_per_item = max(
            self.mm_budget.mm_max_toks_per_item[m] for m in model.vision_modalities
        )
        return SpyreEncoderGraphManager(
            self.vllm_config, self._spyre_device, self._model_dtype(), model
        )

    @torch.inference_mode()
    def _warmup_input_embedding(self, num_tokens: int) -> None:
        """Compile the embedding step a dummy run skips.

        ``_dummy_run`` hands the model a zeroed ``inputs_embeds`` slice instead of
        producing it the way ``_preprocess`` does, and producing it is what compiles.
        """
        # Only a decoder-only multimodal model embeds through ``embed_input_ids``: a
        # text-only signature takes no multimodal arguments, so the call would raise.
        mm_config = getattr(self.model_config, "multimodal_config", None)
        if (
            not self.supports_mm_inputs
            or self.model_config.is_encoder_decoder
            or (mm_config is not None and mm_config.mm_encoder_only)
        ):
            return
        model = cast(_SpyreModelWrapper, self.model)
        # int32 to match upstream's `input_ids` buffer; 0 is an id every vocab holds.
        model.embed_input_ids(torch.zeros(num_tokens, dtype=torch.int32))
        logger.info_once("Warming the input embedding through embed_input_ids.")

    @torch.inference_mode()
    def _warmup_multimodal_encoder(self) -> None:
        """Compile the vision tower at every image-batch width it can see.

        Nothing else traces the tower before the compile guard arms: the dummy runs
        reach the text body only, and upstream's ``profile_run`` (one pass at
        ``mm_max_items_per_batch``) never runs, as ``TorchSpyreWorker`` overrides
        ``determine_available_memory``. Under ``dynamic=False`` every width is its own
        graph, for marked encoder blocks and ``SpyreConv2d`` alike, so each runs here
        (spyre-inference#1182).

        A generative VLM is enumerated only when its tower's blocks are compiled. An
        eager tower's ops compile through torch-spyre's per-op kernels, which the guard
        expects on the serving path, and enumerating them would dominate startup.
        """
        if not self.supports_mm_inputs or (
            self.model_config.runner_type != "pooling" and not self._compiled_tower_blocks
        ):
            return
        mm_budget = getattr(self, "mm_budget", None)
        max_items = getattr(mm_budget, "mm_max_items_per_batch", None)
        model = cast(_SpyreModelWrapper, self.model)
        if not max_items or not hasattr(model, "embed_multimodal"):
            return

        widths = [(modality, int(n)) for modality, n in max_items.items() if int(n) >= 1]
        # Images only for a generative tower: an image item is one tower row of a fixed shape;
        # a video or audio item expands to a frame or chunk count the request sets.
        if self.model_config.runner_type != "pooling":
            widths = [(modality, n) for modality, n in widths if modality == "image"]
        if not widths:
            return
        logger.info(
            "Warming multimodal encoder batch widths: %s.",
            ", ".join(f"{modality} 1..{n}" for modality, n in widths),
        )
        t0 = time.time()
        for modality, n_max in widths:
            for n in range(1, n_max + 1):
                model.embed_multimodal(**self._get_mm_dummy_batch(modality, n))
        logger.info("Multimodal encoder warmup done in %.3fs.", time.time() - t0)

    @torch.inference_mode()
    def _record_attention_graphs(self) -> None:
        """Pre-compile the attention kernels, per-sequence and batched decode.

        The model-level warmup above cannot cover these: ``_dummy_run`` delegates
        upstream, which passes ``attn_metadata=None``, so ``forward`` returns
        before touching a kernel. Left lazy, each new variant pays a full
        Inductor compile mid-serving.
        """
        if not envs.SPYRE_ATTN_RECORD:
            logger.info("Attention graph recording disabled (SPYRE_ATTN_RECORD=0)")
            return
        if self.compilation_config.mode is CompilationMode.NONE:
            logger.info("Attention graph recording disabled (CompilationMode.NONE)")
            return
        assert self._spyre_kv_caches, (
            "Attention graph recording needs the KV cache, but "
            "_spyre_kv_caches is empty: initialize_kv_cache_tensors() must run first."
        )

        # Every layer keeps its own kernel cache, so each is recorded separately;
        # layers sharing a head configuration trace to the same graph and only
        # the first pays a full Inductor compile.
        static_ctx = self.compilation_config.static_forward_context
        t0 = time.time()
        total = 0
        builders = self._attn_metadata_builders()
        with _set_spyre_compilation_settings(self.vllm_config):
            for layer_name, kv_cache in self._spyre_kv_caches.items():
                layer = static_ctx.get(layer_name)
                impl = getattr(layer, "impl", None)
                if not isinstance(impl, SpyreAttentionImpl):
                    continue
                builder = builders.get(layer_name)
                # A KV-cache layer on this impl is always in an attention group whose
                # backend builds SpyreAttentionMetadataBuilder, so a miss is a wiring or
                # ordering bug (recording before initialize_attn_backend), not a config.
                assert builder is not None, (
                    f"Layer {layer_name} has a Spyre attention impl and a KV cache but no "
                    "Spyre metadata builder; initialize_attn_backend() must run first."
                )
                logger.info("Recording attention graphs for layer %s...", layer_name)
                total += impl.record_graphs(layer, kv_cache, builder)
        logger.info(
            "Attention graph recording complete: %d graphs in %.3fs.",
            total,
            time.time() - t0,
        )

    def _attn_metadata_builders(self) -> dict[str, SpyreAttentionMetadataBuilder]:
        """Each layer's metadata builder: attention groups' specs (block size, sliding
        window) need not agree, and a group's ubatch builders differ only in internal
        buffers, so the first stands for all."""
        builders: dict[str, SpyreAttentionMetadataBuilder] = {}
        for group in self._attn_group_iterator():
            builder = group.metadata_builders[0] if group.metadata_builders else None
            if not isinstance(builder, SpyreAttentionMetadataBuilder):
                continue
            for layer_name in group.layer_names:
                builders[layer_name] = builder
        return builders

    def initialize_attn_backend(self, kv_cache_config, is_profiling: bool = False) -> None:
        super().initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
        self._split_attn_groups_by_layer_window()

    def _split_attn_groups_by_layer_window(self) -> None:
        from vllm.config import get_layers_from_vllm_config

        split_attn_groups_by_window(
            self.attn_groups, get_layers_from_vllm_config(self.vllm_config, Attention)
        )

    def _build_attention_metadata(self, *args, **kwargs):
        """Pick the encoder path for this step and attach its plan to the metadata.

        Upstream returns ``(per_layer_metadata, common_metadata)``; the plan hangs off
        the per-layer objects, and its *type* is the path, so every layer in the stack
        runs the same one. Built here rather than in ``forward`` because the builder
        does a D2H read and an H2D convert, which inside a traced region become graph
        nodes.

        Runs for warmup's dummy batches as well as real ones: the impl reads the path
        off the plan, so a dummy run without one would fall through to the wrong path
        and warm the wrong graph.
        """
        out = super()._build_attention_metadata(*args, **kwargs)
        if self.model_config.runner_type != "pooling":
            return out
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            EncoderRectPlan,
            plan_encoder_step,
            publish_encoder_grid,
        )

        rows = int(kwargs.get("num_tokens_padded") or self._encoder_buffer_rows or 0)
        # A rectangle *is* the body buffer, so a step the bucketer did not pad to the
        # declared row count has no rectangle to land on -- eager runs, and any step
        # before the bucketer is warmed. Those take the packed path, which works at
        # any row count.
        rectangles = self._encoder_rectangles if rows == self._encoder_budget else []
        if self._force_encoder_ragged:
            rectangles = []

        per_layer = out[0] if isinstance(out, tuple) else out
        groups = per_layer if isinstance(per_layer, list) else [per_layer]
        plan = plan_encoder_step(
            (md for group in groups if isinstance(group, dict) for md in group.values()),
            rectangles=rectangles,
            width_cap_for=self._encoder_width_caps,
            # self.device is CPU by design (see the class docstring); the real device is
            # _spyre_device.
            device=self._spyre_device,
            dtype=self._model_dtype(),
            # Grouping only pays off through the compiled kernels; the eager per-request
            # loop has no launch overhead to amortise.
            batched=self.compilation_config.mode is CompilationMode.STOCK_TORCH_COMPILE,
            forced_rect=self._forced_encoder_rect,
        )
        # A rectangle's mask shape is how a traced forward reads the grid; the ragged path
        # publishes nothing and so stays opaque.
        publish_encoder_grid(plan)
        # A stale grid would silently mislay this step's tokens in `_preprocess`.
        self._encoder_grid = (
            (plan.extent, plan.width, plan.query_lens)
            if isinstance(plan, EncoderRectPlan)
            else None
        )
        self.spyre_encoder_stats.record(plan, rows)
        return out

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool = False,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        """Inject bucket padding via upstream's padding mechanism.

        Upstream uses this method to determine CUDA graph padding. We reuse
        the same interface to return our shape-bucketed padding so that the
        rest of execute_model (slot_mapping, attention metadata, _preprocess)
        handles padded vs unpadded counts correctly without mutating
        scheduler_output.total_num_scheduled_tokens.

        Decoder: 1D ``compile_sizes`` after warmup. Pooling: one shape, the token
        budget, whichever encoder path the step then takes.
        """
        pad = self._spyre_bucket_batch_descriptor(num_tokens, num_reqs, num_scheduled_tokens_np)
        if pad is not None:
            self._encoder_buffer_rows = pad.num_tokens
            return CUDAGraphMode.NONE, pad, False, None, None
        self._encoder_buffer_rows = num_tokens

        return super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    def _spyre_bucket_batch_descriptor(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> BatchDescriptor | None:
        """Padded ``BatchDescriptor`` for a warmed 1D body bucket, or None.

        Decoder and pooling share this path; the buckets are ``compile_sizes``. Pooling
        declares just one of them, the token budget, so every pooling step pads to that
        same row count.
        """
        del num_reqs, num_scheduled_tokens_np
        bucketer = self.spyre_shape_bucketer
        padded = bucketer.warmed_bucket(num_tokens) if bucketer is not None else None
        return None if padded is None else BatchDescriptor(num_tokens=padded)

    @torch.inference_mode()
    def _dummy_run(self, *args, **kwargs):
        """Force D2H during dummy forward (upstream logits index is CPU).

        Pooling must pass ``force_attention=True``. Upstream skips attention
        metadata unless that flag or a FULL cudagraph is set; encoder impl
        then does ``if attn_metadata is None: return output`` and never
        compiles pack/SDPA. Real ``execute_model`` always has metadata.

        Decoder warmup also publishes null KV slots so the scatter-in-graph
        path (#610) sees the same binding as a real step.
        """
        if self.model_config.runner_type == "pooling":
            kwargs.setdefault("force_attention", True)
        # Read out of the passthrough rather than named in the signature, which would
        # pin this override to upstream's parameter order across vLLM bumps.
        num_tokens = kwargs.get("num_tokens", args[0] if args else None)
        if num_tokens is not None:
            attn_layer.publish_null_slots(num_tokens)
        wrapper = self.model
        keep = isinstance(wrapper, _SpyreModelWrapper) and wrapper._keep_outputs_on_device
        if keep:
            object.__setattr__(wrapper, "_keep_outputs_on_device", False)
        try:
            hidden_states, last_hidden_states = super()._dummy_run(*args, **kwargs)
        finally:
            if keep:
                object.__setattr__(wrapper, "_keep_outputs_on_device", True)

        if (
            keep
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.numel() > 0
            and hidden_states.device.type != "spyre"
        ):
            hidden_states = convert(hidden_states, self._spyre_device)
        return hidden_states, last_hidden_states

    @torch.inference_mode()
    def _warm_pooler_row_widths(self, hidden_states: torch.Tensor) -> None:
        """Compile the pooler's row gather at every row count serving can present.

        ``_dummy_pooler_run`` only ever pools ``min(num_tokens, max_num_seqs)``
        rows, but serving pools one row per *request* -- any count from 1 up. That
        gather specializes on the exact index length, so every unseen count paid a
        full Inductor compile mid-request; it was the whole residual warm-up gap
        once the attention kernels were covered. The poolers round their row count
        to a power of two (``pad_row_count_to_bucket``), so this sweep is short.

        ``SpyreSequencePooler`` trims only after the classifier/embed head, so the head
        sees these same bucket widths and is warmed at each. ``SpyreMeanPool``'s
        rectangle reduce specializes on ``(extent, rows)`` and is warmed per declared
        rectangle.
        """
        if not self._pooling_on_spyre:
            return
        rows = hidden_states.shape[0]
        # poolers_by_task is a plain dict; nn.Module.modules() doesn't descend into it.
        pooler = cast(VllmModelForPooling, self.get_model()).pooler
        pooler_modules = list(_iter_modules(pooler))
        mean_pooling = any(isinstance(m, SpyreMeanPool) for m in pooler_modules)
        classifier_types = SpyreRobertaClassificationHead | SpyreClassifierLinear
        classifier_heads = [
            m
            for m in pooler_modules
            if isinstance(getattr(m, "classifier", None), classifier_types)
        ]
        embed_heads = [m for m in pooler_modules if isinstance(m, SpyreEmbeddingPoolerHead)]
        token_poolers = [m for m in pooler_modules if isinstance(m, SpyreTokenPooler)]
        # Up to the power of two at or above the limit, which is not itself always one:
        # 6 sequences round up to 8 rows, so stopping at the limit misses that width.
        limit = 1 << max(0, self.scheduler_config.max_num_seqs - 1).bit_length()
        widths = [1 << i for i in range(limit.bit_length()) if 1 << i <= rows]
        # MEAN on a rectangle reduces its first min(bucket, lanes) lanes; a grid with a
        # non-power-of-two lane count hands the head that count instead of the bucket.
        grid_shapes: set[tuple[int, int]] = set()
        if mean_pooling:
            grid_shapes = {
                (extent, min(width, lanes))
                for extent, lanes in self._encoder_rectangles
                if extent * lanes == rows
                for width in widths
            }
        for width in sorted(set(widths) | {lanes for _, lanes in grid_shapes}):
            pooled = select_rows(hidden_states, torch.zeros(width, dtype=torch.int64))
            for head in classifier_heads:
                logits = head.classifier(pooled)
                if head.activation is not None:
                    head.activation(logits)
            for head in embed_heads:
                embeddings = head.projector(pooled) if head.projector is not None else pooled
                if head.activation is not None:
                    head.activation(embeddings)
        if grid_shapes:
            device = hidden_states.device
            mask = convert(torch.zeros(rows, 1, dtype=torch.float16), device)
            prod = _mean_pool_row_mask_mul(hidden_states, mask).clone()
            for extent, lanes in grid_shapes:
                lens = convert(torch.ones(lanes, 1, dtype=torch.float32), device)
                _mean_pool_grid_reduce(prod, lens, extent)
        for token_pooler in token_poolers:
            self._warm_token_pool_widths(token_pooler, hidden_states)

    def _warm_token_pool_widths(
        self, token_pooler: SpyreTokenPooler, hidden_states: torch.Tensor
    ) -> None:
        """Compile ``SpyreAllPool``'s bucketed gather (and its head) at every declared length.

        ``SpyreAllPool`` rounds each request's real token count up to the nearest
        entry in ``len_ladder`` instead of tracking it exactly (see its docstring);
        nothing before this swept that same ladder, so the first request landing on
        an unseen length paid a full Inductor compile mid-request.
        """
        all_pool = token_pooler.pooling
        if not isinstance(all_pool, SpyreAllPool):
            return
        head = token_pooler.head
        rows = hidden_states.shape[0]
        for length in sorted(set(all_pool.len_ladder)):
            if length > rows:
                continue
            chunk = select_rows(hidden_states, torch.arange(length, dtype=torch.int64))
            if head is None:
                continue
            classifier = getattr(head, "classifier", None)
            logits = classifier(chunk) if classifier is not None else chunk
            projector = getattr(head, "projector", None)
            if projector is not None:
                logits = projector(logits)
            activation = getattr(head, "activation", None)
            if activation is not None:
                activation(logits)

    @torch.inference_mode()
    def _warm_encoder_inline_paths(self) -> None:
        """One block graph per declared rectangle, plus the one that keeps attention opaque.

        With attention traced in, the block graph specialises on ``(width, extent)``, so the
        single body-shape dummy above covers only whichever rectangle its own batch landed
        on. The trailing ragged run compiles the opaque-attention block graph and is what
        reaches ``warm_kernels`` for the group shapes, since a traced rectangle never calls
        the impl; it is skipped when the group table is empty, which is the bucketer's own
        statement that no batch can miss its rectangle.
        """
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            encoder_inline_active,
        )

        if not encoder_inline_active() or not self._encoder_rectangles:
            return
        groups = encoder_group_shapes(self.vllm_config)
        logger.info(
            "Warming %d encoder rectangle block graph(s)%s.",
            len(self._encoder_rectangles),
            " plus the opaque-attention one" if groups else "",
        )
        for rect in sorted(self._encoder_rectangles, reverse=True):
            self._forced_encoder_rect = rect
            try:
                self._dummy_run(self._encoder_budget, force_attention=True)
            finally:
                self._forced_encoder_rect = None
        if not groups:
            return
        self._force_encoder_ragged = True
        try:
            self._dummy_run(self._encoder_budget, force_attention=True)
        finally:
            self._force_encoder_ragged = False

    @torch.inference_mode()
    def _warm_encoder_unpack(self, hidden_states: torch.Tensor) -> None:
        """Compile the rectangular path's re-compaction gather.

        ``unpad_encoder_grid`` runs in ``_pool``, which no dummy run reaches, so
        without this the first request needing packed-order re-compaction pays its compile.
        One shape only: the gather deliberately keeps the buffer's row count.
        """
        if not self._pooling_on_spyre or not self._encoder_rectangles:
            return
        select_rows(hidden_states, torch.zeros(hidden_states.shape[0], dtype=torch.int64))

    @torch.inference_mode()
    def _warm_encoder_embed_gather(self) -> None:
        """Compile ``_preprocess``'s embeddings-to-grid gather once per rectangle.

        Only a multimodal pooling model (e.g. CLIP) reaches it, and ``_dummy_run`` hands
        the model its own ``inputs_embeds`` without going through ``_preprocess``.
        """
        buffer = getattr(self, "inputs_embeds", None)
        if (
            not self.supports_mm_inputs
            or not self._pooling_on_spyre
            or not self._encoder_rectangles
            or buffer is None
        ):
            return
        body = buffer.gpu[: self._encoder_budget]
        for extent, width in self._encoder_rectangles:
            select_rows(body, torch.zeros(width * extent, dtype=torch.int64))

    def _preprocess(self, *args, **kwargs):
        """Expand the ragged body into the dense grid on the rectangular path.

        Upstream writes rows contiguously and the padding hook only sets the trailing
        pad count, so the interior per-sequence padding a rectangle needs has to
        happen here, through ``expand_body_to_grid``.

        On the packed path, absolute-position RoBERTa still offsets positions here
        before the embedding gather. ``query_start_loc`` and ``seq_lens`` keep the
        real ragged lengths, which attention's mask needs.
        """
        out = super()._preprocess(*args, **kwargs)
        grid = self._encoder_grid
        # A model that masks multimodal rows after embedding (CLIP, models/clip.py)
        # needs its mask in the same row layout as the embeddings.
        align_token_mask = getattr(self.model, "spyre_align_token_mask", None)
        if grid is None:
            if align_token_mask is not None:
                scheduler_output = args[0] if args else kwargs["scheduler_output"]
                align_token_mask(scheduler_output.total_num_scheduled_tokens, None)
            return self._offset_preprocess_positions(out)
        input_ids, inputs_embeds, positions, *rest = out
        extent, width, query_lens = grid
        num_tokens = sum(query_lens)
        if align_token_mask is not None:
            align_token_mask(num_tokens, grid)
        hf_config = getattr(self.model_config, "hf_config", None)
        position_offset = roberta_position_delta(hf_config)

        grid_ids, grid_embeds, grid_pos = expand_body_to_grid(
            input_ids,
            inputs_embeds,
            positions,
            grid,
            pad_token_id=encoder_pad_token_id(self.model_config),
            position_offset=position_offset,
        )

        # token_type_ids rides through `rest` in model_kwargs and is one value per packed
        # token, so it needs the same rearrangement or every sequence past the first gets
        # another's segment ids. Scanned for rather than indexed, to avoid pinning this
        # override to upstream's return arity.
        regrouped = []
        for item in rest:
            if not isinstance(item, dict):
                regrouped.append(item)
                continue
            model_kwargs = cast(dict[str, Any], item)
            token_types = model_kwargs.get("token_type_ids")
            if token_types is not None:
                model_kwargs = {
                    **model_kwargs,
                    "token_type_ids": convert(
                        expand_packed_token_types(
                            token_types[:num_tokens].cpu(), query_lens, width, extent
                        ),
                        token_types.device,
                    ),
                }
            regrouped.append(model_kwargs)

        return (grid_ids, grid_embeds, grid_pos, *regrouped)

    def _offset_preprocess_positions(self, out: Any) -> Any:
        """Add the RoBERTa offset on the packed path, before the wrapper's H2D.

        The rectangular path adds it inside ``expand_packed_to_encoder_grid``.
        Here the body stays packed, so the position tensor upstream returned is
        rewritten in place in the tuple.
        """
        hf_config = getattr(self.model_config, "hf_config", None)
        delta = roberta_position_delta(hf_config)
        if delta == 0 or not isinstance(out, tuple) or len(out) < 3:
            return out
        positions = out[2]
        if not isinstance(positions, torch.Tensor):
            return out
        return (out[0], out[1], offset_host_positions(positions, delta), *out[3:])

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        """Same as GPU dummy pooler, but the cursor stays on CPU like ``_pool``."""
        if not self._pooling_on_spyre:
            return super()._dummy_pooler_run_task(hidden_states, task)

        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs
        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            prompt_token_ids_cpu=dummy_token_ids.cpu(),
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=torch.device("cpu"),
        )
        return model.pooler(hidden_states=hidden_states, pooling_metadata=dummy_metadata)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Pool on the activation device; copy only the pooled vectors back.

        FP32 linear heads keep the pooler on CPU and use
        ``GPUModelRunner._pool``. Do not crop here: Spyre dim-0 slices are
        unsafe, and each method gathers itself (CLS/LAST rows, MEAN
        per-request rows, token AllPool ranges).
        """
        assert not self.use_async_scheduling, (
            "async scheduling is unsupported while pooling on Spyre"
        )

        if not self._pooling_on_spyre:
            hidden_states = unpad_encoder_grid(convert(hidden_states, "cpu"), self._encoder_grid)
            return super()._pool(
                hidden_states,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                kv_connector_output,
            )

        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        # Build the cursor on CPU: upstream does ``cumsum[1:] - 1`` for
        # last_token_indices; that offset-1 view is not stick-aligned on
        # Spyre (copy_from_d2d fails). So the ``*_gpu`` fields here are host
        # tensors, despite the name, and the Spyre pools read them as such.
        # A Spyre-specific PoolingMetadata (like SpyreAttentionMetadata) would
        # make that explicit.
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=torch.device("cpu"),
        )

        hidden_states = convert(hidden_states, self._spyre_device)
        model = cast(VllmModelForPooling, self.model)
        hidden_states = pool_rows_for(
            hidden_states, model.pooler, pooling_metadata, self._encoder_grid
        )

        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        model_runner_output.pooler_output = copy_pooler_output_to_cpu(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )
        self._sync_device()
        return model_runner_output

    # --- KV cache allocation ---

    def _model_dtype(self) -> torch.dtype:
        """The activation dtype the platform settled on (float16 for every engine run;
        see ``TorchSpyrePlatform.apply_config_platform_defaults``)."""
        dtype = self.model_config.dtype
        return dtype if isinstance(dtype, torch.dtype) else torch.float16

    def initialize_kv_cache_tensors(
        self,
        kv_cache_config,
        kernel_block_sizes,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ):
        """Allocate KV cache as one dense paged tensor per layer on Spyre.

        Each layer gets its own SpyrePagedKVCache(k_pages, v_pages), in the shape and
        device layout its attention impl's `allocate_pages` chooses. The attention kernel
        selects a page by indexing with a one-element device tensor, so the page read is
        a real indirect access.

        ``kernel_block_sizes`` is ignored: the backend accepts any 64-multiple, so the
        kernel block size always equals the manager's.
        """
        from vllm.v1.worker.utils import bind_kv_cache

        static_ctx = self.compilation_config.static_forward_context
        # vLLM's `bind_kv_cache` types this dict as `dict[str, torch.Tensor]`,
        # but the matching `SpyreAttentionImpl.forward` consumes the
        # SpyrePagedKVCache — see the suppression on `bind_kv_cache(...)` below.
        with kv_cache_allocation_context or nullcontext():
            kv_caches = allocate_spyre_kv_caches(
                kv_cache_config, static_ctx, self.shared_kv_cache_layers, self._spyre_device
            )

        bind_kv_cache(
            kv_caches,  # ty: ignore[invalid-argument-type]
            static_ctx,
            self.kv_caches,
        )
        self._spyre_kv_caches = dict(kv_caches)
        return kv_caches

    # --- Stubs copied from CPUModelRunner ---
    # These are trivial overrides that GPUModelRunner expects.

    def _sync_device(self) -> None:
        # Wait for outstanding async H2D from SpyreCpuGpuBuffer.copy_to_gpu
        # (and any other non_blocking copies) before the runner consumes
        # Spyre tensors. torch.spyre is registered by torch-spyre autoload.
        torch.spyre.synchronize(self._spyre_device)

    def get_model(self) -> nn.Module:
        # Return the unwrapped model for isinstance checks
        # (e.g. is_text_generation_model in get_supported_tasks).
        model = self.model
        if isinstance(model, _SpyreModelWrapper):
            model = model._model
        # Unwrap torch.compile's OptimizedModule (has _orig_mod attribute)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        assert isinstance(model, nn.Module)
        return model

    # --- Buffer management ---

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> SpyreCpuGpuBuffer:
        """Create a SpyreCpuGpuBuffer with float tensors on Spyre.

        - Float dtypes: .cpu on CPU, .gpu on Spyre at the model dtype
        - Int/bool dtypes: .gpu aliased to .cpu (stays on CPU)
        """
        if dtype.is_floating_point:
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=self._model_dtype(),
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Int/bool → CPU-only (aliased)
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=numpy,
        )


@contextmanager
def _set_spyre_compilation_settings(config: VllmConfig):
    """Context manager for Spyre-specific compilation settings during warmup.

    Similar to _set_global_compilation_settings in cpu_model_runner.py but
    adapted for Spyre's compilation requirements.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            # Freezing folds per-block weights into each block's graph, defeating
            # artifact sharing. Warn rather than override an explicit max_autotune.
            if _compile_granularity() == "block" and not config.model_config.enforce_eager:
                logger.warning(
                    "max_autotune enables Inductor freezing, which folds per-block "
                    "weights into each block's graph and defeats per-block artifact "
                    "sharing. Compile time will grow with layer count."
                )
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
