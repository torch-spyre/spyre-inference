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

"""Per-block vs whole-model compile (SPYRE_COMPILE_GRANULARITY), shared by both model runners."""

import torch.nn as nn
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference import envs
from spyre_inference.multimodal.utils import blocks_marked_compilable
from spyre_inference.v1.worker import compile_guard

SPYRE_COMPILE_GRANULARITIES = ("block", "model")


def compile_granularity() -> str:
    granularity = envs.SPYRE_COMPILE_GRANULARITY
    if granularity not in SPYRE_COMPILE_GRANULARITIES:
        raise ValueError(
            f"Unsupported SPYRE_COMPILE_GRANULARITY={granularity!r}. "
            f"Expected one of {SPYRE_COMPILE_GRANULARITIES}."
        )
    return granularity


def block_sharing_defeated_by() -> str | None:
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


def repeated_block_lists(model: nn.Module) -> list[tuple[nn.ModuleList, bool]]:
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


def compile_blocks(block_lists: list[tuple[nn.ModuleList, bool]]) -> tuple[int, int]:
    """Compile every block in place; returns (decoder blocks, marked tower blocks)."""
    # Only decoder blocks are counted first: the count decides the whole-model fallback,
    # which an opted-in tower must not pre-empt (it compiles beside either form).
    num_blocks = num_tower = 0
    # Models that re-register a slice of `layers` (e.g. gemma-4's self-/cross-decoder)
    # alias blocks across lists; recompiling one is harmless but would double the count.
    seen: set[int] = set()
    for blocks, marked in block_lists:
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
    return num_blocks, num_tower
