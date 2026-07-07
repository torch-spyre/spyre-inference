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

"""Spyre OOT replacement for VocabParallelEmbedding.

The embedding layer is kept ENTIRELY ON CPU. Rationale: the Spyre backend
registers `aten.embedding.default` as a `register_fallback`-based CPU
fallback (see `torch_spyre/ops/fallbacks.py:245`). That fallback wrapper
moves the embedding inputs to CPU, runs `torch.embedding` there, and
copies the result back to Spyre via `spyre_copy_from`. When this happens
inside an Inductor-compiled forward graph, the H2D copy violates Inductor's
view-tracking and corrupts the heap — Granite-3.3-8B crashes with
`malloc_consolidate(): invalid chunk size` at the first forward call
(see `embedding_heap_corruption_findings.md`).

Two-part fix here:

1. Override `_apply` so the embedding weight tensor never moves to Spyre
   in the first place (mirrors the Attention pattern at
   `spyre_model_runner.py:269`). The weight stays on CPU as a normal
   `nn.Parameter`. No per-call D2H of a 4 GB weight tensor; no Spyre↔
   CPU traffic for the weight ever.

2. Wrap `forward` as a `direct_register_custom_op`
   (`torch.ops.vllm.spyre_vocab_parallel_embedding_native`). The
   embedding op then appears as a single opaque FallbackKernel to
   Inductor — invisible to the compile pipeline. Only the
   already-on-CPU `F.embedding(input_cpu, weight_cpu)` runs in the op
   body; the only Spyre↔CPU traffic per call is the small input_ids
   tensor (D2H) and the resulting hidden_states (H2D), both routed
   through the existing opaque `spyre_convert` custom op.
"""

from functools import lru_cache

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, get_layer, register_layer

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for Spyre.

    Embedding weight stays on CPU; the entire layer's forward runs on CPU
    and only the result is moved to Spyre via `spyre_convert`.
    """

    # Marker read by the model runner after `model.to(spyre)`: force this
    # module's params/buffers back to CPU. Necessary because when the weight
    # is TIED to a Spyre-native layer (lm_head with tie_word_embeddings), the
    # `_apply` no-op below is bypassed and the shared storage gets moved to
    # Spyre in place. See spyre_model_runner.load_model.
    _spyre_keep_on_cpu = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )
        self._spyre_layer_name = register_layer(
            self, "spyre_vocab_parallel_embedding"
        )

    def _apply(self, fn, recurse: bool = True):
        # No-op: keep all parameters and buffers (notably `self.weight`) on
        # the device they were initialized on (CPU). Mirrors the Attention
        # CPU-stay pattern at spyre_model_runner.py:269. This makes
        # `self.model.to("spyre")` skip this module entirely, so embedding
        # never goes through Spyre's `_copy_from` for its weight.
        #
        # NB: when the weight is TIED to lm_head (tie_word_embeddings=True),
        # this override is bypassed and the shared storage is moved to Spyre
        # anyway; the `_spyre_keep_on_cpu` marker + the model runner's
        # post-`.to()` sweep corrects that.
        return self

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # The whole forward is one opaque FallbackKernel as far as Inductor
        # is concerned. See module docstring for the why.
        return torch.ops.vllm.spyre_vocab_parallel_embedding_native(
            input_, self._spyre_layer_name
        )

    def _forward_impl(self, input_: torch.Tensor) -> torch.Tensor:
        """Inner body called by the opaque-wrapping custom op.

        The weight is already CPU-resident (see `_apply` override). We
        just need to land `input_` on CPU and send the result back to
        whatever device the caller provided.
        """
        input_cpu = convert(input_, device="cpu")

        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_cpu,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_cpu
            input_mask = None

        # Weight is on CPU (kept there by the `_apply` override above plus the
        # model runner's post-`.to()` CPU sweep for tied weights).
        # Run the lookup directly via aten.embedding on CPU — no Spyre
        # device dispatch, so the unstable Spyre CPU-fallback path in
        # torch_spyre/ops/fallbacks.py:245 never runs.
        output_cpu = torch.nn.functional.embedding(
            masked_input.long(), self.weight.data
        )

        if self.tp_size > 1 and input_mask is not None:
            keep = (~input_mask).to(output_cpu.dtype).unsqueeze(-1)
            output_cpu = output_cpu * keep

        # Route the (only) H2D copy through the existing opaque
        # spyre_convert custom op so Inductor sees one FallbackKernel
        # boundary, not a bare `.to(spyre_device)`.
        output = convert(output_cpu, device=input_.device, dtype=output_cpu.dtype)

        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


def _vocab_parallel_embedding_native_op_func(
    input_: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_layer(layer_name)
    out = layer._forward_impl(input_)
    # Custom-op contract: fresh, non-aliasing storage for each returned tensor.
    return out.contiguous()


def _vocab_parallel_embedding_native_op_fake(
    input_: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_layer(layer_name)
    embed_weight = layer.weight
    hidden_size = embed_weight.shape[1]
    out_shape = list(input_.shape) + [hidden_size]
    return torch.empty(out_shape, dtype=embed_weight.dtype, device=input_.device)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_parallel_embedding_native custom op."""
    direct_register_custom_op(
        op_name="spyre_vocab_parallel_embedding_native",
        op_func=_vocab_parallel_embedding_native_op_func,
        fake_impl=_vocab_parallel_embedding_native_op_fake,
        mutates_args=[],
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once(
        "Registered custom op: spyre_vocab_parallel_embedding_native"
    )
