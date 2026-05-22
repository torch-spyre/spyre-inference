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

"""Paged KV-cache implementation of AttentionBackend using torch-spyre.

This backend implements attention using only PyTorch native operations (matmul,
softmax, etc.) and supports vLLM's paged KV cache. Spyre does not yet support
advanced tensor indexing, in-device transpose+contiguous, or simultaneous
dtype+device conversion, so all cache operations use mask-based alternatives:
elementwise scatter, matmul-based gather, and compiled attention.

Required configuration:
  - max_num_seqs=1
  - num_gpu_blocks_override must be set (e.g. 64)
  - max_model_len <= num_gpu_blocks_override * block_size (e.g. 1024 with block_size=16)

Terminology:
  - aligned_num_physical_blocks: num_blocks rounded up to multiple of 64 for Spyre stick alignment.
  - block_width: block_size * head_size — number of columns per head in cache tensors
    (all tokens in a block concatenated across the head dimension).
  - aligned_max_seq_len: max_seq_len rounded up to a multiple of
    KV_LENGTH_ALIGNMENT (256), bucketing sequence lengths to reduce
    torch.compile recompilations.
  - num_tokens: number of real (non-padding) tokens inserted into the KV cache this
    forward — i.e. num_actual_tokens from AttentionMetadata. 1 during decode, chunk
    size during prefill. Drives the shapes of the per-call scatter selectors.
  - padded_query_len: max_query_len rounded up to a multiple of QUERY_CHUNK_SIZE (32),
    keeping the attention kernel's tensor shapes constant across steps to avoid
    torch.compile recompilation.

Naming:
  - `_dev` suffix: tensor resides on Spyre (device), not CPU.
  - `k_cache_dev` / `v_cache_dev`: the full paged KV cache (Spyre-resident, persists across steps).
  - `k_new_*` / `v_new_*`: new token K/V for the current forward, to be scattered into the cache.
  - `compact_k` / `compact_v`: dense K/V for the active sequence, gathered from the cache.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

KV_LENGTH_ALIGNMENT = 256
QUERY_CHUNK_SIZE = 32


def _maybe_compile(fn):
    """Compile fn unless vLLM's compilation config disables it.

    Mirrors the gating in CustomOp.maybe_compile without requiring CustomOp
    inheritance: returns fn unchanged when compilation mode is NONE or the
    backend is "eager", otherwise wraps it with torch.compile.
    """
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    cfg = get_cached_compilation_config()
    if cfg.mode == CompilationMode.NONE:
        return fn
    if cfg.backend == "eager":
        return fn
    return torch.compile(fn, dynamic=False)


# --- Scatter: values built on CPU, then transferred to Spyre ---
# Tokens land at different block rows and column offsets, requiring
# advanced indexing to construct the values tensor (not supported on Spyre).


def _scatter_cache(cache, mask, values):
    return cache * (1.0 - mask) + values


# --- Attention ---


def _attn_4d(q, k, v, scale, mask):
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale
    scores = scores + mask
    p = scores.softmax(dim=-1)
    return p @ v


@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for PyTorch native attention computation on Spyre."""

    # Batch information
    num_actual_tokens: int
    num_seqs: int
    max_query_len: int
    max_seq_len: int

    # Sequence lengths
    seq_lens: torch.Tensor  # [num_seqs]
    query_start_loc: torch.Tensor  # [num_seqs + 1]

    # Block table for paged KV cache
    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    block_size: int

    # Slot mapping for KV cache updates
    slot_mapping: torch.Tensor  # [num_actual_tokens]

    # Whether causal masking is needed (True when max_query_len > 1)
    apply_causal_mask: bool = False

    # For grouped-query attention
    num_kv_heads: int = 0
    num_heads: int = 0

    # --- Precomputed by builder (shared across all layers) ---

    # Additive attention mask on Spyre, ready for the compiled attention op.
    # Shape: [num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len]
    # fp16, masked positions = -65504, unmasked = 0.
    attention_mask: torch.Tensor | None = None

    # Gather selection mask on Spyre:
    # [num_kv_heads, aligned_num_logical_blocks, aligned_num_physical_blocks]
    gather_sel_mask_dev: torch.Tensor | None = None

    # Aligned max seq len used by gather (needed for reshape)
    aligned_max_seq_len: int = 0

    # Scatter position mask on Spyre: [num_kv_heads, aligned_num_physical_blocks, block_width]
    scatter_mask_dev: torch.Tensor | None = None

    # Scatter row selector on Spyre: [num_kv_heads, aligned_num_physical_blocks, num_tokens]
    # one-hot: row_sel[h, r, t] = 1 iff block_indices[t] == r (broadcast-identical across heads)
    scatter_row_sel_dev: torch.Tensor | None = None

    # Scatter column placement selector on Spyre: [num_tokens, head_size, block_width]
    # one-hot: col_sel[t, d, c] = 1 iff c == block_offsets[t]*head_size + d
    scatter_col_sel_dev: torch.Tensor | None = None

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


class SpyreAttentionMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionMetadata]):
    """Builds attention metadata with precomputed masks."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.head_size = kv_cache_spec.head_size

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)

        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
        if num_blocks is None:
            num_blocks = vllm_config.cache_config.num_gpu_blocks
        assert num_blocks is not None, "num_gpu_blocks not yet determined"

        self.num_blocks = num_blocks
        self.aligned_num_physical_blocks = ((num_blocks + 63) // 64) * 64
        self.block_width = self.block_size * self.head_size

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        apply_causal_mask: bool,
        max_query_len: int,
        aligned_max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build additive attention mask, fully shaped and transferred to Spyre.

        Returns: [num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len]
        """
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_seqs = len(seq_lens)
        num_kv_heads = self.num_kv_heads

        padded_query_len = (
            (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )

        q_pos = torch.arange(max_query_len, device=device)
        kv_pos = torch.arange(aligned_max_seq_len, device=device)

        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)

        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        if apply_causal_mask:
            context_lens = seq_lens - query_lens
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit
            attend = attend & causal_ok

        mask_bool = ~attend  # [num_seqs, max_query_len, aligned_max_seq_len]

        if padded_query_len > max_query_len:
            padding = torch.ones(
                num_seqs,
                padded_query_len - max_query_len,
                aligned_max_seq_len,
                dtype=torch.bool,
                device=device,
            )
            mask_bool = torch.cat([mask_bool, padding], dim=1)

        mask_additive = torch.where(
            mask_bool,
            torch.tensor(-65504.0, dtype=self._target_dtype, device=device),
            torch.tensor(0.0, dtype=self._target_dtype, device=device),
        )

        mask_4d = (
            mask_additive.unsqueeze(1)
            .expand(-1, num_kv_heads, -1, -1)
            .reshape(num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len)
            .contiguous()
        )

        return mask_4d.to(device=self._target_device)

    def _build_gather_sel_mask(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        aligned_max_seq_len: int,
    ) -> torch.Tensor:
        block_size = self.block_size
        aligned_num_physical_blocks = self.aligned_num_physical_blocks
        num_kv_heads = self.num_kv_heads

        aligned_num_logical_blocks = aligned_max_seq_len // block_size

        num_kv_blocks = (int(seq_lens[0].item()) + block_size - 1) // block_size

        sel_mask_2d = torch.zeros(
            aligned_num_logical_blocks, aligned_num_physical_blocks, dtype=self._target_dtype
        )
        logical = torch.arange(num_kv_blocks, device=block_table.device)
        physical = block_table[0, :num_kv_blocks].long()
        sel_mask_2d[logical, physical] = 1.0

        sel_mask_3d = sel_mask_2d.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
        return sel_mask_3d.to(device=self._target_device)

    def _build_scatter_mask_and_selectors(
        self,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build scatter mask and on-device selectors for two-bmm scatter.

        Returns (scatter_mask_dev, scatter_row_sel_dev, scatter_col_sel_dev):
          - scatter_mask_dev:    [num_kv_heads, aligned_num_physical_blocks, block_width]
                                  — positions written by this step
          - scatter_row_sel_dev: [num_kv_heads, aligned_num_physical_blocks, num_tokens]
                                  — row_sel[h,r,t] = 1 iff block_indices[t]==r
          - scatter_col_sel_dev: [num_tokens, head_size, block_width]
                                  — col_sel[t,d,c] = 1 iff c == block_offsets[t]*head_size + d
        """
        block_size = self.block_size
        aligned_num_physical_blocks = self.aligned_num_physical_blocks
        block_width = self.block_width
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        num_tokens = len(slot_mapping)
        d_range = torch.arange(head_size)
        row_idx = block_indices.unsqueeze(1).expand(num_tokens, head_size).reshape(-1)
        col_idx = ((block_offsets * head_size).unsqueeze(1) + d_range.unsqueeze(0)).reshape(-1)

        mask = torch.zeros(
            num_kv_heads, aligned_num_physical_blocks, block_width, dtype=self._target_dtype
        )
        mask[:, row_idx, col_idx] = 1.0
        mask_dev = mask.to(device=self._target_device)

        # row_sel: [aligned_num_physical_blocks, num_tokens], one-hot on block_indices
        row_sel_2d = torch.zeros(aligned_num_physical_blocks, num_tokens, dtype=self._target_dtype)
        t_range = torch.arange(num_tokens)
        row_sel_2d[block_indices.long(), t_range] = 1.0
        row_sel_3d = row_sel_2d.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()
        row_sel_dev = row_sel_3d.to(device=self._target_device)

        # col_sel: [num_tokens, head_size, block_width],
        #     one-hot on block_offsets*head_size + d_range
        col_sel = torch.zeros(num_tokens, head_size, block_width, dtype=self._target_dtype)
        base_col = (block_offsets * head_size).long()  # [num_tokens]
        t_idx = t_range.unsqueeze(1).expand(num_tokens, head_size)  # [num_tokens, head_size]
        d_idx = d_range.unsqueeze(0).expand(num_tokens, head_size)  # [num_tokens, head_size]
        c_idx = base_col.unsqueeze(1) + d_idx  # [num_tokens, head_size]
        col_sel[t_idx.reshape(-1), d_idx.reshape(-1), c_idx.reshape(-1)] = 1.0
        col_sel_dev = col_sel.to(device=self._target_device)

        return mask_dev, row_sel_dev, col_sel_dev

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
        assert common_attn_metadata.num_reqs == 1, (
            "Spyre attention requires max_num_seqs=1, "
            f"got {common_attn_metadata.num_reqs} sequences in batch"
        )

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        max_seq_len = common_attn_metadata.max_seq_len
        max_query_len = common_attn_metadata.max_query_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        apply_causal_mask = common_attn_metadata.causal and max_query_len > 1

        aligned_max_seq_len = (
            (max_seq_len + KV_LENGTH_ALIGNMENT - 1) // KV_LENGTH_ALIGNMENT * KV_LENGTH_ALIGNMENT
        )

        attention_mask = self._build_attention_mask(
            seq_lens,
            query_start_loc,
            apply_causal_mask,
            max_query_len,
            aligned_max_seq_len,
            seq_lens.device,
        )

        gather_sel_mask_dev = self._build_gather_sel_mask(
            block_table,
            seq_lens,
            aligned_max_seq_len,
        )

        scatter_mask_dev, scatter_row_sel_dev, scatter_col_sel_dev = (
            self._build_scatter_mask_and_selectors(slot_mapping)
        )

        return SpyreAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_seqs=common_attn_metadata.num_reqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask=attention_mask,
            gather_sel_mask_dev=gather_sel_mask_dev,
            aligned_max_seq_len=aligned_max_seq_len,
            scatter_mask_dev=scatter_mask_dev,
            scatter_row_sel_dev=scatter_row_sel_dev,
            scatter_col_sel_dev=scatter_col_sel_dev,
        )


class SpyreAttentionBackend(AttentionBackend):
    """Paged KV-cache attention backend for Spyre."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreAttentionImpl"]:
        return SpyreAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreAttentionMetadataBuilder"]:
        return SpyreAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 64 == 0

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class SpyreAttentionImpl(AttentionImpl[SpyreAttentionMetadata]):
    """On-device KV cache attention, specialized for num_seqs=1.

    Masks are precomputed by the metadata builder — forward() only does
    per-layer work: fill K/V values, scatter, bmm gather, attention.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        self._attn_4d = _maybe_compile(_attn_4d)

        self._block_size: int | None = None
        self._num_blocks: int | None = None
        self._aligned_num_physical_blocks: int | None = None
        self._block_width: int | None = None
        self._k_cache_dev: torch.Tensor | None = None
        self._v_cache_dev: torch.Tensor | None = None

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported yet")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window not supported yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported yet")

    def _init_device_cache(self, num_blocks: int, block_size: int) -> None:
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._aligned_num_physical_blocks = ((num_blocks + 63) // 64) * 64
        self._block_width = block_size * self.head_size

        self._k_cache_dev = torch.zeros(
            self.num_kv_heads,
            self._aligned_num_physical_blocks,
            self._block_width,
            dtype=self._target_dtype,
            device=self._target_device,
        )
        self._v_cache_dev = torch.zeros(
            self.num_kv_heads,
            self._aligned_num_physical_blocks,
            self._block_width,
            dtype=self._target_dtype,
            device=self._target_device,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output

        assert attn_metadata.num_seqs == 1, (
            f"Spyre attention requires num_seqs=1, got {attn_metadata.num_seqs}"
        )

        # The value still resides on CPU, as it is
        # created in the GraniteAttention through a split
        # operation, which has to be carried out on CPU
        query = convert(query, device="cpu")
        key = convert(key, device="cpu")

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self._k_cache_dev is None:
            self._init_device_cache(kv_cache.shape[0], attn_metadata.block_size)

        # Step 1: Scatter — selectors precomputed by builder, transfer compact K/V, place on Spyre
        self._scatter_to_device_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            attn_metadata.scatter_mask_dev,
            attn_metadata.scatter_row_sel_dev,
            attn_metadata.scatter_col_sel_dev,
            attn_metadata.slot_mapping[:num_actual_tokens],
        )

        # Step 2: Gather — sel_mask is precomputed by builder and lives on Spyre
        compact_k, compact_v = self._gather_from_device_cache(
            attn_metadata.gather_sel_mask_dev,
            attn_metadata.aligned_max_seq_len,
        )

        # Step 3: Add batch dim (num_seqs=1)
        query_per_seq = query[:num_actual_tokens].unsqueeze(0)

        # Step 4: Attention mask is precomputed by builder — use directly
        mask = attn_metadata.attention_mask

        # Step 5: Compute attention
        attn_output = self._compute_attention(
            query_per_seq, compact_k, compact_v, mask, query.device, query.dtype
        )

        # Step 6: Extract actual tokens (num_seqs=1 → simple slice)
        if output.device.type == "spyre":
            # Workaround for torch-spyre bug: a CPU->Spyre `.copy_()` into a
            # destination whose rank differs from its allocation rank (here,
            # vLLM allocates `output` as 2-D and views it as 3-D) crashes in
            # spyre::get_device_stride_infos. Instead, build the full output
            # on CPU, transfer it to a freshly-allocated Spyre tensor (works),
            # and finish with a Spyre->Spyre copy, which dispatches through
            # torch.ops.spyre.copy_from_d2d and avoids the broken path.
            full_cpu = torch.zeros(output.shape, dtype=output.dtype, device="cpu")
            full_cpu[:num_actual_tokens] = attn_output[0]
            full_spyre = convert(full_cpu.contiguous(), "spyre", output.dtype)
            output.copy_(full_spyre)
        else:
            output[:num_actual_tokens].copy_(attn_output[0])
        return output

    def _scatter_to_device_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        scatter_mask_dev: torch.Tensor,
        row_sel_dev: torch.Tensor,
        col_sel_dev: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter: transfer compact K/V to Spyre, place via two bmms against one-hot selectors.

        SPYRE_SCATTER_USE_OVERWRITE=1 selects an alternative path that
        uses spyre.overwrite_f per token — no bmm/transpose, all on
        device. Requires PR #2084 (specialize_int=True) applied to
        torch-spyre or the kernel will reuse the first call's offsets.
        """
        # Compact per-layer transfer: [num_tokens, num_kv_heads, head_size] on Spyre
        k_new_dev = (
            key.to(self._target_dtype).to(self._target_device).contiguous()
        )  # [num_tokens, num_kv_heads, head_size]
        v_new_dev = value.to(self._target_dtype).to(self._target_device).contiguous()

        if envs.SPYRE_SCATTER_USE_OVERWRITE:
            # Per-token spyre.overwrite_f path. For each token t, write
            # k_new_dev[t, :, :] (shape [num_kv_heads, head_size]) into
            # cache[:, block_indices[t], block_offsets[t]*head_size :
            #       block_offsets[t]*head_size + head_size]
            # via overwrite_f with dims=[1, 2], offsets=[br, cs].
            # Input must be [num_kv_heads, 1, head_size] to match the
            # output's rank (3D) with the listed dims being singleton.
            head_size = self.head_size
            block_size = self._block_size
            assert block_size is not None
            sm = slot_mapping.detach().to("cpu")
            block_indices = (sm // block_size).tolist()
            block_offsets = (sm % block_size).tolist()
            num_tokens = k_new_dev.shape[0]

            # Build per-token inputs on CPU and upload as fresh tensors.
            # Slicing k_new_dev[t] produces a Spyre view whose device
            # layout doesn't match what overwrite_f's bundle compiler
            # expects (triggers patch (A)'s narrowed-view guard). A
            # fresh CPU->Spyre upload reshapes cleanly.
            k_cpu = key.to(self._target_dtype).contiguous()
            v_cpu = value.to(self._target_dtype).contiguous()

            k_cache = self._k_cache_dev
            v_cache = self._v_cache_dev
            for t in range(num_tokens):
                br = int(block_indices[t])
                cs = int(block_offsets[t]) * head_size
                k_tok = k_cpu[t].unsqueeze(1).contiguous().to(self._target_device)
                v_tok = v_cpu[t].unsqueeze(1).contiguous().to(self._target_device)
                k_cache = torch.ops.spyre.overwrite_f(
                    input=k_tok, output=k_cache, dims=[1, 2], offsets=[br, cs]
                )
                v_cache = torch.ops.spyre.overwrite_f(
                    input=v_tok, output=v_cache, dims=[1, 2], offsets=[br, cs]
                )
            self._k_cache_dev = k_cache
            self._v_cache_dev = v_cache
            return

        # bmm 1: place each token's head_size-vector into its block-width slot.
        # [num_tokens, num_kv_heads, head_size] @ [num_tokens, head_size, block_width]
        #   -> [num_tokens, num_kv_heads, block_width]
        k_spread = torch.bmm(k_new_dev, col_sel_dev)
        v_spread = torch.bmm(v_new_dev, col_sel_dev)

        # Permute to [num_kv_heads, num_tokens, block_width] for the row-placement bmm
        k_spread = k_spread.transpose(0, 1).contiguous()
        v_spread = v_spread.transpose(0, 1).contiguous()

        # bmm 2: route each token row to its physical block row.
        # [num_kv_heads, aligned_num_physical_blocks, num_tokens]
        #   @ [num_kv_heads, num_tokens, block_width]
        #   -> [num_kv_heads, aligned_num_physical_blocks, block_width]
        k_vals = torch.bmm(row_sel_dev, k_spread)
        v_vals = torch.bmm(row_sel_dev, v_spread)

        self._k_cache_dev = _scatter_cache(self._k_cache_dev, scatter_mask_dev, k_vals)
        self._v_cache_dev = _scatter_cache(self._v_cache_dev, scatter_mask_dev, v_vals)

    def _gather_from_device_cache(
        self,
        sel_mask_dev: torch.Tensor,
        aligned_max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather dense KV using precomputed selection mask."""
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size

        gathered_k = torch.bmm(sel_mask_dev, self._k_cache_dev)
        gathered_v = torch.bmm(sel_mask_dev, self._v_cache_dev)

        gathered_k = gathered_k.reshape(num_kv_heads, aligned_max_seq_len, head_size)
        gathered_v = gathered_v.reshape(num_kv_heads, aligned_max_seq_len, head_size)

        gathered_k = gathered_k.unsqueeze(0)
        gathered_v = gathered_v.unsqueeze(0)

        return gathered_k, gathered_v

    def _compute_attention(
        self,
        query: torch.Tensor,
        compact_k: torch.Tensor,
        compact_v: torch.Tensor,
        mask: torch.Tensor,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_seqs, max_query_len, num_heads, head_size = query.shape
        _, num_kv_heads, kv_len, _ = compact_k.shape
        num_queries_per_kv = self.num_queries_per_kv

        padded_query_len = (
            (max_query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
        )

        if padded_query_len > max_query_len:
            padding_size = padded_query_len - max_query_len
            query = torch.nn.functional.pad(
                query, (0, 0, 0, 0, 0, padding_size), mode="constant", value=0.0
            )

        q = query.transpose(1, 2).contiguous()
        q = q.reshape(num_seqs * num_kv_heads, num_queries_per_kv, padded_query_len, head_size)

        k = compact_k.reshape(num_seqs * num_kv_heads, 1, kv_len, head_size)
        v = compact_v.reshape(num_seqs * num_kv_heads, 1, kv_len, head_size)

        q_spyre = convert(q, self._target_device, self._target_dtype)

        output_spyre = self._attn_4d(q_spyre, k, v, self.scale, mask)

        output_4d = convert(output_spyre, output_device, output_dtype)
        output_reshaped = output_4d.reshape(num_seqs, num_heads, padded_query_len, head_size)
        output = output_reshaped.transpose(1, 2).contiguous()
        return output[:, :max_query_len, :, :]
