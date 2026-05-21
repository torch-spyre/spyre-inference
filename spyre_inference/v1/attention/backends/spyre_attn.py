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

"""Paged KV-cache attention backend for Spyre using list-of-pages and online softmax.

This backend implements attention using individual page tensors (one per KV block)
and FlashAttention-style online softmax that iterates over pages. Each page is a
complete tensor [num_kv_heads, block_size, head_size] on the Spyre device, passed
directly to bmm without any slicing.

Required configuration:
  - num_gpu_blocks_override must be set (e.g. 64)
  - max_model_len <= num_gpu_blocks_override * block_size (e.g. 1024 with block_size=16)

Architecture:
  - KV cache: two separate lists (k_pages, v_pages), managed by the model runner.
  - reshape_and_cache: writes new K/V tokens into pages via overwrite_f (all on device).
  - Attention: online softmax iterating over pages — no gather/materialization step.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from spyre_inference import envs

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


# ---------------------------------------------------------------------------
# Spyre workarounds
# ---------------------------------------------------------------------------


def _seq_slice(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Slice a tensor along dim 0 (token dim).

    Workaround for Spyre narrowed-view read bug — on Spyre we round-trip
    through host.
    """
    if t.device.type == "spyre":
        return t.cpu()[lo:hi].contiguous().to(t.device)
    return t[lo:hi]

@dataclass
class SpyreAttentionMetadata(AttentionMetadata):
    """Metadata for paged online-softmax attention on Spyre."""

    num_actual_tokens: int
    num_seqs: int
    max_query_len: int
    max_seq_len: int

    seq_lens: torch.Tensor  # [num_seqs]
    query_start_loc: torch.Tensor  # [num_seqs + 1]

    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    block_size: int

    slot_mapping: torch.Tensor  # [num_actual_tokens]

    # Precomputed from slot_mapping (avoids CPU round-trip during forward)
    slot_block_indices: list[int] | None = None
    slot_block_offsets: list[int] | None = None

    apply_causal_mask: bool = False

    num_kv_heads: int = 0
    num_heads: int = 0

    # Additive attention mask on Spyre:
    # [num_seqs * num_kv_heads, 1, padded_query_len, aligned_max_seq_len]
    attention_mask: torch.Tensor | None = None

    aligned_max_seq_len: int = 0

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


class SpyreAttentionMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionMetadata]):
    """Builds attention metadata — only the attention mask is precomputed."""

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

        try:
            torch.randn(1, device=torch.device("spyre"))
            self._target_device = torch.device("spyre")
        except Exception:
            self._target_device = torch.device("cpu")
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
        """Build additive attention mask on Spyre.

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionMetadata:
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

        # Precompute slot indices on CPU to avoid CPU round-trip during forward
        sm_cpu = slot_mapping.detach().cpu()
        slot_block_indices = (sm_cpu // self.block_size).tolist()
        slot_block_offsets = (sm_cpu % self.block_size).tolist()

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
            slot_block_indices=slot_block_indices,
            slot_block_offsets=slot_block_offsets,
            apply_causal_mask=apply_causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            attention_mask=attention_mask,
            aligned_max_seq_len=aligned_max_seq_len,
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
    """Online-softmax paged attention iterating over KV pages.

    KV cache is a tuple (k_pages, v_pages) where each is a list of tensors
    of shape [num_kv_heads, block_size, head_size] on Spyre. No monolithic
    cache tensor, no gather masks.
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

        self._target_device: torch.device | None = None
        self._target_dtype = torch.float16

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported yet")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window not supported yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported yet")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[list[torch.Tensor], list[torch.Tensor]],
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output

        k_pages, v_pages = kv_cache
        num_actual_tokens = attn_metadata.num_actual_tokens
        block_size = attn_metadata.block_size

        # Derive target device from the pages themselves
        if self._target_device is None:
            self._target_device = k_pages[0].device

        # Step 1: Reshape and cache — write new tokens into pages (on device)
        self._reshape_and_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata.slot_block_indices[:num_actual_tokens],
            attn_metadata.slot_block_offsets[:num_actual_tokens],
        )

        # Step 2: Online softmax attention over pages (varlen)
        # Writes directly into output buffer — avoids torch.cat
        output = self._online_softmax_attention(
            query[:num_actual_tokens],
            k_pages,
            v_pages,
            attn_metadata,
            output,
        )

        return output

    def _reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        block_indices: list[int],
        block_offsets: list[int],
    ) -> None:
        """Write new K/V tokens into their respective pages.

        key, value: [num_tokens, num_kv_heads, head_size] — already on target device
        k_pages, v_pages: list[Tensor], each [num_kv_heads, block_size, head_size]
        block_indices, block_offsets: precomputed from slot_mapping in metadata builder
        """
        num_tokens = key.shape[0]
        use_overwrite_f = envs.SPYRE_USE_OVERWRITE_F

        for t in range(num_tokens):
            block_idx = block_indices[t]
            block_offset = block_offsets[t]

            if use_overwrite_f:
                k_tok = key[t].unsqueeze(1)
                v_tok = value[t].unsqueeze(1)

                k_pages[block_idx] = torch.ops.spyre.overwrite_f(
                    input=k_tok, output=k_pages[block_idx],
                    dims=[1], offsets=[block_offset],
                )
                v_pages[block_idx] = torch.ops.spyre.overwrite_f(
                    input=v_tok, output=v_pages[block_idx],
                    dims=[1], offsets=[block_offset],
                )
            else:
                k_pages[block_idx][:, block_offset, :] = key[t]
                v_pages[block_idx][:, block_offset, :] = value[t]

    def _online_softmax_attention(
        self,
        query: torch.Tensor,
        k_pages: list[torch.Tensor],
        v_pages: list[torch.Tensor],
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """FlashAttention-style online softmax iterating over KV pages (varlen).

        Handles multiple sequences using query_start_loc for the varlen layout.
        Each k_page/v_page is [num_kv_heads, block_size, head_size] — a complete
        tensor on Spyre, passed to bmm directly without slicing.

        Writes results directly into the output buffer (avoids torch.cat).
        Returns the output tensor (may be a new tensor when using overwrite_f).

        query: [num_actual_tokens, num_heads, head_size] — already on target device
        output: [>=num_actual_tokens, num_heads, head_size] — pre-allocated buffer
        """
        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        num_queries_per_kv = self.num_queries_per_kv
        block_size = attn_metadata.block_size

        num_seqs = attn_metadata.num_seqs
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        attention_mask = attn_metadata.attention_mask

        use_overwrite_f = envs.SPYRE_USE_OVERWRITE_F

        for seq_idx in range(num_seqs):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            query_len = q_end - q_start
            kv_len = int(seq_lens[seq_idx].item())

            q_seq = _seq_slice(query, q_start, q_end)

            # Pad query to QUERY_CHUNK_SIZE
            padded_query_len = (
                (query_len + QUERY_CHUNK_SIZE - 1) // QUERY_CHUNK_SIZE * QUERY_CHUNK_SIZE
            )
            if padded_query_len > query_len:
                q_seq = torch.nn.functional.pad(
                    q_seq, (0, 0, 0, 0, 0, padded_query_len - query_len),
                    mode="constant", value=0.0,
                )

            # Reshape: [padded_query_len, num_heads, head_size]
            #   → [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            q = q_seq.unsqueeze(0).transpose(1, 2).contiguous()
            q = q.reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)

            # Per-sequence mask slice:
            # attention_mask is [num_seqs * num_kv_heads, 1, padded_max_query_len, aligned_max_seq_len]
            mask_start = seq_idx * num_kv_heads
            mask_end = mask_start + num_kv_heads
            seq_mask = attention_mask[mask_start:mask_end, :, :padded_query_len, :]

            num_blocks_needed = (kv_len + block_size - 1) // block_size

            tile_max: torch.Tensor | None = None
            tile_sum: torch.Tensor | None = None
            tile_output: torch.Tensor | None = None

            for i in range(num_blocks_needed):
                page_idx = int(block_table[seq_idx, i])
                k_page = k_pages[page_idx]
                v_page = v_pages[page_idx]

                # Expand for GQA: [num_kv_heads, 1, block_size, head_size]
                k_page_4d = k_page.unsqueeze(1)
                v_page_4d = v_page.unsqueeze(1)

                # Q @ K^T: [num_kv_heads, num_queries_per_kv, padded_query_len, block_size]
                scores = torch.matmul(q, k_page_4d.transpose(-2, -1)) * self.scale

                # Apply per-tile mask slice
                mask_tile = seq_mask[:, :, :, i * block_size:(i + 1) * block_size]
                scores = scores + mask_tile

                scores_max = scores.max(dim=-1, keepdim=True)[0]

                if i == 0:
                    tile_max = scores_max
                    tile_probs = torch.exp(scores - tile_max)
                    tile_output = torch.matmul(tile_probs, v_page_4d)
                    tile_sum = tile_probs.sum(dim=-1, keepdim=True)
                else:
                    old_max = tile_max
                    tile_max = torch.maximum(tile_max, scores_max)
                    rescale = torch.exp(old_max - tile_max)
                    tile_output = tile_output * rescale
                    tile_sum = tile_sum * rescale
                    tile_probs = torch.exp(scores - tile_max)
                    tile_output = tile_output + torch.matmul(tile_probs, v_page_4d)
                    tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)

            # Normalize
            assert tile_output is not None and tile_sum is not None
            result = tile_output / tile_sum

            # Reshape back: [num_kv_heads, num_queries_per_kv, padded_query_len, head_size]
            #   → [query_len, num_heads, head_size]
            result = result.reshape(1, num_heads, padded_query_len, head_size)
            result = result.transpose(1, 2).contiguous()
            seq_result = result[0, :query_len, :, :]

            # Write into output buffer without torch.cat
            if use_overwrite_f:
                output = torch.ops.spyre.overwrite_f(
                    input=seq_result, output=output,
                    dims=[0], offsets=[q_start],
                )
            else:
                output[q_start:q_end].copy_(seq_result)

        return output
