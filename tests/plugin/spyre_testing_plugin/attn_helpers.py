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

# These live in the installed plugin, not a tests/ module, so consumers import them
# by absolute name regardless of rootdir: `tests` is a PEP 420 namespace that
# resolves only while the repo root is on sys.path, which the two-tree rootdir the
# upstream job builds breaks.

from unittest.mock import Mock

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionMetadataBuilder,
)


def _fused_qkv_kv_views(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V as the backend receives them: strided last-dim views of a fused QKV
    on ``device``, which contiguous k/v would not exercise."""
    num_tokens = query.shape[0]
    slabs = [t.reshape(num_tokens, -1) for t in (query, key, value)]
    qkv = convert(torch.cat(slabs, dim=-1), device)
    _, k_view, v_view = qkv.split([s.shape[-1] for s in slabs], dim=-1)
    return (
        k_view.view(num_tokens, key.shape[1], key.shape[2]),
        v_view.view(num_tokens, value.shape[1], value.shape[2]),
    )


def _build_metadata(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    sliding_window: int | None = None,
    model_num_kv_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
):
    """Use the real SpyreAttentionMetadataBuilder to construct metadata.

    ``model_num_kv_heads`` is what ``model_config.get_num_kv_heads()`` reports, for
    the per-layer-head-count models where that differs from the spec's; it defaults
    to agreeing with ``num_kv_heads``.
    """
    from vllm.config import get_current_vllm_config

    # Reuse the VllmConfig set up by the `default_vllm_config` fixture and
    # stub the head-count methods the builder reads.
    vllm_config = get_current_vllm_config()
    vllm_config.model_config.get_num_attention_heads = Mock(return_value=num_query_heads)
    vllm_config.model_config.get_num_kv_heads = Mock(
        return_value=num_kv_heads if model_num_kv_heads is None else model_num_kv_heads
    )
    # The builder asserts these agree, and derives its padding buckets from the
    # cache_config one, so a test block_size has to be set in both places.
    vllm_config.cache_config.block_size = block_size

    if sliding_window is not None:
        kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=dtype,
            sliding_window=sliding_window,
        )
    else:
        kv_cache_spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )

    builder = SpyreAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layers.0.self_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    query_lens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
    max_query_len = int(query_lens_per_seq.max().item())
    max_seq_len = int(seq_lens.max().item())
    num_actual_tokens = int(query_start_loc[-1].item())

    common_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=len(seq_lens),
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        is_prefilling=(query_lens_per_seq > 1),
    )

    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )


def assert_close_outliers(
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_outliers: int = 0,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    *,
    outlier_atol: float | None = None,
    outlier_rtol: float | None = None,
) -> None:
    """Assert tensors are close, allowing up to *max_outliers* elements to exceed tolerance.

    Arguments beyond *max_outliers* are forwarded to ``torch.testing.assert_close``.

    Args:
        actual: tensor under test.
        expected: reference tensor.
        max_outliers: number of elements that may exceed the base tolerances.
        atol: absolute tolerance for the bulk of elements.
        rtol: relative tolerance for the bulk of elements.
        outlier_atol: absolute tolerance for outlier elements (defaults to *atol*,
            meaning outliers only need to be finite, not within any tighter bound).
        outlier_rtol: relative tolerance for outlier elements.
        msg: additional context for the failure message.
    """
    # `NaN > tol` is False, so a non-finite actual scores zero outliers and passes
    # the check below. Attention output is always finite, so reject it up front.
    n_nonfinite = int((~torch.isfinite(actual)).sum())
    if n_nonfinite:
        raise AssertionError(
            f"{n_nonfinite}/{actual.numel()} element(s) of actual are non-finite "
            f"(NaN or inf); the value was never written or the kernel diverged."
        )

    diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    outlier_mask = diff > tol
    n_outliers = outlier_mask.sum().item()

    if n_outliers <= max_outliers and max_outliers > 0:
        # Check that outliers are still within the relaxed bound (or simply finite)
        if outlier_atol is not None or outlier_rtol is not None:
            outlier_tol = (outlier_atol if outlier_atol is not None else atol) + (
                outlier_rtol if outlier_rtol is not None else rtol
            ) * expected.abs()
            if diff[outlier_mask].gt(outlier_tol[outlier_mask]).any():
                worst = diff[outlier_mask].max().item()
                raise AssertionError(
                    f"{n_outliers} outlier(s) exceed base tolerances, "
                    f"and at least one outlier also exceeds the relaxed bound "
                    f"(worst diff={worst:.4g})."
                )
        if n_outliers > 0:
            print(
                f"  [assert_close_outliers] {n_outliers}/{actual.numel()} element(s) "
                f"exceed base tolerance but remain within relaxed bound — acceptable."
            )
        return  # acceptable number of outliers within relaxed bounds

    # Fall through to standard assert_close for a clear error message
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        prefix = (
            f"{n_outliers} elements exceed atol={atol}, rtol={rtol}. "
            if n_outliers > max_outliers
            else ""
        )
        raise AssertionError(
            f"{prefix}"
            f"max_outliers={max_outliers} was specified "
            f"but {n_outliers} element(s) exceed tolerance.\n"
            f"{e}"
        ) from e


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    block_size: int,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    alibi_slopes: list[float] | None = None,
) -> torch.Tensor:
    """Reference implementation of attention for validation."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        # Pages are token-major, so dim 0 of the concat is the token axis.
        k_blocks = [key_cache[idx] for idx in block_indices]
        v_blocks = [value_cache[idx] for idx in block_indices]
        k = torch.cat(k_blocks, dim=0)[:kv_len]  # [kv_len, num_kv_heads, head_size]
        v = torch.cat(v_blocks, dim=0)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1)
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if alibi_slopes is not None:
            # bias[h, q, k] = slope[h] * (k_abs_pos - q_abs_pos), applied before mask.
            # Under strict causal decoding the q_abs_pos term cancels through
            # softmax, so any per-row-constant simplification is equivalent —
            # keep the full form here for clarity in the reference.
            slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
            context_len = kv_len - query_len
            q_abs = torch.arange(query_len, dtype=torch.float32) + context_len
            kv_abs = torch.arange(kv_len, dtype=torch.float32)
            rel = kv_abs.unsqueeze(0) - q_abs.unsqueeze(1)  # [query_len, kv_len]
            bias = slopes.view(-1, 1, 1) * rel.unsqueeze(0)  # [num_heads, q, k]
            attn = attn + bias
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _padded_mask_metadata(
    seq_lens: list[tuple[int, int]],
    block_size: int = 64,
    sliding_window: int | None = None,
    num_query_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    max_num_blocks: int | None = None,
):
    """Build metadata on CPU for a list of (query_len, kv_len) sequences.

    ``max_num_blocks`` is the block-table width build() pads onto; it defaults
    to no headroom, so a test wanting padding to actually happen must pass a
    wider table, as a real engine's is.
    """
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [kv for _, kv in seq_lens]
    num_seqs = len(seq_lens)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    if max_num_blocks is None:
        max_num_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_table = torch.arange(num_seqs * max_num_blocks, dtype=torch.int32).reshape(
        num_seqs, max_num_blocks
    )
    slot_mapping = torch.arange(sum(query_lens), dtype=torch.int64)

    return _build_metadata(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        seq_lens=kv_lens_tensor,
        query_start_loc=cu_query_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        sliding_window=sliding_window,
    )
