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

"""The backend-selection gates ``SpyreAttentionBackend`` answers to vLLM.

``AttentionBackend.validate_configuration`` consults these statics while vLLM
picks a backend for a model config, so a head size that misses fp16 stick
alignment (a multiple of 64) or an unlisted KV cache dtype has to be refused
here, at config time, rather than reach the card and fault. Nothing else in
the suite executes them: the pooling models that ship 32-wide heads are padded
to 64 before construction, so the rejection path is never taken end to end.
"""

import pytest
import torch
from vllm.config import get_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType, MultipleOf

from spyre_inference.v1.attention.backends.spyre_attn import (
    FP8_KV_CACHE_DTYPES,
    SpyreAttentionBackend,
)


@pytest.mark.parametrize("head_size", [64, 128, 192, 1024])
def test_head_sizes_that_fill_a_stick_are_supported(head_size):
    assert SpyreAttentionBackend.supports_head_size(head_size)


@pytest.mark.parametrize("head_size", [1, 32, 63, 100, 8190])
def test_head_sizes_off_stick_alignment_are_rejected(head_size):
    """32 is the width the pooling models ship; padded to 64 before construction,
    and unpadded it has to be refused here, not fault on the card."""
    assert not SpyreAttentionBackend.supports_head_size(head_size)


@pytest.mark.parametrize(
    "kv_cache_dtype", [None, "auto", "float16", "bfloat16", *FP8_KV_CACHE_DTYPES]
)
def test_declared_kv_cache_dtypes_pass_the_gate(kv_cache_dtype):
    assert SpyreAttentionBackend.supports_kv_cache_dtype(kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["int8", "uint8", "fp8_e5m2"])
def test_other_kv_cache_dtypes_are_rejected(kv_cache_dtype):
    assert not SpyreAttentionBackend.supports_kv_cache_dtype(kv_cache_dtype)


def _selection_reasons(**overrides):
    kwargs = dict(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(major=1, minor=0),
        attn_type=AttentionType.DECODER,
    )
    return SpyreAttentionBackend.validate_configuration(**(kwargs | overrides))


def test_an_unpadded_head_size_is_rejected_at_selection_time():
    """The rejection as vLLM runs it: the head size is the only thing at fault."""
    assert _selection_reasons(head_size=32) == ["head_size not supported"]


def test_an_unlisted_kv_cache_dtype_is_rejected_at_selection_time():
    assert _selection_reasons(kv_cache_dtype="int8") == ["kv_cache_dtype not supported"]


def test_a_stick_aligned_config_is_selected_without_complaint():
    assert _selection_reasons() == []


def test_kernel_block_sizes_without_a_config_are_the_stick_multiple(monkeypatch):
    """No live VllmConfig: the 128-byte stick is the only constraint left. The
    session's platform init leaves one set, so the branch needs the patch."""
    import spyre_inference.v1.attention.backends.spyre_attn as spyre_attn

    monkeypatch.setattr(spyre_attn, "get_current_vllm_config_or_none", lambda: None)
    assert SpyreAttentionBackend.get_supported_kernel_block_sizes() == [MultipleOf(64)]


def test_kernel_block_sizes_follow_the_configured_block_size(default_vllm_config):
    """With one, pages and recorded kernels exist for cache_config.block_size only."""
    assert SpyreAttentionBackend.get_supported_kernel_block_sizes() == [
        get_current_vllm_config().cache_config.block_size
    ]
