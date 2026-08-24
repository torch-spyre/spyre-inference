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

"""Tests for the HuggingFace Transformers backend (model_impl='transformers').

TODO: Delete this file once https://github.com/torch-spyre/spyre-inference/issues/324
is resolved and re-enable the upstream tests in upstream_tests.yaml.
"""

from __future__ import annotations

import pytest
import torch


def test_rope_frequencies_rebuilt_at_the_pre_pad_head_dim():
    """HF derives inv_freq from the widened head_dim, so the rebuild has to undo it."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from spyre_inference.hf_adapters import _rope_at_original_head_dim

    orig, padded = 4, 128
    cfg = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        head_dim=orig,
        max_position_embeddings=256,
    )
    expected = LlamaRotaryEmbedding(config=cfg).inv_freq.clone()
    assert expected.shape == (orig // 2,)

    cfg.head_dim = padded
    padded_rope = LlamaRotaryEmbedding(config=cfg)
    # What HF built off the padded config: too many frequencies, wrong spacing.
    assert padded_rope.inv_freq.shape == (padded // 2,)
    assert not torch.equal(padded_rope.inv_freq[: orig // 2], expected)

    shim = _rope_at_original_head_dim(cfg, padded_rope, orig)

    assert torch.equal(shim.inv_freq, expected)
    assert cfg.head_dim == padded, "the padded width must be restored for the model"


def test_padded_qk_logits_match_the_unpadded_reference():
    """Weight padding + rebuilt rotation + 1/sqrt(orig) scale must leave the logits
    unchanged versus stock HF at the native head_dim."""
    from hf_adapters.hf_common import PrecomputedRotaryEmbedding, apply_rope_matmul
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from spyre_inference.custom_ops.head_pad import _pad_weight
    from spyre_inference.hf_adapters import _rope_at_original_head_dim

    orig, padded = 4, 128
    n_heads, hidden, seq = 4, 16, 6
    torch.manual_seed(0)

    cfg = LlamaConfig(
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        num_hidden_layers=1,
        head_dim=orig,
        max_position_embeddings=64,
    )
    x = torch.randn(1, seq, hidden)
    position_ids = torch.arange(seq).unsqueeze(0)
    q_w, k_w = torch.randn(n_heads * orig, hidden), torch.randn(n_heads * orig, hidden)

    def heads(inputs, weight, head_dim):
        # [B, L, hidden] -> [B, H, L, head_dim], the layout RoPE and attention use.
        return (inputs @ weight.T).view(1, seq, n_heads, head_dim).transpose(1, 2)

    hf_rope = LlamaRotaryEmbedding(config=cfg)
    cos, sin = hf_rope(x, position_ids)
    q_ref, k_ref = apply_rotary_pos_emb(heads(x, q_w, orig), heads(x, k_w, orig), cos, sin)
    logits_ref = (q_ref @ k_ref.transpose(-1, -2)) * orig**-0.5

    cfg.head_dim = padded
    q_pad = heads(x, _pad_weight("q_proj.weight", q_w, n_heads, n_heads, orig, padded), padded)
    k_pad = heads(x, _pad_weight("k_proj.weight", k_w, n_heads, n_heads, orig, padded), padded)

    spyre_rope = PrecomputedRotaryEmbedding(
        _rope_at_original_head_dim(cfg, hf_rope, orig), padded_head_dim=padded
    )
    # Drive the cache directly: forward() ships the result to the Spyre device.
    spyre_rope.set_dtype(torch.float32)
    spyre_rope._extend_cache(seq)
    rotation = spyre_rope._freq_cache[position_ids]

    q_rot, k_rot = apply_rope_matmul(q_pad, rotation), apply_rope_matmul(k_pad, rotation)
    logits_pad = (q_rot @ k_rot.transpose(-1, -2)) * orig**-0.5

    torch.testing.assert_close(logits_pad, logits_ref, rtol=1e-5, atol=1e-5)

    half, padded_half = orig // 2, padded // 2
    assert torch.allclose(q_rot[..., :half], q_ref[..., :half], atol=1e-6)
    assert torch.allclose(
        q_rot[..., padded_half : padded_half + half], q_ref[..., half:], atol=1e-6
    )
    assert not q_rot[..., half:padded_half].any()
    assert not q_rot[..., padded_half + half :].any()


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        # head_dim=64 -> padded; micro-g3.3 is 128 -> unpadded. Covers both branches.
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_generate(model: str) -> None:
    """Verify model_impl='transformers' loads and generates non-empty output."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        model_impl="transformers",
    )
    model_config = llm.llm_engine.model_config
    assert model_config.using_transformers_backend()

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    outputs = llm.generate(["Hello, world!"], sp)
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
