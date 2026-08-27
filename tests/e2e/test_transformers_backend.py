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

Stands in for upstream's ``tests/models/transformers/test_backend.py``, which is
disabled in ``upstream_tests.yaml``: it compares against an HF CPU reference over 32
tokens of logprobs, which fp16 on Spyre is unlikely to satisfy. The native Spyre path
is the reference here instead.
"""

from __future__ import annotations

import json

import pytest
import torch


def test_rope_frequencies_rebuilt_at_the_pre_pad_head_dim():
    """HF derives inv_freq from the widened head_dim, so the rebuild has to undo it."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from spyre_inference.transformers_backend import _rope_at_original_head_dim

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

    rebuilt = _rope_at_original_head_dim(cfg, padded_rope, orig)

    assert torch.equal(rebuilt.inv_freq, expected)
    assert cfg.head_dim == padded, "the padded width must be restored for the model"


def test_padded_qk_logits_match_the_unpadded_reference():
    """Weight padding + rebuilt rotation + 1/sqrt(orig) scale must leave the logits
    unchanged versus stock HF at the native head_dim."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from spyre_inference.custom_ops.head_pad import _pad_weight
    from spyre_inference.transformers_backend import (
        _rope_at_original_head_dim,
        _spyre_apply_rotary,
        _SpyreRotaryEmbedding,
    )

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

    spyre_rope = _SpyreRotaryEmbedding(
        _rope_at_original_head_dim(cfg, hf_rope, orig),
        cfg.max_position_embeddings,
        padded,
        torch.float32,
    )
    rotation, _ = spyre_rope(x, position_ids)

    q_rot, k_rot = _spyre_apply_rotary(q_pad, k_pad, rotation)
    logits_pad = (q_rot @ k_rot.transpose(-1, -2)) * orig**-0.5

    torch.testing.assert_close(logits_pad, logits_ref, rtol=1e-5, atol=1e-5)

    half, padded_half = orig // 2, padded // 2
    assert torch.allclose(q_rot[..., :half], q_ref[..., :half], atol=1e-6)
    assert torch.allclose(
        q_rot[..., padded_half : padded_half + half], q_ref[..., half:], atol=1e-6
    )
    assert not q_rot[..., half:padded_half].any()
    assert not q_rot[..., padded_half + half :].any()


def _gemma3_text_config():
    """A Gemma 3 text config with both attention types, so its rope is layer-typed and
    the two types rotate at different thetas."""
    from transformers import Gemma3TextConfig

    return Gemma3TextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=4,
        intermediate_size=64,
        vocab_size=100,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
    )


def test_layer_typed_rope_rotates_each_layer_type_at_its_own_frequencies():
    """Layer-typed ropes key their frequencies on a third ``layer_type`` argument, so the
    replacement needs one cache per type; one cache rotates sliding layers globally."""
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from spyre_inference.transformers_backend import (
        _rope_frequencies,
        _spyre_apply_rotary,
        _SpyreRotaryEmbedding,
    )

    cfg = _gemma3_text_config()
    hf_rope = Gemma3RotaryEmbedding(cfg)
    assert sorted(_rope_frequencies(hf_rope)) == ["full_attention", "sliding_attention"]

    torch.manual_seed(0)
    seq = 5
    x = torch.randn(2, seq, cfg.hidden_size)
    position_ids = torch.arange(seq).expand(2, seq)
    q = torch.randn(2, cfg.num_attention_heads, seq, cfg.head_dim)
    k = torch.randn(2, cfg.num_key_value_heads, seq, cfg.head_dim)

    spyre_rope = _SpyreRotaryEmbedding(hf_rope, cfg.max_position_embeddings, None, torch.float32)

    for layer_type in ("full_attention", "sliding_attention"):
        cos, sin = hf_rope(x, position_ids, layer_type)
        q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)

        # Third argument positional, the way Gemma3TextModel.forward passes it.
        rotation, second = spyre_rope(x, position_ids, layer_type)
        assert second is None
        q_rot, k_rot = _spyre_apply_rotary(q, k, rotation)

        torch.testing.assert_close(q_rot, q_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_rot, k_ref, rtol=1e-5, atol=1e-5)

    assert not torch.allclose(
        spyre_rope._caches["full_attention"], spyre_rope._caches["sliding_attention"]
    ), "one cache serving both layer types is the bug this guards"


def test_patched_apply_rotary_leaves_stock_hf_callers_working():
    """The patch is never lifted from sys.modules, so an HF model built later in the same
    process — a CPU reference next to the vLLM one — has to keep getting HF's rotation."""
    from transformers import LlamaConfig
    from transformers.models.llama import modeling_llama

    from spyre_inference.transformers_backend import (
        _rope_dispatch,
        _spyre_apply_rotary,
        _SpyreRotaryEmbedding,
    )

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=100,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = modeling_llama.LlamaModel(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        reference = model(input_ids=input_ids).last_hidden_state.clone()

    original = modeling_llama.apply_rotary_pos_emb
    patched = _rope_dispatch(original)
    assert patched._spyre_patched, "the marker stops _patch_rope wrapping twice"

    modeling_llama.apply_rotary_pos_emb = patched
    try:
        with torch.no_grad():
            after = model(input_ids=input_ids).last_hidden_state
        torch.testing.assert_close(after, reference, rtol=0, atol=0)

        q = torch.randn(1, 4, 6, cfg.head_dim)
        k = torch.randn(1, 4, 6, cfg.head_dim)
        spyre_rope = _SpyreRotaryEmbedding(
            model.rotary_emb, cfg.max_position_embeddings, None, torch.float32
        )
        rotation, second = spyre_rope(q, torch.arange(6).unsqueeze(0))
        expected = _spyre_apply_rotary(q, k, rotation)
        for got, want in zip(patched(q, k, rotation, second), expected):
            assert torch.equal(got, want), "a Spyre rotation must still take the matmul path"
    finally:
        modeling_llama.apply_rotary_pos_emb = original


PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# The two paths are not bit-identical: they run different module code (HF's vs vLLM's)
# and round their rotation caches differently, so greedy sampling eventually tie-breaks
# apart. The failure mode being guarded against diverges from the first token or two.
MAX_TOKENS = 8


def _generate_greedy(model: str, model_impl: str) -> list[list[int]]:
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        model_impl=model_impl,
    )
    assert llm.llm_engine.model_config.using_transformers_backend() == (
        model_impl == "transformers"
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]

    del llm
    cleanup_dist_env_and_memory()
    return token_ids


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        # head_dim=64 -> padded; micro-g3.3 is 128 -> unpadded. Covers both branches.
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_matches_native(model: str) -> None:
    """The Transformers backend must generate what the native Spyre path does.

    Content, not just non-empty output: a broken RoPE or a norm falling back to an
    unsupported fp32 promotion still yields fluent text, just unrelated to the prompt.
    """
    transformers_ids = _generate_greedy(model, "transformers")
    native_ids = _generate_greedy(model, "vllm")

    assert transformers_ids == native_ids


def _model_repo(path, *, mistral_format: bool) -> str:
    """A local llama repo; with *mistral_format*, also the ``params.json`` and
    ``consolidated*.safetensors`` that make ``config_format="auto"`` pick Mistral."""
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
        vocab_size=1000,
        head_dim=64,
        max_position_embeddings=128,
    )
    (path / "config.json").write_text(json.dumps(hf_config.to_dict()))

    if mistral_format:
        (path / "params.json").write_text(
            json.dumps(
                {
                    "dim": 256,
                    "n_layers": 2,
                    "n_heads": 4,
                    "n_kv_heads": 4,
                    "hidden_dim": 512,
                    "head_dim": 64,
                    "vocab_size": 1000,
                    "norm_eps": 1e-5,
                    "rope_theta": 10000.0,
                    "max_position_embeddings": 128,
                    "dtype": "float16",
                }
            )
        )
        # is_mistral_model_repo() only checks the filename, never the contents.
        (path / "consolidated.safetensors").write_bytes(b"")

    return str(path)


def _vllm_config(model: str):
    from vllm.config import LoadConfig, ModelConfig, VllmConfig

    model_config = ModelConfig(
        model=model, trust_remote_code=False, dtype="float16", seed=0, max_model_len=128
    )
    return VllmConfig(model_config=model_config, load_config=LoadConfig())


def test_mistral_format_repo_parses_to_a_config_hf_cannot_build_from(tmp_path):
    """The premise of _fix_generic_config: for a repo carrying both params.json and
    config.json, vLLM checks Mistral first and ends at a bare PretrainedConfig."""
    from transformers import AutoModel
    from transformers.configuration_utils import PretrainedConfig

    from vllm.transformers_utils.config import get_config

    hf_config = get_config(_model_repo(tmp_path, mistral_format=True), trust_remote_code=False)

    assert type(hf_config) is PretrainedConfig
    assert hf_config.model_type == "transformer"
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModel.from_config(hf_config)


def test_fix_generic_config_re_resolves_and_forces_hf_weights(tmp_path):
    from transformers import AutoModel, LlamaConfig

    from spyre_inference.transformers_backend import SpyreTransformersForCausalLM

    vllm_config = _vllm_config(_model_repo(tmp_path, mistral_format=True))
    assert vllm_config.load_config.load_format == "auto"

    SpyreTransformersForCausalLM._fix_generic_config(vllm_config)

    resolved = vllm_config.model_config.hf_config
    assert type(resolved) is LlamaConfig
    assert vllm_config.model_config.hf_text_config is resolved
    # The re-resolved config only describes the HF-format weights, so the load format
    # has to follow it.
    assert vllm_config.load_config.load_format == "hf"

    # Fields the Mistral parser and the platform set must carry over; head_dim in
    # particular sizes the KV cache.
    assert resolved.vocab_size == 1000
    assert resolved.head_dim == 128
    assert resolved._spyre_orig_head_dim == 64

    assert type(AutoModel.from_config(resolved)).__name__ == "LlamaModel"


def test_fix_generic_config_leaves_an_hf_format_repo_alone(tmp_path):
    from transformers import LlamaConfig

    from spyre_inference.transformers_backend import SpyreTransformersForCausalLM

    vllm_config = _vllm_config(_model_repo(tmp_path, mistral_format=False))
    assert type(vllm_config.model_config.hf_config) is LlamaConfig

    before = vllm_config.model_config.hf_config
    SpyreTransformersForCausalLM._fix_generic_config(vllm_config)

    assert vllm_config.model_config.hf_config is before
    assert vllm_config.load_config.load_format == "auto"
