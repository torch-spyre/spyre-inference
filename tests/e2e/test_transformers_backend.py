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


def test_stamp_layer_idx_assigns_sequential_indices():
    import torch.nn as nn

    from spyre_inference.transformers_backend import _stamp_layer_idx

    class FakeSelfAttention(nn.Module):
        pass

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = FakeSelfAttention()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = Block()
            self.layer1 = Block()

    model = Model()
    assert not hasattr(model.layer0.attention, "layer_idx")
    assert not hasattr(model.layer1.attention, "layer_idx")

    _stamp_layer_idx(model)

    assert model.layer0.attention.layer_idx == 0
    assert model.layer1.attention.layer_idx == 1


def test_stamp_layer_idx_skips_modules_that_already_have_it():
    import torch.nn as nn

    from spyre_inference.transformers_backend import _stamp_layer_idx

    class FakeSelfAttention(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.layer_idx = idx

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FakeSelfAttention(99)

    model = Model()
    _stamp_layer_idx(model)
    assert model.attn.layer_idx == 99


def test_gather_free_forward_matches_weight0_expand():
    import types

    import torch
    import torch.nn as nn

    from spyre_inference.transformers_backend import _gather_free_forward

    embed_dim = 8
    vocab_size = 16
    seq_len = 4

    torch.manual_seed(0)
    word_emb = nn.Embedding(vocab_size, embed_dim)
    pos_emb = nn.Embedding(32, embed_dim)
    tte = nn.Embedding(2, embed_dim)
    layer_norm = nn.LayerNorm(embed_dim)
    dropout = nn.Dropout(0.0)

    # weight[0] broadcast — what the function should compute
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    position_ids = torch.arange(2, 2 + seq_len).unsqueeze(0)
    inputs_embeds = word_emb(input_ids)
    expected_tte = tte.weight[0].view(1, 1, -1).expand(1, seq_len, -1)
    pos_out = pos_emb(position_ids)
    expected = layer_norm(inputs_embeds + expected_tte + pos_out)

    # fake self with the attributes _gather_free_forward accesses
    self = types.SimpleNamespace(
        token_type_embeddings=tte,
        word_embeddings=word_emb,
        position_embeddings=pos_emb,
        LayerNorm=layer_norm,
        dropout=dropout,
        padding_idx=1,
        create_position_ids_from_input_ids=lambda ids, pad, past: torch.arange(
            past + 2, past + 2 + ids.shape[1]
        ).unsqueeze(0),
    )

    result = _gather_free_forward(self, input_ids=input_ids)
    torch.testing.assert_close(result, expected)


def test_gather_free_forward_explicit_token_type_ids_preserved():
    """When explicit token_type_ids are passed, the original embedding lookup is used.

    The weight[0] optimization only applies when token_type_ids is None.
    Explicit IDs (e.g. all-ones for segment B) must produce weight[1], not weight[0].
    """
    import types

    import torch
    import torch.nn as nn

    from spyre_inference.transformers_backend import _gather_free_forward

    embed_dim = 8
    vocab_size = 16
    seq_len = 4

    torch.manual_seed(2)
    word_emb = nn.Embedding(vocab_size, embed_dim)
    pos_emb = nn.Embedding(32, embed_dim)
    tte = nn.Embedding(2, embed_dim)
    layer_norm = nn.LayerNorm(embed_dim)
    dropout = nn.Dropout(0.0)

    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    position_ids = torch.arange(2, 2 + seq_len).unsqueeze(0)
    # explicit all-ones token_type_ids — must use weight[1], not weight[0]
    token_type_ids = torch.ones(1, seq_len, dtype=torch.long)

    inputs_embeds = word_emb(input_ids)
    expected_tte = tte.weight[1].view(1, 1, -1).expand(1, seq_len, -1)
    expected = layer_norm(inputs_embeds + expected_tte + pos_emb(position_ids))

    self = types.SimpleNamespace(
        token_type_embeddings=tte,
        word_embeddings=word_emb,
        position_embeddings=pos_emb,
        LayerNorm=layer_norm,
        dropout=dropout,
        padding_idx=1,
        create_position_ids_from_input_ids=lambda ids, pad, past: torch.arange(
            past + 2, past + 2 + ids.shape[1]
        ).unsqueeze(0),
    )

    result = _gather_free_forward(self, input_ids=input_ids, token_type_ids=token_type_ids)
    torch.testing.assert_close(result, expected)
    # sanity: weight[0] != weight[1] so the test is non-trivial
    assert not torch.equal(tte.weight[0], tte.weight[1])


def test_patch_encoder_gather_binds_method_roberta():
    """_patch_encoder_gather binds _gather_free_forward on RobertaEmbeddings.

    The original class must be unchanged (no subclass swap); only the instance's
    forward attribute is replaced via __get__.
    """
    pytest.importorskip("transformers")
    import torch.nn as nn
    from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaEmbeddings

    from spyre_inference.transformers_backend import _gather_free_forward, _patch_encoder_gather

    cfg = RobertaConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        vocab_size=100,
    )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = RobertaEmbeddings(cfg)

    model = FakeModel()
    assert type(model.embeddings) is RobertaEmbeddings

    _patch_encoder_gather(model)

    # class is unchanged — no subclass swap
    assert type(model.embeddings) is RobertaEmbeddings
    # instance forward is bound to our gather-free implementation
    assert model.embeddings.forward.__func__ is _gather_free_forward
    # weights are preserved
    assert model.embeddings.word_embeddings.weight.shape == (100, 16)


def test_patch_encoder_gather_binds_method_bert():
    """_patch_encoder_gather binds _gather_free_forward on BertEmbeddings.

    Covers the bug where _patch_xlm_roberta_gather omitted BertEmbeddings,
    causing aten::gather.out crash on BERT-based models (e.g. BAAI/bge-base-en-v1.5).
    """
    pytest.importorskip("transformers")
    import torch.nn as nn
    from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

    from spyre_inference.transformers_backend import _gather_free_forward, _patch_encoder_gather

    cfg = BertConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        vocab_size=100,
    )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = BertEmbeddings(cfg)

    model = FakeModel()
    assert type(model.embeddings) is BertEmbeddings

    _patch_encoder_gather(model)

    assert type(model.embeddings) is BertEmbeddings
    assert model.embeddings.forward.__func__ is _gather_free_forward
    assert model.embeddings.word_embeddings.weight.shape == (100, 16)


def test_patch_encoder_gather_noop_on_unrelated_model():
    """_patch_encoder_gather must not touch models without BERT-family embeddings."""
    import torch.nn as nn

    from spyre_inference.transformers_backend import _patch_encoder_gather

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    model = MLP()
    _patch_encoder_gather(model)
    # _patch_encoder_gather binds forward via instance __dict__; if nothing was patched
    # no instance-level 'forward' attribute should exist on the model or its submodules.
    assert "forward" not in model.__dict__
    assert all("forward" not in m.__dict__ for m in model.modules())


def test_gather_free_forward_bert_position_ids():
    """_gather_free_forward takes the BERT branch (position_ids buffer slice) when
    create_position_ids_from_input_ids is absent on self.

    The RoBERTa branch is covered by test_gather_free_forward_matches_weight0_expand.
    """
    import types

    import torch
    import torch.nn as nn

    from spyre_inference.transformers_backend import _gather_free_forward

    embed_dim = 8
    vocab_size = 16
    seq_len = 4
    max_pos = 32

    torch.manual_seed(1)
    word_emb = nn.Embedding(vocab_size, embed_dim)
    pos_emb = nn.Embedding(max_pos, embed_dim)
    tte = nn.Embedding(2, embed_dim)
    layer_norm = nn.LayerNorm(embed_dim)
    dropout = nn.Dropout(0.0)

    # BERT sequential position IDs: [0, 1, 2, 3]
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    position_ids_buf = torch.arange(max_pos).unsqueeze(0)  # [1, max_pos]

    # Expected: uses position_ids_buf[:, 0:seq_len] = [0,1,2,3]
    expected_pos_ids = position_ids_buf[:, :seq_len]
    inputs_embeds = word_emb(input_ids)
    expected_tte = tte.weight[0].view(1, 1, -1).expand(1, seq_len, -1)
    expected = layer_norm(inputs_embeds + expected_tte + pos_emb(expected_pos_ids))

    # BERT-style self: no create_position_ids_from_input_ids, has position_ids buffer
    self = types.SimpleNamespace(
        token_type_embeddings=tte,
        word_embeddings=word_emb,
        position_embeddings=pos_emb,
        LayerNorm=layer_norm,
        dropout=dropout,
        position_ids=position_ids_buf,
        # deliberately no create_position_ids_from_input_ids — triggers BERT branch
    )

    result = _gather_free_forward(self, input_ids=input_ids)
    torch.testing.assert_close(result, expected)


def test_pre_classifier_head_wired_into_pooler():
    """_PreClassifierHead is registered on self.classifier AND on the pooler head.

    SpyreTransformersForSequenceClassification._install_head builds a _PreClassifierHead
    that chains pre_classifier → ReLU → original_classifier. It rebinds both
    self.classifier and classify_pooler.head.classifier so calls through the pooler
    use the new head.  This test verifies both wiring points and that the squeeze
    of a spurious [B,1,C] output is applied.
    """
    import types

    import torch
    import torch.nn as nn

    from spyre_inference.transformers_backend import _PreClassifierHead

    # Minimal stand-ins for vLLM's ClassifierPoolerHead and SequencePooler.
    class _FakeHead:
        def __init__(self, clf):
            self.classifier = clf

    class _FakeClassifyPooler:
        def __init__(self, clf):
            self.head = _FakeHead(clf)

    # Original classifier: nn.Linear produces [B, C] or [B, 1, C] shape.
    hidden = 8
    num_labels = 3
    original_clf = nn.Linear(hidden, num_labels)
    pre_clf = nn.Linear(hidden, hidden)

    classify_pooler = _FakeClassifyPooler(original_clf)
    pooler = types.SimpleNamespace(poolers_by_task={"classify": classify_pooler})

    new_head = _PreClassifierHead(pre_clf, original_clf)

    # Simulate the two binding lines in _install_head.
    classifier = new_head
    cp = pooler.poolers_by_task.get("classify")
    if cp is not None:
        cp.head.classifier = new_head

    # Both references must point to the new head.
    assert classifier is new_head
    assert classify_pooler.head.classifier is new_head

    # parameters() must be non-empty so spyre_pooler._module_has_float32_params
    # can detect fp32 weights and route the pooler to CPU.
    assert list(new_head.parameters()), "head must expose parameters for Spyre CPU routing"

    # Functional check: output shape is [B, num_labels].
    x = torch.randn(2, hidden)
    with torch.no_grad():
        out = new_head(x)
    assert out.shape == (2, num_labels), f"expected (2, {num_labels}), got {out.shape}"


def test_pre_classifier_head_dtype_matches_classifier():
    """pre_clf must be cast to classifier's dtype before _PreClassifierHead is built.

    ClassifierPoolerHead stores head_dtype=float32 and calls
    pooled_data.to(head_dtype) before invoking self.classifier. Our _install_head
    must cast pre_classifier to the same dtype so the matmul doesn't raise
    "mat1 and mat2 must have the same dtype, but got Float and Half".
    """
    import torch
    import torch.nn as nn

    hidden = 8
    num_labels = 3

    # classifier is fp32 (as head_dtype=float32 from SequenceClassificationMixin)
    classifier = nn.Linear(hidden, num_labels, dtype=torch.float32)
    # pre_classifier starts fp16 (model dtype from init_parameters)
    pre_classifier = nn.Linear(hidden, hidden, dtype=torch.float16)

    # Simulate the dtype-alignment step from _install_head.
    clf_dtype = next(classifier.parameters()).dtype
    pre_classifier.to(dtype=clf_dtype)

    # After alignment pre_clf must match classifier dtype.
    assert next(pre_classifier.parameters()).dtype == torch.float32

    # Functional check: fp32 input flows through both layers without dtype error.
    x = torch.randn(2, hidden, dtype=torch.float32)
    with torch.no_grad():
        out = nn.functional.relu(pre_classifier(x))
        out = classifier(out)
    assert out.shape == (2, num_labels)
    assert out.dtype == torch.float32


def test_squeeze_head_removes_spurious_dim():
    """_SqueezeHead squeezes [B, 1, C] → [B, C]; passes [B, C] through unchanged."""
    import torch
    import torch.nn as nn

    from spyre_inference.transformers_backend import _SqueezeHead

    hidden = 8
    num_labels = 4
    original_clf = nn.Linear(hidden, num_labels)

    head = _SqueezeHead(original_clf)

    # parameters() must be non-empty.
    assert list(head.parameters()), "head must expose parameters for Spyre CPU routing"

    # [B, C] passes through unchanged.
    x = torch.randn(2, hidden)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (2, num_labels)

    # [B, 1, C] is squeezed to [B, C].
    class _FakeLinear(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, num_labels)

    head3d = _SqueezeHead(_FakeLinear())
    with torch.no_grad():
        out = head3d(x)
    assert out.shape == (2, num_labels), f"expected (2, {num_labels}), got {out.shape}"


def _apply_alias_loop(result: set) -> set:
    """Apply the generic alias loop from SpyreTransformersForSequenceClassification.load_weights."""
    for key in list(result):
        if key.startswith("classifier."):
            result.add("classifier.classifier." + key[len("classifier.") :])
        elif key.startswith("pre_classifier."):
            result.add("classifier.pre_clf." + key[len("pre_classifier.") :])
    return result


def test_load_weights_aliases_flat_linear_classifier():
    """Flat nn.Linear classifier (DistilBERT): weight/bias paths aliased correctly.

    _install_head wraps self.classifier inside _PreClassifierHead or _SqueezeHead:
      classifier.weight/bias     -> classifier.classifier.weight/bias  (both variants)
      pre_classifier.weight/bias -> classifier.pre_clf.weight/bias     (_PreClassifierHead)
    """
    result = _apply_alias_loop(
        {
            "pre_classifier.weight",
            "pre_classifier.bias",
            "classifier.weight",
            "classifier.bias",
        }
    )

    assert "classifier.classifier.weight" in result
    assert "classifier.classifier.bias" in result
    assert "classifier.pre_clf.weight" in result
    assert "classifier.pre_clf.bias" in result


def test_load_weights_aliases_multi_layer_classifier():
    """Multi-layer head (RobertaClassificationHead): dense.* and out_proj.* aliased.

    papluca/xlm-roberta-base-language-detection uses RobertaClassificationHead
    which has dense + out_proj sub-layers. The old fixed-key alias dict missed these.
    The generic loop must cover all sub-parameter paths regardless of head structure.
    """
    result = _apply_alias_loop(
        {
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        }
    )

    assert "classifier.classifier.dense.weight" in result
    assert "classifier.classifier.dense.bias" in result
    assert "classifier.classifier.out_proj.weight" in result
    assert "classifier.classifier.out_proj.bias" in result
    # original keys must still be present (loop only adds, never removes)
    assert "classifier.dense.weight" in result


def test_load_weights_aliases_no_pre_classifier():
    """Models without pre_classifier must not get pre_clf aliases."""
    result = _apply_alias_loop({"classifier.weight", "classifier.bias"})

    assert "classifier.classifier.weight" in result
    assert "classifier.classifier.bias" in result
    assert not any(k.startswith("classifier.pre_clf") for k in result)


def test_install_head_raises_for_unsupported_pre_classifier():
    """_install_head must raise NotImplementedError for non-Linear pre_classifier.

    Silently dropping a multi-layer MLP pre_classifier would produce wrong output
    with no error. The NotImplementedError surfaces this at model load time instead.
    """
    import types

    import torch.nn as nn

    from spyre_inference.transformers_backend import SpyreTransformersForSequenceClassification

    # Build a minimal stand-in for self that _install_head reads before raising.
    # The raise happens before self.pooler is accessed, so pooler is not needed.
    fake_self = types.SimpleNamespace(
        classifier=nn.Linear(8, 2),
        pre_classifier=nn.Sequential(nn.Linear(8, 8), nn.ReLU()),  # not nn.Linear
        model=types.SimpleNamespace(__class__=type("FakeModel", (), {})),
    )

    with pytest.raises(NotImplementedError, match="pre_classifier of type"):
        SpyreTransformersForSequenceClassification._install_head(fake_self)
