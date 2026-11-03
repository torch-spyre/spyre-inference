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

"""CPU contract tests using the real HF backbone and vLLM weight loader."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import GraniteConfig, GraniteModel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from spyre_inference.hf_adapters import (
    AttentionBinding,
    PreparedBody,
    prepare_execution,
    uses_hf_adapters,
)
from spyre_inference.platform import TorchSpyrePlatform
from spyre_inference.transformers_backend import SpyreTransformersForCausalLM


def execution_result(head_size=64, **overrides):
    fields = dict(
        head_size=head_size,
        forward_backbone=Mock(),
        attention_bindings=tuple(
            AttentionBinding(index, 2, 2, head_size, 0.125, Mock()) for index in range(2)
        ),
        compile_body=Mock(),
    )
    fields.update(overrides)
    return PreparedBody(**fields)


def prepare_fixture(model, *, vllm_config):
    return execution_result()


def pad_fixture(model, *, vllm_config):
    # Geometry-only fixture; no forward is run with these unpadded test weights.
    return execution_result(head_size=128)


class RecordedAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.settings = kwargs
        self.processed = False

    def process_weights_after_loading(self, dtype):
        self.processed = True


class RecordedLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=torch.float16))
        self.weight.weight_loader = self.load
        self.bias = None

    @staticmethod
    def load(param, loaded):
        with torch.no_grad():
            param.copy_(loaded)


@pytest.fixture
def config_factory(tmp_path):
    def create(*, adapter=True, tied=False, hf_config=None, hf_overrides=None, **kwargs):
        hf = hf_config or GraniteConfig(
            architectures=["GraniteForCausalLM"],
            hidden_size=128,
            intermediate_size=257,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=256,
            max_position_embeddings=64,
            attention_multiplier=0.125,
            logits_scaling=16.0,
            tie_word_embeddings=tied,
        )
        hf.save_pretrained(tmp_path)
        additional = {"spyre_model_preparation": "hf_adapters"} if adapter else {}
        engine_args = EngineArgs(
            model=str(tmp_path),
            model_impl="transformers",
            dtype="float16",
            max_model_len=64,
            enforce_eager=True,
            skip_tokenizer_init=True,
            hf_overrides=hf_overrides or {},
            additional_config=additional,
        )
        TorchSpyrePlatform.pre_register_and_update()
        model = engine_args.create_model_config()
        return VllmConfig(model_config=model, additional_config=additional, **kwargs)

    return create


@pytest.fixture
def backend_factory(monkeypatch, config_factory):
    group = SimpleNamespace(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr("vllm.model_executor.models.transformers.base.get_pp_group", lambda: group)
    monkeypatch.setattr("vllm.model_executor.models.transformers.base.get_tp_group", lambda: group)
    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.causal.ParallelLMHead", RecordedLMHead
    )
    monkeypatch.setattr(
        SpyreTransformersForCausalLM, "_decorate_for_torch_compile", lambda self: None
    )
    monkeypatch.setattr(
        SpyreTransformersForCausalLM, "_get_attn_cls", lambda self: RecordedAttention
    )

    def create(*, preparer=prepare_fixture, **kwargs):
        monkeypatch.setattr("spyre_inference.hf_adapters.prepare_execution", preparer)
        config = config_factory(**kwargs)
        with set_current_vllm_config(config), torch.device("cpu"):
            backend = SpyreTransformersForCausalLM(vllm_config=config)
        return backend

    return create


def test_platform_padding_is_per_engine(config_factory):
    adapter = config_factory()
    normal = config_factory(adapter=False)
    assert adapter.model_config.get_head_size() == 64
    assert adapter.model_config.hf_text_config.intermediate_size == 257
    assert normal.model_config.get_head_size() == 128
    assert normal.model_config.hf_text_config.intermediate_size > 257
    TorchSpyrePlatform.check_and_update_config(adapter)
    assert adapter.model_config.get_head_size() == 64


@pytest.mark.parametrize(
    "additional",
    [
        {"spyre_model_preparation": "typo"},
        {"spyre_hf_adapters_preparer": "m:f"},
        {"spyre_model_preparation": "hf_adapters", "spyre_hf_adapters_preparer": "missing_colon"},
    ],
)
def test_invalid_selection_rejected(additional):
    with pytest.raises(ValueError):
        uses_hf_adapters(SimpleNamespace(additional_config=additional))


def test_body_weights_and_rope_survive_until_preparation(backend_factory, monkeypatch):
    backend = backend_factory()
    monkeypatch.setattr(backend, "_patch_rope", Mock(side_effect=AssertionError("bridge RoPE ran")))
    rope = backend.model.rotary_emb
    assert not backend.attention_fusers
    for layer in backend.model.layers:
        assert type(layer.self_attn.q_proj) is nn.Linear
        assert type(layer.mlp.gate_proj) is nn.Linear
        assert type(layer.input_layernorm).__name__ == "GraniteRMSNorm"
        assert not hasattr(layer.self_attn, "attn")
    assert type(backend.model.get_input_embeddings()) is nn.Embedding
    reference = GraniteModel(backend.model.config).to(torch.float16)
    weights = {f"model.{name}": p.detach().clone() for name, p in reference.named_parameters()}
    weights["lm_head.weight"] = torch.randn_like(backend.lm_head.weight)
    loaded = backend.load_weights(iter(weights.items()))
    assert loaded == set(weights)
    actual = dict(backend.named_parameters())
    for name, tensor in weights.items():
        torch.testing.assert_close(actual[name], tensor)
    backend.prepare_hf_adapters()
    assert backend.model.rotary_emb is rope
    assert not backend.attention_fusers
    assert len(backend.adapter_attentions) == 2
    for executor, binding in zip(
        backend.adapter_attentions, backend._prepared_body.attention_bindings
    ):
        assert executor.attn.settings["head_size"] == 64
        assert executor.attn.settings["scale"] == 0.125
        assert executor.attn.processed
        binding.bind.assert_called_once_with(executor)
    with pytest.raises(RuntimeError, match="exactly once"):
        backend.prepare_hf_adapters()
    with pytest.raises(NotImplementedError, match="Reloading"):
        backend.load_weights([])


def test_attention_uses_prepared_geometry_and_original_scale(backend_factory):
    backend = backend_factory(preparer=pad_fixture)
    assert backend.model_config.get_head_size() == 64
    backend.prepare_hf_adapters()
    assert backend.model_config.get_head_size() == 128
    for executor in backend.adapter_attentions:
        assert executor.attn.settings["head_size"] == 128
        assert executor.attn.settings["scale"] == 0.125


def test_tied_head_is_loaded_from_raw_hf_embedding(backend_factory):
    backend = backend_factory(tied=True)
    weights = {f"model.{name}": torch.randn_like(p) for name, p in backend.model.named_parameters()}
    loaded = backend.load_weights(iter(weights.items()))
    assert "lm_head.weight" in loaded
    torch.testing.assert_close(backend.lm_head.weight, backend.model.get_input_embeddings().weight)


def test_real_granite_adapter_prepares_loaded_body(backend_factory):
    pytest.importorskip("hf_adapters.execution")
    from hf_adapters.hf_common import PrecomputedRotaryEmbedding

    backend = backend_factory(preparer=prepare_execution)
    backend.eval()
    weights = {f"model.{name}": torch.randn_like(p) for name, p in backend.model.named_parameters()}
    weights["lm_head.weight"] = torch.randn_like(backend.lm_head.weight)
    backend.load_weights(iter(weights.items()))
    original_head = backend.lm_head.weight.detach().clone()
    original_config = backend.model.config
    backend.prepare_hf_adapters()
    assert backend.model.config is original_config
    assert backend.logits_processor.scale == 1 / 16
    assert backend.model_config.get_head_size() == 128
    assert callable(backend._prepared_body.compile_body)
    assert isinstance(backend.model._spyre_rope, PrecomputedRotaryEmbedding)
    torch.testing.assert_close(backend.lm_head.weight, original_head)
    for layer, executor in zip(backend.model.layers, backend.adapter_attentions):
        assert type(layer.self_attn.q_proj) is nn.Linear
        assert layer.self_attn.q_proj.weight.shape == (256, 128)
        assert executor.attn.settings["head_size"] == 128
        assert executor.attn.settings["scale"] == 0.125
        assert executor.attn.processed
        assert layer.mlp.gate_proj.weight.shape == (257, 128)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_prepared_logits_scale_is_rejected(config_factory, monkeypatch, scale):
    from spyre_inference.hf_adapters import prepare_body

    monkeypatch.setattr(
        "spyre_inference.hf_adapters.prepare_execution",
        lambda *args, **kwargs: execution_result(logits_scale=scale),
    )
    with pytest.raises(ValueError, match="logits_scale"):
        prepare_body(nn.Identity(), config_factory())


def test_adapter_forward_requires_preparation(backend_factory):
    backend = backend_factory()
    with pytest.raises(RuntimeError, match="not been prepared"):
        backend(torch.tensor([1]), torch.tensor([0]))


def test_native_forward_still_delegates_to_transformers(backend_factory, monkeypatch):
    backend = backend_factory()
    backend._hf_adapters = False
    native = Mock(return_value=torch.zeros(1, 128))
    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.TransformersForCausalLM.forward", native
    )
    ids, positions = torch.tensor([1]), torch.tensor([0])
    assert backend(ids, positions) is native.return_value
    native.assert_called_once_with(ids, positions, None, None)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("model_impl", "vllm", "model_impl"),
        ("quantization", "fp8", "unquantized"),
        ("runner_type", "pooling", "text generation"),
    ],
)
def test_unsupported_model_options(config_factory, field, value, error):
    from spyre_inference.hf_adapters import validate_hf_adapters_config

    config = config_factory()
    setattr(config.model_config, field, value)
    with pytest.raises(ValueError, match=error):
        validate_hf_adapters_config(config)


@pytest.mark.parametrize("granularity", ["block", "model"])
def test_adapter_compile_skips_native_wrapping_but_not_eager(
    backend_factory, monkeypatch, granularity
):
    from vllm.config import CompilationMode, get_current_vllm_config

    from spyre_inference.v1.worker.compile_granularity import compile_for_spyre

    monkeypatch.setenv("SPYRE_COMPILE_GRANULARITY", granularity)
    monkeypatch.setattr(torch, "compile", Mock(side_effect=AssertionError("double compilation")))
    monkeypatch.setattr(
        "spyre_inference.v1.worker.compile_granularity.compile_blocks",
        Mock(side_effect=AssertionError("double compilation")),
    )
    backend = backend_factory()

    def compile_body(model):
        assert get_current_vllm_config() is backend.vllm_config

    callback = Mock(side_effect=compile_body)
    backend._prepared_body = execution_result(compile_body=callback)
    config = backend.vllm_config
    assert compile_for_spyre(backend, config) == (backend, 0)
    callback.assert_not_called()
    backend.model_config.enforce_eager = False
    config.compilation_config.mode = CompilationMode.NONE
    assert compile_for_spyre(backend, config) == (backend, 0)
    callback.assert_not_called()
    config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    assert compile_for_spyre(backend, config) == (backend, 0)
    callback.assert_called_once_with(backend.model)


@pytest.mark.parametrize("adapter", [False, True])
def test_runner_prepares_before_placement_and_skips_only_body_passes(
    config_factory, monkeypatch, adapter
):
    from vllm.config import get_current_vllm_config

    from spyre_inference.v1.worker import spyre_model_runner as module

    config = config_factory(adapter=adapter)
    events = []
    body = nn.Module()

    def prepare():
        assert get_current_vllm_config() is config
        events.append("prepare")

    body.prepare_hf_adapters = prepare
    loader = SimpleNamespace(load_model=lambda **kwargs: events.append("load") or body)
    monkeypatch.setattr(module, "get_model_loader", lambda cfg: loader)
    monkeypatch.setattr(module, "check_spyre_model", lambda *args: None)
    monkeypatch.setattr(module, "get_model_architecture", lambda cfg: (None, "unused"))
    for name, event in [
        ("install_pad_loaders", "pad_loader"),
        ("fix_padded_model", "fix_padding"),
        ("resolve_kv_scales", "kv_scales"),
        ("move_model_to_spyre", "move"),
        ("apply_multimodal_patches", "multimodal"),
    ]:
        monkeypatch.setattr(module, name, lambda *args, event=event: events.append(event))
    monkeypatch.setattr(
        module, "compile_for_spyre", lambda model, cfg: (events.append("compile") or model, 0)
    )
    monkeypatch.setattr(module.SpyreShapeBucketer, "for_config", lambda cfg: None)
    monkeypatch.setattr(module, "_SpyreModelWrapper", lambda model, *args, **kwargs: model)
    runner = SimpleNamespace(
        vllm_config=config,
        model_config=config.model_config,
        load_config=config.load_config,
        lora_config=None,
        speculative_config=None,
        compilation_config=config.compilation_config,
        num_spec_tokens=0,
        use_aux_hidden_state_outputs=False,
        _setup_eagle3_aux_hidden_state_outputs=lambda: None,
        _spyre_device="cpu",
        max_num_reqs=2,
        inputs_embeds=SimpleNamespace(gpu=None),
        _model_dtype=lambda: torch.float16,
    )
    module.TorchSpyreModelRunner.load_model(runner)
    expected = (
        ["load", "prepare", "kv_scales", "move", "compile"]
        if adapter
        else ["pad_loader", "load", "fix_padding", "kv_scales", "move", "multimodal", "compile"]
    )
    assert events == expected
    assert runner.model is body


@pytest.mark.parametrize(
    "result,error",
    [
        (None, TypeError),
        (execution_result(65), ValueError),
        (execution_result(0), ValueError),
        (execution_result(compile_body="not callable"), TypeError),
        (execution_result(compile_body=None), TypeError),
        (execution_result(forward_backbone=None), TypeError),
        (execution_result(attention_bindings=()), ValueError),
    ],
)
def test_invalid_provider_result_is_rejected(config_factory, monkeypatch, result, error):
    from spyre_inference.hf_adapters import prepare_body

    monkeypatch.setattr("spyre_inference.hf_adapters.prepare_execution", lambda *a, **k: result)
    with pytest.raises(error):
        prepare_body(nn.Module(), config_factory())


def test_prelaunch_preserves_registration_but_skips_model_overrides(monkeypatch):
    from spyre_inference import models
    from spyre_inference.models import clip, gemma4

    register = Mock()
    gemma = Mock()
    clip_override = Mock()
    monkeypatch.setattr(models, "register_models", register)
    monkeypatch.setattr(gemma4, "force_text_backbone", gemma)
    monkeypatch.setattr(clip, "force_disable_chunked_prefill", clip_override)
    models.apply_prelaunch_overrides(
        SimpleNamespace(
            additional_config={
                "spyre_model_preparation": "hf_adapters",
            }
        )
    )
    register.assert_called_once()
    gemma.assert_not_called()
    clip_override.assert_not_called()


@pytest.mark.parametrize(
    "family", ["granite", "phi3", "gpt2", "gpt_neo", "gpt_neox", "opt", "ministral"]
)
def test_existing_adapter_execution_uses_explicit_attention(backend_factory, monkeypatch, family):
    import copy
    import importlib

    import torch.nn.functional as F
    from hf_adapters.hf_common import get_backbone
    from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config

    class NumericalAttention(RecordedAttention):
        def forward(self, query, key, value):
            heads = self.settings["num_heads"]
            kv_heads = self.settings["num_kv_heads"]
            width = self.settings["head_size"]
            q = query.view(1, -1, heads, width).transpose(1, 2)
            k = key.view(1, -1, kv_heads, width).transpose(1, 2)
            v = value.view(1, -1, kv_heads, width).transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.settings["scale"], enable_gqa=True
            )
            return out.transpose(1, 2).reshape(-1, heads * width)

    monkeypatch.setattr(
        SpyreTransformersForCausalLM, "_get_attn_cls", lambda self: NumericalAttention
    )
    hf = None
    if family == "phi3":
        hf = Phi3Config(
            architectures=["Phi3ForCausalLM"],
            hidden_size=256,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=256,
            max_position_embeddings=64,
            original_max_position_embeddings=64,
            partial_rotary_factor=0.75,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=1,
        )
    elif family in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
        hf = AutoConfig.for_model(
            family,
            architectures=[
                {
                    "gpt2": "GPT2LMHeadModel",
                    "gpt_neo": "GPTNeoForCausalLM",
                    "gpt_neox": "GPTNeoXForCausalLM",
                    "opt": "OPTForCausalLM",
                }[family]
            ],
            hidden_size=128,
            intermediate_size=256,
            ffn_dim=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            vocab_size=256,
            max_position_embeddings=64,
            word_embed_proj_dim=128,
            **({"attention_types": [[["global", "local"], 1]]} if family == "gpt_neo" else {}),
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    elif family == "ministral":
        hf = AutoConfig.for_model(
            family,
            architectures=["MinistralForCausalLM"],
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=256,
            max_position_embeddings=64,
            sliding_window=64,
            layer_types=["full_attention", "sliding_attention"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    backend = backend_factory(preparer=prepare_execution, hf_config=hf)
    assert backend.model.config.architectures == ["TransformersForCausalLM"]
    assert backend.model.config.model_type == family
    assert backend.text_config._attn_implementation == ("eager" if family == "gpt_neo" else "vllm")
    backend.eval()
    reference_config = copy.deepcopy(backend.model.config)
    reference_config._attn_implementation = "eager" if family == "gpt_neo" else "sdpa"
    reference = AutoModelForCausalLM.from_config(reference_config).eval().to(torch.float16)
    if family == "gpt2":
        with torch.no_grad():
            for layer in reference.transformer.h:
                layer.attn.c_attn.bias.normal_(std=0.1)
    if family == "gpt_neo":
        with torch.no_grad():
            for layer in reference.transformer.h:
                layer.attn.attention.out_proj.bias.normal_(std=0.1)
    weights = {name: p.detach().clone() for name, p in reference.named_parameters()}
    if family == "gpt_neo":
        weights["transformer.h.0.attn.attention.bias"] = torch.ones(1, 1, 64, 64)
        weights["transformer.h.0.attn.attention.masked_bias"] = torch.tensor(-1e9)
    if family == "gpt2":
        weights["transformer.h.0.attn.bias"] = torch.ones(1, 1, 64, 64)
    backend.load_weights(iter(weights.items()))
    for name, parameter in backend.model.named_parameters():
        torch.testing.assert_close(parameter, get_backbone(reference).get_parameter(name))
    ids = torch.tensor([[1, 3, 5, 7]])
    positions = torch.tensor([[5, 6, 7, 8]])
    with torch.inference_mode():
        expected = get_backbone(reference)(
            ids, position_ids=positions, use_cache=False
        ).last_hidden_state
    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.fusers.AttentionFuser.match",
        Mock(side_effect=AssertionError("Adapter execution must not discover HF forwards")),
    )
    backend.prepare_hf_adapters()
    monkeypatch.setattr(
        backend.model, "forward", Mock(side_effect=AssertionError("HF forward ran"))
    )
    assert not backend.attention_fusers
    assert len(backend.adapter_attentions) == 2
    expected_width = 64 if family in ("gpt2", "gpt_neo", "opt") else 128
    for executor in backend.adapter_attentions:
        assert executor.attn.processed
        assert executor.attn.settings["head_size"] == expected_width
        if family in ("gpt2", "gpt_neo", "gpt_neox", "opt"):
            assert executor.attn.settings["num_kv_heads"] == 2
    adapter = importlib.import_module(f"hf_adapters.hf_{family}")
    assert backend._prepared_body.forward_backbone.__self__.driver is adapter._run_backbone_forward
    with torch.inference_mode():
        actual = backend(ids[0], positions[0])
    torch.testing.assert_close(actual, expected[0], atol=0.006, rtol=0.006)
    assert backend.model_config.get_head_size() == expected_width
    with pytest.raises(ValueError, match="token IDs"):
        backend(
            None,
            positions[0],
            inputs_embeds=torch.zeros(4, backend.text_config.hidden_size),
        )


@pytest.mark.parametrize("failure", ["duplicate", "width", "scale", "head_count"])
def test_malformed_explicit_bindings_rejected(config_factory, monkeypatch, failure):
    from spyre_inference.hf_adapters import AttentionBinding, prepare_body

    first = AttentionBinding(0, 2, 2, 128, 0.125, lambda module: None)
    second = AttentionBinding(1, 2, 2, 128, 0.125, lambda module: None)
    if failure == "duplicate":
        second = first
    elif failure == "width":
        second = AttentionBinding(1, 2, 2, 64, 0.125, lambda module: None)
    elif failure == "scale":
        second = AttentionBinding(1, 2, 2, 128, float("nan"), lambda module: None)
    elif failure == "head_count":
        second = AttentionBinding(1, 4, 2, 128, 0.125, lambda module: None)
    result = PreparedBody(
        128,
        compile_body=lambda model: None,
        forward_backbone=lambda ids, pos: ids,
        attention_bindings=(first, second),
    )
    monkeypatch.setattr("spyre_inference.hf_adapters.prepare_execution", lambda *a, **k: result)
    with pytest.raises(ValueError):
        prepare_body(nn.Module(), config_factory())


@pytest.mark.parametrize("window,accepted", [(63, False), (64, True), (65, True), (None, False)])
def test_adapter_sliding_layers_require_covering_context(config_factory, window, accepted):
    from spyre_inference.hf_adapters import validate_hf_adapters_config

    config = config_factory()
    config.model_config.hf_text_config.layer_types = [
        "full_attention",
        "sliding_attention",
    ]
    config.model_config.hf_text_config.sliding_window = window
    if accepted:
        validate_hf_adapters_config(config)
    else:
        with pytest.raises(ValueError, match="dense full-attention decoder"):
            validate_hf_adapters_config(config)


def test_adapter_non_attention_layer_rejected_with_covering_window(config_factory):
    from spyre_inference.hf_adapters import validate_hf_adapters_config

    config = config_factory()
    config.model_config.hf_text_config.layer_types = [
        "linear_attention",
        "full_attention",
    ]
    config.model_config.hf_text_config.sliding_window = 64
    with pytest.raises(ValueError, match="dense full-attention decoder"):
        validate_hf_adapters_config(config)


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize(
    "family",
    [
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "granite",
        "llama",
        "ministral",
        "mistral",
        "olmo",
        "olmo2",
        "opt",
        "phi3",
        "qwen2",
        "qwen3",
        "smollm3",
    ],
)
def test_hf_attention_selection_is_per_engine(adapter, family):
    from transformers import AutoConfig

    backend = SpyreTransformersForCausalLM.__new__(SpyreTransformersForCausalLM)
    nn.Module.__init__(backend)
    backend.config = backend.text_config = AutoConfig.for_model(family)
    backend._hf_adapters = adapter
    backend._patch_config()
    expected = "eager" if adapter and family == "gpt_neo" else "vllm"
    assert backend.text_config._attn_implementation == expected


@pytest.mark.parametrize("capability", ["is_backend_compatible", "_can_set_attn_implementation"])
def test_gpt_neo_keeps_backend_when_hf_advertises_support(monkeypatch, capability):
    from transformers import GPTNeoConfig, GPTNeoModel

    monkeypatch.setattr(GPTNeoModel, capability, classmethod(lambda cls: True))
    backend = SpyreTransformersForCausalLM.__new__(SpyreTransformersForCausalLM)
    nn.Module.__init__(backend)
    backend.config = backend.text_config = GPTNeoConfig()
    backend._hf_adapters = True
    backend._patch_config()
    assert backend.text_config._attn_implementation == "vllm"


def test_gpt_neo_default_transformers_eligibility_is_unchanged(config_factory):
    from transformers import GPTNeoConfig, GPTNeoForCausalLM

    assert not GPTNeoForCausalLM.is_backend_compatible()
    config = GPTNeoConfig(
        architectures=["GPTNeoForCausalLM"],
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        attention_types=[[["global", "local"], 1]],
        vocab_size=256,
        max_position_embeddings=64,
    )
    with pytest.raises(ValueError, match="not compatible with vLLM"):
        config_factory(adapter=False, hf_config=config)


@pytest.mark.parametrize("kind", ["dict", "callable", "replacement"])
def test_automatic_wrapper_preserves_user_overrides_and_engine_isolation(tmp_path, kind):
    import copy

    config = GraniteConfig(
        architectures=["GraniteForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )
    config.save_pretrained(tmp_path)
    settings = {
        "architectures": ["GraniteForCausalLM"],
        "rms_norm_eps": 3e-5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 12345.0},
    }
    before = copy.deepcopy(settings)
    calls = []

    def override(config):
        calls.append(config)
        if kind == "replacement":
            config = copy.deepcopy(config)
        config.architectures = list(settings["architectures"])
        config.rms_norm_eps = settings["rms_norm_eps"]
        config.rope_parameters = {
            **(getattr(config, "rope_parameters", None) or {}),
            **settings["rope_parameters"],
        }
        return config

    user_overrides = settings if kind == "dict" else override
    args = EngineArgs(
        model=str(tmp_path),
        model_impl="transformers",
        max_model_len=64,
        skip_tokenizer_init=True,
        hf_overrides=user_overrides,
        additional_config={"spyre_model_preparation": "hf_adapters"},
    )
    TorchSpyrePlatform.pre_register_and_update()
    calls_per_creation = []
    for _ in range(2):
        before_calls = len(calls)
        resolved = args.create_model_config()
        calls_per_creation.append(len(calls) - before_calls)
        assert resolved.architectures == ["TransformersForCausalLM"]
        assert resolved.hf_config.model_type == "granite"
        assert resolved.hf_config.rms_norm_eps == 3e-5
        assert resolved.hf_config.rope_parameters["rope_theta"] == 12345.0
        assert resolved.hf_config.rope_parameters["rope_type"] == "default"
        assert args.hf_overrides is user_overrides
        assert settings == before
    if kind != "dict":
        assert calls_per_creation[0] > 0
        assert calls_per_creation[0] == calls_per_creation[1]
    args.additional_config = {}
    ordinary = args.create_model_config()
    assert ordinary.architectures == ["GraniteForCausalLM"]
    assert ordinary.hf_config.rms_norm_eps == 3e-5
    assert args.hf_overrides is user_overrides


def test_automatic_wrapper_restores_overrides_after_config_failure(tmp_path):
    GraniteConfig(architectures=["GraniteForCausalLM"]).save_pretrained(tmp_path)

    def broken_override(config):
        raise ValueError("user override failed")

    args = EngineArgs(
        model=str(tmp_path),
        model_impl="transformers",
        skip_tokenizer_init=True,
        hf_overrides=broken_override,
        additional_config={"spyre_model_preparation": "hf_adapters"},
    )
    TorchSpyrePlatform.pre_register_and_update()
    with pytest.raises(ValueError, match="user override failed"):
        args.create_model_config()
    assert args.hf_overrides is broken_override


def test_generic_wrapper_cannot_hide_nested_multimodal_config(config_factory):
    from spyre_inference.hf_adapters import validate_hf_adapters_config

    config = config_factory()
    assert not config.model_config.is_multimodal_model
    config.model_config.hf_config = SimpleNamespace(text_config=config.model_config.hf_text_config)
    with pytest.raises(ValueError, match="text generation only"):
        validate_hf_adapters_config(config)


def test_generic_wrapper_does_not_enable_unsupported_adapter(config_factory):
    from transformers import BertConfig, BertModel

    hf = BertConfig(
        architectures=["BertForMaskedLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )
    config = config_factory(hf_config=hf)
    with pytest.raises(ValueError, match="does not support BertModel"):
        prepare_execution(BertModel(hf).eval(), vllm_config=config)
