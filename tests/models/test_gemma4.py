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

"""`force_text_backbone` must keep repairing plain text gemma-4 checkpoints (the
transformers>=5.16 heterogeneous-config issue) while leaving a real multimodal
Gemma4Config (populated vision_config/audio_config) alone, so vLLM's own
Gemma4ForConditionalGeneration can resolve normally. No card needed: this is pure
EngineArgs/config-object plumbing, mocked at the vllm.transformers_utils.config.get_config
seam `force_text_backbone` imports locally.
"""

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.models import gemma4
from spyre_inference.models.gemma4 import (
    GEMMA4_TEXT_BACKBONE_OVERRIDE,
    configure_gemma4_moe_layers,
    force_text_backbone,
    repair_head_dim_access,
)


def _engine_args(**overrides):
    defaults = dict(
        hf_overrides=None,
        hf_config_path=None,
        model="google/gemma-4-26B-A4B",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        config_format="auto",
        hf_token=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_get_config(monkeypatch, hf_config):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        lambda *args, **kwargs: hf_config,
    )


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_text"])
def test_force_text_backbone_applies_for_pure_text_checkpoint(monkeypatch, model_type):
    """A plain text checkpoint (no vision_config/audio_config) still gets the fix."""
    hf_config = SimpleNamespace(model_type=model_type, vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides == GEMMA4_TEXT_BACKBONE_OVERRIDE


def test_force_text_backbone_skips_for_vlm_checkpoint_with_vision_config(monkeypatch):
    """A vision checkpoint must keep its architectures -- forcing the text backbone would
    strip the tower. It needs no override at all: the head-dim repair runs from the
    platform hook."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=None,
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


def test_audio_only_checkpoint_is_rejected(monkeypatch):
    """Falling through to the text backbone would silently drop the tower, leaving a
    model that looks fine but ignores its audio inputs."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=None,
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    with pytest.raises(NotImplementedError, match="audio is not supported"):
        force_text_backbone(_engine_args())


def test_vision_plus_audio_checkpoint_is_allowed_for_its_vision_path(monkeypatch):
    """Both towers present: the vision path still works, so this warns rather than
    refusing to load."""
    hf_config = SimpleNamespace(
        model_type="gemma4",
        vision_config=SimpleNamespace(model_type="gemma4_vision"),
        audio_config=SimpleNamespace(model_type="gemma4_audio"),
    )
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


def test_repair_head_dim_access_reaches_the_nested_text_config():
    text_config = SimpleNamespace(model_type="gemma4_text")
    config = SimpleNamespace(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
    )

    repair_head_dim_access(config)

    assert config.architectures == ["Gemma4ForConditionalGeneration"]
    assert config.allow_global_per_layer_attribute_access is True
    assert text_config.allow_global_per_layer_attribute_access is True


def test_repair_head_dim_access_ignores_an_unrelated_config():
    """It runs from `check_and_update_config`, i.e. for every model Spyre loads."""
    config = SimpleNamespace(model_type="llama")

    repair_head_dim_access(config)

    assert not hasattr(config, "allow_global_per_layer_attribute_access")


def test_force_text_backbone_ignores_unrelated_model_type(monkeypatch):
    hf_config = SimpleNamespace(model_type="llama", vision_config=None, audio_config=None)
    _patch_get_config(monkeypatch, hf_config)

    engine_args = _engine_args()
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is None


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(model_type="gemma4", vision_config=object(), audio_config=None), True),
        # Audio is out of scope, so an audio-only config is not "multimodal" here.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=object()), False),
        # Text-only gemma-4 is already validated in fp16 here; don't move it.
        (SimpleNamespace(model_type="gemma4", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="gemma4_text", vision_config=None, audio_config=None), False),
        (SimpleNamespace(model_type="llama", vision_config=None, audio_config=None), False),
    ],
)
def test_is_multimodal_gemma4_requires_a_vision_tower(hf_config, expected):
    """``force_text_backbone`` gates on this: a checkpoint with a tower keeps the
    multimodal architecture, one without it is pinned to the text backbone."""
    from spyre_inference.models.gemma4 import is_multimodal_gemma4

    assert is_multimodal_gemma4(hf_config) is expected


def test_multimodal_gemma4_delegates_its_text_half_through_the_model_registry():
    """Why ``Gemma4ForConditionalGeneration`` needs no ``_ADAPTED_ARCHS`` entry: it
    resolves its text half through ``ModelRegistry``, so it already lands on
    ``SpyreGemma4ForCausalLM``. If upstream ever built that model directly, the PLE and
    MoE adaptations would vanish silently -- hence a tripwire rather than a comment.
    """
    import inspect

    gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

    source = inspect.getsource(gemma4_mm.Gemma4ForConditionalGeneration.__init__)
    assert "init_vllm_registered_model" in source, (
        "Gemma4ForConditionalGeneration no longer resolves its text model through "
        "the vLLM registry; SpyreGemma4ForCausalLM's MoE/PLE adaptations are being "
        "bypassed. Register a Spyre subclass of it in models/_ADAPTED_ARCHS."
    )
    assert '"Gemma4ForCausalLM"' in source, (
        "Gemma4ForConditionalGeneration no longer names Gemma4ForCausalLM as its "
        "text architecture; check which architecture it resolves now and make sure "
        "spyre_models() adapts that one."
    )


def test_force_text_backbone_respects_user_supplied_hf_overrides(monkeypatch):
    """Skipped entirely (not even a get_config call) when the user already set
    hf_overrides. Safe because the head-dim repair no longer rides on this override:
    `repair_head_dim_access` runs from the platform hook, so a user override -- a dict
    such as GEMMA4_TEXT_BACKBONE_OVERRIDE included -- still gets it.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("get_config should not be called when hf_overrides is already set")

    monkeypatch.setattr("vllm.transformers_utils.config.get_config", _boom)

    def _user_override(config):
        return config

    engine_args = _engine_args(hf_overrides=_user_override)
    force_text_backbone(engine_args)

    assert engine_args.hf_overrides is _user_override


def test_moe_recipe_reads_the_decoder_experts_and_router_scale(monkeypatch):
    """vLLM's decoder holds `experts` and `router.per_expert_scale` directly; a dense
    layer has `experts = None`. A rename must not silently skip the recipe."""
    calls = []
    monkeypatch.setattr(gemma4, "configure_spyre_moe_layer", lambda *a: calls.append(a))
    scale = torch.tensor([2.0, 3.0])
    moe = SimpleNamespace(
        experts=SimpleNamespace(routed_experts="routed"),
        router=SimpleNamespace(per_expert_scale=scale),
    )

    assert configure_gemma4_moe_layers([moe, SimpleNamespace(experts=None)]) == 1
    ((routed, recipe),) = calls
    assert routed == "routed"
    folded = recipe.prepare_down_weight(torch.ones(2, 1, 1))
    torch.testing.assert_close(folded.flatten(), scale)


@pytest.mark.parametrize(
    ("flag", "raises"),
    [("enable_moe_block", True), ("use_second_mlp_block", True), (None, False)],
)
def test_moe_checkpoint_without_experts_raises(monkeypatch, flag, raises):
    """Either flag makes upstream build the MoE block, so either has to trip the check
    when no layer exposes `experts`; a dense checkpoint configures none and is fine."""
    from torch import nn
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

    def parent_init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(**({flag: True} if flag else {}))
        self.model = SimpleNamespace(layers=[SimpleNamespace(experts=None)])

    monkeypatch.setattr(Gemma4ForCausalLM, "__init__", parent_init)
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    if raises:
        with pytest.raises(RuntimeError, match="no decoder layer exposes"):
            gemma4.SpyreGemma4ForCausalLM(vllm_config=vllm_config)
    else:
        gemma4.SpyreGemma4ForCausalLM(vllm_config=vllm_config)


class _Branch(torch.nn.Module):
    def __init__(self, seed, reduce, world):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(8, 8, generator=g)
        self.reduce, self.world = reduce, world

    def forward(self, x, router_logits=None):
        out = x @ self.weight
        if router_logits is not None:
            out = out * router_logits.sigmoid()
        return out * self.world if self.reduce else out


def _decoder_layer(*, fused, world=4):
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    norms = {
        name: _Branch(i, reduce=False, world=world)
        for i, name in enumerate(
            (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_2",
                "post_feedforward_layernorm",
            )
        )
    }
    layer = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    torch.nn.Module.__init__(layer)
    for name, mod in norms.items():
        setattr(layer, name, mod)
    attn = _Branch(20, reduce=True, world=world)
    layer.self_attn = lambda positions, hidden_states, **_: attn(hidden_states)
    layer.mlp = _Branch(21, reduce=not fused, world=world)
    layer.router = _Branch(22, reduce=False, world=world)
    layer.experts = _Branch(23, reduce=not fused, world=world)
    layer.enable_moe_block = True
    layer.per_layer_input_gate = None
    layer.register_buffer("layer_scalar", torch.tensor([0.5]))
    return layer


def test_fused_ffn_forward_matches_upstreams_forward(monkeypatch):
    """Fails when upstream's forward changes and the copy no longer matches it."""
    world = 4
    monkeypatch.setattr(gemma4, "tensor_model_parallel_all_reduce", lambda x: x * world)
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(7))

    expected, _ = _decoder_layer(fused=False, world=world).forward(None, x, None)
    got, _ = gemma4._fused_ffn_forward(_decoder_layer(fused=True, world=world), None, x, None)

    torch.testing.assert_close(got, expected)


def test_fuse_gemma4_ffn_allreduce_turns_off_both_branch_reductions():
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    down_proj = SimpleNamespace(reduce_results=True)
    moe_config = SimpleNamespace(skip_final_all_reduce=False)
    moe = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    torch.nn.Module.__init__(moe)
    moe.mlp = SimpleNamespace(down_proj=down_proj)
    moe.experts = SimpleNamespace(moe_config=moe_config)
    dense = Gemma4DecoderLayer.__new__(Gemma4DecoderLayer)
    torch.nn.Module.__init__(dense)
    dense.experts = None

    assert gemma4.fuse_gemma4_ffn_allreduce([moe, dense]) == 1
    assert not down_proj.reduce_results
    assert moe_config.skip_final_all_reduce
    assert moe.forward.__func__ is gemma4._fused_ffn_forward


@pytest.mark.parametrize(("tp", "fused"), [(1, 0), (4, 1)])
def test_ffn_allreduce_is_fused_only_under_tensor_parallelism(monkeypatch, tp, fused):
    from torch import nn
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

    def parent_init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(enable_moe_block=True)
        self.model = SimpleNamespace(layers=["layer"])

    calls = []
    monkeypatch.setattr(Gemma4ForCausalLM, "__init__", parent_init)
    monkeypatch.setattr(gemma4, "configure_gemma4_moe_layers", lambda layers: 1)
    monkeypatch.setattr(gemma4, "fuse_gemma4_ffn_allreduce", calls.append)
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp))

    gemma4.SpyreGemma4ForCausalLM(vllm_config=vllm_config)

    assert len(calls) == fused
