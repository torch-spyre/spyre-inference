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

"""``install_bert_head_pad`` widens BertSelfAttention/BertAttention at construction.

CPU-only: constructs the real vLLM classes under a TP=1 group. The end-to-end check
that the padded model produces correct embeddings is
``tests/e2e/test_encoder_models.py``.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.custom_ops import bert_head_pad
from spyre_inference.custom_ops.bert_head_pad import install_bert_head_pad


@pytest.fixture(autouse=True)
def _restore_bert_classes():
    """``install_bert_head_pad`` patches process-global vLLM classes; undo it so
    tests in this file (and any other test that later constructs a Bert model)
    don't see a patch left behind by a previous test."""
    from vllm.model_executor.models.bert import BertAttention, BertSelfAttention

    orig_self_attn_init = BertSelfAttention.__init__
    orig_attn_init = BertAttention.__init__
    orig_patched_flag = getattr(BertSelfAttention, bert_head_pad._PATCHED_ATTR, False)
    yield
    BertSelfAttention.__init__ = orig_self_attn_init
    BertAttention.__init__ = orig_attn_init
    setattr(BertSelfAttention, bert_head_pad._PATCHED_ATTR, orig_patched_flag)


def _model_config(*, orig, padded, runner_type="pooling"):
    hf_config = SimpleNamespace(head_dim=padded, _spyre_orig_head_dim=orig)
    return SimpleNamespace(hf_config=hf_config, runner_type=runner_type)


def test_pads_a_sub_stick_head_dim(default_vllm_config, tp_group):
    from vllm.model_executor.models.bert import BertAttention

    install_bert_head_pad(_model_config(orig=32, padded=64))

    attn = BertAttention(hidden_size=384, num_attention_heads=12, layer_norm_eps=1e-5)

    assert attn.self.head_dim == 64
    assert attn.self.q_size == 12 * 64
    assert attn.self.qkv_proj.weight.shape == (3 * 12 * 64, 384)
    assert attn.self.attn.impl.scale == pytest.approx(32**-0.5)
    assert attn.output.dense.weight.shape == (384, 12 * 64)


def test_leaves_an_already_aligned_head_dim_alone(default_vllm_config, tp_group):
    """Once installed, the patch stays correct for a later, unpadded model too."""
    from vllm.model_executor.models.bert import BertAttention

    install_bert_head_pad(_model_config(orig=32, padded=64))  # install, as above
    install_bert_head_pad(_model_config(orig=64, padded=64))  # a second, aligned model

    attn = BertAttention(hidden_size=768, num_attention_heads=12, layer_norm_eps=1e-5)

    assert attn.self.head_dim == 64
    assert attn.self.qkv_proj.weight.shape == (3 * 12 * 64, 768)
    assert attn.output.dense.weight.shape == (768, 768)


def test_noop_when_head_padding_is_not_active(default_vllm_config, tp_group):
    from vllm.model_executor.models.bert import BertAttention

    model_config = SimpleNamespace(hf_config=SimpleNamespace(head_dim=32), runner_type="pooling")
    install_bert_head_pad(model_config)

    attn = BertAttention(hidden_size=384, num_attention_heads=12, layer_norm_eps=1e-5)

    assert attn.self.head_dim == 32
    assert attn.output.dense.weight.shape == (384, 384)


def test_noop_for_a_non_pooling_model(default_vllm_config, tp_group):
    """Padding is only meant for the pooling path; a decoder shares this hf_config
    shape when the RoPE branch of ``_maybe_pad_head_dim`` fired instead."""
    from vllm.model_executor.models.bert import BertAttention

    install_bert_head_pad(_model_config(orig=32, padded=64, runner_type="generate"))

    attn = BertAttention(hidden_size=384, num_attention_heads=12, layer_norm_eps=1e-5)

    assert attn.self.head_dim == 32
    assert attn.output.dense.weight.shape == (384, 384)
