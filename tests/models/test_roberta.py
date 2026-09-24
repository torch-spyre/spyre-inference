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

"""``max_model_len`` has to fit the table an offset position embedding indexes into.

RoBERTa gathers ``position_ids + pad_token_id + 1``, so a 514-row table holds 512 usable
positions. The encoder's rectangular path pads every sequence to the declared length, so
the pad rows alone reach the top of the range on every request.
"""

from types import SimpleNamespace

import pytest

from spyre_inference.models.roberta import cap_max_model_len_for_position_offset


def _model_config(
    *,
    architectures=("XLMRobertaForSequenceClassification",),
    max_model_len=514,
    rows=514,
    pad_token_id=1,
    position_embedding_type="absolute",
):
    hf_config = SimpleNamespace(
        architectures=list(architectures),
        max_position_embeddings=rows,
        pad_token_id=pad_token_id,
        position_embedding_type=position_embedding_type,
    )
    return SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)


def test_caps_to_what_the_offset_table_can_index():
    config = _model_config(max_model_len=514)
    cap_max_model_len_for_position_offset(config)
    assert config.max_model_len == 512  # 514 rows - pad_token_id 1 - 1


def test_leaves_a_fitting_max_model_len_alone():
    config = _model_config(max_model_len=512)
    cap_max_model_len_for_position_offset(config)
    assert config.max_model_len == 512


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"architectures": ("BertForSequenceClassification",)}, id="not_roberta"),
        pytest.param({"position_embedding_type": "rotary"}, id="not_absolute"),
        pytest.param({"rows": None}, id="rows_unknown"),
        pytest.param({"pad_token_id": None}, id="pad_token_unknown"),
    ],
)
def test_no_op_when_it_does_not_apply(kwargs):
    config = _model_config(max_model_len=4096, **kwargs)
    cap_max_model_len_for_position_offset(config)
    assert config.max_model_len == 4096
