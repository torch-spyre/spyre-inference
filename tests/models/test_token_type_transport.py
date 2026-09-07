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

"""The out-of-band ``token_type_ids`` buffer must always match ``input_ids``.

The buffer is reused across steps so the compiled graph sees one tensor, which
makes a length change the interesting case: the embedding adds the buffer to the
word embeddings inside the compiled region, so a stale length is a hard failure
there rather than a wrong answer here.
"""

import torch

from spyre_inference.models._token_type import SpyreTokenTypeEmbedding


class _Embedding(SpyreTokenTypeEmbedding):
    """Stands in for Bert/RobertaEmbedding: only the segment lookup is needed."""

    def token_type_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        return token_type_ids.unsqueeze(-1).expand(-1, 4).float()


def _segment_ids(embedding: _Embedding, input_ids: torch.Tensor) -> torch.Tensor:
    """The segment ids the embedding actually consumes for ``input_ids``."""
    return embedding.spyre_token_type_embeddings(input_ids)[:, 0].to(torch.int64)


def test_padded_segment_ids_are_zero_extended():
    embedding = _Embedding()
    input_ids = torch.arange(6)

    embedding.set_spyre_token_type_ids(input_ids, torch.tensor([0, 0, 1, 1]))

    assert _segment_ids(embedding, input_ids).tolist() == [0, 0, 1, 1, 0, 0]


def test_buffer_follows_a_shorter_padded_length():
    """A shorter batch must not read the tail of the previous one."""
    embedding = _Embedding()
    embedding.set_spyre_token_type_ids(torch.arange(6), torch.tensor([0, 0, 1, 1, 1, 1]))

    short_ids = torch.arange(4)
    embedding.set_spyre_token_type_ids(short_ids, torch.tensor([0, 1]))

    assert _segment_ids(embedding, short_ids).tolist() == [0, 1, 0, 0]


def test_batch_without_segment_ids_resizes_the_buffer():
    """The regression: the None path used to zero the buffer without resizing it,
    so the next embedding added a previous-length buffer inside the graph."""
    embedding = _Embedding()
    embedding.set_spyre_token_type_ids(torch.arange(6), torch.tensor([0, 0, 1, 1]))

    long_ids = torch.arange(10)
    embedding.set_spyre_token_type_ids(long_ids, None)

    assert embedding.spyre_token_type_ids is not None
    assert embedding.spyre_token_type_ids.shape == long_ids.shape
    assert _segment_ids(embedding, long_ids).tolist() == [0] * 10


def test_single_segment_model_never_allocates_a_buffer():
    """No batch ever supplies segment ids: the embedding falls back to zeros."""
    embedding = _Embedding()
    input_ids = torch.arange(5)

    embedding.set_spyre_token_type_ids(input_ids, None)

    assert embedding.spyre_token_type_ids is None
    assert _segment_ids(embedding, input_ids).tolist() == [0] * 5
