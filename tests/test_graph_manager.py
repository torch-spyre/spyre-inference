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

"""Unit tests for SpyreGraphManager bucket dispatch."""

import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from spyre_inference.v1.worker.spyre_graph_manager import (
    SpyreGraphManager,
)


@pytest.fixture()
def mock_vllm_config():
    """Create a minimal VllmConfig mock with compile_sizes."""
    config = MagicMock()
    config.compilation_config.compile_sizes = [1, 2, 4, 8, 16]
    return config


@pytest.fixture()
def graph_manager(mock_vllm_config):
    return SpyreGraphManager(mock_vllm_config)


class TestFindBucket:
    def test_exact_match(self, graph_manager):
        assert graph_manager.find_bucket(8) == 8

    def test_rounds_up_to_next_bucket(self, graph_manager):
        assert graph_manager.find_bucket(3) == 4
        assert graph_manager.find_bucket(5) == 8
        assert graph_manager.find_bucket(9) == 16

    def test_smallest_token_count(self, graph_manager):
        assert graph_manager.find_bucket(1) == 1

    def test_exceeds_max_returns_none(self, graph_manager):
        assert graph_manager.find_bucket(17) is None
        assert graph_manager.find_bucket(100) is None

    def test_zero_tokens(self, graph_manager):
        assert graph_manager.find_bucket(0) == 1


class TestDispatch:
    def test_returns_descriptor_with_padding(self, graph_manager):
        desc = graph_manager.dispatch(5)
        assert desc is not None
        assert desc.actual_num_tokens == 5
        assert desc.padded_num_tokens == 8

    def test_exact_match_no_padding(self, graph_manager):
        desc = graph_manager.dispatch(4)
        assert desc is not None
        assert desc.actual_num_tokens == 4
        assert desc.padded_num_tokens == 4

    def test_exceeds_max_returns_none(self, graph_manager):
        assert graph_manager.dispatch(20) is None

    def test_descriptor_is_frozen(self, graph_manager):
        desc = graph_manager.dispatch(3)
        with pytest.raises(FrozenInstanceError):
            desc.actual_num_tokens = 10


class TestManagerState:
    def test_initial_state_not_captured(self, graph_manager):
        assert not graph_manager.is_captured

    def test_mark_captured(self, graph_manager):
        graph_manager.mark_captured()
        assert graph_manager.is_captured

    def test_bucket_sizes_sorted(self, graph_manager):
        assert graph_manager.bucket_sizes == [1, 2, 4, 8, 16]

    def test_max_bucket_size(self, graph_manager):
        assert graph_manager._max_bucket_size == 16


class TestEdgeCases:
    def test_empty_compile_sizes(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = []
        mgr = SpyreGraphManager(config)
        assert mgr.bucket_sizes == []
        assert mgr._max_bucket_size == 0
        assert mgr.find_bucket(1) is None
        assert mgr.dispatch(1) is None

    def test_single_bucket(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [8]
        mgr = SpyreGraphManager(config)
        assert mgr.find_bucket(1) == 8
        assert mgr.find_bucket(8) == 8
        assert mgr.find_bucket(9) is None

    def test_unsorted_input_gets_sorted(self):
        config = MagicMock()
        config.compilation_config.compile_sizes = [16, 2, 8, 1, 4]
        mgr = SpyreGraphManager(config)
        assert mgr.bucket_sizes == [1, 2, 4, 8, 16]
