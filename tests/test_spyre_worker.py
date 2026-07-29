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

"""Unit tests for TorchSpyreWorker (spyre_inference/v1/worker/spyre_worker.py).

Tests target the environment variable propagation, memory pool context
override, and `determine_available_memory` logic — all of which are
testable on CPU without the Spyre runtime.
"""

import os
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestTorchSpyreWorkerMemoryPool:
    """Tests for _maybe_get_memory_pool_context override."""

    def test_returns_nullcontext(self):
        """Spyre workers don't use host-side cumem allocators."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        ctx = worker._maybe_get_memory_pool_context("test_tag")
        assert isinstance(ctx, nullcontext)

    def test_nullcontext_is_usable(self):
        """The returned context manager is usable in a with-statement."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        ctx = worker._maybe_get_memory_pool_context("weights")
        with ctx:
            pass  # should not raise


class TestTorchSpyreWorkerDetermineMemory:
    """Tests for determine_available_memory."""

    def test_returns_kv_cache_memory_bytes(self):
        """determine_available_memory returns the configured kv_cache_memory_bytes."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        # Simulate the cache_config attribute
        mock_cache_config = MagicMock()
        mock_cache_config.kv_cache_memory_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
        worker.cache_config = mock_cache_config

        result = worker.determine_available_memory()
        assert result == 4 * 1024 * 1024 * 1024

    def test_raises_when_kv_cache_memory_bytes_is_none(self):
        """determine_available_memory asserts kv_cache_memory_bytes is not None."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        mock_cache_config = MagicMock()
        mock_cache_config.kv_cache_memory_bytes = None
        worker.cache_config = mock_cache_config

        with pytest.raises(AssertionError):
            worker.determine_available_memory()


class TestTorchSpyreWorkerInitDevice:
    """Tests for init_device environment variable propagation.

    These test the env-var setup logic without actually calling torch_spyre
    or distributed init.
    """

    def test_env_vars_set_from_config(self):
        """init_device sets RANK/WORLD_SIZE/LOCAL_RANK/LOCAL_WORLD_SIZE."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.rank = 1
        worker.local_rank = 1

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.world_size = 4
        worker.vllm_config = mock_vllm_config
        worker.model_config = MagicMock()
        worker.model_config.seed = 42
        worker.distributed_init_method = "tcp://127.0.0.1:12345"

        # Patch all external deps to isolate the env var logic
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spyre_inference.v1.worker.spyre_worker.register_all"),
            patch("spyre_inference.v1.worker.spyre_worker.init_worker_distributed_environment"),
            patch("spyre_inference.v1.worker.spyre_worker.set_random_seed"),
            patch("spyre_inference.v1.worker.spyre_worker.TorchSpyreModelRunner"),
            patch.dict("sys.modules", {"torch_spyre": MagicMock()}),
            patch("torch.spyre", create=True) as mock_spyre,
        ):
            mock_spyre.set_device = MagicMock()
            worker.init_device()

            assert os.environ["RANK"] == "1"
            assert os.environ["WORLD_SIZE"] == "4"
            assert os.environ["LOCAL_RANK"] == "1"
            assert os.environ["LOCAL_WORLD_SIZE"] == "4"

    def test_env_vars_use_setdefault(self):
        """init_device uses setdefault — does not override torchrun-supplied values."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.rank = 1
        worker.local_rank = 1

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.world_size = 4
        worker.vllm_config = mock_vllm_config
        worker.model_config = MagicMock()
        worker.model_config.seed = 42
        worker.distributed_init_method = "tcp://127.0.0.1:12345"

        # Pre-set env vars as torchrun would
        with (
            patch.dict(os.environ, {"RANK": "7", "WORLD_SIZE": "8"}, clear=True),
            patch("spyre_inference.v1.worker.spyre_worker.register_all"),
            patch("spyre_inference.v1.worker.spyre_worker.init_worker_distributed_environment"),
            patch("spyre_inference.v1.worker.spyre_worker.set_random_seed"),
            patch("spyre_inference.v1.worker.spyre_worker.TorchSpyreModelRunner"),
            patch.dict("sys.modules", {"torch_spyre": MagicMock()}),
            patch("torch.spyre", create=True) as mock_spyre,
        ):
            mock_spyre.set_device = MagicMock()
            worker.init_device()

            # Pre-existing values should NOT be overwritten
            assert os.environ["RANK"] == "7"
            assert os.environ["WORLD_SIZE"] == "8"
            # Not pre-set → should be populated
            assert os.environ["LOCAL_RANK"] == "1"
            assert os.environ["LOCAL_WORLD_SIZE"] == "4"

    def test_sets_spyre_device(self):
        """init_device calls torch.spyre.set_device(local_rank)."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.rank = 0
        worker.local_rank = 2

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.world_size = 4
        worker.vllm_config = mock_vllm_config
        worker.model_config = MagicMock()
        worker.model_config.seed = 42
        worker.distributed_init_method = "tcp://127.0.0.1:12345"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spyre_inference.v1.worker.spyre_worker.register_all"),
            patch("spyre_inference.v1.worker.spyre_worker.init_worker_distributed_environment"),
            patch("spyre_inference.v1.worker.spyre_worker.set_random_seed"),
            patch("spyre_inference.v1.worker.spyre_worker.TorchSpyreModelRunner"),
            patch.dict("sys.modules", {"torch_spyre": MagicMock()}),
            patch("torch.spyre", create=True) as mock_spyre,
        ):
            mock_spyre.set_device = MagicMock()
            worker.init_device()
            mock_spyre.set_device.assert_called_once_with(2)

    def test_calls_register_all(self):
        """init_device registers all custom ops before model loading."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.rank = 0
        worker.local_rank = 0

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.world_size = 1
        worker.vllm_config = mock_vllm_config
        worker.model_config = MagicMock()
        worker.model_config.seed = 42
        worker.distributed_init_method = "tcp://127.0.0.1:12345"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "spyre_inference.v1.worker.spyre_worker.register_all"
            ) as mock_register,
            patch("spyre_inference.v1.worker.spyre_worker.init_worker_distributed_environment"),
            patch("spyre_inference.v1.worker.spyre_worker.set_random_seed"),
            patch("spyre_inference.v1.worker.spyre_worker.TorchSpyreModelRunner"),
            patch.dict("sys.modules", {"torch_spyre": MagicMock()}),
            patch("torch.spyre", create=True) as mock_spyre,
        ):
            mock_spyre.set_device = MagicMock()
            worker.init_device()
            mock_register.assert_called_once()

    def test_constructs_model_runner_with_spyre_device(self):
        """init_device creates TorchSpyreModelRunner with torch.device('spyre')."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.rank = 0
        worker.local_rank = 0

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config.world_size = 1
        worker.vllm_config = mock_vllm_config
        worker.model_config = MagicMock()
        worker.model_config.seed = 42
        worker.distributed_init_method = "tcp://127.0.0.1:12345"

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spyre_inference.v1.worker.spyre_worker.register_all"),
            patch("spyre_inference.v1.worker.spyre_worker.init_worker_distributed_environment"),
            patch("spyre_inference.v1.worker.spyre_worker.set_random_seed"),
            patch(
                "spyre_inference.v1.worker.spyre_worker.TorchSpyreModelRunner"
            ) as mock_runner_cls,
            patch.dict("sys.modules", {"torch_spyre": MagicMock()}),
            patch("torch.spyre", create=True) as mock_spyre,
        ):
            mock_spyre.set_device = MagicMock()
            worker.init_device()

            mock_runner_cls.assert_called_once_with(
                mock_vllm_config,
                torch.device("spyre"),
            )


class TestTorchSpyreWorkerSleepWake:
    """Tests for sleep/wake_up (no-op stubs)."""

    def test_sleep_is_noop(self):
        """sleep() does not raise."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.sleep(level=1)  # should not raise

    def test_wake_up_is_noop(self):
        """wake_up() does not raise."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.wake_up(tags=["model"])  # should not raise

    def test_wake_up_none_tags(self):
        """wake_up(tags=None) does not raise."""
        from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

        worker = TorchSpyreWorker.__new__(TorchSpyreWorker)
        worker.wake_up(tags=None)  # should not raise
