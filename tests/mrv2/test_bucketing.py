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

"""SpyreBucketManager.dispatch pads every token count to a body bucket and
never selects a cudagraph."""

import bisect
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils

from spyre_inference.v1.worker.spyre.bucketing import SpyreBucketManager

_DECODER_SIZES = [1, 2, 4, 8, 16, 32, 64, 512]


@pytest.fixture(autouse=True)
def _single_pp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    # CudaGraphManager reads the PP group at construction; no process group here.
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )


def _config(
    compile_sizes: list[int] | None,
    *,
    runner_type: str = "generate",
    enforce_eager: bool = False,
    max_num_seqs: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type=runner_type, enforce_eager=enforce_eager),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=1),
        compilation_config=SimpleNamespace(
            compile_sizes=compile_sizes,
            # A GPU-style config: the manager must ignore it, not capture these.
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_cudagraph_capture_size=8,
        ),
        speculative_config=None,
    )


def _manager(config: SimpleNamespace, lora_capture_cases: list[int] | None = None):
    return SpyreBucketManager(
        config,  # ty: ignore[invalid-argument-type]
        torch.device("cpu"),
        decode_query_len=1,
        lora_capture_cases=lora_capture_cases,
    )


def test_every_decoder_token_count_pads_to_its_bucket() -> None:
    manager = _manager(_config(_DECODER_SIZES))
    for num_tokens in range(1, _DECODER_SIZES[-1] + 1):
        desc = manager.dispatch(
            num_reqs=min(num_tokens, 32),
            num_tokens=num_tokens,
            uniform_token_count=None,
            num_active_loras=0,
        )
        expected = _DECODER_SIZES[bisect.bisect_left(_DECODER_SIZES, num_tokens)]
        assert desc.num_tokens == expected, num_tokens
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert desc.num_reqs == min(num_tokens, 32)


def test_uniform_decode_batches_still_take_no_cudagraph() -> None:
    manager = _manager(_config(_DECODER_SIZES))
    for num_reqs in (1, 3, 8, 17):
        desc = manager.dispatch(num_reqs, num_reqs, uniform_token_count=1, num_active_loras=0)
        assert desc.cg_mode == CUDAGraphMode.NONE
        assert desc.uniform_token_count is None
    assert not manager.needs_capture()


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_pooling_pads_every_count_to_the_budget_even_eager(enforce_eager: bool) -> None:
    """The rectangular encoder path needs the body at exactly the budget."""
    budget = 1024
    manager = _manager(_config([budget], runner_type="pooling", enforce_eager=enforce_eager))
    for num_tokens in (1, 63, 64, 500, budget):
        desc = manager.dispatch(4, num_tokens, None, 0)
        assert desc.num_tokens == budget


@pytest.mark.parametrize(
    "config",
    [
        _config(_DECODER_SIZES, enforce_eager=True),
        _config([]),
        _config([], runner_type="pooling"),
    ],
    ids=["eager-decoder", "no-compile-sizes", "pooling-opted-out"],
)
def test_no_bucketer_means_no_padding(config: SimpleNamespace) -> None:
    manager = _manager(config)
    assert manager.shape_bucketer is None
    for num_tokens in (1, 7, 100):
        desc = manager.dispatch(1, num_tokens, None, 0)
        assert (desc.num_tokens, desc.cg_mode) == (num_tokens, CUDAGraphMode.NONE)


def test_empty_and_oversized_batches_stay_unpadded() -> None:
    manager = _manager(_config(_DECODER_SIZES))
    assert manager.bucket(0) == 0
    assert manager.bucket(_DECODER_SIZES[-1] + 1) == _DECODER_SIZES[-1] + 1


def test_later_dispatch_kwargs_are_accepted() -> None:
    """Spyre has no DBO, so only num_ubatches=1 is accepted."""
    manager = _manager(_config(_DECODER_SIZES))
    assert manager.dispatch(2, 3, None, 0, max_query_len=2, num_ubatches=1).num_tokens == 4
    with pytest.raises(NotImplementedError, match="DBO"):
        manager.dispatch(2, 3, None, 0, num_ubatches=2)


def test_active_loras_map_to_a_capture_case() -> None:
    manager = _manager(_config(_DECODER_SIZES), lora_capture_cases=[0, 2, 4])
    assert manager.dispatch(1, 1, None, num_active_loras=3).num_active_loras == 4
    assert manager.dispatch(1, 1, None, num_active_loras=0).num_active_loras == 0


def test_from_runner_reads_the_runner() -> None:
    runner = SimpleNamespace(
        vllm_config=_config(_DECODER_SIZES),
        device=torch.device("cpu"),
        decode_query_len=1,
        lora_capture_cases=[0],
    )
    manager = SpyreBucketManager.from_runner(runner)  # ty: ignore[invalid-argument-type]
    assert manager.bucket(5) == 8
    assert manager.device == torch.device("cpu")
