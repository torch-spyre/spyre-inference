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

"""TorchSpyreWorker sizes the KV cache through upstream's Worker.determine_available_memory
(spyre-inference#901), capped at what a full batch at max_model_len can use. Only the
card is faked: torch.accelerator's memory view, with free bytes the test sets.

The cap and the device cost come from upstream's own KV cache grouping, so a
sliding-window or Mamba group counts what it holds per request, and every layer's own
page counts towards the device bytes, as allocate_spyre_kv_caches allocates them."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.utils import request_memory

from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

GiB = 1 << 30
TOTAL = 96 * GiB
BLOCK_BYTES = 2 * 2 * 4 * 128 * 128  # K+V, fp16, 4 kv heads, head 128, block 128


def _full(dtype: torch.dtype = torch.float16) -> FullAttentionSpec:
    return FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=dtype)


def _two_full_layers(dtype: torch.dtype = torch.float16) -> dict:
    return {f"layers.{i}.attn": _full(dtype) for i in range(2)}


def _worker(
    monkeypatch,
    *,
    free_after_load: int,
    max_num_seqs: int = 4,
    max_model_len: int = 1024,
    kv_cache_memory_bytes: int | None = None,
    dtype: torch.dtype = torch.float16,
    kv_cache_spec: dict | None = None,
    hybrid_kv_cache_manager: bool = False,
) -> TorchSpyreWorker:
    """A worker that started on an empty card and whose model now leaves free_after_load."""
    card = SimpleNamespace(free=TOTAL)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *_: (card.free, TOTAL))
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda *_: {})
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda *_: None)

    specs = kv_cache_spec if kv_cache_spec is not None else _two_full_layers(dtype)
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(max_model_len=max_model_len, multimodal_config=None)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.block_size = 128
    vllm_config.cache_config.kv_cache_memory_bytes = kv_cache_memory_bytes
    vllm_config.cache_config.gpu_memory_utilization = 0.9
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = not hybrid_kv_cache_manager
    worker = object.__new__(TorchSpyreWorker)
    worker.vllm_config = vllm_config
    worker.parallel_config = SimpleNamespace(data_parallel_size=1)
    worker.cache_config = vllm_config.cache_config
    worker.model_config = vllm_config.model_config
    worker.scheduler_config = vllm_config.scheduler_config
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: dict(specs),
        profile_run=lambda **_: None,
        model_memory_usage=0,
    )
    worker.init_snapshot = MemorySnapshot(device=torch.device("cpu"))
    worker.requested_memory = request_memory(worker.init_snapshot, worker.cache_config)
    card.free = free_after_load
    return worker


def test_a_short_batch_is_capped_at_the_batch_area(monkeypatch):
    """Memory for far more pages than max_num_seqs × max_model_len is not handed out:
    the single block pool could never allocate them."""
    worker = _worker(monkeypatch, free_after_load=80 * GiB)

    budget = worker.determine_available_memory()

    assert budget // (2 * BLOCK_BYTES) == 4 * (1024 // 128) + 1


def test_the_budget_is_the_requested_share_minus_what_loading_consumed(monkeypatch):
    """In whole blocks: vLLM divides the budget by the bytes per block anyway."""
    worker = _worker(monkeypatch, free_after_load=60 * GiB, max_num_seqs=1024, max_model_len=131072)

    blocks = worker.determine_available_memory() // (2 * BLOCK_BYTES)

    assert blocks == (worker.requested_memory - 36 * GiB) // (2 * BLOCK_BYTES)


def test_an_explicit_kv_cache_memory_bytes_skips_the_measurement(monkeypatch):
    worker = _worker(
        monkeypatch,
        free_after_load=0,
        max_num_seqs=1024,
        max_model_len=131072,
        kv_cache_memory_bytes=3 * GiB,
    )

    assert worker.determine_available_memory() == 3 * GiB


def test_an_explicit_kv_cache_memory_bytes_is_still_capped(monkeypatch):
    """A leftover VLLM_CPU_KVCACHE_SPACE=64 must not turn into 64 GiB of pages."""
    worker = _worker(monkeypatch, free_after_load=0, kv_cache_memory_bytes=64 * GiB)

    assert worker.determine_available_memory() // (2 * BLOCK_BYTES) == 4 * (1024 // 128) + 1


def test_loading_past_the_requested_share_leaves_nothing(monkeypatch):
    """vLLM then fails at init, before any allocation, naming gpu_memory_utilization."""
    worker = _worker(monkeypatch, free_after_load=5 * GiB)

    assert worker.determine_available_memory() <= 0


def test_an_fp8_cache_fits_twice_the_blocks(monkeypatch):
    """The budget is bytes and vLLM divides it by the spec's page size, so a 1-byte
    cache holds twice the tokens in the same memory."""
    blocks = {}
    for dtype in (torch.float16, torch.uint8):
        worker = _worker(
            monkeypatch,
            free_after_load=60 * GiB,
            max_num_seqs=1024,
            max_model_len=131072,
            dtype=dtype,
        )
        page = next(iter(worker.model_runner.get_kv_cache_spec().values())).page_size_bytes
        blocks[dtype] = worker.determine_available_memory() // (2 * page)

    assert blocks[torch.uint8] == 2 * blocks[torch.float16]


def _swa(window: int = 512) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=128, num_kv_heads=4, head_size=128, dtype=torch.float16, sliding_window=window
    )


def test_sliding_window_layers_count_at_full_length_with_one_pool(monkeypatch):
    """With the hybrid manager off, vLLM promotes sliding-window layers to full
    attention, so each holds max_model_len per request like a full layer."""
    specs = {**{f"layers.{i}.attn": _swa() for i in range(5)}, "layers.5.attn": _full()}
    worker = _worker(monkeypatch, free_after_load=80 * GiB, kv_cache_spec=specs)

    budget = worker.determine_available_memory()

    assert budget // (6 * BLOCK_BYTES) == 4 * (1024 // 128) + 1


def test_sliding_window_groups_reserve_their_window_with_the_hybrid_manager(monkeypatch):
    """With per-type groups, a sliding-window group needs only its window per request
    (upstream's admission bound), not max_model_len: the cap shrinks accordingly."""
    specs = {**{f"layers.{i}.attn": _swa() for i in range(5)}, "layers.5.attn": _full()}
    max_model_len = 8192
    worker = _worker(
        monkeypatch,
        free_after_load=80 * GiB,
        max_model_len=max_model_len,
        kv_cache_spec=specs,
        hybrid_kv_cache_manager=True,
    )

    blocks = worker.determine_available_memory() // BLOCK_BYTES

    # One full-attention group plus five single-layer sliding-window groups.
    full = max_model_len // 128
    window = SlidingWindowSpec.max_admission_blocks_per_request(
        _swa(),
        max_in_flight_tokens=worker.vllm_config.max_in_flight_tokens,
        max_model_len=max_model_len,
    )
    assert window < full
    assert blocks == 4 * (full + 5 * window) + 1


def _mamba_hybrid(num_mamba: int = 6) -> dict:
    page = _full().page_size_bytes
    mamba = MambaSpec(
        shapes=((3, 1024), (8, 64, 128)),
        dtypes=(torch.float16, torch.float16),
        block_size=8192,
        page_size_padded=page,
        mamba_cache_mode="none",
    )
    return {**{f"layers.{i}.mixer": mamba for i in range(num_mamba)}, "layers.99.attn": _full()}


def test_a_mamba_hybrid_reserves_one_state_block_per_group(monkeypatch):
    """Each Mamba group holds one block per request, on top of the attention pages:
    counting attention pages only would leave room for no full-length request."""
    max_model_len = 8192
    worker = _worker(
        monkeypatch,
        free_after_load=80 * GiB,
        max_model_len=max_model_len,
        kv_cache_spec=_mamba_hybrid(),
        hybrid_kv_cache_manager=True,
    )

    blocks = worker.determine_available_memory() // BLOCK_BYTES

    assert blocks == 4 * (max_model_len // 128 + 6) + 1


def test_a_mamba_hybrid_is_sized_by_its_per_layer_device_cost(monkeypatch):
    """vLLM's pool needs one shared page per block for all groups, but Spyre gives every
    layer its own page: the budget converts the measured device bytes into blocks at
    Spyre's cost, so they fit on the card."""
    specs = _mamba_hybrid()
    worker = _worker(
        monkeypatch,
        free_after_load=40 * GiB,
        max_num_seqs=1024,
        max_model_len=131072,
        kv_cache_spec=specs,
        hybrid_kv_cache_manager=True,
    )
    mamba_bytes = specs["layers.0.mixer"].real_page_size_bytes
    spyre_bytes_per_block = 6 * mamba_bytes + BLOCK_BYTES

    blocks = worker.determine_available_memory() // BLOCK_BYTES

    measured = worker.requested_memory - (TOTAL - 40 * GiB)
    assert blocks == measured // spyre_bytes_per_block
    assert blocks * spyre_bytes_per_block <= measured


@pytest.mark.parametrize("hybrid_kv_cache_manager", [False, True])
def test_a_homogeneous_model_is_capped_the_same_with_either_manager(
    monkeypatch, hybrid_kv_cache_manager
):
    worker = _worker(
        monkeypatch, free_after_load=80 * GiB, hybrid_kv_cache_manager=hybrid_kv_cache_manager
    )

    assert worker.determine_available_memory() // (2 * BLOCK_BYTES) == 4 * (1024 // 128) + 1
