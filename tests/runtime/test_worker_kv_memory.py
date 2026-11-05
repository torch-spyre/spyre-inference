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
(spyre-inference#901), with every block the measured memory pays for. Only the card is
faked: torch.accelerator's memory view, with free bytes the test sets.

The device cost per block comes from upstream's own KV cache grouping, and every layer's
own page counts towards it, as allocate_spyre_kv_caches allocates them."""

import logging
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import request_memory

from spyre_inference.v1.worker import spyre_model_runner
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

GiB = 1 << 30
TOTAL = 96 * GiB
BLOCK_BYTES = 2 * 2 * 4 * 128 * 128  # K+V, fp16, 4 kv heads, head 128, block 128
BATCH_BLOCKS = 4 * (1024 // 128) + 1  # _worker's default max_num_seqs at max_model_len
NUM_LAYERS = 32


def _full(dtype: torch.dtype = torch.float16) -> FullAttentionSpec:
    return FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=dtype)


def _full_layers(dtype: torch.dtype = torch.float16, num_layers: int = NUM_LAYERS) -> dict:
    return {f"layers.{i}.attn": _full(dtype) for i in range(num_layers)}


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

    specs = kv_cache_spec if kv_cache_spec is not None else _full_layers(dtype)
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        max_model_len=max_model_len, multimodal_config=None, supports_kv_cache_grouping=False
    )
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


def test_the_budget_is_the_requested_share_minus_what_loading_consumed(monkeypatch):
    """In whole blocks, as vLLM divides the budget by the bytes per block anyway, and
    past what a full batch needs: the prefix cache keeps the rest for reuse."""
    worker = _worker(monkeypatch, free_after_load=60 * GiB)

    blocks = worker.determine_available_memory() // (NUM_LAYERS * BLOCK_BYTES)

    assert blocks == (worker.requested_memory - 36 * GiB) // (NUM_LAYERS * BLOCK_BYTES)
    assert blocks > BATCH_BLOCKS


def test_an_explicit_kv_cache_memory_bytes_skips_the_measurement(monkeypatch):
    worker = _worker(monkeypatch, free_after_load=0, kv_cache_memory_bytes=3 * GiB)

    assert worker.determine_available_memory() == 3 * GiB


def test_loading_past_the_requested_share_leaves_nothing(monkeypatch):
    """vLLM then fails at init, before any allocation, naming gpu_memory_utilization."""
    worker = _worker(monkeypatch, free_after_load=5 * GiB)

    assert worker.determine_available_memory() <= 0


def test_an_fp8_cache_fits_twice_the_blocks(monkeypatch):
    """The budget is bytes and vLLM divides it by the spec's page size, so a 1-byte
    cache holds twice the tokens in the same memory."""
    blocks = {}
    for dtype in (torch.float16, torch.uint8):
        worker = _worker(monkeypatch, free_after_load=60 * GiB, dtype=dtype)
        page = next(iter(worker.model_runner.get_kv_cache_spec().values())).page_size_bytes
        blocks[dtype] = worker.determine_available_memory() // (NUM_LAYERS * page)

    assert blocks[torch.uint8] == 2 * blocks[torch.float16]


def _pack(tensor_bytes: list[int], free_regions: list[int]) -> bool:
    """Place each tensor in the region with the most room, as the allocator does."""
    for nbytes in sorted(tensor_bytes, reverse=True):
        i = max(range(len(free_regions)), key=free_regions.__getitem__)
        if free_regions[i] < nbytes:
            return False
        free_regions[i] -= nbytes
    return True


@pytest.mark.parametrize("weights_region", [0, 5])
def test_a_shallow_models_kv_tensors_fit_the_16_gib_regions(monkeypatch, weights_region):
    """A tensor never spans two of the card's six 16 GiB regions. Four layers at the
    default budget would make eight 10.75 GiB K/V tensors, of which only six fit."""
    worker = _worker(
        monkeypatch, free_after_load=95 * GiB, kv_cache_spec=_full_layers(num_layers=4)
    )
    worker.cache_config.gpu_memory_utilization = 0.92
    worker.requested_memory = request_memory(worker.init_snapshot, worker.cache_config)
    region = 16 * GiB
    regions = [region] * 6
    regions[weights_region] -= 1 * GiB

    blocks = worker.determine_available_memory() // (4 * BLOCK_BYTES)

    assert blocks * 4 * BLOCK_BYTES < worker.requested_memory - 1 * GiB
    assert _pack([blocks * BLOCK_BYTES // 2] * 8, regions)


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


def test_a_mamba_hybrid_is_sized_by_its_per_layer_device_cost(monkeypatch):
    """vLLM's pool needs one shared page per block for all groups, but Spyre gives every
    layer its own page: the budget converts the measured device bytes into blocks at
    Spyre's cost, so they fit on the card."""
    specs = _mamba_hybrid()
    worker = _worker(
        monkeypatch, free_after_load=20 * GiB, kv_cache_spec=specs, hybrid_kv_cache_manager=True
    )
    mamba_bytes = specs["layers.0.mixer"].real_page_size_bytes
    spyre_bytes_per_block = 6 * mamba_bytes + BLOCK_BYTES

    blocks = worker.determine_available_memory() // BLOCK_BYTES

    measured = worker.requested_memory - (TOTAL - 20 * GiB)
    assert blocks == measured // spyre_bytes_per_block
    assert blocks * spyre_bytes_per_block <= measured


@pytest.mark.parametrize(("spare_blocks", "warns"), [(-1, True), (0, False)])
def test_requests_queue_for_pages_only_below_the_batch_area(
    monkeypatch, caplog, spare_blocks, warns
):
    worker = _worker(monkeypatch, free_after_load=TOTAL)
    worker.requested_memory = (BATCH_BLOCKS + spare_blocks) * NUM_LAYERS * BLOCK_BYTES

    with caplog.at_level(logging.WARNING, logger="spyre_inference.v1.worker.spyre_worker"):
        caplog.clear()
        blocks = worker.determine_available_memory() // (NUM_LAYERS * BLOCK_BYTES)

    assert blocks == BATCH_BLOCKS + spare_blocks
    assert any("queue for pages" in r.getMessage() for r in caplog.records) == warns


@pytest.mark.parametrize(("override", "warns"), [(BATCH_BLOCKS - 1, True), (BATCH_BLOCKS, False)])
def test_the_batch_area_warning_checks_num_gpu_blocks_override(
    monkeypatch, caplog, override, warns
):
    """vLLM replaces the measured block count with the override later, so the override,
    not the card, decides whether a full batch fits."""
    worker = _worker(monkeypatch, free_after_load=TOTAL)
    worker.requested_memory = 2 * BATCH_BLOCKS * NUM_LAYERS * BLOCK_BYTES
    worker.cache_config.num_gpu_blocks_override = override

    with caplog.at_level(logging.WARNING, logger="spyre_inference.v1.worker.spyre_worker"):
        caplog.clear()
        worker.determine_available_memory()

    assert any("queue for pages" in r.getMessage() for r in caplog.records) == warns


def test_sizing_a_sliding_window_model_logs_no_vllm_warning(monkeypatch, caplog):
    """vLLM warns once that the hybrid manager is off for such a model, in
    get_kv_cache_configs; sizing must not repeat it on every worker."""
    specs = {
        "layers.0.attn": _full(),
        "layers.1.attn": SlidingWindowSpec(
            block_size=128, num_kv_heads=4, head_size=128, dtype=torch.float16, sliding_window=512
        ),
    }
    worker = _worker(monkeypatch, free_after_load=60 * GiB, kv_cache_spec=specs)
    kv_logger = logging.getLogger("vllm.v1.core.kv_cache_planning")
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)

    with caplog.at_level(logging.WARNING, logger="vllm.v1.core.kv_cache_planning"):
        caplog.clear()
        assert worker.determine_available_memory() > 0

    assert not [r for r in caplog.records if r.name == kv_logger.name]
    assert not kv_logger.disabled


def test_sizing_allocates_the_staging_buffers_and_runs_no_forward(monkeypatch):
    """On Spyre the first forward is the trace; run before the KV cache exists, it would
    trace no KV write and warmup would retrace every block. The staging buffers are
    allocated inside the measurement instead: they scale with max_num_batched_tokens, and
    allocated after the KV cache took the measured memory they would not fit."""

    def forward(*args, **kwargs):
        raise AssertionError("determine_available_memory ran a forward")

    staged = []
    monkeypatch.setattr(GPUModelRunner, "profile_run", forward)
    monkeypatch.setattr(TorchSpyreModelRunner, "_dummy_run", forward)
    monkeypatch.setattr(
        spyre_model_runner, "allocate_staging_buffers", lambda *args: staged.append(args)
    )
    worker = _worker(monkeypatch, free_after_load=60 * GiB)
    runner = object.__new__(TorchSpyreModelRunner)
    runner.model_memory_usage = 0
    runner.get_kv_cache_spec = worker.model_runner.get_kv_cache_spec
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner._spyre_device = torch.device("cpu")
    worker.model_runner = runner

    assert worker.determine_available_memory() > 0
    assert staged == [({}, torch.device("cpu"))]
