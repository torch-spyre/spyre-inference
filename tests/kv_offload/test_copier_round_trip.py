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

"""Byte-exact round-trip tests for ``SpyreKvDmaCopier``.

Offloads a KV page from the Spyre device into a shared host-pool slot (attached by
name), mutates the device page, reloads from the slot, and asserts the reloaded
bytes are identical to the original. Covers real micro-g3.3-8b KV-page geometry
including a multi-MB page, and asserts the copy methods never allocate.

Runs against the real Spyre device when available; otherwise skips.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from transformers import AutoConfig

from spyre_inference.v1.kv_offload import SpyreKvDmaCopier

pytestmark = pytest.mark.compile

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"


def _round_trip(copier: SpyreKvDmaCopier, original: torch.Tensor, pool_name: str) -> None:
    """Offload ``original`` to a fresh pool slot, then reload and assert byte-exact."""
    slot_bytes = copier.slot_bytes_for(original)
    pool = copier.create_or_attach_pool(pool_name, num_slots=1, slot_bytes=slot_bytes)
    slot_id = 0

    # D2H: offload the device page into the shared host pool slot.
    copier.copy_d2h(original, pool, slot_id)

    # H2D: reload into a separate zeroed device tensor. Starting from zeros means a
    # no-op reload would leave zeros and fail the assert, so the data must genuinely
    # come back from the host slot.
    reload = torch.zeros_like(original)
    copier.copy_h2d(reload, pool, slot_id)

    assert torch.equal(reload.to("cpu"), original.to("cpu"))


def test_round_trip_small_buffer() -> None:
    if not spyre_available():
        pytest.skip("Spyre device not available")

    torch.manual_seed(0)
    original = torch.randn(10, device="spyre", dtype=torch.float16)
    _round_trip(SpyreKvDmaCopier(), original, "M1-F2-kv-offload-small")


def test_round_trip_real_model_small_slot() -> None:
    """Real micro-g3.3-8b KV geometry, 16-token page (~64 KiB)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    cfg = AutoConfig.from_pretrained(MODEL)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    # 2 (K & V) x block_size x num_key_value_heads x head_dim
    shape = (2, 16, cfg.num_key_value_heads, head_dim)

    torch.manual_seed(0)
    original = torch.randn(shape, device="spyre", dtype=torch.float16)
    _round_trip(SpyreKvDmaCopier(), original, "M1-F2-kv-offload-real-small")


def test_round_trip_real_model_large_slot() -> None:
    """Real micro-g3.3-8b KV geometry, 1024-token page (multi-MB, ~4 MiB)."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    cfg = AutoConfig.from_pretrained(MODEL)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    shape = (2, 1024, cfg.num_key_value_heads, head_dim)

    torch.manual_seed(0)
    original = torch.randn(shape, device="spyre", dtype=torch.float16)
    _round_trip(SpyreKvDmaCopier(), original, "M1-F2-kv-offload-real-large")


def test_copy_never_allocates(monkeypatch: pytest.MonkeyPatch) -> None:
    """copy_d2h / copy_h2d must not allocate device or host tensors."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    copier = SpyreKvDmaCopier()
    original = torch.randn(10, device="spyre", dtype=torch.float16)
    slot_bytes = copier.slot_bytes_for(original)
    pool = copier.create_or_attach_pool(
        "M1-F2-kv-offload-noalloc", num_slots=1, slot_bytes=slot_bytes
    )
    reload = torch.zeros_like(original)

    # Any tensor-creating entry point firing during a copy is a failure: the copier
    # must operate purely on the caller-provided device tensor and pool slot. Guard
    # the constructors it could plausibly reach for a destination buffer, then run
    # only the two copies under the guards.
    for fn in ("empty", "zeros", "empty_like", "zeros_like", "randn", "tensor"):

        def _forbidden(*args, _fn=fn, **kwargs):
            raise AssertionError(f"copier allocated via torch.{_fn}")

        monkeypatch.setattr(torch, fn, _forbidden)

    copier.copy_d2h(original, pool, 0)
    copier.copy_h2d(reload, pool, 0)

    monkeypatch.undo()
    assert torch.equal(reload.to("cpu"), original.to("cpu"))
