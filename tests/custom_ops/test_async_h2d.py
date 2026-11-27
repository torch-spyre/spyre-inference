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

"""Non-blocking host-to-Spyre uploads through `convert`."""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

POISON = -7


@pytest.mark.parametrize(
    "extra",
    [{"device_layout": object()}, {"dynamic": {0: {"min": 8, "max": 16}}}],
    ids=["layout", "dynamic"],
)
def test_layout_and_dynamic_uploads_do_not_wait(extra, monkeypatch):
    """`convert` asks `Tensor.to` for a non-blocking copy and forwards the Spyre arguments."""
    from spyre_inference.custom_ops import utils

    calls = []

    def record_to(tensor, *args, **kwargs):
        calls.append(kwargs)
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    utils.convert(torch.zeros(8, 4), device="spyre", **extra)
    assert calls == [{"dtype": torch.float32, "non_blocking": True, **extra}]


@pytest.mark.parametrize(
    "entry,dtype",
    [
        ("convert", torch.int32),
        ("convert", torch.float32),
        ("convert", torch.int64),
        ("convert", torch.float16),
        ("layout", torch.int32),
        ("dynamic", torch.int32),
    ],
    ids=lambda v: str(v).removeprefix("torch."),
)
def test_upload_ignores_source_rewrite_after_enqueue(entry, dtype):
    """A caller rewriting its buffer while the upload is queued must not reach the card."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops import utils

    utils.register()
    dev = torch.device("spyre")
    a = torch.randn(2048, 2048, dtype=torch.float16).to(dev)
    mm = torch.compile(lambda x: x @ x, dynamic=False)
    mm(a)
    torch.spyre.synchronize()

    # The flat and row-outermost uploads are dense on the card, the copy torch-spyre submits
    # without converting; that source is the one a DMA would read after `convert` returns.
    # The reserved upload keeps the default layout, as `_to_device_with_page_range` does.
    width = 128 // dtype.itemsize * 2
    for _ in range(20):
        # Small integers: exact in every device format, including fp16's 9-bit mantissa.
        src = (torch.arange(8 * width) % 31 + 40).to(dtype).reshape(8, width)
        want = src.clone()
        for _ in range(40):
            mm(a)
        if entry == "layout":
            dst = utils.convert(
                src, device=dev, device_layout=utils.row_outermost_layout(src.shape, dtype)
            )
        elif entry == "dynamic":
            dst = utils.convert(src, device=dev, dynamic={0: {"min": 8, "max": 8}})
        else:
            dst = utils.convert(src.reshape(-1), device=dev)
        src.fill_(POISON)
        torch.spyre.synchronize()
        assert torch.equal(dst.cpu().reshape(src.shape), want)
