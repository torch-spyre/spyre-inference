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

from spyre_inference.custom_ops.utils import _async_source

POISON = -7


@pytest.mark.parametrize(
    "dtype,owned,copied",
    [
        (torch.int32, False, True),
        (torch.float32, False, True),
        (torch.int32, True, False),
        (torch.int64, False, False),
        (torch.float16, False, False),
    ],
)
def test_async_source_copies_only_unstaged_caller_buffers(dtype, owned, copied):
    src = torch.zeros(8, dtype=dtype)
    assert (_async_source(src, owned).data_ptr() != src.data_ptr()) == copied


@pytest.mark.parametrize(
    "source,device,dtype",
    [
        (torch.zeros(4, dtype=torch.int32), "cpu", None),
        (torch.zeros(4, dtype=torch.float32), "spyre", torch.float16),
        (torch.zeros(8, dtype=torch.int32)[::2], "spyre", None),
    ],
    ids=["cpu-target", "dtype-change", "strided"],
)
def test_layout_convert_falls_back_to_tensor_to(source, device, dtype, monkeypatch):
    from spyre_inference.custom_ops import utils

    def no_async(*args, **kwargs):
        raise AssertionError("must not take the async layout upload")

    layouts = []

    def record_to(tensor, *args, **kwargs):
        layouts.append(kwargs.get("device_layout"))
        return tensor

    monkeypatch.setattr(utils, "_to_layout_async", no_async)
    monkeypatch.setattr(torch.Tensor, "to", record_to)
    sentinel = object()
    utils.convert(source, device=device, dtype=dtype, device_layout=sentinel)
    assert layouts == [sentinel]


@pytest.mark.parametrize(
    "layout,dtype",
    [
        (False, torch.int32),
        (False, torch.float32),
        # Passed through uncloned: these fail if torch-spyre stops staging them.
        (False, torch.int64),
        (False, torch.float16),
        (True, torch.int32),
    ],
    ids=["convert-int32", "convert-fp32", "convert-int64", "convert-fp16", "layout-int32"],
)
def test_upload_ignores_source_rewrite_after_enqueue(layout, dtype):
    """A caller rewriting its buffer while the upload is queued must not reach the card."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from spyre_inference.custom_ops import utils
    from spyre_inference.v1.attention.ops.layout import temporary_chunk_major_page_index_layout

    utils.register()
    dev = torch.device("spyre")
    a = torch.randn(2048, 2048, dtype=torch.float16).to(dev)
    mm = torch.compile(lambda x: x @ x, dynamic=False)
    mm(a)
    torch.spyre.synchronize()

    for _ in range(20):
        # Two whole sticks: a same-format upload of this size skips torch-spyre's staging.
        n = 8 if layout else 128 // dtype.itemsize * 2
        src = torch.arange(n, dtype=dtype) + 40
        want = src.clone()
        for _ in range(40):
            mm(a)
        if layout:
            dst = utils.convert(
                src.reshape(2, 4, 1),
                device=dev,
                device_layout=temporary_chunk_major_page_index_layout(2, 4),
            )
        else:
            dst = utils.convert(src, device=dev)
        src.fill_(POISON)
        torch.spyre.synchronize()
        assert torch.equal(dst.cpu().reshape(-1), want)
