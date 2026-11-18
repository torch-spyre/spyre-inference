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

"""The attention kernels reject positional arguments, so a swapped pair cannot slip through."""

import pytest

from spyre_inference.v1.attention.ops.batched_decode import (
    _merge_entry_local,
    batched_decode_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_decode import (
    page_attn_decode_kernel,
)
from spyre_inference.v1.attention.ops.page_attn_prefill import (
    page_attn_prefill_kernel,
)
from spyre_inference.v1.attention.ops.reshape_and_cache import (
    reshape_and_cache_kernel,
)


@pytest.mark.parametrize(
    "kernel",
    [
        page_attn_prefill_kernel,
        page_attn_decode_kernel,
        batched_decode_kernel,
        _merge_entry_local,
    ],
)
def test_kernel_rejects_positional_arguments(kernel):
    with pytest.raises(TypeError, match="positional argument"):
        kernel(None)


def test_store_takes_kv_scales_by_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        reshape_and_cache_kernel(None, None, None, None, None, 0.25, 0.5)
