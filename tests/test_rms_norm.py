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

"""Test SpyreRMSNorm OOT registration.

Numerical correctness is covered end-to-end (tests/test_vllm_spyre_next.py):
forward_oot delegates to upstream forward_native, whose fp32 upcast lives in the
opaque ``ir.ops.rms_norm`` op and is only lowered under the worker's Spyre
compile context, so it cannot be reproduced by a standalone forward_oot call.
"""

import pytest


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch():
    """RMSNorm OOT registration: class swap + forward_oot dispatch."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    assert isinstance(layer, SpyreRMSNorm)
    assert layer._forward_method == layer.forward_oot
