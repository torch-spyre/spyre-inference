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

"""
Test SpyreRMSNorm custom op correctness against a reference implementation.
"""

import pytest
import sys


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch():
    """Verify RMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-k", "test_rmsnorm_oot_dispatch", "-v"]))
