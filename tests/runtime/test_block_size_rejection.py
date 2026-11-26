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

"""The power-of-two half of ``TorchSpyrePlatform``'s block-size alignment.

``test_block_size`` in ``test_platform.py`` covers the aligned outcomes; the
rejection of a user-supplied multiple of 64 that is not a power of two lives
here, in its own file, because that file's registry cases read gated HF repos.
"""

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.compilation import CompilationConfig


@pytest.mark.parametrize("requested", [192, 320])
def test_non_power_of_two_block_size_is_rejected(requested):
    """Every KV ladder keeps the powers of two starting at ``block_size``, so a
    non-power-of-two multiple of 64 (already aligned, so the rounding-up half of
    ``_align_block_size`` cannot catch it) must be refused rather than silently
    start the bucket counts off-grid."""
    from spyre_inference.platform import TorchSpyrePlatform

    cache_config = CacheConfig(block_size=requested)
    assert cache_config.user_specified_block_size
    assert requested % 64 == 0 and requested & (requested - 1)

    # VllmConfig construction runs check_and_update_config too, so the whole
    # build sits inside the raises block.
    with pytest.raises(ValueError, match="power of two"):
        vllm_config = VllmConfig(
            model_config=ModelConfig(
                model="Qwen/Qwen3-0.6B",
                max_model_len=1,
                dtype=torch.float16,
                trust_remote_code=True,
            ),
            cache_config=cache_config,
            compilation_config=CompilationConfig(custom_ops=["all"]),
        )
        TorchSpyrePlatform.check_and_update_config(vllm_config)
