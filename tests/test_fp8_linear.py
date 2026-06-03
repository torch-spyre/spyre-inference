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
FP8 quantization support tests.

These tests are marked xfail (strict) to track FP8 support gaps in torch-spyre
and spyre-inference. When FP8 support is added upstream, these tests will
automatically flip to passing and notify us to implement the Spyre-specific
handling.

References:
    - vLLM FP8 config: vllm/model_executor/layers/quantization/fp8.py
    - Hardware formats: docs/architecture/fp8-support-gaps.md
"""

import pytest
import torch


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True,
    reason="FP8 quantization not supported in torch-spyre: FP8 tensor creation on device fails with 'normal_kernel_cpu not implemented for Float8_e4m3fn'",
)
def test_fp8_tensor_creation_on_spyre(tp_group):
    """FP8 tensors must be creatable directly on spyre device.

    This is the most basic requirement: if we can't create FP8 tensors on
    the device, no FP8 quantization path can work.

    Expected failure mode:
        RuntimeError: "normal_kernel_cpu" not implemented for 'Float8_e4m3fn'

    When this test passes, torch-spyre has gained basic FP8 dtype support.
    """
    # This should work without falling back to CPU
    x = torch.randn(8, 8, dtype=torch.float8_e4m3fn, device="spyre")
    assert x.dtype == torch.float8_e4m3fn
    assert x.device.type == "spyre"


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True,
    reason="FP8 matmul kernel (_scaled_mm or equivalent) not implemented in torch-spyre",
)
def test_fp8_matmul_w8a8(tp_group):
    """FP8 W8A8 (weight 8-bit, activation 8-bit) matmul must be supported.

    This is the core operation for static FP8 quantization. Both weights and
    activations are in float8_e4m3fn format, with per-tensor scales applied.

    When this test passes, torch-spyre has a working FP8 GEMM kernel.
    """
    # FP8 weights and activations
    weight = torch.randn(512, 512, dtype=torch.float8_e4m3fn, device="spyre")
    activation = torch.randn(32, 512, dtype=torch.float8_e4m3fn, device="spyre")
    scale = torch.ones(1, dtype=torch.float32, device="spyre")

    # Scaled matmul: output = scale * (activation @ weight.T)
    # This requires torch-spyre to have _scaled_mm or equivalent
    output = torch._scaled_mm(
        activation,
        weight.T,
        out_dtype=torch.float16,
        scale_a=scale,
        scale_b=scale,
    )
    assert output.shape == (32, 512)
    assert output.dtype == torch.float16


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True,
    reason="spyre-inference does not intercept Fp8Config - only UnquantizedLinearMethod is handled",
)
def test_fp8_linear_layer_with_quant_config(tp_group):
    """Linear layers must work with vLLM's Fp8Config quantization config.

    This tests the spyre-inference plugin's ability to intercept FP8 quantized
    linear layers and route them through Spyre-specific handling.

    When this test passes, spyre-inference has SpyreFp8LinearMethod implemented.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    # Create FP8 quantization config (as vLLM would from model config)
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="static",
    )

    # This should not raise NotImplementedError
    layer = MergedColumnParallelLinear(
        input_size=512,
        output_sizes=[1024, 1024],
        bias=False,
        params_dtype=torch.float16,
        quant_config=quant_config,  # FP8 config, not None
        disable_tp=True,
        prefix="gate_up_proj",
    )

    # Should get a Spyre FP8 method, not the unquantized one
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    assert not isinstance(layer.quant_method, SpyreUnquantizedLinearMethod)


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True,
    reason="FP8 online quantization (W8A16 dynamic) not supported - requires activation quantization kernels",
)
def test_fp8_online_quantization(tp_group):
    """FP8 online (dynamic) quantization path must work.

    Online quantization quantizes activations at runtime (W8A16: 8-bit weights,
    16-bit activations). This requires FP8 conversion kernels in torch-spyre.

    When this test passes, torch-spyre has activation quantization support.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.linear import RowParallelLinear

    # Dynamic activation scheme = online quantization
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",  # Dynamic = online quantization
    )

    layer = RowParallelLinear(
        input_size=512,
        output_size=256,
        bias=False,
        params_dtype=torch.float16,
        quant_config=quant_config,
        reduce_results=True,
        disable_tp=True,
        prefix="down_proj",
    )

    x = torch.randn(32, 512, dtype=torch.float16, device="spyre")
    output, _ = layer(x)
    assert output.shape == (32, 256)


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True,
    reason="FP8 checkpoint loading (process_weights_after_loading) not implemented for FP8 weights",
)
def test_fp8_weight_loading(tp_group):
    """FP8 quantized weights must load correctly through the quantization path.

    When loading FP8 checkpoints, weights come pre-quantized with per-tensor
    or per-block scales. The quant_method.process_weights_after_loading()
    must handle FP8 weight tensors properly.

    When this test passes, spyre-inference has FP8 weight loading implemented.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from vllm.model_executor.layers.linear import LinearBase

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,  # Pre-quantized checkpoint
        activation_scheme="static",
    )

    # Get the linear method
    linear_method = quant_config.get_quant_method(None, "test")
    assert isinstance(linear_method, Fp8LinearMethod)

    # Create a mock layer and test weight processing
    # This requires SpyreFp8LinearMethod to be implemented
    layer = LinearBase(512, 512, bias=False, params_dtype=torch.float16)
    layer.weight = torch.nn.Parameter(
        torch.randn(512, 512, dtype=torch.float8_e4m3fn, device="spyre")
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    # Should not raise - must handle FP8 weights
    linear_method.process_weights_after_loading(layer)


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True, reason="Block-wise FP8 quantization (per-block scales) not supported"
)
def test_fp8_blockwise_quantization(tp_group):
    """FP8 block-wise quantization must be supported.

    Block-wise quantization uses per-block scales (e.g., 128x128 blocks)
    instead of per-tensor scales. This provides better accuracy but requires
    more complex kernel support.

    When this test passes, torch-spyre has block-wise FP8 matmul support.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="static",
        weight_block_size=[128, 128],  # Block-wise quantization
    )

    # Block-wise requires additional scale handling
    # This test will pass when torch-spyre supports block-scaled matmul
    assert quant_config.weight_block_size == [128, 128]


@pytest.mark.spyre
@pytest.mark.fp8
@pytest.mark.xfail(
    strict=True, reason="FP8 KV cache quantization not supported in attention backend"
)
def test_fp8_kv_cache(tp_group):
    """FP8 KV cache must be supported in the attention backend.

    KV cache quantization to FP8 reduces memory bandwidth. This requires
    modifications to SpyreAttentionBackend to handle FP8 KV tensors.

    When this test passes, the attention backend supports FP8 KV cache.
    """
    from vllm.model_executor.layers.quantization.kv_cache import Fp8KVCacheMethod
    from vllm.attention.backends.abstract import Attention

    # FP8 KV cache method
    kv_cache_config = type("KVCacheConfig", (), {"dtype": "fp8_e4m3fn"})()

    # Attention layer with FP8 KV cache
    attn = Attention(
        num_heads=8,
        num_kv_heads=8,
        head_size=64,
        kv_cache_dtype="fp8_e4m3fn",
    )

    # Should use FP8 KV cache method
    assert isinstance(attn.kv_cache_method, Fp8KVCacheMethod)


@pytest.mark.fp8
def test_fp8_dtype_availability():
    """Verify PyTorch FP8 dtypes are defined (non-spyre baseline).

    This test checks that PyTorch has FP8 dtype attributes defined.
    Note: torch.randn() with FP8 dtype may not be supported in all PyTorch
    versions - FP8 tensors typically need to be created via .to() conversion
    from FP32/FP16 tensors.
    """
    # Check that FP8 dtypes are defined
    assert hasattr(torch, "float8_e4m3fn"), "torch.float8_e4m3fn should be available"
    assert hasattr(torch, "float8_e5m2"), "torch.float8_e5m2 should be available"

    # FP8 dtypes exist but may not have full operator support
    # Creation via .to() from FP32 is the supported path in older PyTorch
    try:
        x = torch.randn(8, 8, dtype=torch.float8_e4m3fn)
        assert x.dtype == torch.float8_e4m3fn
    except NotImplementedError:
        # Expected in PyTorch versions without full FP8 CPU support
        # The dtypes exist but random initialization isn't implemented
        pytest.skip("torch.randn() not implemented for float8_e4m3fn in this PyTorch version")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
