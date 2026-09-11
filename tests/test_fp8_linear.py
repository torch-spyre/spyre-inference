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

"""Tests for Spyre FP8 linear kernel — aten._scaled_mm path."""

import warnings

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.fp8_linear_kernel import (
    FP8_E4M3FN_MAX,
    SpyreFp8LinearKernel,
    register_spyre_fp8_linear_kernel,
)

FP8_E4M3FN_MIN = -FP8_E4M3FN_MAX


def _quantize_weight_fp8(weight_fp16: torch.Tensor):
    """Quantize float16 weight to float8_e4m3fn with a per-tensor scale."""
    amax = weight_fp16.abs().amax()
    scale = (amax / FP8_E4M3FN_MAX).to(torch.float16)
    weight_fp8 = (weight_fp16 / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
    return weight_fp8, scale


def _quantize_weight_fp8_per_channel(weight_kn: torch.Tensor):
    """Quantize ``[K, N]`` weight with one scale per output column (Granite)."""
    amax = weight_kn.abs().amax(dim=0).clamp(min=1e-12)
    scale = (amax / FP8_E4M3FN_MAX).to(torch.float16)
    weight_fp8 = (weight_kn / scale).clamp(FP8_E4M3FN_MIN, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
    return weight_fp8, scale


def _make_kernel(*, granite_channel: bool = False):
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel

    if granite_channel:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTokenSym,
            kFp8StaticChannelSym,
        )

        return init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            weight_shape=(64, 128),
            input_dtype=torch.float16,
            out_dtype=torch.float16,
            module_name="TestSpyreFp8Granite",
        )

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    method = Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
    return init_fp8_linear_kernel(
        activation_quant_key=method.activation_quant_key,
        weight_quant_key=method.weight_quant_key,
        weight_shape=(64, 128),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
        module_name="TestSpyreFp8",
    )


@pytest.mark.fp8
class TestSpyreFp8LinearKernel:
    def test_register(self):
        assert register_spyre_fp8_linear_kernel()
        assert SpyreFp8LinearKernel is not None

    def test_kernel_selected_for_oot(self):
        """init_fp8_linear_kernel finds the Spyre OOT scaled_mm kernel."""
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")
        assert isinstance(kernel, SpyreFp8LinearKernel), (
            f"Expected SpyreFp8LinearKernel, got {type(kernel).__name__}"
        )

    def test_process_weights_keeps_fp8(self):
        """Weights stay FP8 for aten._scaled_mm (no CPU dequant / weight_t)."""
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(42)
        weight_kn = torch.randn(64, 128, dtype=torch.float16) * 0.05
        weight_fp8, weight_scale = _quantize_weight_fp8(weight_kn)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale.reshape(1), requires_grad=False)
        kernel.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight.shape == (64, 128)
        assert layer.weight_scale.numel() == 1
        assert getattr(layer, "weight_t", None) is None

    def test_can_implement_per_channel(self):
        """Granite compressed-tensors channel weights are accepted."""
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel(granite_channel=True)
        except ImportError:
            pytest.skip("vLLM FP8 channel QuantKey unavailable")
        assert isinstance(kernel, SpyreFp8LinearKernel)
        assert kernel._per_token_act

    def test_process_weights_keeps_per_channel_scale(self):
        """Per-channel scales stay as N values (not folded to a scalar)."""
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel(granite_channel=True)
        except ImportError:
            pytest.skip("vLLM FP8 channel QuantKey unavailable")

        torch.manual_seed(42)
        in_features, out_features = 64, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        weight_fp8, weight_scale = _quantize_weight_fp8_per_channel(weight_kn)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        # compressed-tensors channel layout before our reshape: [N, 1]
        layer.weight_scale = torch.nn.Parameter(
            weight_scale.reshape(out_features, 1), requires_grad=False
        )
        kernel.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight.shape == (in_features, out_features)
        assert layer.weight_scale.shape == (1, out_features)
        assert layer.weight_scale.numel() == out_features
        torch.testing.assert_close(
            layer.weight_scale.reshape(-1).cpu(),
            weight_scale.cpu(),
            atol=0.0,
            rtol=0.0,
        )
        assert getattr(layer, "weight_t", None) is None

    def _prepare_spyre_apply_layer(self, kernel, weight_kn, *, per_channel: bool):
        """Load-time process on CPU, then move FP16 weight to Spyre.

        Forward quantizes with ``quantize_weight_fp8_with_scale`` (``qfp8wt``)
        inside the same compiled graph as ``_scaled_mm`` — torch-spyre's
        ``test_fp8_scaled_mm_cpu``. CPU ``float8.to("spyre")`` is the wrong layout.
        """
        if per_channel:
            _weight_fp8, weight_scale = _quantize_weight_fp8_per_channel(weight_kn)
            weight_scale = weight_scale.reshape(-1, 1)
        else:
            _weight_fp8, weight_scale = _quantize_weight_fp8(weight_kn)
            weight_scale = weight_scale.reshape(1)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(_weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        kernel.process_weights_after_loading(layer)
        layer.weight = torch.nn.Parameter(weight_kn.contiguous().to("spyre"), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.to("spyre"), requires_grad=False
        )
        return layer

    def _run_spyre_apply(self, kernel, layer, x):
        """Call apply_weights on Spyre (one inductor graph: qfp8ch + qfp8wt + mm)."""
        from torch_spyre.ops.fallbacks import FallbackWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", FallbackWarning)
            actual = kernel.apply_weights(layer, x)
        assert actual.device.type == "spyre", actual.device
        return actual

    @pytest.mark.parametrize("num_tokens", [1, 4, 128])
    def test_scaled_mm_apply(self, num_tokens):
        """apply_weights runs aten._scaled_mm on Spyre."""
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(42)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        layer = self._prepare_spyre_apply_layer(kernel, weight_kn, per_channel=False)

        x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="spyre")
        actual = self._run_spyre_apply(kernel, layer, x)
        assert actual.dtype == torch.float16
        assert actual.shape == (num_tokens, out_features)

    @pytest.mark.parametrize("num_tokens", [1, 4, 128])
    def test_scaled_mm_apply_per_channel(self, num_tokens):
        """apply_weights with Granite per-channel weight scales + per-token acts."""
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel(granite_channel=True)
        except ImportError:
            pytest.skip("vLLM FP8 channel QuantKey unavailable")

        torch.manual_seed(42)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        layer = self._prepare_spyre_apply_layer(kernel, weight_kn, per_channel=True)

        x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="spyre")
        actual = self._run_spyre_apply(kernel, layer, x)
        assert actual.dtype == torch.float16
        assert actual.shape == (num_tokens, out_features)

    def test_qkv_constructs_with_fp8_config(self, tp_group):
        """Real QKVParallelLinear + Fp8Config constructs (kernel selection works)."""
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()

        try:
            from vllm.model_executor.layers.linear import QKVParallelLinear
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        except ImportError:
            pytest.skip("vLLM Fp8Config not available")

        layer = QKVParallelLinear(
            hidden_size=128,
            head_size=64,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            params_dtype=torch.float16,
            quant_config=Fp8Config(is_checkpoint_fp8_serialized=True),
            prefix="test.qkv_proj",
        )
        assert isinstance(layer.quant_method.fp8_linear, SpyreFp8LinearKernel)


class TestFp8TileHelpers:
    """SuperDSC-legal M/N splits from the Granite torch-spyre probe."""

    def test_m_tiles_4096_wide(self):
        from spyre_inference.custom_ops.fp8_linear_kernel import _m_tiles

        assert _m_tiles(1, 4096, 4096) == [1]
        assert _m_tiles(4, 4096, 4096) == [4]
        assert _m_tiles(16, 4096, 4096) == [4, 4, 4, 4]
        assert _m_tiles(3, 4096, 4096) == [4]
        assert _m_tiles(6, 4096, 4096) == [4, 4]
        assert _m_tiles(128, 128, 128) == [128]

    def test_n_tiles_granite(self):
        from spyre_inference.custom_ops.fp8_linear_kernel import _n_tiles

        assert _n_tiles(4096) == [4096]
        assert _n_tiles(1024) == [1024]
        assert _n_tiles(6144) == [4096, 1024, 1024]
        assert _n_tiles(25600) == [4096] * 6 + [1024]
        assert _n_tiles(12800) == [4096, 4096, 4096, 128, 128, 128, 128]


class TestDetectFp8ForCompile:
    """``_compile_for_spyre`` must use fullgraph=False for Granite FP8."""

    def test_detects_fp8_weight_dtype(self):
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        layer = torch.nn.Linear(8, 8, bias=False)
        layer.weight = torch.nn.Parameter(
            torch.empty(8, 8, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        model = torch.nn.Sequential(layer)
        assert TorchSpyreModelRunner._model_has_spyre_fp8(model)

    def test_detects_compressed_tensors_scheme_kernel(self):
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        class _Scheme:
            fp8_linear = object.__new__(SpyreFp8LinearKernel)

        class _QuantMethod:
            scheme = _Scheme()

        layer = torch.nn.Linear(4, 4, bias=False)
        layer.quant_method = _QuantMethod()
        model = torch.nn.Sequential(layer)
        assert TorchSpyreModelRunner._model_has_spyre_fp8(model)

    def test_plain_fp16_is_not_fp8(self):
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        assert not TorchSpyreModelRunner._model_has_spyre_fp8(model)
