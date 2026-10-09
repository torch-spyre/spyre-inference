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

"""Tests for Spyre FP8 linear kernel — ``spyre.scaled_mm`` path.

Spyre ``qfp8ch``/``qfp8wt`` and SuperDSC ``scaled_mm`` are not IEEE
``float8_e4m3fn`` / CPU ``_scaled_mm``. These tests check shapes, layouts, and
that QFP8WT is installed before the forward. Numerical checks compare that
load-time weight to quantizing fp16 inside a GEMM graph, not a CPU golden.
"""

import warnings

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.custom_ops.fp8_linear_kernel import (
    FP8_E4M3FN_MAX,
    SpyreFp8LinearKernel,
    _fp8_gemm_epilogue,
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


@torch.compile(backend="inductor", dynamic=False)
def _in_graph_qfp8wt_mm(
    x: torch.Tensor,
    weight_fp16: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    scale_a = torch.ops.spyre.quantscalepertokenfp8(x, FP8_E4M3FN_MAX)
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale_a)
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(weight_fp16, weight_scale)
    y = torch.ops.spyre.scaled_mm(x_fp8, w_fp8, out_dtype=torch.float16)
    return _fp8_gemm_epilogue(y, scale_a, weight_scale, None)


@torch.compile(backend="inductor", dynamic=False)
def _in_graph_qfp8wt_mm_static(
    x: torch.Tensor,
    scale_a: torch.Tensor,
    weight_fp16: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale_a)
    w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(weight_fp16, weight_scale)
    y = torch.ops.spyre.scaled_mm(x_fp8, w_fp8, out_dtype=torch.float16)
    return _fp8_gemm_epilogue(y, scale_a, weight_scale, None)


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

    def test_process_weights_dmas_checkpoint_fp8(self):
        """Checkpoint FP8 is DMA'd to QFP8WT; model.to('spyre') then skips it."""
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
        weight_kn = torch.randn(64, 128, dtype=torch.float16) * 0.05
        weight_fp8, weight_scale = _quantize_weight_fp8(weight_kn)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale.reshape(1), requires_grad=False)
        kernel.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight.device.type == "spyre"
        assert layer.weight.shape == (64, 128)
        self._assert_qfp8wt(layer.weight)
        assert layer.weight_scale.dtype == torch.float16
        assert layer.weight_scale.device.type == "cpu"
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
        assert layer.weight.device.type == "spyre"
        assert layer.weight.shape == (in_features, out_features)
        self._assert_qfp8wt(layer.weight)
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
        """Normalize scales, move the fp16 weight onto Spyre, quantize once.

        The weight passed through ``process_weights_after_loading`` is fp16, so
        this does not take the checkpoint DMA. ``install_qfp8wt`` is the
        post-``model.to`` step; ``apply_weights`` does not quantize.
        """
        if per_channel:
            _, weight_scale = _quantize_weight_fp8_per_channel(weight_kn)
            weight_scale = weight_scale.reshape(-1, 1)
        else:
            _, weight_scale = _quantize_weight_fp8(weight_kn)
            weight_scale = weight_scale.reshape(1)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_kn.contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        kernel.process_weights_after_loading(layer)
        layer.weight = torch.nn.Parameter(weight_kn.contiguous().to("spyre"), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.to("spyre"), requires_grad=False
        )
        kernel.install_qfp8wt(layer)
        return layer

    def _run_spyre_apply(self, kernel, layer, x):
        """Call apply_weights inside a compile, as the block graph does.

        ``spyre.scaled_mm`` has no eager kernel. Serving traces this method
        from the block; a bare call returns None.
        """
        from torch_spyre.ops.fallbacks import FallbackWarning

        # Checkpoint tests stop after process_weights, which leaves the scale
        # on CPU. install_qfp8wt is the post-model.to step: move that scale,
        # and quantize when the weight is still fp16.
        kernel.install_qfp8wt(layer)
        compiled = torch.compile(
            lambda inp: kernel.apply_weights(layer, inp),
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FallbackWarning)
            actual = compiled(x)
        assert actual.device.type == "spyre", actual.device
        return actual

    def _assert_qfp8wt(self, weight):
        layout = weight.device_tensor_layout()
        if layout is None:
            pytest.fail("cached weight has no device_tensor_layout")
        from torch_spyre._C import ElementArrangement

        assert layout.element_arrangement == ElementArrangement.QFP8WT, layout.element_arrangement

    @pytest.mark.parametrize("num_tokens", [1, 4, 128])
    def test_scaled_mm_apply(self, num_tokens):
        """apply_weights runs spyre.scaled_mm plus one scale epilogue on Spyre."""
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

    @pytest.mark.parametrize(
        "batch, seq_len",
        [
            (1, 5),
            (5, 1),
            (13, 10),
            (10, 13),
        ],
    )
    def test_scaled_mm_3d_matches_2d(self, batch, seq_len):
        """3-D input ``(B, S, K)`` must produce the same values as 2-D ``(B*S, K)``."""
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

        x2d = torch.randn(batch * seq_len, in_features, dtype=torch.float16, device="spyre")
        x3d = x2d.reshape(batch, seq_len, in_features)

        out2d = self._run_spyre_apply(kernel, layer, x2d)
        out3d = self._run_spyre_apply(kernel, layer, x3d)

        assert out2d.shape == (batch * seq_len, out_features)
        assert out3d.shape == (batch, seq_len, out_features)
        torch.testing.assert_close(out3d.reshape_as(out2d), out2d, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_tokens", [1, 4, 5, 128, 130])
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

    @pytest.mark.parametrize("per_channel", [False, True])
    def test_qfp8wt_installed_once_and_reused(self, per_channel):
        """install_qfp8wt quantizes once; later applies read that same weight."""
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel(granite_channel=per_channel)
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(42)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        layer = self._prepare_spyre_apply_layer(kernel, weight_kn, per_channel=per_channel)
        weight = layer.weight
        assert weight.dtype == torch.float8_e4m3fn
        assert weight.device.type == "spyre"
        assert weight.shape == (in_features, out_features)
        self._assert_qfp8wt(weight)
        assert getattr(layer, "_qfp8wt_for_mm", None) is None

        x = torch.randn(4, in_features, dtype=torch.float16, device="spyre")
        actual = self._run_spyre_apply(kernel, layer, x)
        again = self._run_spyre_apply(kernel, layer, x)
        assert layer.weight is weight
        assert again.dtype == torch.float16
        assert again.shape == actual.shape

    def test_checkpoint_fp8_reuses_dmad_qfp8wt(self):
        """apply uses the QFP8WT tensor process_weights DMA'd, with no requant."""
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(7)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        weight_fp8, weight_scale = _quantize_weight_fp8(weight_kn)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale.reshape(1), requires_grad=False)
        kernel.process_weights_after_loading(layer)
        loaded = layer.weight
        assert loaded.dtype == torch.float8_e4m3fn
        assert loaded.device.type == "spyre"
        self._assert_qfp8wt(loaded)

        x = torch.randn(4, in_features, dtype=torch.float16, device="spyre")
        actual = self._run_spyre_apply(kernel, layer, x)
        assert actual.shape == (4, out_features)
        assert layer.weight is loaded

    def test_transposed_checkpoint_matches_contiguous_dma(self):
        """``weight.t()`` must DMA like a contiguous ``[K, N]``.

        vLLM's FP8 post-load returns that non-contiguous view. The QFP8WT copy
        assumes row-major host strides, so leaving the view uncompacted
        scrambles every linear.
        """
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(7)
        in_features, out_features = 128, 256
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        weight_fp8, weight_scale = _quantize_weight_fp8(weight_kn)
        stored_nk = weight_fp8.t().contiguous()
        transposed = stored_nk.t()
        assert not transposed.is_contiguous()
        assert tuple(transposed.shape) == (in_features, out_features)

        def _layer(weight: torch.Tensor) -> torch.nn.Module:
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(weight_scale.reshape(1), requires_grad=False)
            kernel.process_weights_after_loading(layer)
            return layer

        x = torch.randn(4, in_features, dtype=torch.float16, device="spyre")
        out_contig = self._run_spyre_apply(kernel, _layer(weight_fp8), x)
        out_view = self._run_spyre_apply(kernel, _layer(transposed), x)
        torch.testing.assert_close(out_view, out_contig, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("num_tokens", [1, 4, 128])
    def test_installed_qfp8wt_matches_in_graph_qfp8wt(self, num_tokens):
        """Load-time QFP8WT matches compiling qfp8wt in the GEMM graph.

        Same check as spyre-inference#934 ``test_prequant_result_matches_fallback``
        without an env-gated kernel path. The two quantize independently, so
        fp8 rounding can differ by a few ULPs; one ULP dequanted to fp16 is
        ~scale * 2/448 ≈ 2e-4, and 0.1 covers accumulation over K=128.
        """
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel()
        except ImportError:
            pytest.skip("vLLM FP8 APIs unavailable")

        torch.manual_seed(13)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        weight_fp16 = weight_kn.contiguous().to("spyre")
        layer = self._prepare_spyre_apply_layer(kernel, weight_kn, per_channel=False)
        x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="spyre")
        self._assert_installed_matches_in_graph(
            kernel, layer, x, weight_fp16, num_tokens=num_tokens
        )

    @pytest.mark.parametrize("num_tokens", [1, 4, 128])
    def test_installed_qfp8wt_per_channel_matches_in_graph(self, num_tokens):
        """Per-channel load-time qfp8wt matches in-graph qfp8wt (PR #934)."""
        if not spyre_available():
            pytest.skip("Spyre device not available")
        if SpyreFp8LinearKernel is None:
            pytest.skip("vLLM FP8 kernel base unavailable")

        register_spyre_fp8_linear_kernel()
        try:
            kernel = _make_kernel(granite_channel=True)
        except ImportError:
            pytest.skip("vLLM FP8 channel QuantKey unavailable")

        torch.manual_seed(21)
        in_features, out_features = 128, 128
        weight_kn = torch.randn(in_features, out_features, dtype=torch.float16) * 0.05
        weight_fp16 = weight_kn.contiguous().to("spyre")
        layer = self._prepare_spyre_apply_layer(kernel, weight_kn, per_channel=True)
        x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="spyre")
        self._assert_installed_matches_in_graph(
            kernel, layer, x, weight_fp16, num_tokens=num_tokens
        )

    def _assert_installed_matches_in_graph(
        self, kernel, layer, x, weight_fp16, *, num_tokens: int
    ) -> None:
        from torch_spyre.ops.fallbacks import FallbackWarning

        from spyre_inference.custom_ops.fp8_linear_kernel import _per_tensor_activation_scale

        out_installed = self._run_spyre_apply(kernel, layer, x).cpu()
        scale = layer.weight_scale
        with warnings.catch_warnings():
            warnings.simplefilter("error", FallbackWarning)
            if kernel._per_token_act:
                out_graph = _in_graph_qfp8wt_mm(x, weight_fp16, scale).cpu()
            else:
                out_graph = _in_graph_qfp8wt_mm_static(
                    x, _per_tensor_activation_scale(x), weight_fp16, scale
                ).cpu()
        assert out_installed.shape == out_graph.shape
        max_diff = (out_installed.float() - out_graph.float()).abs().max().item()
        assert max_diff < 0.1, (
            f"load-time qfp8wt vs in-graph qfp8wt max_diff={max_diff:.6f} (num_tokens={num_tokens})"
        )


def test_require_qfp8wt_fails_without_layout():
    """Missing device_tensor_layout must error, not skip the QFP8WT check."""
    from spyre_inference.custom_ops.fp8_linear_kernel import _require_qfp8wt

    with pytest.raises(RuntimeError, match="device_tensor_layout"):
        _require_qfp8wt(torch.zeros(2, 2, dtype=torch.float16))


def test_apply_weights_is_traced():
    """The forward is plain code so the block graph can use fullgraph=True."""
    assert not getattr(SpyreFp8LinearKernel.apply_weights, "_torchdynamo_disable", False)


def test_install_qfp8wt_rejects_cpu_fp16():
    """fp16 quantization happens after the Spyre move, not on a CPU weight."""
    if SpyreFp8LinearKernel is None:
        pytest.skip("vLLM FP8 kernel base unavailable")

    register_spyre_fp8_linear_kernel()
    try:
        kernel = _make_kernel()
    except ImportError:
        pytest.skip("vLLM FP8 APIs unavailable")

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(64, 128, dtype=torch.float16), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
    kernel.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="on Spyre"):
        kernel.install_qfp8wt(layer)
