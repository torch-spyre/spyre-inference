# FP8 Support Gaps

This page documents the current state of FP8 (float8) quantization support in the spyre-inference stack, including specific gaps, blockers, and required work to enable FP8 models in vLLM.

**Status: Not Supported** — FP8 quantization is explicitly excluded from current implementation

**Related:** [Hardware Precision Formats](../../wiki/foundations/hardware/precision-formats.md), [Model Enablement](../../wiki/stack/model-enablement.md), [Torch-Spyre](../../wiki/stack/torch-spyre.md)

## Summary

**Hardware capability:** The AIU accelerator supports FP8 natively — both E4M3 (1-4-3) and E5M2 (1-5-2) formats are available on AIU 1.0 and AIU 1.5.

**Software reality:** The spyre-inference plugin and torch-spyre stack do **not** support FP8 quantization. All custom ops explicitly exclude quantized paths, and torch-spyre lacks the necessary kernel support for FP8 matmul operations.

## Current State

### spyre-inference Plugin

**Custom linear layers** (`spyre_inference/custom_ops/linear.py`):

- Only handles `UnquantizedLinearMethod` — quantized paths are not intercepted
- `SpyreUnquantizedLinearMethod.apply()` performs plain `F.linear()` with no FP8 handling
- No FP8-specific linear method implementations exist

**LM head layer** (`spyre_inference/custom_ops/parallel_lm_head.py`):

- Explicitly documents: "No quantization support: only UnquantizedEmbeddingMethod is replaced"
- Raises `NotImplementedError` if quantization method other than unquantized is detected

**Test coverage:**

- All existing tests use `quant_config=None`
- No FP8 or quantization tests exist
- Tests run exclusively in `torch.float16` dtype

### torch-spyre Backend

**FP8 dtype support:**

- PyTorch provides `torch.float8_e4m3fn` and `torch.float8_e5m2` dtypes
- Basic FP8 tensor creation works on CPU
- **FP8 tensors cannot be created on spyre device**: `"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'` — random initialization kernels don't support FP8

**Matmul kernel support:**

- torch-spyre uses `_attn_transposed` for attention matmul operations
- No FP8-specific matmul kernels are registered
- No `scaled_mm` (scaled matrix multiply) implementations for FP8 W8A8 or W8A16 paths

**Decompositions and lowerings:**

- No FP8-specific decompositions registered in `enable_spyre_decompositions()`
- No FP8 lowerings in `enable_spyre_lowerings()`
- FP8 quantization ops fall back to eager CPU execution

### vLLM Integration

vLLM's FP8 quantization infrastructure (`vllm/model_executor/layers/quantization/fp8.py`) requires:

1. **Fp8LinearMethod** — performs weight dequantization or FP8 matmul
2. **Fp8OnlineLinearMethod** — online quantization path with activation quantization
3. **FP8 kernel dispatch** — `init_fp8_linear_kernel()` selects backend-specific kernels
4. **Scale tensor handling** — per-tensor or per-block weight/activation scales

None of these components have Spyre-specific implementations.

## Required Work

### Phase 1: Basic FP8 Support (W8A8 Static)

**torch-spyre changes:**

1. **FP8 matmul kernel** — Implement or wrap a Spyre-compatible FP8 GEMM kernel
   - Must handle `float8_e4m3fn` weights and activations
   - Output accumulation in `float32` or `float16`
   - Support per-tensor activation and weight scales

2. **FP8 dtype tensor ops** — Register kernels for:
   - FP8 tensor initialization (zeros, empty, copy from CPU)
   - FP8 → FP16/FP32 conversion (for dequantization paths)
   - FP8 elementwise ops if needed (ReLU, etc.)

3. **Scale tensor handling** — Ensure scale tensors (typically `float32`) work correctly with matmul

**spyre-inference changes:**

1. **SpyreFp8LinearMethod** — Custom linear method that:
   - Routes FP8 matmul through torch-spyre kernel
   - Handles scale application (activation_scale × weight_scale)
   - Manages FP8 weight loading and layout

2. **Quantization config detection** — Intercept `Fp8Config` and return appropriate method

3. **Test coverage** — Add FP8-specific tests comparing against CPU reference

### Phase 2: Advanced FP8 Features

**Online quantization (W8A16 dynamic):**

- Activation quantization at runtime
- Per-token or per-128-element-block quantization
- Requires FP8 conversion kernels in torch-spyre

**Block-wise quantization:**

- Weight block sizes (e.g., 128×128) with per-block scales
- More complex weight loading and scale management
- Potentially better accuracy

**KV cache quantization:**

- FP8 KV cache for attention
- Requires attention backend modifications
- Memory bandwidth savings

## Blockers

### Known torch-spyre Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| FP8 tensor creation on device | ❌ Blocked | `"normal_kernel_cpu" not implemented` |
| FP8 matmul kernel | ❌ Missing | No `_scaled_mm` or equivalent |
| FP8 decompositions | ❌ Missing | Not registered in patches.py |
| FP8 lowerings | ❌ Missing | Not in lowering registry |
| Scale tensor ops | ⚠️ Unknown | Need to verify basic support |

### spyre-inference Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| Quantization method interception | ❌ Missing | Only UnquantizedLinearMethod handled |
| FP8 weight loading | ❌ Missing | No process_weights_after_loading for FP8 |
| FP8 test infrastructure | ❌ Missing | No reference implementation tests |

## Monitoring torch-spyre Upstream

To detect when FP8 support lands in torch-spyre:

1. **Watch for new kernels** in `torch_spyre/ops/` related to:
   - `scaled_mm`, `fp8_matmul`, `float8_gemm`
   - FP8 conversion ops

2. **Check decomposition registry** in `torch_spyre/_inductor/patches.py`:
   - Look for `aten.quantize_per_tensor`, `aten.dequantize`
   - FP8-specific decompositions

3. **Test FP8 tensor creation**:
   ```python
   torch.randn(8, 8, dtype=torch.float8_e4m3fn, device='spyre')
   ```

4. **Monitor vLLM FP8 tests** — when they start passing on spyre, the backend support is ready

## References

- vLLM FP8 implementation: `vllm/model_executor/layers/quantization/fp8.py`
- vLLM FP8 kernels: `vllm/model_executor/kernels/linear/scaled_mm.py`
- PyTorch FP8 docs: `torch.float8_e4m3fn`, `torch.float8_e5m2`
