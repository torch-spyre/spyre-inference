# FP8 Support Gaps

This page documents the current state of FP8 (float8) quantization support in the spyre-inference stack, including specific gaps, blockers, and required work to enable FP8 models in vLLM.

**Status: Not Supported** — FP8 quantization is explicitly excluded from current implementation

**Related:** See the spyre-knowledgebase MCP server for hardware precision formats and torch-spyre architecture details. Use `mcp__spyre-knowledgebase__search(query="FP8 float8 precision")` to query.

## torch-spyre Issue Tracker Summary

As of June 2026, there are **9+ open issues and PRs** related to FP8 in torch-spyre. The work is actively in progress but blocked on several key items.

| Type | # | Title | Status | Notes |
|------|---|-------|--------|-------|
| PR | [#2401](https://github.com/torch-spyre/torch-spyre/pull/2401) | feat: Add FP8 quantization support for torch-spyre | Open, pending approval | Adds `quantize_fp8_with_scale` op and inductor lowering. Review Q: why custom reciprocal? |
| PR | [#2286](https://github.com/torch-spyre/torch-spyre/pull/2286) | FP8 scaled_mm enablement | Open | Enables `torch.ops.aten._scaled_mm()` on Spyre, fixes SDSC codegen. Resumes #2077. |
| Issue | [#2276](https://github.com/torch-spyre/torch-spyre/issues/2276) | FP8 test coverage and validation | Open | Test infrastructure for FP8 paths |
| Issue | [#2186](https://github.com/torch-spyre/torch-spyre/issues/2186) | FP8 quantization operations not functioning correctly in deeptools | **BLOCKER** | `qfp8`, `qfp8wt`, `qfp8mb` fail with "Fixed layout with too many dimensions". `qfp8ch` loses data when hidden > 8. |
| Issue | [#2185](https://github.com/torch-spyre/torch-spyre/issues/2185) | "Anywhere valid" issue causing data corruption in FP16 to FP8 dtype conversion | Open | Data fidelity bug in dtype conversion |
| Issue | [#1958](https://github.com/torch-spyre/torch-spyre/issues/1958) | Support scaled_bmm FP8 operator | Open | Batched matmul support needed |
| Issue | [#1957](https://github.com/torch-spyre/torch-spyre/issues/1957) | Support scaled_mm FP8 operator | Open | Core FP8 GEMM support |
| Issue | [#1956](https://github.com/torch-spyre/torch-spyre/issues/1956) | Support FP8 quantization operation | Open | Base quantization op |
| Issue | [#1474](https://github.com/torch-spyre/torch-spyre/issues/1474) | Support operations on Float8 (SEN143_FP8) tensors in Spyre backend | Open | Tags: inductor compatibility |

**Key takeaways:**

- Core `_scaled_mm` support is in-flight (#2286) but not yet merged
- Deeptools has a **blocker** (#2186) with FP8 ops failing with layout errors — this must be resolved before any FP8 quantization can work
- FP8 quantization PR (#2401) exists but is pending approval and depends on #2286
- Multiple foundational ops (#1956, #1957, #1958) are still open — these are prerequisites for vLLM FP8 integration

## Summary

**Hardware capability:** The AIU accelerator supports FP8 natively — both E4M3 (1-4-3) and E5M2 (1-5-2) formats are available on AIU 1.0 and AIU 1.5. These are the **standard** FP8 formats (same as NVIDIA H100+), not the FNUZ variants used by some other accelerators.

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

- PyTorch provides `torch.float8_e4m3fn` and `torch.float8_e5m2` dtypes (standard formats, not FNUZ variants)
- Basic FP8 tensor creation works on CPU
- **FP8 tensors cannot be directly initialized on spyre device**: Random initialization kernels like `torch.randn()` don't support FP8 dtypes on spyre (`"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'`). However, FP8 tensors can be created on CPU (e.g., `torch.rand().to(dtype=torch.float8_e4m3fn)`) and transferred to spyre successfully.

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
| FP8 tensor creation on device | ⚠️ Partial | Direct initialization (`torch.randn`) unsupported; CPU→spyre transfer works |
| FP8 matmul kernel | ❌ Missing | No `_scaled_mm` or equivalent |
| FP8 decompositions | ❌ Missing | Not registered in patches.py |
| FP8 lowerings | ❌ Missing | Not in lowering registry |
| FP8 ops trigger InductorError | ⚠️ Known issue | Issue #1474: "Support operations on Float8 (SEN143_FP8) tensors" |
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
   # Direct initialization (currently fails):
   torch.randn(8, 8, dtype=torch.float8_e4m3fn, device='spyre')

   # CPU creation + transfer (works):
   torch.rand(8, 8, dtype=torch.float8_e4m3fn).to('spyre')
   ```

4. **Monitor vLLM FP8 tests** — when they start passing on spyre, the backend support is ready

## References

- vLLM FP8 implementation: `vllm/model_executor/layers/quantization/fp8.py`
- vLLM FP8 kernels: `vllm/model_executor/kernels/linear/scaled_mm.py`
- PyTorch FP8 docs: `torch.float8_e4m3fn`, `torch.float8_e5m2` (standard formats; AIU does **not** use FNUZ variants)
- Hardware precision formats: spyre-knowledgebase `wiki/foundations/hardware/precision-formats.md`
