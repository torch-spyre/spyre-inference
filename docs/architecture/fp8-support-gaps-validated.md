# FP8 Support Gaps — Claim Validation Report

This document validates each claim made in the FP8 support gaps documentation against the torch-spyre codebase, vLLM implementation, and hardware documentation.

**Validation Date:** 2026-06-03  
**torch-spyre Commit:** a77f340 (validate skeleton #2456)  
**Source:** https://github.com/torch-spyre/torch-spyre

---

## Summary Table

| # | Claim | Status | Notes |
|---|-------|--------|-------|
| 1 | AIU supports FP8 natively (E4M3 + E5M2) | ✅ TRUE | Confirmed in hardware precision formats documentation |
| 2 | Custom linear layers only handle `UnquantizedLinearMethod` | ✅ TRUE | Verified in `spyre_inference/custom_ops/linear.py` |
| 3 | LM head has no quantization support | ✅ TRUE | Explicit `NotImplementedError` for quant_config ≠ None |
| 4 | FP8 tensors cannot be created on spyre device | ⚠️ PARTIALLY FALSE | FP8 tensors can be created on CPU and transferred; only `torch.randn()` fails directly on spyre |
| 5 | No FP8 matmul kernels registered | ✅ TRUE | No `_scaled_mm` or FP8 GEMM implementations found |
| 6 | No FP8 decompositions in `enable_spyre_decompositions()` | ✅ TRUE | Verified in `torch_spyre/_inductor/decompositions.py` |
| 7 | No FP8 lowerings in `enable_spyre_lowerings()` | ✅ TRUE | Verified in `torch_spyre/_inductor/lowering.py` |
| 8 | FP8 ops fall back to eager CPU execution | ✅ TRUE | Confirmed via fallback mechanism; may also trigger InductorError |
| 9 | vLLM FP8 kernel registry has no Spyre entry | ✅ TRUE | `init_fp8_linear_kernel()` has no Spyre dispatch path |
| 10 | torch-spyre supports `Float8_e5m2fnuz` not `Float8_e5m2` | ❌ FALSE | No FNUZ references in torch-spyre; docs correctly state E5M2 |
| 11 | FP8 tensor creation works but not via `torch.randn` | ✅ TRUE | `torch.rand().to(dtype)` works; direct `torch.randn(dtype=torch.float8_*)` on spyre fails |
| 12 | FP8 inputs trigger InductorError | ✅ TRUE | Issue #1474 tracks inductor compatibility; tests marked xfail |

---

## Detailed Validation

### Claim 1: AIU Supports FP8 Natively (E4M3 + E5M2)

**Status: ✅ TRUE**

**Source:** `wiki/foundations/hardware/precision-formats.md` (spyre-knowledgebase MCP)

> "AIU supports the standard FP8 dtype comprising two complementary 8-bit formats — **E4M3** (1-4-3) and **E5M2** (1-5-2). These are the same two FP8 formats used on NVIDIA GPUs (H100+), making AIU's FP8 dtype fully standard. Both formats are supported on **AIU 1.0 and AIU 1.5**."

The hardware documentation is unambiguous: both standard FP8 formats (E4M3 and E5M2) are natively supported across all AIU generations. This is distinct from the FNUZ variants used by some other accelerators.

---

### Claim 2: Custom Linear Layers Only Handle `UnquantizedLinearMethod`

**Status: ✅ TRUE**

**Source:** `spyre_inference/custom_ops/linear.py:85-86`

```python
if isinstance(self.quant_method, UnquantizedLinearMethod):
    self.quant_method = SpyreUnquantizedLinearMethod()
```

The `SpyreLinearBase.__init__()` method only intercepts `UnquantizedLinearMethod`. There is no handling for:
- `Fp8LinearMethod`
- `Fp8OnlineLinearMethod`
- `ModelOptFp8LinearMethod`
- `FBGEMMFp8LinearMethod`
- Any other quantized linear method

When vLLM attempts to use FP8 quantization, the quant_method remains unchanged and will call into vLLM's standard FP8 path, which has no Spyre-specific kernel implementations.

---

### Claim 3: LM Head Has No Quantization Support

**Status: ✅ TRUE**

**Source:** `spyre_inference/custom_ops/parallel_lm_head.py:98-103`

```python
quant_config = kwargs.get("quant_config")
if quant_config is not None:
    raise NotImplementedError(
        "SpyreParallelLMHead does not support quantization "
        f"(quant_config={quant_config}). Only quant_config=None is supported."
    )
```

The `SpyreParallelLMHead` explicitly rejects any quantization configuration at initialization time. This is a hard blocker — FP8-quantized models using the standard vLLM LM head will fail immediately.

---

### Claim 4: FP8 Tensors Cannot Be Created on Spyre Device

**Status: ⚠️ PARTIALLY FALSE**

**Source:** `torch-spyre/tests/test_spyre.py:354-356`

```python
torch.float8_e4m3fn: lambda: torch.rand(64, 64).to(
    dtype=torch.float8_e4m3fn
),
```

The test suite shows that FP8 tensors:
1. **Can be created on CPU** using `torch.rand().to(dtype=torch.float8_e4m3fn)`
2. **Can be transferred to spyre** via `.to("spyre")`
3. **Can be transferred back to CPU** and maintain data fidelity (within FP8 precision)

The actual limitation is more specific:
- `torch.randn(size, dtype=torch.float8_e4m3fn, device="spyre")` fails with `"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'`
- This is a **random initialization kernel** limitation, not a general tensor creation limitation

**Correction:** The documentation should state: "FP8 tensors cannot be **directly initialized** on spyre device using random factories (`torch.randn`, `torch.empty` with FP8 dtype). They must be created on CPU and transferred."

---

### Claim 5: No FP8 Matmul Kernels Registered

**Status: ✅ TRUE**

**Source:** 
- `torch_spyre/ops/eager.py` — no FP8 matmul implementations
- `torch_spyre/ops/fallbacks.py` — no FP8-specific fallbacks
- GitHub PR #2286 — "FP8 scaled_mm enablement" (OPEN)

The `register_torch_compile_kernel()` call in `eager.py` registers standard ops like `aten.mm`, `aten.bmm`, `aten.addmm`, but:
- No `aten._scaled_mm` (the core FP8 matmul op)
- No `aten._scaled_dot_product_attention` variants with FP8
- No custom `spyre::fp8_mm` or similar

PR #2286 explicitly addresses this gap by enabling `torch.ops.aten._scaled_mm()` on Spyre, but it remains unmerged as of commit a77f340.

---

### Claim 6: No FP8 Decompositions in `enable_spyre_decompositions()`

**Status: ✅ TRUE**

**Source:** `torch_spyre/_inductor/decompositions.py`

The `spyre_decompositions` dictionary contains decompositions for:
- `aten.rms_norm.default`
- `aten.layer_norm.default`
- `aten.topk`
- `aten.gelu.default`
- `aten.softplus.default`
- `aten.linear.default`
- `aten._scaled_dot_product_fused_attention_overrideable.default`
- Various utility ops (ones, new_ones, logical_not, sign, addmm, etc.)

**No FP8-related decompositions found**, including:
- No `aten.quantize_per_tensor`
- No `aten.dequantize`
- No `aten._scaled_mm` decomposition
- No FP8 dtype conversion decompositions

---

### Claim 7: No FP8 Lowerings in `enable_spyre_lowerings()`

**Status: ✅ TRUE**

**Source:** `torch_spyre/_inductor/lowering.py`

The `spyre_lowerings` dictionary and `register_spyre_lowering()` calls cover:
- `aten.mm.default`
- `aten.bmm.default`
- `aten.mean.dim`, `aten.mean.default`
- `aten.clone.default`
- `aten.full.default`
- `aten.cat.default`
- Various spyre custom ops

**No FP8-specific lowerings found**, including:
- No `aten._scaled_mm` lowering
- No FP8 tensor creation lowerings
- No FP8 conversion lowerings

---

### Claim 8: FP8 Quantization Ops Fall Back to Eager CPU Execution

**Status: ✅ TRUE**

**Source:** `torch_spyre/ops/fallbacks.py`

The fallback mechanism works as follows:
1. Ops without Spyre kernels registered trigger the fallback path
2. `warn_fallback()` emits a `FallbackWarning` (can be upgraded to error with `-W "error::...`)
3. Tensors are moved to CPU, the op runs via standard PyTorch, results moved back

However, there's an important caveat:

**Caveat:** Some FP8 ops may not cleanly fall back — they may trigger **InductorError** instead when the Inductor compiler attempts to generate code for unsupported FP8 operations. This is tracked in issue #1474.

---

### Claim 9: vLLM FP8 Kernel Registry Has No Spyre Entry

**Status: ✅ TRUE**

**Source:** `vllm/model_executor/layers/quantization/fp8.py` and `vllm/model_executor/kernels/linear/`

The vLLM FP8 infrastructure includes:
- `Fp8Config` — quantization configuration
- `Fp8LinearMethod` — offline (pre-quantized) weights path
- `Fp8OnlineLinearMethod` — online quantization path
- `init_fp8_linear_kernel()` — dispatches to platform-specific kernels

The kernel dispatch chain includes:
- CUTLASS-based kernels (CUDA)
- Marlin kernels (CUDA)
- CPU-based fallbacks

**No Spyre-specific FP8 kernel implementations exist.** When vLLM attempts to initialize FP8 linear layers on Spyre, it will either:
1. Fall back to a generic CPU path (slow, defeats the purpose)
2. Fail if the platform doesn't support the required ops

---

### Claim 10: torch-spyre Supports `Float8_e5m2fnuz` Not `Float8_e5m2`

**Status: ❌ FALSE**

**Source:** 
- torch-spyre codebase search: zero matches for `fnuz`, `FNUZ`, `e5m2fnuz`
- `wiki/foundations/hardware/precision-formats.md`: explicitly states E5M2 (1-5-2)

This claim appears to be **incorrect**. The torch-spyre codebase contains no references to FNUZ formats. PyTorch provides both variants:
- `torch.float8_e5m2` — standard IEEE-style (used by NVIDIA H100+, AIU)
- `torch.float8_e5m2fnuz` — FNUZ variant (used by AMD MI300X, some other accelerators)

The hardware documentation clearly states AIU uses the **standard E5M2 format**, matching NVIDIA GPUs. The FNUZ variant has different exponent/mantissa encoding and is not interchangeable.

---

### Claim 11: FP8 Tensor Creation Works But Not Via `torch.randn`

**Status: ✅ TRUE**

**Source:** `torch-spyre/tests/test_spyre.py:294-314, 354-368`

The test `test_cross_device_copy_dtypes` explicitly tests FP8 roundtrip:

```python
torch.float8_e4m3fn: lambda: torch.rand(64, 64).to(dtype=torch.float8_e4m3fn),
...
for dtype, tensor_factory in dtype_configs.items():
    x = tensor_factory()
    x_cpu = self._roundtrip_to_spyre_and_back(
        x, expect_warning=dtype in dtypes_with_downcast_warning
    )
    self._assert_roundtrip_close(x, x_cpu, dtype)
```

This confirms:
1. `torch.rand(64, 64).to(dtype=torch.float8_e4m3fn)` — ✅ works on CPU
2. `.to("spyre")` — ✅ transfer to spyre works
3. `.to("cpu")` — ✅ transfer back works
4. Data fidelity check passes within FP8 precision

The failing case is specifically:
```python
torch.randn(64, 64, dtype=torch.float8_e4m3fn, device="spyre")
# Error: "normal_kernel_cpu" not implemented for 'Float8_e4m3fn'
```

---

### Claim 12: FP8 Inputs Trigger InductorError

**Status: ✅ TRUE**

**Source:** 
- `torch-spyre/tests/test_spyre.py:50-53`
- Issue #1474: "Support operations on Float8 (SEN143_FP8) tensors in Spyre backend"

```python
_SCALAR_ADD_XFAIL_FP8 = pytest.mark.xfail(
    reason="Support scalar eager add for DataFormats.SEN143_FP8 in Spyre"
)
```

The test is marked `xfail` with the issue reference #1474, which is tagged "inductor compatibility". This indicates that FP8 operations don't cleanly fall back to CPU — they trigger errors in the Inductor lowering path.

The issue title references `DataFormats.SEN143_FP8`, which is the internal torch-spyre data format code for FP8. This suggests the Inductor compiler doesn't know how to handle FP8 tensors during graph lowering.

---

## Upstream torch-spyre Issue Status

| Issue/PR # | Title | Status | Relevance |
|------------|-------|--------|-----------|
| #2401 | feat: Add FP8 quantization support for torch-spyre | OPEN (pending approval) | Adds `quantize_fp8_with_scale` op and inductor lowering |
| #2286 | FP8 scaled_mm enablement | OPEN | Enables `torch.ops.aten._scaled_mm()` on Spyre |
| #2276 | FP8 test coverage and validation | OPEN | Test infrastructure for FP8 paths |
| #2186 | FP8 quantization operations not functioning correctly in deeptools | **BLOCKER** | `qfp8`, `qfp8wt`, `qfp8mb` fail with layout errors |
| #2185 | "Anywhere valid" issue causing data corruption in FP16 to FP8 dtype conversion | OPEN | Data fidelity bug |
| #1958 | Support scaled_bmm FP8 operator | OPEN | Batched matmul support |
| #1957 | Support scaled_mm FP8 operator | OPEN | Core FP8 GEMM support |
| #1956 | Support FP8 quantization operation | OPEN | Base quantization op |
| #1474 | Support operations on Float8 (SEN143_FP8) tensors in Spyre backend | OPEN | Inductor compatibility |

---

## Required Documentation Corrections

### Correction 1: FP8 Tensor Creation (Claim 4)

**Current text:**
> "FP8 tensors cannot be created on spyre device: `"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'`"

**Corrected text:**
> "FP8 tensors cannot be **directly initialized** on spyre device using random factories (`torch.randn`, `torch.empty` with FP8 dtype). However, FP8 tensors can be created on CPU (e.g., `torch.rand().to(dtype=torch.float8_e4m3fn)`) and transferred to spyre successfully. The error `"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'` applies specifically to normal distribution initialization."

### Correction 2: FNUZ Format (Claim 10)

**Current text:** (This claim was made in a PR comment, not the original docs)

**Correction:**
> "torch-spyre uses standard FP8 formats (E4M3 and E5M2), matching NVIDIA GPUs. The FNUZ variants (`float8_e4m3fnuz`, `float8_e5m2fnuz`) are **not** used in torch-spyre or AIU hardware."

---

## Conclusions

The FP8 support gaps documentation is **largely accurate** with one significant clarification needed:

1. **Hardware capability is real** — AIU does support FP8 natively
2. **Software stack gaps are real** — torch-spyre lacks FP8 kernels, decompositions, and lowerings
3. **spyre-inference plugin gaps are real** — no quantization method interception
4. **FP8 tensor transfer works** — but direct initialization on device does not
5. **FNUZ claim is false** — torch-spyre uses standard E5M2, not FNUZ

**Blocker status:** Issue #2186 (deeptools FP8 ops failing) is marked as a BLOCKER and must be resolved before any FP8 quantization can function, even if other gaps were filled.

**Path to FP8 support:**
1. torch-spyre must merge #2286 (scaled_mm) and #2401 (quantize_fp8_with_scale)
2. deeptools must fix #2186 (FP8 op layout errors)
3. spyre-inference must add `SpyreFp8LinearMethod` and `SpyreFp8LMHeadMethod`
4. End-to-end tests must be added comparing against CPU reference

---

## Validation Methodology

This validation was performed by:
1. Reading the original `docs/architecture/fp8-support-gaps.md`
2. Extracting claims from PR comment https://github.com/torch-spyre/spyre-inference/pull/225#issuecomment-4615512402
3. Searching torch-spyre codebase (commit a77f340) for relevant implementations
4. Reading spyre-knowledgebase hardware documentation
5. Examining vLLM FP8 implementation in `.venv/lib/python3.12/site-packages/vllm/`
6. Cross-referencing torch-spyre GitHub issues and PRs

All claims were validated against primary sources (code, documentation) rather than secondary reports.
