// Copyright 2026 The Spyre-Inference Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions adapted from vLLM (https://github.com/vllm-project/vllm),
// Copyright contributors to the vLLM project, Apache-2.0.

// Vectorised expf over vec_op::FP32Vec16, adapted from the DEFINE_FAST_EXP
// macro in vLLM csrc/cpu/cpu_arch_macros.h (AVX512 from vllm#62440, AVX2 from
// vllm#43452). The s390x path is the same polynomial on VXE lanes. Defined
// only on ISAs with a SIMD FP32Vec16; callers keep a scalar expf path for
// the rest.
#ifndef SPYRE_CPU_FAST_EXP_HPP
#define SPYRE_CPU_FAST_EXP_HPP

#include "cpu_types.hpp"

#include <cstring>

namespace spyre_cpu {

#if defined(__x86_64__) && defined(__AVX512F__)

// 5-term polynomial for 2^frac, scaled by 2^n; inputs below ln(FLT_MIN)
// flush to 0. Needs AVX512F.
inline __attribute__((always_inline)) vec_op::FP32Vec16 fast_exp(
    const vec_op::FP32Vec16& vec) {
  const __m512 c1 = _mm512_set1_ps(0.999999701f);
  const __m512 c2 = _mm512_set1_ps(0.499991506f);
  const __m512 c3 = _mm512_set1_ps(0.166676521f);
  const __m512 c4 = _mm512_set1_ps(0.0418978221f);
  const __m512 c5 = _mm512_set1_ps(0.00828929059f);
  const __m512 log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
  const __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));
  const __m512 ln_flt_min = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
  const __m512 ln_flt_max = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 one = _mm512_set1_ps(1.f);
  const __m512 two = _mm512_set1_ps(2.f);
  const __m512 zero = _mm512_setzero_ps();
  const __m512i bias = _mm512_set1_epi32(127);
  constexpr int kMantissaBits = 23;

  const __m512 x = vec.reg;
  const __mmask16 underflow = _mm512_cmp_ps_mask(x, ln_flt_min, _CMP_LT_OS);
  __m512 src = _mm512_max_ps(_mm512_min_ps(x, ln_flt_max), ln_flt_min);

  __m512 fx = _mm512_fmadd_ps(src, log2ef, half);
  const __m512i fx_i =
      _mm512_cvt_roundps_epi32(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  fx = _mm512_cvtepi32_ps(fx_i);

  const __m512 r = _mm512_fnmadd_ps(fx, ln2f, src);
  __m512 poly = _mm512_fmadd_ps(r, c5, c4);
  poly = _mm512_fmadd_ps(r, poly, c3);
  poly = _mm512_fmadd_ps(r, poly, c2);
  poly = _mm512_fmadd_ps(r, poly, c1);
  poly = _mm512_fmadd_ps(r, poly, one);

  // 2^(fx - 1) via the exponent bits; the final *2 restores 2^fx without
  // overflowing the exponent at fx = 128.
  __m512i pow2_i = _mm512_cvtps_epi32(_mm512_sub_ps(fx, one));
  pow2_i = _mm512_slli_epi32(_mm512_add_epi32(pow2_i, bias), kMantissaBits);
  __m512 pow2 = _mm512_castsi512_ps(pow2_i);
  pow2 = _mm512_mask_blend_ps(underflow, pow2, zero);

  return vec_op::FP32Vec16(_mm512_mul_ps(_mm512_mul_ps(poly, pow2), two));
}

#elif defined(__x86_64__) && defined(__AVX2__)

// Same algorithm as the AVX512 path on two 8-lane halves. The pragma defines
// __FMA__ so the fmadd intrinsics exist; the target attribute matches the one
// on _top_p_row_core and top_p_row, which is what lets this inline there. The
// AVX2 extension is built with -mno-fma: omitting -mfma is not enough when GCC
// defaults to x86-64-v3, which leaves FMA on after -mavx2. Gumbel and rejection
// stay clear of it.
#pragma GCC push_options
#pragma GCC target("avx2,f16c,fma")

__attribute__((target("avx2,f16c,fma"), always_inline)) inline __m256 fast_exp8(
    __m256 x) {
  const __m256 c1 = _mm256_set1_ps(0.999999701f);
  const __m256 c2 = _mm256_set1_ps(0.499991506f);
  const __m256 c3 = _mm256_set1_ps(0.166676521f);
  const __m256 c4 = _mm256_set1_ps(0.0418978221f);
  const __m256 c5 = _mm256_set1_ps(0.00828929059f);
  const __m256 log2ef = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));
  const __m256 ln2f = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));
  const __m256 ln_flt_min = _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));
  const __m256 ln_flt_max = _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.f);
  const __m256 two = _mm256_set1_ps(2.f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256i bias = _mm256_set1_epi32(127);
  constexpr int kMantissaBits = 23;

  const __m256 underflow = _mm256_cmp_ps(x, ln_flt_min, _CMP_LT_OS);
  __m256 src = _mm256_max_ps(_mm256_min_ps(x, ln_flt_max), ln_flt_min);

  // AVX2 has no cvt_roundps_epi32: floor first, then convert.
  __m256 fx = _mm256_fmadd_ps(src, log2ef, half);
  fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  fx = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(fx));

  const __m256 r = _mm256_fnmadd_ps(fx, ln2f, src);
  __m256 poly = _mm256_fmadd_ps(r, c5, c4);
  poly = _mm256_fmadd_ps(r, poly, c3);
  poly = _mm256_fmadd_ps(r, poly, c2);
  poly = _mm256_fmadd_ps(r, poly, c1);
  poly = _mm256_fmadd_ps(r, poly, one);

  __m256i pow2_i = _mm256_cvtps_epi32(_mm256_sub_ps(fx, one));
  pow2_i = _mm256_slli_epi32(_mm256_add_epi32(pow2_i, bias), kMantissaBits);
  __m256 pow2 = _mm256_castsi256_ps(pow2_i);
  // blendv picks the second operand where the mask's sign bit is set.
  pow2 = _mm256_blendv_ps(pow2, zero, underflow);

  return _mm256_mul_ps(_mm256_mul_ps(poly, pow2), two);
}

__attribute__((target("avx2,f16c,fma"), always_inline)) inline vec_op::FP32Vec16
fast_exp(const vec_op::FP32Vec16& vec) {
  return vec_op::FP32Vec16(fast_exp8(vec.reg_low), fast_exp8(vec.reg_high));
}

#pragma GCC pop_options

#elif defined(__s390x__)

// Same polynomial as the x86 paths, one VXE register (4 lanes) at a time.
// Needs -mvx -mzvector.
inline __attribute__((always_inline)) __vector float splat_f32_bits(
    unsigned int bits) {
  // Both ends store the integer and the float with the same endianness, so
  // copying the object representation keeps the IEEE bits.
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return vec_splats(value);
}

inline __attribute__((always_inline)) __vector float fast_exp4(__vector float x) {
  const __vector float c1 = vec_splats(0.999999701f);
  const __vector float c2 = vec_splats(0.499991506f);
  const __vector float c3 = vec_splats(0.166676521f);
  const __vector float c4 = vec_splats(0.0418978221f);
  const __vector float c5 = vec_splats(0.00828929059f);
  const __vector float log2ef = splat_f32_bits(0x3fb8aa3b);
  const __vector float ln2f = splat_f32_bits(0x3f317218);
  const __vector float ln_flt_min = splat_f32_bits(0xc2aeac50);
  const __vector float ln_flt_max = splat_f32_bits(0x42b17218);
  const __vector float half = vec_splats(0.5f);
  const __vector float one = vec_splats(1.f);
  const __vector float two = vec_splats(2.f);
  const __vector float zero = vec_splats(0.f);

  const auto underflow = vec_cmplt(x, ln_flt_min);
  __vector float src = vec_max(vec_min(x, ln_flt_max), ln_flt_min);

  __vector float fx = vec_madd(src, log2ef, half);
  fx = vec_floor(fx);

  // src - fx * ln2, fused.
  const __vector float r = vec_nmsub(fx, ln2f, src);
  __vector float poly = vec_madd(r, c5, c4);
  poly = vec_madd(r, poly, c3);
  poly = vec_madd(r, poly, c2);
  poly = vec_madd(r, poly, c1);
  poly = vec_madd(r, poly, one);

  // 2^(fx - 1) via the exponent bits. fx is an exact integer here, so the
  // toward-zero conversion matches the x86 round-to-nearest convert. The
  // final *2 restores 2^fx without overflowing the exponent at fx = 128.
  __vector signed int pow2_i = vec_signed(vec_sub(fx, one));
  pow2_i = vec_add(pow2_i, vec_splats((signed int)127));
  __vector unsigned int pow2_bits = (__vector unsigned int)pow2_i;
  pow2_bits = vec_sl(pow2_bits, vec_splats((unsigned int)23));
  __vector float pow2 = (__vector float)pow2_bits;
  // vec_sel takes the second operand where the mask is true.
  pow2 = vec_sel(pow2, zero, underflow);

  return vec_mul(vec_mul(poly, pow2), two);
}

inline __attribute__((always_inline)) vec_op::FP32Vec16 fast_exp(
    const vec_op::FP32Vec16& vec) {
  vec_op::f32x4x4_t out;
  out.val[0] = fast_exp4(vec.reg.val[0]);
  out.val[1] = fast_exp4(vec.reg.val[1]);
  out.val[2] = fast_exp4(vec.reg.val[2]);
  out.val[3] = fast_exp4(vec.reg.val[3]);
  return vec_op::FP32Vec16(out);
}

#elif defined(__powerpc__)

// FP32Vec16::exp() in cpu_types_vsx.hpp is already a vectorised minimax
// polynomial.
inline __attribute__((always_inline)) vec_op::FP32Vec16 fast_exp(
    const vec_op::FP32Vec16& vec) {
  return vec.exp();
}

#endif

}  // namespace spyre_cpu

#endif  // SPYRE_CPU_FAST_EXP_HPP
