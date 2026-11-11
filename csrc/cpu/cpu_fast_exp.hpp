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
// vllm#43452). Defined only on ISAs with a SIMD FP32Vec16; callers keep a
// scalar expf path for the rest.
#ifndef SPYRE_CPU_FAST_EXP_HPP
#define SPYRE_CPU_FAST_EXP_HPP

#include "cpu_types.hpp"

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

// Same algorithm as the AVX512 path on two 8-lane halves. Needs AVX2 + FMA.
inline __attribute__((always_inline)) __m256 fast_exp8(__m256 x) {
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

inline __attribute__((always_inline)) vec_op::FP32Vec16 fast_exp(
    const vec_op::FP32Vec16& vec) {
  return vec_op::FP32Vec16(fast_exp8(vec.reg_low), fast_exp8(vec.reg_high));
}

#elif defined(__powerpc__) || defined(__s390x__)

// FP32Vec16::exp() in cpu_types_vsx.hpp / cpu_types_vxe.hpp is already a
// vectorised minimax polynomial.
inline __attribute__((always_inline)) vec_op::FP32Vec16 fast_exp(
    const vec_op::FP32Vec16& vec) {
  return vec.exp();
}

#endif

}  // namespace spyre_cpu

#endif  // SPYRE_CPU_FAST_EXP_HPP
