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

#ifndef CPU_TYPES_X86_HPP
#define CPU_TYPES_X86_HPP

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifndef __AVX2__
static_assert(false, "AVX2 must be supported for the current implementation.");
#endif

namespace vec_op {

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    __m256 reg;
    float values[VEC_ELEM_NUM];
  };

  __m256 reg;

  explicit FP32Vec8(float v) : reg(_mm256_set1_ps(v)) {}

  explicit FP32Vec8() : reg(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec8(const float* ptr) : reg(_mm256_loadu_ps(ptr)) {}

  explicit FP32Vec8(__m256 data) : reg(data) {}

  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, &ar](int i) { result += ar.values[i]; });

    return result;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(expf(ar.values[7]), expf(ar.values[6]),
                                  expf(ar.values[5]), expf(ar.values[4]),
                                  expf(ar.values[3]), expf(ar.values[2]),
                                  expf(ar.values[1]), expf(ar.values[0])));
  }

  FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(tanhf(ar.values[7]), tanhf(ar.values[6]),
                                  tanhf(ar.values[5]), tanhf(ar.values[4]),
                                  tanhf(ar.values[3]), tanhf(ar.values[2]),
                                  tanhf(ar.values[1]), tanhf(ar.values[0])));
  }

  FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(erf(ar.values[7]), erf(ar.values[6]),
                                  erf(ar.values[5]), erf(ar.values[4]),
                                  erf(ar.values[3]), erf(ar.values[2]),
                                  erf(ar.values[1]), erf(ar.values[0])));
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_mul_ps(reg, b.reg));
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_add_ps(reg, b.reg));
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_sub_ps(reg, b.reg));
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_div_ps(reg, b.reg));
  }

  void save(float* ptr) const { _mm256_storeu_ps(ptr, reg); }
};

#ifdef __AVX512F__
struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    __m512 reg;
    float values[VEC_ELEM_NUM];
  };

  __m512 reg;

  explicit FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}

  explicit FP32Vec16() : reg(_mm512_set1_ps(0.0)) {}

  // normal load
  explicit FP32Vec16(const float* ptr) : reg(_mm512_loadu_ps(ptr)) {}

  // non-temporal load
  explicit FP32Vec16(bool, void* ptr)
      : reg((__m512)_mm512_stream_load_si512(ptr)) {}

  explicit FP32Vec16(__m512 data) : reg(data) {}

  // de-pack 4 bit values
  explicit FP32Vec16(int64_t value, const FP32Vec16& lut) {
    int64_t mask_0 = 0x0F0F0F0F0F0F0F0F;
    int64_t mask_1 = 0xF0F0F0F0F0F0F0F0;
    int64_t value_0 = value & mask_0;
    int64_t value_1 = value & mask_1;
    __m128i vec_0 = _mm_movpi64_epi64((__m64)value_0);
    __m128i vec_1 = _mm_movpi64_epi64((__m64)value_1);
    vec_0 = _mm_cvtepu8_epi16(vec_0);
    vec_1 = _mm_cvtepu8_epi16(vec_1);
    vec_1 = _mm_slli_epi16(vec_1, 4);
    __m128i vec = _mm_or_si128(vec_0, vec_1);
    __m512i vec_i32 = _mm512_cvtepu8_epi32(vec);
    reg = _mm512_permutexvar_ps(vec_i32, lut.reg);
  }

  explicit FP32Vec16(const FP32Vec8& data)
      : reg((__m512)_mm512_inserti32x8(
            _mm512_castsi256_si512((__m256i)data.reg), (__m256i)data.reg, 1)) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_mul_ps(reg, b.reg));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_add_ps(reg, b.reg));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_sub_ps(reg, b.reg));
  }

  FP32Vec16 operator-() const {
    return FP32Vec16(_mm512_xor_ps(reg, _mm512_set1_ps(-0.0f)));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_div_ps(reg, b.reg));
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(_mm512_min_ps(max.reg, _mm512_max_ps(min.reg, reg)));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_max_ps(reg, b.reg));
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_max_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_min_ps(reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_min_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 abs() const { return FP32Vec16(_mm512_abs_ps(reg)); }

  float reduce_sum() const { return _mm512_reduce_add_ps(reg); }

  float reduce_max() const { return _mm512_reduce_max_ps(reg); }

  float reduce_min() const { return _mm512_reduce_min_ps(reg); }

  float get_last_elem() const { return _mm512_cvtss_f32(reg); }

  void save(float* ptr) const { _mm512_storeu_ps(ptr, reg); }

  void save(float* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm512_mask_storeu_ps(ptr, mask, reg);
  }
};
#else
struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    __m256 reg;
    float values[8];
  };

  __m256 reg_low;
  __m256 reg_high;

  explicit FP32Vec16(float v)
      : reg_low(_mm256_set1_ps(v)), reg_high(_mm256_set1_ps(v)) {}

  explicit FP32Vec16()
      : reg_low(_mm256_set1_ps(0.0)), reg_high(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec16(const float* ptr)
      : reg_low(_mm256_loadu_ps(ptr)), reg_high(_mm256_loadu_ps(ptr + 8)) {}

  explicit FP32Vec16(__m256 low, __m256 high) : reg_low(low), reg_high(high) {}

  explicit FP32Vec16(const FP32Vec8& data)
      : reg_low(data.reg), reg_high(data.reg) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_mul_ps(reg_low, b.reg_low),
                     _mm256_mul_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_add_ps(reg_low, b.reg_low),
                     _mm256_add_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_sub_ps(reg_low, b.reg_low),
                     _mm256_sub_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator-() const {
    const __m256 neg = _mm256_set1_ps(-0.0f);
    return FP32Vec16(_mm256_xor_ps(reg_low, neg), _mm256_xor_ps(reg_high, neg));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_div_ps(reg_low, b.reg_low),
                     _mm256_div_ps(reg_high, b.reg_high));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_max_ps(reg_low, b.reg_low),
                     _mm256_max_ps(reg_high, b.reg_high));
  }

  float reduce_max() const {
    __m256 v = _mm256_max_ps(reg_low, reg_high);
    // Permute to compare elements within 128-bit lanes
    __m256 v_shuffled = _mm256_permute_ps(
        v, 0b00001011);  // Swap halves within each 128-bit lane
    __m256 v_max = _mm256_max_ps(v, v_shuffled);

    v_shuffled = _mm256_permute_ps(
        v_max, 0b00000001);  // Shuffle elements within each 128-bit lane
    v_max = _mm256_max_ps(v_max, v_shuffled);

    // Permute to compare elements between 128-bit lanes
    v_shuffled =
        _mm256_permute2f128_ps(v_max, v_max, 0b00000001);  // Swap 128-bit lanes
    v_max = _mm256_max_ps(v_max, v_shuffled);

    // At this point, the maximum value is present in all elements of v_max.
    // Extract the first element for the scalar result.
    return _mm256_cvtss_f32(v_max);  // Extract the lowest 32-bit float
  }

  float reduce_sum() const {
    FP32Vec8 low = FP32Vec8(reg_low);
    FP32Vec8 high = FP32Vec8(reg_high);
    return low.reduce_sum() + high.reduce_sum();
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    float sum = 0.0;
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr uint32_t base_mask = (0xFFFF >> (16 - group_size));
    uint32_t mask = base_mask << (idx * group_size);

    AliasReg ar;

    auto func = [&sum, &mask, &ar](int i) {
      int flag = mask & 0x1;
      mask = mask >> 1;
      if (flag != 0) sum += ar.values[i];
    };

    ar.reg = reg_low;
    unroll_loop<int, 8>(func);

    ar.reg = reg_high;
    unroll_loop<int, 8>(func);

    return sum;
  }

  void save(float* ptr) const {
    _mm256_storeu_ps(ptr, reg_low);
    _mm256_storeu_ps(ptr + 8, reg_high);
  }

  void save(float* ptr, const int elem_num) const {
    // Partial store: cmpgt produces a sign-bit mask (0xFFFFFFFF/0 per lane)
    // for the first elem_num lanes, applied across the two 8-wide halves.
    if (elem_num <= 8) {
      __m256i mask =
          _mm256_cmpgt_epi32(_mm256_set1_epi32(elem_num),
                             _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      _mm256_maskstore_ps(ptr, mask, reg_low);
    } else {
      _mm256_storeu_ps(ptr, reg_low);
      __m256i mask =
          _mm256_cmpgt_epi32(_mm256_set1_epi32(elem_num - 8),
                             _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      _mm256_maskstore_ps(ptr + 8, mask, reg_high);
    }
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(
        _mm256_min_ps(max.reg_low, _mm256_max_ps(min.reg_low, reg_low)),
        _mm256_min_ps(max.reg_high, _mm256_max_ps(min.reg_high, reg_high)));
  }

  FP32Vec16 abs() const {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    return FP32Vec16(_mm256_andnot_ps(sign_mask, reg_low),
                     _mm256_andnot_ps(sign_mask, reg_high));
  }

  FP32Vec16 tanh() const {
    FP32Vec8 low(reg_low);
    FP32Vec8 high(reg_high);
    return FP32Vec16(low.tanh().reg, high.tanh().reg);
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_min_ps(reg_low, b.reg_low),
                     _mm256_min_ps(reg_high, b.reg_high));
  }

  // Partial element-wise min over the first elem_num lanes only (tail path).
  // Scalar via AliasReg: AVX2 has no masked vminps, so we spill, loop, reload.
  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    AliasReg ar_this_low, ar_this_high, ar_b_low, ar_b_high;
    ar_this_low.reg = reg_low;
    ar_this_high.reg = reg_high;
    ar_b_low.reg = b.reg_low;
    ar_b_high.reg = b.reg_high;
    for (int i = 0; i < elem_num && i < 8; ++i)
      ar_this_low.values[i] =
          std::min(ar_this_low.values[i], ar_b_low.values[i]);
    for (int i = 0; i < elem_num - 8 && i < 8; ++i)
      ar_this_high.values[i] =
          std::min(ar_this_high.values[i], ar_b_high.values[i]);
    return FP32Vec16(ar_this_low.reg, ar_this_high.reg);
  }

  // Partial element-wise max over the first elem_num lanes only (tail path).
  // Scalar via AliasReg: AVX2 has no masked vmaxps, so we spill, loop, reload.
  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    AliasReg ar_this_low, ar_this_high, ar_b_low, ar_b_high;
    ar_this_low.reg = reg_low;
    ar_this_high.reg = reg_high;
    ar_b_low.reg = b.reg_low;
    ar_b_high.reg = b.reg_high;
    for (int i = 0; i < elem_num && i < 8; ++i)
      ar_this_low.values[i] =
          std::max(ar_this_low.values[i], ar_b_low.values[i]);
    for (int i = 0; i < elem_num - 8 && i < 8; ++i)
      ar_this_high.values[i] =
          std::max(ar_this_high.values[i], ar_b_high.values[i]);
    return FP32Vec16(ar_this_low.reg, ar_this_high.reg);
  }

  float reduce_min() const {
    __m256 v = _mm256_min_ps(reg_low, reg_high);
    __m256 v_shuffled = _mm256_permute_ps(v, 0b00001011);
    __m256 v_min = _mm256_min_ps(v, v_shuffled);
    v_shuffled = _mm256_permute_ps(v_min, 0b00000001);
    v_min = _mm256_min_ps(v_min, v_shuffled);
    v_shuffled = _mm256_permute2f128_ps(v_min, v_min, 0b00000001);
    v_min = _mm256_min_ps(v_min, v_shuffled);
    return _mm256_cvtss_f32(v_min);
  }
};
#endif

};  // namespace vec_op

#endif
