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

#ifndef CPU_TYPES_VSX_HPP
#define CPU_TYPES_VSX_HPP

#include <altivec.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

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
typedef struct f32x4x2_t {
  __vector float val[2];
} f32x4x2_t;

typedef struct f32x4x4_t {
  __vector float val[4];
} f32x4x4_t;

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    f32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x2_t reg;

  explicit FP32Vec8(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
  }

  explicit FP32Vec8() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
  }

  explicit FP32Vec8(const float* ptr) {
    reg.val[0] = vec_xl(0, ptr);
    reg.val[1] = vec_xl(16, ptr);
  }

  explicit FP32Vec8(f32x4x2_t data) : reg(data) {}

  FP32Vec8(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
  }

  float reduce_sum() const {
    // VSX horizontal reduction: 3 vector ops instead of 8 scalar adds.
    // Step 1: pairwise sum of the two 4-wide halves
    __vector float s = vec_add(reg.val[0], reg.val[1]);
    // Step 2: rotate by 8 bytes (2 floats) and add
    s = vec_add(s, vec_sld(s, s, 8));
    // Step 3: rotate by 4 bytes (1 float) and add  => all lanes hold total
    s = vec_add(s, vec_sld(s, s, 4));
    return vec_extract(s, 0);
  }
  FP32Vec8 exp() const {
    f32x4x2_t out;
    const __vector float log2e = vec_splats(1.44269504088896341f);
    const __vector float one = vec_splats(1.0f);
    const __vector float min_x = vec_splats(-87.3f);
    const __vector float max_x = vec_splats(88.7f);

    // 5th-degree minimax polynomial for 2^r (r in [0,1))
    const __vector float c1 = vec_splats(0.6931471805599453f);
    const __vector float c2 = vec_splats(0.240226506959101f);
    const __vector float c3 = vec_splats(0.05550410866482158f);
    const __vector float c4 = vec_splats(0.009618129107628477f);
    const __vector float c5 = vec_splats(0.0013333558146428443f);

    for (int i = 0; i < 2; i++) {
      __vector float x = reg.val[i];
      x = vec_max(x, min_x);
      x = vec_min(x, max_x);

      __vector float y = vec_mul(x, log2e);

      __vector float kf = vec_floor(y);
      __vector float r = vec_sub(y, kf);

      // Convert float to signed integer. Use vec_cts for PowerPC AltiVec
      // compatibility.
      __vector signed int k = vec_cts(kf, 0);
      const __vector signed int min_k = vec_splats((signed int)-126);
      const __vector signed int max_k = vec_splats((signed int)127);
      k = vec_min(vec_max(k, min_k), max_k);

      // Build 2^k from exponent bits
      __vector signed int exp_int = vec_add(k, vec_splats((signed int)127));
      __vector unsigned int bits = (__vector unsigned int)exp_int;
      bits = vec_sl(bits, vec_splats((unsigned int)23));
      __vector float pow2k = (__vector float)bits;

      // Improved minimax polynomial
      __vector float poly = vec_madd(c5, r, c4);
      poly = vec_madd(poly, r, c3);
      poly = vec_madd(poly, r, c2);
      poly = vec_madd(poly, r, c1);
      poly = vec_madd(poly, r, one);

      out.val[i] = vec_mul(pow2k, poly);
    }
    return FP32Vec8(out);
  }

  FP32Vec8 tanh() const {
    const __vector float one = vec_splats(1.0f);
    const __vector float two = vec_splats(2.0f);
    const __vector float zero = vec_splats(0.0f);
    const __vector float sat = vec_splats(9.0f);

    f32x4x2_t out;

    for (int i = 0; i < 2; i++) {
      __vector float x = reg.val[i];
      __vector float ax = vec_abs(x);

      __vector bool int mask = vec_cmpge(x, zero);
      __vector float sign = vec_sel(vec_splats(-1.0f), one, mask);

      __vector bool int saturated = vec_cmpge(ax, sat);

      __vector float two_x = vec_mul(x, two);
      f32x4x2_t tmp;
      tmp.val[0] = two_x;
      tmp.val[1] = two_x;
      FP32Vec8 temp_vec(tmp);
      FP32Vec8 exp_vec = temp_vec.exp();
      vector float e = temp_vec.exp().reg.val[0];

      vector float num = vec_sub(e, one);
      vector float den = vec_add(e, one);
      vector float t = vec_div(num, den);

      out.val[i] = vec_sel(t, sign, saturated);
    }
    return FP32Vec8(out);
  }

  FP32Vec8 er() const {
    const vector float a1 = vec_splats(0.254829592f);
    const vector float a2 = vec_splats(-0.284496736f);
    const vector float a3 = vec_splats(1.421413741f);
    const vector float a4 = vec_splats(-1.453152027f);
    const vector float a5 = vec_splats(1.061405429f);
    const vector float p = vec_splats(0.3275911f);
    const vector float one = vec_splats(1.0f);
    const vector float zero = vec_splats(0.0f);
    const vector float sat = vec_splats(6.0f);

    f32x4x2_t ret;

    for (int i = 0; i < 2; i++) {
      vector float x = reg.val[i];
      vector float ax = vec_abs(x);

      vector bool int mask = vec_cmpge(x, zero);
      vector float sign = vec_sel(vec_splats(-1.0f), one, mask);

      vector bool int saturated = vec_cmpge(ax, sat);

      vector float t = vec_div(one, vec_madd(p, ax, one));

      vector float poly = a5;
      poly = vec_madd(poly, t, a4);
      poly = vec_madd(poly, t, a3);
      poly = vec_madd(poly, t, a2);
      poly = vec_madd(poly, t, a1);
      poly = vec_mul(poly, t);

      vector float x_squared = vec_mul(x, x);
      vector float neg_x_squared = vec_mul(vec_splats(-1.0f), x_squared);
      f32x4x2_t tmp;
      tmp.val[0] = neg_x_squared;
      tmp.val[1] = neg_x_squared;
      FP32Vec8 exp_input(tmp);
      vector float exp_term = exp_input.exp().reg.val[0];

      vector float y = vec_nmsub(poly, exp_term, one);
      vector float erf_val = vec_mul(sign, y);

      ret.val[i] = vec_sel(erf_val, sign, saturated);
    }
    return FP32Vec8(ret);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_mul(reg.val[0], b.reg.val[0]), vec_mul(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_add(reg.val[0], b.reg.val[0]), vec_add(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_sub(reg.val[0], b.reg.val[0]), vec_sub(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_div(reg.val[0], b.reg.val[0]), vec_div(reg.val[1], b.reg.val[1])});
  }

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    f32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x4_t reg;

  explicit FP32Vec16(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
    reg.val[2] = vec_splats(v);
    reg.val[3] = vec_splats(v);
  }

  explicit FP32Vec16() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
    reg.val[2] = vec_splats(0.0f);
    reg.val[3] = vec_splats(0.0f);
  }

  explicit FP32Vec16(const float* ptr) {
    reg.val[0] = vec_xl(0, ptr);
    reg.val[1] = vec_xl(16, ptr);
    reg.val[2] = vec_xl(32, ptr);
    reg.val[3] = vec_xl(48, ptr);
  }

  explicit FP32Vec16(bool, const float* ptr) : FP32Vec16(ptr) {}
  explicit FP32Vec16(f32x4x4_t data) : reg(data) {}

  FP32Vec16(const FP32Vec16& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[2];
    reg.val[3] = data.reg.val[3];
  }

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  }

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_mul(reg.val[0], b.reg.val[0]),
                                vec_mul(reg.val[1], b.reg.val[1]),
                                vec_mul(reg.val[2], b.reg.val[2]),
                                vec_mul(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_add(reg.val[0], b.reg.val[0]),
                                vec_add(reg.val[1], b.reg.val[1]),
                                vec_add(reg.val[2], b.reg.val[2]),
                                vec_add(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator-() const {
    const __vector float zero = vec_splats(0.0f);
    return FP32Vec16(
        f32x4x4_t({vec_sub(zero, reg.val[0]), vec_sub(zero, reg.val[1]),
                   vec_sub(zero, reg.val[2]), vec_sub(zero, reg.val[3])}));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_sub(reg.val[0], b.reg.val[0]),
                                vec_sub(reg.val[1], b.reg.val[1]),
                                vec_sub(reg.val[2], b.reg.val[2]),
                                vec_sub(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_div(reg.val[0], b.reg.val[0]),
                                vec_div(reg.val[1], b.reg.val[1]),
                                vec_div(reg.val[2], b.reg.val[2]),
                                vec_div(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(f32x4x4_t(
        {vec_min(max.reg.val[0], vec_max(min.reg.val[0], reg.val[0])),
         vec_min(max.reg.val[1], vec_max(min.reg.val[1], reg.val[1])),
         vec_min(max.reg.val[2], vec_max(min.reg.val[2], reg.val[2])),
         vec_min(max.reg.val[3], vec_max(min.reg.val[3], reg.val[3]))}));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_max(reg.val[0], b.reg.val[0]),
                                vec_max(reg.val[1], b.reg.val[1]),
                                vec_max(reg.val[2], b.reg.val[2]),
                                vec_max(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 max(const FP32Vec16& b, int elem_num) const {
    FP32Vec16 result;

    __vector unsigned int indices = {0, 1, 2, 3};
    __vector unsigned int elem_num_vec =
        vec_splats(static_cast<unsigned int>(elem_num));

    __vector unsigned int chunk_offset0 = {0, 0, 0, 0};
    __vector unsigned int chunk_offset1 = {4, 4, 4, 4};
    __vector unsigned int chunk_offset2 = {8, 8, 8, 8};
    __vector unsigned int chunk_offset3 = {12, 12, 12, 12};

    __vector bool int mask0 = vec_cmplt(indices + chunk_offset0, elem_num_vec);
    __vector bool int mask1 = vec_cmplt(indices + chunk_offset1, elem_num_vec);
    __vector bool int mask2 = vec_cmplt(indices + chunk_offset2, elem_num_vec);
    __vector bool int mask3 = vec_cmplt(indices + chunk_offset3, elem_num_vec);

    result.reg.val[0] = vec_sel(this->reg.val[0],
                                vec_max(this->reg.val[0], b.reg.val[0]), mask0);
    result.reg.val[1] = vec_sel(this->reg.val[1],
                                vec_max(this->reg.val[1], b.reg.val[1]), mask1);
    result.reg.val[2] = vec_sel(this->reg.val[2],
                                vec_max(this->reg.val[2], b.reg.val[2]), mask2);
    result.reg.val[3] = vec_sel(this->reg.val[3],
                                vec_max(this->reg.val[3], b.reg.val[3]), mask3);

    return FP32Vec16(result.reg);
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_min(reg.val[0], b.reg.val[0]),
                                vec_min(reg.val[1], b.reg.val[1]),
                                vec_min(reg.val[2], b.reg.val[2]),
                                vec_min(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 min(const FP32Vec16& b, int elem_num) const {
    FP32Vec16 result;

    vector unsigned int indices = {0, 1, 2, 3};
    vector unsigned int elem_num_vec =
        vec_splats(static_cast<unsigned int>(elem_num));

    vector unsigned int chunk_offset0 = {0, 0, 0, 0};
    vector unsigned int chunk_offset1 = {4, 4, 4, 4};
    vector unsigned int chunk_offset2 = {8, 8, 8, 8};
    vector unsigned int chunk_offset3 = {12, 12, 12, 12};

    vector bool int mask0 = vec_cmplt(indices + chunk_offset0, elem_num_vec);
    vector bool int mask1 = vec_cmplt(indices + chunk_offset1, elem_num_vec);
    vector bool int mask2 = vec_cmplt(indices + chunk_offset2, elem_num_vec);
    vector bool int mask3 = vec_cmplt(indices + chunk_offset3, elem_num_vec);

    result.reg.val[0] = vec_sel(this->reg.val[0],
                                vec_min(this->reg.val[0], b.reg.val[0]), mask0);
    result.reg.val[1] = vec_sel(this->reg.val[1],
                                vec_min(this->reg.val[1], b.reg.val[1]), mask1);
    result.reg.val[2] = vec_sel(this->reg.val[2],
                                vec_min(this->reg.val[2], b.reg.val[2]), mask2);
    result.reg.val[3] = vec_sel(this->reg.val[3],
                                vec_min(this->reg.val[3], b.reg.val[3]), mask3);

    return FP32Vec16(result.reg);
  }

  FP32Vec16 abs() const {
    return FP32Vec16(f32x4x4_t({vec_abs(reg.val[0]), vec_abs(reg.val[1]),
                                vec_abs(reg.val[2]), vec_abs(reg.val[3])}));
  }

  FP32Vec16 exp() const {
    FP32Vec8 lo(f32x4x2_t{reg.val[0], reg.val[1]});
    FP32Vec8 hi(f32x4x2_t{reg.val[2], reg.val[3]});
    auto lo_e = lo.exp();
    auto hi_e = hi.exp();
    return FP32Vec16(f32x4x4_t{lo_e.reg.val[0], lo_e.reg.val[1],
                               hi_e.reg.val[0], hi_e.reg.val[1]});
  }

  FP32Vec16 tanh() const {
    FP32Vec8 lo(f32x4x2_t{reg.val[0], reg.val[1]});
    FP32Vec8 hi(f32x4x2_t{reg.val[2], reg.val[3]});
    auto lo_tanh = lo.tanh();
    auto hi_tanh = hi.tanh();
    return FP32Vec16(f32x4x4_t{lo_tanh.reg.val[0], lo_tanh.reg.val[1],
                               hi_tanh.reg.val[0], hi_tanh.reg.val[1]});
  }

  FP32Vec16 er() const {
    FP32Vec8 lo(f32x4x2_t{reg.val[0], reg.val[1]});
    FP32Vec8 hi(f32x4x2_t{reg.val[2], reg.val[3]});
    auto lo_er = lo.er();
    auto hi_er = hi.er();
    return FP32Vec16(f32x4x4_t{lo_er.reg.val[0], lo_er.reg.val[1],
                               hi_er.reg.val[0], hi_er.reg.val[1]});
  }

  float reduce_max() {
    __vector float max01 = vec_max(reg.val[0], reg.val[1]);
    __vector float max23 = vec_max(reg.val[2], reg.val[3]);
    __vector float max_all = vec_max(max01, max23);
    __vector float temp = vec_max(max_all, vec_sld(max_all, max_all, 8));
    temp = vec_max(temp, vec_sld(temp, temp, 4));
    return vec_extract(temp, 0);
  }

  float reduce_min() {
    __vector float min01 = vec_min(reg.val[0], reg.val[1]);
    __vector float min23 = vec_min(reg.val[2], reg.val[3]);
    __vector float min_all = vec_min(min01, min23);
    __vector float temp = vec_min(min_all, vec_sld(min_all, min_all, 8));
    temp = vec_min(temp, vec_sld(temp, temp, 4));
    return vec_extract(temp, 0);
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, &ar](int i) { result += ar.values[i]; });

    return result;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&result, &start, ar](int i) { result += ar.values[start + i]; });

    return result;
  }

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
    vec_xst(reg.val[2], 32, ptr);
    vec_xst(reg.val[3], 48, ptr);
  }

  void save(float* ptr, const int elem_num) const {
    const int elements_in_chunk1 =
        (elem_num >= 0) ? ((elem_num >= 4) ? 4 : elem_num) : 0;
    const int elements_in_chunk2 =
        (elem_num > 4) ? ((elem_num >= 8) ? 4 : elem_num - 4) : 0;
    const int elements_in_chunk3 =
        (elem_num > 8) ? ((elem_num >= 12) ? 4 : elem_num - 8) : 0;
    const int elements_in_chunk4 =
        (elem_num > 12) ? ((elem_num >= 16) ? 4 : elem_num - 12) : 0;

    const size_t bytes_chunk1 =
        static_cast<size_t>(elements_in_chunk1 * sizeof(float));
    const size_t bytes_chunk2 =
        static_cast<size_t>(elements_in_chunk2 * sizeof(float));
    const size_t bytes_chunk3 =
        static_cast<size_t>(elements_in_chunk3 * sizeof(float));
    const size_t bytes_chunk4 =
        static_cast<size_t>(elements_in_chunk4 * sizeof(float));

    vec_xst_len(reg.val[0], ptr, bytes_chunk1);
    vec_xst_len(reg.val[1],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 16),
                bytes_chunk2);
    vec_xst_len(reg.val[2],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 32),
                bytes_chunk3);
    vec_xst_len(reg.val[3],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 48),
                bytes_chunk4);
  }
};

}  // namespace vec_op

#endif
