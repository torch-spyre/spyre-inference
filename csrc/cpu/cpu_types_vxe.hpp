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

#ifndef CPU_TYPES_VXE_HPP
#define CPU_TYPES_VXE_HPP

#include <vecintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace vec_op {

#define vec_neg(a) (-(a))
#define vec_add(a, b) ((a) + (b))
#define vec_sub(a, b) ((a) - (b))
#define vec_mul(a, b) ((a) * (b))
#define vec_div(a, b) ((a) / (b))
#define vec_sr(a, b) ((a) >> (b))  // Vector Shift Right Algebraic
#define vec_sl(a, b) ((a) << (b))  // Vector Shift Left

#define FORCE_INLINE __attribute__((always_inline)) inline

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
    __vector float sum = vec_add(reg.val[0], reg.val[1]);
    __vector float hi = vec_sld(sum, sum, 8);
    sum = vec_add(sum, hi);
    __vector float lo = vec_sld(sum, sum, 4);
    sum = vec_add(sum, lo);
    return vec_extract(sum, 0);
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

      __vector signed int k = vec_signed(kf);
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
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    const __vector float one = vec_splats(1.0f);
    const __vector float two = vec_splats(2.0f);
    const __vector float zero = vec_splats(0.0f);
    const __vector float sat =
        vec_splats(9.0f);  // beyond this, tanh(x) ~ sign(x)

    f32x4x2_t out;

    for (int i = 0; i < 2; i++) {
      __vector float x = reg.val[i];
      __vector float ax = vec_abs(x);

      // sign(x): +1 or -1
      __vector float sign = vec_sel(vec_splats(-1.0f), one, vec_cmpgt(x, zero));

      // saturation mask: |x| > sat
      __vector __bool int saturated = vec_cmpgt(ax, sat);

      // 2x
      __vector float two_x = vec_mul(x, two);

      // Build a temporary FP32Vec8 with both lanes = 2x, reuse exp()
      f32x4x2_t tmp;
      tmp.val[0] = two_x;
      tmp.val[1] = two_x;
      FP32Vec8 exp_2x_vec(tmp);

      FP32Vec8 e2x = exp_2x_vec.exp();
      __vector float e = e2x.reg.val[i];

      // tanh(x) = (e - 1) / (e + 1)
      __vector float num = vec_sub(e, one);
      __vector float den = vec_add(e, one);

      __vector float t = vec_div(num, den);

      // For large |x|, clamp to sign(x)
      out.val[i] = vec_sel(t, sign, saturated);
    }

    return FP32Vec8(out);
  }

  FP32Vec8 er() const {
    // A&S 7.1.26 approximation:
    // erf(x) = sign(x) * (1 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t *
    // exp(-x^2)) t = 1 / (1 + p*|x|),  p = 0.3275911

    const __vector float one = vec_splats(1.0f);
    const __vector float zero = vec_splats(0.0f);
    const __vector float p = vec_splats(0.3275911f);

    // Polynomial coeffs
    const __vector float a1 = vec_splats(0.254829592f);
    const __vector float a2 = vec_splats(-0.284496736f);
    const __vector float a3 = vec_splats(1.421413741f);
    const __vector float a4 = vec_splats(-1.453152027f);
    const __vector float a5 = vec_splats(1.061405429f);

    // Threshold where erf(x) ~ sign(x)
    const __vector float sat = vec_splats(6.0f);

    f32x4x2_t out;

    for (int lane = 0; lane < 2; lane++) {
      __vector float x = reg.val[lane];
      __vector float ax = vec_abs(x);

      // sign(x)
      __vector float sign = vec_sel(vec_splats(-1.0f), one, vec_cmpgt(x, zero));

      // |x| > 6 → erf(x) = ±1
      __vector __bool int saturated = vec_cmpgt(ax, sat);

      // t = 1 / (1 + p * |x|)
      __vector float t = vec_madd(p, ax, one);
      t = vec_div(one, t);

      // poly = a5
      __vector float poly = a5;
      poly = vec_madd(poly, t, a4);
      poly = vec_madd(poly, t, a3);
      poly = vec_madd(poly, t, a2);
      poly = vec_madd(poly, t, a1);

      // full polynomial: poly = poly * t
      poly = vec_mul(poly, t);

      // Compute exp(-x^2)
      __vector float x2 = vec_mul(x, x);
      __vector float neg_x2 = vec_neg(x2);

      f32x4x2_t tmp;
      tmp.val[0] = neg_x2;
      tmp.val[1] = neg_x2;
      FP32Vec8 exp_neg_x2(tmp);

      FP32Vec8 e = exp_neg_x2.exp();
      __vector float ex = e.reg.val[lane];

      // erf(x) = sign * (1 - poly * exp(-x^2))
      __vector float term = vec_mul(poly, ex);
      __vector float y = vec_sub(one, term);
      y = vec_mul(y, sign);

      // saturated → ±1
      __vector float sat_val = vec_mul(sign, one);
      out.val[lane] = vec_sel(y, sat_val, saturated);
    }

    return FP32Vec8(out);
  }
  // Elementwise sigmoid(x) = 1 / (1 + exp(-x))
  FP32Vec8 sigmoid() const {
    const __vector float one = vec_splats(1.0f);

    f32x4x2_t neg;
    for (int i = 0; i < 2; ++i) {
      neg.val[i] = vec_neg(reg.val[i]);
    }

    FP32Vec8 neg_x(neg);
    FP32Vec8 e = neg_x.exp();  // exp(-x)

    f32x4x2_t denom;
    for (int i = 0; i < 2; ++i) {
      denom.val[i] = vec_add(one, e.reg.val[i]);
    }

    FP32Vec8 denom_vec(denom);
    FP32Vec8 one_vec(1.0f);

    return one_vec / denom_vec;
  }

  // Tanh-based GELU:
  // gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
  FP32Vec8 gelu_tanh() const {
    const __vector float k_s2pi = vec_splats(0.7978845608028654f);  // √(2/π)
    const __vector float k_0_0447 = vec_splats(0.044715f);

    f32x4x2_t x2, x3, inner;
    for (int i = 0; i < 2; ++i) {
      __vector float x = reg.val[i];
      x2.val[i] = vec_mul(x, x);                            // x^2
      x3.val[i] = vec_mul(x2.val[i], x);                    // x^3
      __vector float t = vec_madd(k_0_0447, x3.val[i], x);  // x + 0.044715*x^3
      inner.val[i] = vec_mul(k_s2pi, t);                    // √(2/π)*(...)
    }

    FP32Vec8 inner_vec(inner);
    FP32Vec8 t = inner_vec.tanh();  // tanh part

    FP32Vec8 one_vec(1.0f);
    FP32Vec8 half_vec(0.5f);

    FP32Vec8 x_vec(*this);
    return x_vec * half_vec * (one_vec + t);
  }

  // Erf-based GELU:
  // gelu(x) = 0.5 * x * (1 + erf(x / √2))
  FP32Vec8 gelu_erf() const {
    const __vector float inv_sqrt2 = vec_splats(0.7071067811865476f);  // 1/√2
    FP32Vec8 x_vec(*this);

    f32x4x2_t scaled;
    for (int i = 0; i < 2; ++i) {
      scaled.val[i] = vec_mul(reg.val[i], inv_sqrt2);
    }
    FP32Vec8 x_scaled(scaled);

    FP32Vec8 erf_x = x_scaled.er();

    FP32Vec8 one_vec(1.0f);
    FP32Vec8 half_vec(0.5f);

    return x_vec * half_vec * (one_vec + erf_x);
  }

  // Elementwise reciprocal: 1/x (scalar per lane, for correctness)
  FP32Vec8 rcp() const {
    AliasReg in, out;
    in.reg = reg;

    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      out.values[i] = 1.0f / in.values[i];
    }
    return FP32Vec8(out.reg);
  }

  // Elementwise rsqrt(x) = 1 / sqrt(x) (scalar per lane, for correctness)
  FP32Vec8 rsqrt() const {
    AliasReg in, out;
    in.reg = reg;

    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      out.values[i] = 1.0f / std::sqrt(in.values[i]);
    }
    return FP32Vec8(out.reg);
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

  // De-pack 16 x 4-bit nibbles from a 64-bit value and look each up in a
  // 16-element float LUT. Used by WNA16 (AWQ/GPTQ) dequantization.
  explicit FP32Vec16(int64_t value, const FP32Vec16& lut) {
    uint64_t uval = static_cast<uint64_t>(value);
    uval = (uval >> 32) | (uval << 32);

    // Process 4 floats per output vector register
    for (int v = 0; v < 4; ++v) {
      // Extract 4 nibble indices for this output vector
      uint8_t n0 = (uval >> ((v * 4 + 0) * 4)) & 0xF;
      uint8_t n1 = (uval >> ((v * 4 + 1) * 4)) & 0xF;
      uint8_t n2 = (uval >> ((v * 4 + 2) * 4)) & 0xF;
      uint8_t n3 = (uval >> ((v * 4 + 3) * 4)) & 0xF;

      // Build permute control: index % 8 * 4 gives byte offset within a
      // 32-byte window (two 16-byte LUT vectors concatenated).
      alignas(16) uint8_t ctrl[16] = {
          (uint8_t)((n0 % 8) * 4 + 0), (uint8_t)((n0 % 8) * 4 + 1),
          (uint8_t)((n0 % 8) * 4 + 2), (uint8_t)((n0 % 8) * 4 + 3),
          (uint8_t)((n1 % 8) * 4 + 0), (uint8_t)((n1 % 8) * 4 + 1),
          (uint8_t)((n1 % 8) * 4 + 2), (uint8_t)((n1 % 8) * 4 + 3),
          (uint8_t)((n2 % 8) * 4 + 0), (uint8_t)((n2 % 8) * 4 + 1),
          (uint8_t)((n2 % 8) * 4 + 2), (uint8_t)((n2 % 8) * 4 + 3),
          (uint8_t)((n3 % 8) * 4 + 0), (uint8_t)((n3 % 8) * 4 + 1),
          (uint8_t)((n3 % 8) * 4 + 2), (uint8_t)((n3 % 8) * 4 + 3),
      };
      __vector unsigned char perm =
          (__vector unsigned char)vec_xl(0, (const signed char*)ctrl);

      // Gather from both LUT halves via vec_perm (VPERM)
      __vector unsigned char from_lo =
          vec_perm((__vector unsigned char)lut.reg.val[0],
                   (__vector unsigned char)lut.reg.val[1], perm);
      __vector unsigned char from_hi =
          vec_perm((__vector unsigned char)lut.reg.val[2],
                   (__vector unsigned char)lut.reg.val[3], perm);

      // Build selection mask: 0xFF bytes for indices >= 8, 0x00 otherwise
      uint8_t m0 = (n0 >= 8) ? 0xFF : 0x00;
      uint8_t m1 = (n1 >= 8) ? 0xFF : 0x00;
      uint8_t m2 = (n2 >= 8) ? 0xFF : 0x00;
      uint8_t m3 = (n3 >= 8) ? 0xFF : 0x00;
      alignas(16) uint8_t sel[16] = {
          m0, m0, m0, m0, m1, m1, m1, m1, m2, m2, m2, m2, m3, m3, m3, m3,
      };
      __vector __bool char mask =
          (__vector __bool char)vec_xl(0, (const signed char*)sel);

      reg.val[v] = (__vector float)vec_sel(from_lo, from_hi, mask);
    }
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

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_sub(reg.val[0], b.reg.val[0]),
                                vec_sub(reg.val[1], b.reg.val[1]),
                                vec_sub(reg.val[2], b.reg.val[2]),
                                vec_sub(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator-() const {
    return FP32Vec16(f32x4x4_t({vec_neg(reg.val[0]), vec_neg(reg.val[1]),
                                vec_neg(reg.val[2]), vec_neg(reg.val[3])}));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_div(reg.val[0], b.reg.val[0]),
                                vec_div(reg.val[1], b.reg.val[1]),
                                vec_div(reg.val[2], b.reg.val[2]),
                                vec_div(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 exp() const {
    FP32Vec8 lo(f32x4x2_t{reg.val[0], reg.val[1]});
    FP32Vec8 hi(f32x4x2_t{reg.val[2], reg.val[3]});
    auto lo_exp = lo.exp();
    auto hi_exp = hi.exp();
    return FP32Vec16(f32x4x4_t{lo_exp.reg.val[0], lo_exp.reg.val[1],
                               hi_exp.reg.val[0], hi_exp.reg.val[1]});
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

  float reduce_sum() const {
    __vector float sum = vec_add(vec_add(reg.val[0], reg.val[1]),
                                 vec_add(reg.val[2], reg.val[3]));
    __vector float hi = vec_sld(sum, sum, 8);
    sum = vec_add(sum, hi);
    __vector float lo = vec_sld(sum, sum, 4);
    sum = vec_add(sum, lo);
    return vec_extract(sum, 0);
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

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_max(reg.val[0], b.reg.val[0]),
                                vec_max(reg.val[1], b.reg.val[1]),
                                vec_max(reg.val[2], b.reg.val[2]),
                                vec_max(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_min(reg.val[0], b.reg.val[0]),
                                vec_min(reg.val[1], b.reg.val[1]),
                                vec_min(reg.val[2], b.reg.val[2]),
                                vec_min(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 clamp(const FP32Vec16& min_v, const FP32Vec16& max_v) const {
    return this->max(min_v).min(max_v);
  }

  float reduce_max() const {
    __vector float m = vec_max(vec_max(reg.val[0], reg.val[1]),
                               vec_max(reg.val[2], reg.val[3]));
    __vector float hi = vec_sld(m, m, 8);
    m = vec_max(m, hi);
    __vector float lo = vec_sld(m, m, 4);
    m = vec_max(m, lo);
    return vec_extract(m, 0);
  }

  FP32Vec16 abs() const {
    return FP32Vec16(f32x4x4_t({vec_abs(reg.val[0]), vec_abs(reg.val[1]),
                                vec_abs(reg.val[2]), vec_abs(reg.val[3])}));
  }

  float reduce_min() const {
    __vector float m = vec_min(vec_min(reg.val[0], reg.val[1]),
                               vec_min(reg.val[2], reg.val[3]));
    __vector float h = vec_sld(m, m, 8);
    m = vec_min(m, h);
    __vector float l = vec_sld(m, m, 4);
    m = vec_min(m, l);
    return vec_extract(m, 0);
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    AliasReg ar_this, ar_b;
    ar_this.reg = reg;
    ar_b.reg = b.reg;
    for (int i = 0; i < elem_num && i < VEC_ELEM_NUM; ++i) {
      ar_this.values[i] = std::min(ar_this.values[i], ar_b.values[i]);
    }
    return FP32Vec16(ar_this.reg);
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    AliasReg ar_this, ar_b;
    ar_this.reg = reg;
    ar_b.reg = b.reg;
    for (int i = 0; i < elem_num && i < VEC_ELEM_NUM; ++i) {
      ar_this.values[i] = std::max(ar_this.values[i], ar_b.values[i]);
    }
    return FP32Vec16(ar_this.reg);
  }

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
    vec_xst(reg.val[2], 32, ptr);
    vec_xst(reg.val[3], 48, ptr);
  }

  void save(float* ptr, const int elem_num) const {
    AliasReg ar;
    ar.reg = reg;
    for (int i = 0; i < elem_num && i < VEC_ELEM_NUM; ++i) {
      ptr[i] = ar.values[i];
    }
  }
};

};  // namespace vec_op

#endif
