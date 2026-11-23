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

// z15 vector helpers for the MRV2 host kernels (mrv2_kernels.cpp,
// mrv2_spec_decode_kernels.cpp), 4 float lanes. Compares, selects and arithmetic
// repeat the scalar loops' own operations in the same order, and log, exp and log1p
// call the same libm functions lane by lane, so they return the scalar results bit
// for bit. (SLEEF's vector builds are not an option: torch builds for s390x need not
// export them, and _C would then fail to import.)

#pragma once

#if defined(__s390x__)

  #include <vecintrin.h>

  #include <algorithm>
  #include <cfloat>
  #include <cmath>
  #include <cstdint>
  #include <limits>

namespace mrv2_vx {

using vf32 = __vector float;
using vi32 = __vector signed int;
using vu32 = __vector unsigned int;
using vb32 = __vector __bool int;

constexpr float kLaneNegInf = -std::numeric_limits<float>::infinity();

// (x > m) ? x : m, the scalar loops' std::max(m, x): a NaN x never replaces m.
inline vf32 max_skip_nan(vf32 m, vf32 x) { return vec_sel(m, x, vec_cmpgt(x, m)); }

// The largest non-NaN lane, -inf if none.
inline float lane_max(vf32 m) {
  float r = kLaneNegInf;
  for (int i = 0; i < 4; ++i) r = std::max(r, m[i]);
  return r;
}

// The largest non-NaN value of row[begin, end) (each divided by `div` first when
// Divide), -inf if none. A zero max keeps the sign of the row's first zero, as the
// scalar loop's first-wins std::max does.
template <bool Divide = false>
inline float row_max(const float* row, int64_t begin, int64_t end, float div = 1.0f) {
  auto at = [&](int64_t v) { return Divide ? row[v] / div : row[v]; };
  const vf32 vd = vec_splats(div);
  auto load = [&](int64_t v) {
    const vf32 x = vec_xl(0, row + v);
    return Divide ? x / vd : x;
  };
  vf32 m0 = vec_splats(kLaneNegInf), m1 = m0, m2 = m0, m3 = m0;
  int64_t v = begin;
  for (; v + 16 <= end; v += 16) {
    m0 = max_skip_nan(m0, load(v));
    m1 = max_skip_nan(m1, load(v + 4));
    m2 = max_skip_nan(m2, load(v + 8));
    m3 = max_skip_nan(m3, load(v + 12));
  }
  float m = lane_max(max_skip_nan(max_skip_nan(m0, m1), max_skip_nan(m2, m3)));
  for (; v < end; ++v) m = std::max(m, at(v));
  if (m == 0.0f)
    for (v = begin; v < end; ++v)
      if (at(v) == 0.0f) return at(v);
  return m;
}

// The first index in [v, end) whose value equals m; m must occur there.
inline int64_t first_equal(const float* row, int64_t v, int64_t end, float m) {
  const vf32 vm = vec_splats(m);
  while (v + 4 <= end && !vec_any_eq(vec_xl(0, row + v), vm)) v += 4;
  while (!(row[v] == m)) ++v;
  return v;
}

// tl.max(return_indices=True) over row[begin, end): the first NaN, else the first max.
inline int64_t first_nan_or_max(const float* row, int64_t begin, int64_t end) {
  int64_t v = begin;
  while (v + 4 <= end) {
    const vf32 x = vec_xl(0, row + v);
    if (!vec_all_eq(x, x)) break;
    v += 4;
  }
  for (; v < end; ++v)
    if (row[v] != row[v]) return v;
  return first_equal(row, begin, end, row_max(row, begin, end));
}

// Lane i of a 4-token step is allowed by bit i of `bits`.
inline vb32 lane_bits_set(uint32_t bits) {
  const vu32 lane_bit = {1u, 2u, 4u, 8u};
  return vec_cmpgt(vec_splats(bits) & lane_bit, vec_splats(0u));
}

// No lane of x is finite (|x| <= FLT_MAX fails for +-inf and NaN alike).
inline bool none_finite(vf32 x) { return !vec_any_le(vec_abs(x), vec_splats(FLT_MAX)); }

// NaN lanes read as -inf.
inline vf32 nan_to_neg_inf(vf32 x) { return vec_sel(vec_splats(kLaneNegInf), x, vec_cmpeq(x, x)); }

// libm's float log, exp and log1p of each lane, as the scalar loops call them.
inline vf32 log4(vf32 x) {
  for (int i = 0; i < 4; ++i) x[i] = std::log(float(x[i]));
  return x;
}
inline vf32 exp4(vf32 x) {
  for (int i = 0; i < 4; ++i) x[i] = std::exp(float(x[i]));
  return x;
}
inline vf32 log1p4(vf32 x) {
  for (int i = 0; i < 4; ++i) x[i] = std::log1p(float(x[i]));
  return x;
}

// ------------------------------------------------------ murmur3 Gumbel noise
// murmur3_finish, murmur3_uniform32 and log1p_neg_stable, four offsets at a time.

inline vu32 murmur3_mix4(vu32 h, vu32 k) {
  k *= vec_splats(0xCC9E2D51u);
  k = vec_rli(k, 15);
  k *= vec_splats(0x1B873593u);
  h = vec_rli(h ^ k, 13);
  return h * vec_splats(5u) + vec_splats(0xE6546B64u);
}

inline vu32 murmur3_fmix4(vu32 h) {
  h ^= h >> vec_splats(16u);
  h *= vec_splats(0x85EBCA6Bu);
  h ^= h >> vec_splats(13u);
  h *= vec_splats(0xC2B2AE35u);
  return h ^ (h >> vec_splats(16u));
}

// The uniform in (0, 1) for tokens v .. v+3 of the row with hash prefix `prefix`.
inline vf32 murmur3_uniform4(uint32_t prefix, int64_t v) {
  const vu32 offs = vec_splats(uint32_t(v)) + vu32{0u, 1u, 2u, 3u};
  const vu32 bits = murmur3_fmix4(murmur3_mix4(vec_splats(prefix), offs) ^ vec_splats(16u));
  const vf32 hi16 = vec_float(bits >> vec_splats(16u));
  const vf32 lo16 = vec_float(bits & vec_splats(0xFFFFu));
  return hi16 * vec_splats(1.52587890625e-05f) +
         (lo16 + vec_splats(0.5f)) * vec_splats(2.3283064365386963e-10f);
}

inline vf32 log1p_neg_stable4(vf32 v) {
  vf32 p = vec_splats(float(1.0 / 8.0));
  for (const float c : {float(1.0 / 7.0), float(1.0 / 6.0), float(1.0 / 5.0), float(1.0 / 4.0),
                        float(1.0 / 3.0), float(1.0 / 2.0), 1.0f})
    p = vec_splats(c) + v * p;
  const vf32 series = -v * p;
  const vf32 one_minus = vec_splats(1.0f) - v;
  const vf32 tiny = vec_splats(5.960464477539063e-08f);
  const vf32 direct = log4(vec_sel(one_minus, tiny, vec_cmplt(one_minus, tiny)));
  return vec_sel(direct, series, vec_cmplt(v, vec_splats(0.25f)));
}

// The fp32 Gumbel noise -log(-log1p(-u)) of uniforms u.
inline vf32 gumbel_from_uniform4(vf32 u) { return -log4(-log1p_neg_stable4(u)); }

// The fp32 Gumbel noise of tokens v .. v+3.
inline vf32 gumbel_noise4(uint32_t prefix, int64_t v) {
  return gumbel_from_uniform4(murmur3_uniform4(prefix, v));
}

// ------------------------------------------------------------------ softmax

// The max of x (x / div when Divide) over row[begin, end) and the float sum of
// exp(x - max), added in index order as the scalar loop adds them; 0 if max is -inf.
template <bool Divide>
inline void max_sumexp(const float* row, int64_t begin, int64_t end, float div,
                       float* out_max, float* out_sum) {
  const float m = row_max<Divide>(row, begin, end, div);
  float s = 0.0f;
  if (m > kLaneNegInf) {
    const vf32 vd = vec_splats(div), vm = vec_splats(m);
    int64_t v = begin;
    for (; v + 4 <= end; v += 4) {
      vf32 x = vec_xl(0, row + v);
      if (Divide) x = x / vd;
      const vf32 e = exp4(x - vm);
      for (int i = 0; i < 4; ++i) s += e[i];
    }
    for (; v < end; ++v) s += std::exp((Divide ? row[v] / div : row[v]) - m);
  }
  *out_max = m;
  *out_sum = s;
}

}  // namespace mrv2_vx

#endif  // __s390x__
