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

#pragma once

#include <cmath>
#include <cstdint>

namespace gumbel_noise {

// Gumbel-max: argmax_i(logit_i + g_i) with g_i ~ Gumbel(0, 1) matches
// softmax. Noise is SplitMix64(seed, i), so tokens are independent and
// consecutive seeds are not a 1-token table shift (vllm#59786).
constexpr uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr double GUMBEL_MAX = 37.5;  // 53-bit u in (0,1) => -log(-log u) < 37.5

inline uint64_t splitmix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline uint64_t noise_bits(uint64_t key, int64_t i) {
  return splitmix64(key + uint64_t(i + 1) * SPLITMIX_GAMMA) >> 11;
}

inline double gumbel_from_bits(uint64_t bits) {
  const double u = (double(bits) + 0.5) * 0x1.0p-53;
  return -std::log(-std::log(u));
}

// 1 - u for the same draw, exact in double: -log u >= 1 - u lets a caller
// reject most candidates with one exp instead of two logs.
inline double one_minus_u_from_bits(uint64_t bits) {
  return (double((1ULL << 53) - 1 - bits) + 0.5) * 0x1.0p-53;
}

}  // namespace gumbel_noise
