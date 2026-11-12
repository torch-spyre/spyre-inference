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

#include "cpu_fast_exp.hpp"
#include "cpu_types.hpp"

#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Gumbel-max: argmax_i(logit_i + g_i) with g_i ~ Gumbel(0, 1) matches
// softmax. Noise is SplitMix64(seed, i), so tokens are independent and
// consecutive seeds are not a 1-token table shift (vllm#59786).
constexpr uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr double GUMBEL_MAX = 37.5;  // 53-bit u in (0,1) => -log(-log u) < 37.5

static inline uint64_t splitmix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static inline uint64_t noise_bits(uint64_t key, int64_t i) {
  return splitmix64(key + uint64_t(i + 1) * SPLITMIX_GAMMA) >> 11;
}

static inline double gumbel_from_bits(uint64_t bits) {
  const double u = (double(bits) + 0.5) * 0x1.0p-53;
  return -std::log(-std::log(u));
}

static inline double one_minus_u_from_bits(uint64_t bits) {
  return (double((1ULL << 53) - 1 - bits) + 0.5) * 0x1.0p-53;
}

// Exact step. x + g > best_x + best_g is (x - best_x) + g > best_g. Adding g
// into a 1e38 logit loses it in double ULP, so ties at the float max would
// always keep the lower index.
static inline void consider_token(const float* row, uint64_t key, int64_t i,
                                  double& best_logit, double& best_g,
                                  int64_t& best_idx) {
  const double dx = double(row[i]) - best_logit;
  // The noise is bounded; this also skips every -inf.
  if (dx + GUMBEL_MAX <= best_g) return;
  const uint64_t bits = noise_bits(key, i);
  // Wins iff -log u < exp(dx - best_g); -log u >= 1 - u rejects most tokens
  // with one exp instead of two logs.
  if (one_minus_u_from_bits(bits) >= std::exp(dx - best_g)) return;
  const double g = gumbel_from_bits(bits);
  if (dx + g > best_g) {
    best_logit = row[i];
    best_g = g;
    best_idx = i;
  }
}

// Above this, casting the threshold to float can move exp by more than the
// margin below, so the chunk stays on the exact step.
constexpr double kGumbelFloatThresh = 2048.0;
// fast_exp is a few ulp, and the threshold loses a fraction of a float ulp.
// One percent covers both. A closer token is rechecked with std::exp.
constexpr double kGumbelExpMargin = 1.01;

#if defined(__x86_64__) && defined(__AVX2__) && !defined(__AVX512F__)
#define SPYRE_AVX2_FMA __attribute__((target("avx2,f16c,fma")))
#else
#define SPYRE_AVX2_FMA
#endif

// exp(logit - threshold) for 16 lanes. The AVX2 build keeps FMA off in this
// file; this helper opts in so fast_exp can inline.
SPYRE_AVX2_FMA
static void gumbel_approx_exp16(const float* logits, double threshold,
                                float* out) {
  using spyre_cpu::fast_exp;
  const vec_op::FP32Vec16 x(logits);
  const vec_op::FP32Vec16 t(static_cast<float>(threshold));
  fast_exp(x - t).save(out);
}

// True only when one_minus_u is far enough above the approximation that the
// exact exp would reject too. A low approximation falls through to the exact
// step, so a winner is never dropped.
static inline bool approx_rejects(double one_minus_u, float e) {
  if (!(e >= 0.f) || !std::isfinite(e)) return false;
  return one_minus_u > double(e) * kGumbelExpMargin;
}

static void fused_gumbel_argmax_row(int64_t* __restrict__ output,
                                    const float* __restrict__ row,
                                    uint64_t seed, int64_t vocab_size) {
  const uint64_t key = splitmix64(seed);

  // The first unmasked token seeds the running best; an all -inf row gives 0.
  int64_t first = 0;
  while (first < vocab_size &&
         row[first] == -std::numeric_limits<float>::infinity())
    ++first;
  if (first == vocab_size) {
    *output = 0;
    return;
  }

  double best_logit = row[first];
  double best_g = gumbel_from_bits(noise_bits(key, first));
  int64_t best_idx = first;
  int64_t i = first + 1;
  // The best only gets stricter inside a chunk, so an exp from the chunk's
  // starting threshold rejects fewer tokens, never a real winner.
  for (; i + 16 <= vocab_size;) {
    const double threshold = best_logit + best_g;
    if (threshold > kGumbelFloatThresh || threshold < -kGumbelFloatThresh) {
      for (int lane = 0; lane < 16; ++lane, ++i)
        consider_token(row, key, i, best_logit, best_g, best_idx);
      continue;
    }
    // A masked row is almost all -inf. The noise bound drops a whole chunk
    // before the exp, which is the exact per-token prune.
    const float chunk_max = vec_op::FP32Vec16(row + i).reduce_max();
    if (double(chunk_max) + GUMBEL_MAX <= threshold) {
      i += 16;
      continue;
    }
    float e[16];
    gumbel_approx_exp16(row + i, threshold, e);
    for (int lane = 0; lane < 16; ++lane, ++i) {
      if (approx_rejects(one_minus_u_from_bits(noise_bits(key, i)), e[lane]))
        continue;
      consider_token(row, key, i, best_logit, best_g, best_idx);
    }
  }
  for (; i < vocab_size; ++i)
    consider_token(row, key, i, best_logit, best_g, best_idx);
  *output = best_idx;
}

static void fused_gumbel_argmax_kernel(int64_t* __restrict__ output,
                                       const float* __restrict__ logits,
                                       const int64_t* __restrict__ seeds,
                                       const int64_t batch_size,
                                       const int64_t vocab_size) {
  if (batch_size <= 0) return;
  if (batch_size == 1) {
    fused_gumbel_argmax_row(output, logits, uint64_t(seeds[0]), vocab_size);
    return;
  }
#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    fused_gumbel_argmax_row(output + b, logits + b * vocab_size,
                            uint64_t(seeds[b]), vocab_size);
  }
}

static void greedy_argmax_kernel(int64_t* __restrict__ output,
                                 const float* __restrict__ logits,
                                 const int64_t batch_size,
                                 const int64_t vocab_size) {
  constexpr int VEC_ELEM_NUM = vec_op::FP32Vec16::VEC_ELEM_NUM;
  const int64_t vec_end = vocab_size - (vocab_size % VEC_ELEM_NUM);

#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;

    // Vector max has no consistent NaN rule, so NaNs are caught via the sum:
    // any NaN (or inf + -inf, a harmless false positive) makes it NaN.
    vec_op::FP32Vec16 vmax(-std::numeric_limits<float>::infinity());
    vec_op::FP32Vec16 vsum(0.0f);
    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vec_op::FP32Vec16 v(row + i);
      vmax = vmax.max(v);
      vsum = vsum + v;
    }
    float best_val = vmax.reduce_max();
    bool maybe_nan = std::isnan(vsum.reduce_sum());
    for (int64_t i = vec_end; i < vocab_size; ++i) {
      maybe_nan |= std::isnan(row[i]);
      if (row[i] > best_val) {
        best_val = row[i];
      }
    }

    int64_t best_idx = -1;
    if (maybe_nan) {
      // torch.argmax treats NaN as the maximum: return the first NaN.
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(row[i])) {
          best_idx = i;
          break;
        }
      }
    }
    if (best_idx < 0) {
      best_idx = 0;
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (row[i] == best_val) {
          best_idx = i;
          break;
        }
      }
    }
    output[b] = best_idx;
  }
}

}  // namespace

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.size(1) > 0, "vocab_size must be positive");
  TORCH_CHECK(seeds.device().is_cpu(), "seeds must be a CPU tensor");
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
  TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == logits.size(0),
              "seeds must be 1-D with batch_size elements");

  auto logits_contig = logits.contiguous();
  auto seeds_contig = seeds.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  fused_gumbel_argmax_kernel(output.data_ptr<int64_t>(),
                             logits_contig.data_ptr<float>(),
                             seeds_contig.data_ptr<int64_t>(),
                             logits_contig.size(0), logits_contig.size(1));
  return output;
}

torch::Tensor greedy_argmax(const torch::Tensor& logits) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");

  // A single row is faster in ATen: OpenMP fork/join dominates the scan
  // (~1 ms vs ~50 µs on s390x at vocab=32k).
  if (logits.size(0) <= 1) {
    return logits.argmax(/*dim=*/-1);
  }

  auto logits_contig = logits.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  greedy_argmax_kernel(output.data_ptr<int64_t>(),
                       logits_contig.data_ptr<float>(), logits_contig.size(0),
                       logits_contig.size(1));
  return output;
}
