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
// Translated from vLLM's MRV2 Triton kernels (https://github.com/vllm-project/vllm,
// vllm/v1/worker/gpu), Copyright contributors to the vLLM project, Apache-2.0.
//
// philox is Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
// SC'11), following Triton's tl.rand (python/triton/language/random.py) so the outputs
// match it bit for bit.

// MRV2 spec-decode host kernels: the rejection sampler
// (gpu/spec_decode/rejection_sampler{,_utils}.py) and DFlash input prep
// (gpu/spec_decode/dflash/speculator.py). Same contract as mrv2_kernels.cpp:
// the grid, then the Triton kernel's own arguments by name and in order.
// Block verification and watermarked resampling are refused by name.

#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kTlRandMin = 4.6566127342e-10f;

template <typename T>
T* ptr(const torch::Tensor& t) {
  TORCH_CHECK(t.device().is_cpu(), "MRV2 host kernels need host tensors, got ", t.device());
  return static_cast<T*>(t.data_ptr());
}

// int32/int64 (and bool/uint8 widened) read and written as int64.
class Ints {
 public:
  explicit Ints(const torch::Tensor& t) : data_(ptr<void>(t)), type_(t.scalar_type()) {
    TORCH_CHECK(type_ == at::kLong || type_ == at::kInt || type_ == at::kBool ||
                    type_ == at::kByte,
                "expected an integer tensor, got ", type_);
  }
  int64_t operator[](int64_t i) const {
    switch (type_) {
      case at::kLong: return static_cast<const int64_t*>(data_)[i];
      case at::kInt: return static_cast<const int32_t*>(data_)[i];
      default: return static_cast<const uint8_t*>(data_)[i];
    }
  }
  void set(int64_t i, int64_t v) const {
    switch (type_) {
      case at::kLong: static_cast<int64_t*>(data_)[i] = v; break;
      case at::kInt: static_cast<int32_t*>(data_)[i] = static_cast<int32_t>(v); break;
      default: static_cast<uint8_t*>(data_)[i] = static_cast<uint8_t>(v);
    }
  }

 private:
  void* data_;
  at::ScalarType type_;
};

// Logits rows read as fp32, whatever their storage dtype.
class Floats {
 public:
  explicit Floats(const torch::Tensor& t) : data_(ptr<void>(t)), type_(t.scalar_type()) {}
  float operator[](int64_t i) const {
    switch (type_) {
      case at::kFloat: return static_cast<const float*>(data_)[i];
      case at::kDouble: return static_cast<float>(static_cast<const double*>(data_)[i]);
      case at::kHalf: return static_cast<float>(static_cast<const at::Half*>(data_)[i]);
      case at::kBFloat16: return static_cast<float>(static_cast<const at::BFloat16*>(data_)[i]);
      default: TORCH_CHECK(false, "unsupported logits dtype ", type_);
    }
  }

 private:
  const void* data_;
  at::ScalarType type_;
};

// ------------------------------------------------------------- tl.rand (Philox)

void philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
  for (int round = 0; round < 10; ++round) {
    const uint64_t p0 = uint64_t(0xD2511F53u) * c[0];
    const uint64_t p1 = uint64_t(0xCD9E8D57u) * c[2];
    const uint32_t n0 = uint32_t(p1 >> 32) ^ c[1] ^ k0;
    const uint32_t n2 = uint32_t(p0 >> 32) ^ c[3] ^ k1;
    c[0] = n0;
    c[1] = uint32_t(p1);
    c[2] = n2;
    c[3] = uint32_t(p0);
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
}

// tl_rand32(seed, offset, includes_zero=False) for an int64 offset.
float tl_rand32(int64_t seed, int64_t offset) {
  const uint64_t s = static_cast<uint64_t>(seed), o = static_cast<uint64_t>(offset);
  uint32_t c[4] = {uint32_t(o), uint32_t(o >> 32), 0, 0};
  philox(c, uint32_t(s), uint32_t(s >> 32));
  int32_t x;
  std::memcpy(&x, &c[0], sizeof(x));
  if (x < 0) x = ~x;
  return std::max(static_cast<float>(x) * 4.6566127342e-10f, kTlRandMin);
}

inline int64_t spec_step_seed(int64_t seed, int64_t step) {
  return seed ^ static_cast<int64_t>(static_cast<uint64_t>(step) << 32);
}

// ---------------------------------------------------- gumbel (resample draw)
// sample/gumbel.py's murmur3 noise, as mrv2_kernels.cpp's gumbel_row.

inline uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }

inline uint32_t murmur3_mix(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = rotl32(k, 15);
  k *= 0x1B873593u;
  h ^= k;
  h = rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}

inline uint32_t murmur3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}

inline uint32_t murmur3_prefix(int64_t seed, int64_t pos, uint32_t domain) {
  const uint64_t s = static_cast<uint64_t>(seed);
  uint32_t h = domain;
  h = murmur3_mix(h, static_cast<uint32_t>(s));
  h = murmur3_mix(h, static_cast<uint32_t>(s >> 32));
  return murmur3_mix(h, static_cast<uint32_t>(static_cast<uint64_t>(pos)));
}

inline uint32_t murmur3_finish(uint32_t prefix, uint32_t offset) {
  return murmur3_fmix32(murmur3_mix(prefix, offset) ^ 16u);
}

inline float murmur3_uniform32(uint32_t bits) {
  const float hi16 = static_cast<float>(bits >> 16);
  const float lo16 = static_cast<float>(bits & 0xFFFFu);
  return hi16 * 1.52587890625e-05f + (lo16 + 0.5f) * 2.3283064365386963e-10f;
}

inline float log1p_neg_stable(float v) {
  float p = float(1.0 / 8.0);
  p = float(1.0 / 7.0) + v * p;
  p = float(1.0 / 6.0) + v * p;
  p = float(1.0 / 5.0) + v * p;
  p = float(1.0 / 4.0) + v * p;
  p = float(1.0 / 3.0) + v * p;
  p = float(1.0 / 2.0) + v * p;
  p = 1.0f + v * p;
  const float series = -v * p;
  const float direct = std::log(std::max(1.0f - v, 5.960464477539063e-08f));
  return v < 0.25f ? series : direct;
}

inline double murmur3_uniform64(uint32_t lo_prefix, uint32_t hi_prefix, uint32_t offset) {
  const uint64_t lo = murmur3_finish(lo_prefix, offset);
  const uint64_t hi = murmur3_finish(hi_prefix, offset);
  const uint64_t random53 = ((hi << 32) | lo) >> 11;
  const double u = (static_cast<double>(random53) + 0.5) * 1.1102230246251565e-16;
  return u == 1.0 ? u - 1.1102230246251565e-16 : u;
}

// gumbel_noised_argmax over [begin, end) of an fp32 row, APPLY_TEMPERATURE=False,
// IS_DRAFTING=False: plain argmax at temp 0, first index on ties.
template <typename acc_t>
std::pair<int64_t, acc_t> gumbel_block(const std::vector<float>& row, int64_t begin,
                                       int64_t end, float temp, int64_t seed, int64_t pos) {
  constexpr bool fp64 = std::is_same_v<acc_t, double>;
  const bool noisy = temp != 0.0f;
  const uint32_t lo_prefix = murmur3_prefix(seed, pos, 0);
  const uint32_t hi_prefix = murmur3_prefix(seed, pos, 0x9E3779B9u);
  acc_t best = -std::numeric_limits<acc_t>::infinity();
  int64_t best_idx = 0;
  bool nan_seen = false;
  for (int64_t v = begin; v < end; ++v) {
    acc_t value = static_cast<acc_t>(row[v]);
    if (noisy) {
      if constexpr (fp64) {
        value += -std::log(-std::log(murmur3_uniform64(lo_prefix, hi_prefix, uint32_t(v))));
      } else {
        const float u = murmur3_uniform32(murmur3_finish(lo_prefix, uint32_t(v)));
        value = value + -std::log(-log1p_neg_stable(u));
      }
    }
    // tl.max(return_indices=True): the first NaN wins, else the first max.
    if (nan_seen) continue;
    if (std::isnan(value) || v == begin || value > best) {
      nan_seen = std::isnan(value);
      best = value;
      best_idx = v - begin;
    }
  }
  return {best_idx, best};
}

// Global logsumexp from the per-block (max, sumexp) partials.
float global_lse(const float* maxes, const float* sumexps, int64_t n) {
  float m = kNegInf;
  for (int64_t b = 0; b < n; ++b) m = std::max(m, maxes[b]);
  float s = 0.0f;
  for (int64_t b = 0; b < n; ++b) s += sumexps[b] * std::exp(maxes[b] - m);
  return m + std::log(s);
}

// The first index of the max of non-NaN values (tl.argmax after NaN -> -inf).
int64_t argmax_no_nan(const float* values, int64_t n) {
  float best = kNegInf;
  int64_t idx = 0;
  for (int64_t b = 0; b < n; ++b) {
    const float v = std::isnan(values[b]) ? kNegInf : values[b];
    if (v > best) {
      best = v;
      idx = b;
    }
  }
  return idx;
}

}  // namespace

// ------------------------------------------------------------ rejection_sampler.py

void mrv2_flatten_sampled(std::vector<int64_t> grid, const torch::Tensor& flat_sampled_ptr,
                          const torch::Tensor& sampled_ptr, int64_t sampled_stride,
                          const torch::Tensor& num_sampled_ptr,
                          const torch::Tensor& cu_num_logits_ptr) {
  const Ints flat(flat_sampled_ptr), sampled(sampled_ptr), num_sampled(num_sampled_ptr),
      cu(cu_num_logits_ptr);
  for (int64_t r = 0; r < grid[0]; ++r)
    for (int64_t i = 0; i < num_sampled[r]; ++i)
      flat.set(cu[r] + i, sampled[r * sampled_stride + i]);
}

void mrv2_gather_draft_sampled(std::vector<int64_t> grid, const torch::Tensor& draft_sampled_ptr,
                               const torch::Tensor& pos_ptr, const torch::Tensor& input_ids_ptr,
                               const torch::Tensor& positions_ptr,
                               const torch::Tensor& logits_indices_ptr,
                               const torch::Tensor& expanded_idx_mapping_ptr,
                               const torch::Tensor& expanded_local_pos_ptr,
                               const torch::Tensor& prefill_len_ptr, int64_t num_logits,
                               int64_t BLOCK_SIZE) {
  const Ints draft(draft_sampled_ptr), pos_out(pos_ptr), input_ids(input_ids_ptr),
      positions(positions_ptr), indices(logits_indices_ptr), idx(expanded_idx_mapping_ptr),
      local_pos(expanded_local_pos_ptr), prefill_len(prefill_len_ptr);
  for (int64_t i = 0; i < num_logits; ++i) {
    const int64_t token = indices[i];
    const int64_t p = positions[token];
    const int64_t lp = local_pos[i];
    const bool placeholder = lp > 0 && p - lp < prefill_len[idx[i]];
    draft.set(i, placeholder ? -1 : input_ids[token]);
    pos_out.set(i, p);
  }
}

// ------------------------------------------------------- rejection_sampler_utils.py

void mrv2_compute_local_logits_stats(
    std::vector<int64_t> grid, const torch::Tensor& target_local_argmax_ptr,
    int64_t target_local_argmax_stride, const torch::Tensor& target_local_max_ptr,
    int64_t target_local_max_stride, const torch::Tensor& target_local_sumexp_ptr,
    int64_t target_local_sumexp_stride, const torch::Tensor& draft_local_max_ptr,
    int64_t draft_local_max_stride, const torch::Tensor& draft_local_sumexp_ptr,
    int64_t draft_local_sumexp_stride, const torch::Tensor& target_logits_ptr,
    int64_t target_logits_stride, const std::optional<torch::Tensor>& draft_logits_ptr,
    int64_t draft_logits_stride_0, int64_t draft_logits_stride_1,
    const torch::Tensor& expanded_idx_mapping_ptr, const torch::Tensor& expanded_local_pos_ptr,
    const torch::Tensor& temp_ptr, int64_t vocab_size, int64_t num_logits, int64_t BLOCK_SIZE,
    bool HAS_DRAFT_LOGITS) {
  const Ints idx(expanded_idx_mapping_ptr), local_pos(expanded_local_pos_ptr),
      t_argmax(target_local_argmax_ptr);
  const Floats target(target_logits_ptr);
  std::optional<Floats> draft;
  if (HAS_DRAFT_LOGITS) draft.emplace(*draft_logits_ptr);
  float* t_max = ptr<float>(target_local_max_ptr);
  float* t_sum = ptr<float>(target_local_sumexp_ptr);
  float* d_max = ptr<float>(draft_local_max_ptr);
  float* d_sum = ptr<float>(draft_local_sumexp_ptr);
  const float* temps = ptr<float>(temp_ptr);
  const int64_t num_blocks = grid[1];
#pragma omp parallel for collapse(2) schedule(static) if (num_logits * num_blocks > 1)
  for (int64_t i = 0; i < num_logits; ++i) {
    for (int64_t blk = 0; blk < num_blocks; ++blk) {
      const bool bonus = (i + 1 < num_logits ? local_pos[i + 1] : 0) == 0;
      if (bonus) continue;
      const int64_t req = idx[i];
      const float temp = temps[req];
      const int64_t begin = blk * BLOCK_SIZE, end = std::min(begin + BLOCK_SIZE, vocab_size);
      const int64_t row = i * target_logits_stride;
      if (temp == 0.0f) {
        float best = kNegInf;
        int64_t best_idx = 0;
        bool nan_seen = false;
        for (int64_t v = begin; v < end && !nan_seen; ++v) {
          const float x = target[row + v];
          if (std::isnan(x) || v == begin || x > best) {
            nan_seen = std::isnan(x);
            best = x;
            best_idx = v - begin;
          }
        }
        t_argmax.set(i * target_local_argmax_stride + blk, begin + best_idx);
        t_max[i * target_local_max_stride + blk] = best;
        continue;
      }
      auto max_sumexp = [&](auto load, float* out_max, float* out_sum) {
        float m = kNegInf;
        for (int64_t v = begin; v < end; ++v) m = std::max(m, load(v));
        float s = 0.0f;
        if (m > kNegInf)
          for (int64_t v = begin; v < end; ++v) s += std::exp(load(v) - m);
        *out_max = m;
        *out_sum = s;
      };
      max_sumexp([&](int64_t v) { return target[row + v]; },
                 &t_max[i * target_local_max_stride + blk],
                 &t_sum[i * target_local_sumexp_stride + blk]);
      if (HAS_DRAFT_LOGITS) {
        const int64_t drow = req * draft_logits_stride_0 + local_pos[i] * draft_logits_stride_1;
        max_sumexp([&](int64_t v) { return (*draft)[drow + v] / temp; },
                   &d_max[i * draft_local_max_stride + blk],
                   &d_sum[i * draft_local_sumexp_stride + blk]);
      }
    }
  }
}

void mrv2_rejection(
    std::vector<int64_t> grid, const torch::Tensor& sampled_ptr, int64_t sampled_stride,
    const torch::Tensor& rejected_steps_ptr, const torch::Tensor& target_rejected_logsumexp_ptr,
    const torch::Tensor& draft_rejected_logsumexp_ptr, const torch::Tensor& target_logits_ptr,
    int64_t target_logits_stride, const torch::Tensor& target_local_argmax_ptr,
    int64_t target_local_argmax_stride, const torch::Tensor& target_local_max_ptr,
    int64_t target_local_max_stride, const torch::Tensor& target_local_sumexp_ptr,
    int64_t target_local_sumexp_stride, const torch::Tensor& draft_sampled_ptr,
    const std::optional<torch::Tensor>& draft_logits_ptr, int64_t draft_logits_stride_0,
    int64_t draft_logits_stride_1, const torch::Tensor& draft_local_max_ptr,
    int64_t draft_local_max_stride, const torch::Tensor& draft_local_sumexp_ptr,
    int64_t draft_local_sumexp_stride, const torch::Tensor& cu_num_logits_ptr,
    const torch::Tensor& idx_mapping_ptr, const torch::Tensor& temp_ptr,
    const torch::Tensor& seed_ptr, const torch::Tensor& pos_ptr,
    const std::optional<torch::Tensor>& synthetic_conditional_rates_ptr,
    const std::optional<torch::Tensor>& cumulative_log_p_ptr,
    const std::optional<torch::Tensor>& local_residual_mass_ptr,
    int64_t local_residual_mass_stride, const std::optional<torch::Tensor>& fly_entropy_ptr,
    double fly_entropy_threshold, int64_t vocab_num_blocks, int64_t PADDED_VOCAB_NUM_BLOCKS,
    bool HAS_DRAFT_LOGITS, bool SYNTHETIC_MODE, bool USE_BLOCK_VERIFICATION,
    int64_t FLY_WINDOW_SIZE) {
  TORCH_CHECK(!USE_BLOCK_VERIFICATION,
              "MRV2 host rejection sampler: rejection_sample_method='block' is not supported "
              "on Spyre");
  TORCH_CHECK(FLY_WINDOW_SIZE == 0,
              "MRV2 host rejection sampler: rejection_sample_method='fly' is not supported "
              "on Spyre");
  const Ints sampled(sampled_ptr), rejected(rejected_steps_ptr), t_argmax(target_local_argmax_ptr),
      draft_sampled(draft_sampled_ptr), cu(cu_num_logits_ptr), idx(idx_mapping_ptr),
      seeds(seed_ptr), positions(pos_ptr);
  const Floats target(target_logits_ptr);
  std::optional<Floats> draft;
  if (HAS_DRAFT_LOGITS) draft.emplace(*draft_logits_ptr);
  const float* t_max = ptr<float>(target_local_max_ptr);
  const float* t_sum = ptr<float>(target_local_sumexp_ptr);
  const float* d_max = ptr<float>(draft_local_max_ptr);
  const float* d_sum = ptr<float>(draft_local_sumexp_ptr);
  const float* temps = ptr<float>(temp_ptr);
  const float* rates = SYNTHETIC_MODE ? ptr<float>(*synthetic_conditional_rates_ptr) : nullptr;
  float* target_lse_out = ptr<float>(target_rejected_logsumexp_ptr);
  float* draft_lse_out = ptr<float>(draft_rejected_logsumexp_ptr);
  const int64_t nb = vocab_num_blocks;
  for (int64_t r = 0; r < grid[0]; ++r) {
    const int64_t req = idx[r];
    const int64_t start = cu[r];
    const int64_t num_drafts = cu[r + 1] - start - 1;
    const int64_t seed = seeds[req];
    const float temp = temps[req];
    const bool greedy = temp == 0.0f;
    int64_t accepted_length = 0;
    float target_lse = 0.0f, draft_lse = 0.0f;
    bool verifying = true;
    for (int64_t i = 0; i < num_drafts; ++i) {
      const int64_t logit = start + i;
      int64_t token = draft_sampled[logit + 1];
      const bool valid = token >= 0;
      token = std::max<int64_t>(0, token);
      if (!greedy) verifying = verifying && valid;
      if (!verifying) continue;
      const float u = tl_rand32(seed, positions[logit]);
      bool accepted;
      int64_t target_argmax = 0;
      if (greedy) {
        const int64_t b = argmax_no_nan(t_max + logit * target_local_max_stride, nb);
        target_argmax = t_argmax[logit * target_local_argmax_stride + b];
        accepted = (SYNTHETIC_MODE ? u < rates[i] : target_argmax == token) && valid;
      } else {
        const float t_logit = valid ? target[logit * target_logits_stride + token] : kNegInf;
        const float t_lse = global_lse(t_max + logit * target_local_max_stride,
                                       t_sum + logit * target_local_sumexp_stride, nb);
        const float target_logprob = t_logit - t_lse;
        float draft_logprob = 0.0f, d_lse = 0.0f;
        if (HAS_DRAFT_LOGITS) {
          const float d_logit =
              valid ? (*draft)[req * draft_logits_stride_0 + i * draft_logits_stride_1 + token] /
                          temp
                    : kNegInf;
          d_lse = global_lse(d_max + logit * draft_local_max_stride,
                             d_sum + logit * draft_local_sumexp_stride, nb);
          draft_logprob = d_logit - d_lse;
        }
        target_lse = t_lse;
        draft_lse = d_lse;
        accepted = SYNTHETIC_MODE ? u < rates[i]
                                  : target_logprob > std::log(u) + draft_logprob;
      }
      verifying = accepted;
      accepted_length += accepted;
      sampled.set(r * sampled_stride + i, greedy && !accepted ? target_argmax : token);
    }
    rejected.set(r, accepted_length);
    target_lse_out[r] = target_lse;
    draft_lse_out[r] = draft_lse;
  }
}

void mrv2_resample(
    std::vector<int64_t> grid, const torch::Tensor& resampled_local_argmax_ptr,
    int64_t resampled_local_argmax_stride, const torch::Tensor& resampled_local_max_ptr,
    int64_t resampled_local_max_stride, const torch::Tensor& target_logits_ptr,
    int64_t target_logits_stride, const torch::Tensor& target_rejected_logsumexp_ptr,
    const std::optional<torch::Tensor>& draft_logits_ptr, int64_t draft_logits_stride_0,
    int64_t draft_logits_stride_1, const torch::Tensor& draft_rejected_logsumexp_ptr,
    const torch::Tensor& rejected_step_ptr, const torch::Tensor& cu_num_logits_ptr,
    const torch::Tensor& expanded_idx_mapping_ptr, const torch::Tensor& draft_sampled_ptr,
    const torch::Tensor& temp_ptr, const torch::Tensor& seed_ptr, const torch::Tensor& pos_ptr,
    const std::optional<torch::Tensor>& cumulative_log_p_ptr,
    const std::optional<torch::Tensor>& contexts_ptr, int64_t contexts_stride,
    const std::optional<torch::Tensor>& watermarking_ptr,
    const std::optional<torch::Tensor>& watermarking_skip_mask_ptr, int64_t watermark_key_0,
    int64_t watermark_key_1, int64_t vocab_size, int64_t BLOCK_SIZE, bool HAS_DRAFT_LOGITS,
    bool USE_FP64, bool USE_BLOCK_VERIFICATION, int64_t CONTEXT_WIDTH, bool WATERMARK,
    bool DEDUPLICATE_CONTEXTS) {
  TORCH_CHECK(!USE_BLOCK_VERIFICATION && !WATERMARK,
              "MRV2 host resample: block verification and watermarking are not supported "
              "with speculative decoding on Spyre");
  const Ints rejected(rejected_step_ptr), cu(cu_num_logits_ptr), idx(expanded_idx_mapping_ptr),
      draft_sampled(draft_sampled_ptr), seeds(seed_ptr), positions(pos_ptr),
      out_argmax(resampled_local_argmax_ptr);
  const Floats target(target_logits_ptr);
  std::optional<Floats> draft;
  if (HAS_DRAFT_LOGITS) draft.emplace(*draft_logits_ptr);
  const float* temps = ptr<float>(temp_ptr);
  const float* t_lse = ptr<float>(target_rejected_logsumexp_ptr);
  const float* d_lse = ptr<float>(draft_rejected_logsumexp_ptr);
  const int64_t num_blocks = grid[1];
#pragma omp parallel for schedule(dynamic) if (grid[0] > 1)
  for (int64_t r = 0; r < grid[0]; ++r) {
    const int64_t step = rejected[r];
    const int64_t token_idx = cu[r] + step;
    const int64_t req = idx[token_idx];
    const float temp = temps[req];
    const bool bonus = token_idx == cu[r + 1] - 1;
    if (temp == 0.0f && !bonus) continue;
    const int64_t rejected_token = bonus ? 0 : draft_sampled[token_idx + 1];
    std::vector<float> residual(vocab_size);
    for (int64_t v = 0; v < vocab_size; ++v) {
      const float t = target[token_idx * target_logits_stride + v];
      if (bonus || rejected_token < 0) {
        residual[v] = t;
      } else if (HAS_DRAFT_LOGITS) {
        const float d =
            (*draft)[req * draft_logits_stride_0 + step * draft_logits_stride_1 + v] / temp;
        const float tl = t - t_lse[r];
        const float ratio = std::exp((d - d_lse[r]) - tl);
        residual[v] = ratio < 1.0f ? tl + std::log1p(-ratio) : kNegInf;
      } else {
        residual[v] = v != rejected_token ? t : kNegInf;
      }
    }
    for (int64_t blk = 0; blk < num_blocks; ++blk) {
      const int64_t begin = blk * BLOCK_SIZE, end = std::min(begin + BLOCK_SIZE, vocab_size);
      if (USE_FP64) {
        const auto [i, value] =
            gumbel_block<double>(residual, begin, end, temp, seeds[req], positions[token_idx]);
        out_argmax.set(r * resampled_local_argmax_stride + blk, begin + i);
        ptr<double>(resampled_local_max_ptr)[r * resampled_local_max_stride + blk] = value;
      } else {
        const auto [i, value] =
            gumbel_block<float>(residual, begin, end, temp, seeds[req], positions[token_idx]);
        out_argmax.set(r * resampled_local_argmax_stride + blk, begin + i);
        ptr<float>(resampled_local_max_ptr)[r * resampled_local_max_stride + blk] = value;
      }
    }
  }
}

void mrv2_insert_resampled(std::vector<int64_t> grid, const torch::Tensor& sampled_ptr,
                           int64_t sampled_stride, const torch::Tensor& num_sampled_ptr,
                           const torch::Tensor& resampled_local_argmax_ptr,
                           int64_t resampled_local_argmax_stride,
                           const torch::Tensor& resampled_local_max_ptr,
                           int64_t resampled_local_max_stride, int64_t resample_num_blocks,
                           const torch::Tensor& cu_num_logits_ptr,
                           const torch::Tensor& expanded_idx_mapping_ptr,
                           const torch::Tensor& temp_ptr, int64_t PADDED_RESAMPLE_NUM_BLOCKS) {
  const Ints sampled(sampled_ptr), num_sampled(num_sampled_ptr),
      local_argmax(resampled_local_argmax_ptr), cu(cu_num_logits_ptr),
      idx(expanded_idx_mapping_ptr);
  const float* temps = ptr<float>(temp_ptr);
  const bool fp64 = resampled_local_max_ptr.scalar_type() == at::kDouble;
  for (int64_t r = 0; r < grid[0]; ++r) {
    const int64_t n = num_sampled[r];
    const int64_t token_idx = cu[r] + n;
    const int64_t req = idx[token_idx];
    num_sampled.set(r, n + 1);
    const bool bonus = token_idx == cu[r + 1] - 1;
    if (temps[req] == 0.0f && !bonus) continue;
    int64_t best_block = 0;
    double best = -std::numeric_limits<double>::infinity();
    for (int64_t b = 0; b < resample_num_blocks; ++b) {
      const int64_t at = r * resampled_local_max_stride + b;
      double v = fp64 ? ptr<double>(resampled_local_max_ptr)[at]
                      : ptr<float>(resampled_local_max_ptr)[at];
      if (std::isnan(v)) v = -std::numeric_limits<double>::infinity();
      if (v > best) {
        best = v;
        best_block = b;
      }
    }
    sampled.set(r * sampled_stride + n,
                local_argmax[r * resampled_local_argmax_stride + best_block]);
  }
}

// ---------------------------------------------------------- dflash/speculator.py

void mrv2_prepare_dflash_inputs(
    std::vector<int64_t> grid, const torch::Tensor& out_input_ids_ptr,
    const torch::Tensor& out_query_positions_ptr, const torch::Tensor& out_is_padding_ptr,
    const torch::Tensor& out_query_start_loc_ptr, const torch::Tensor& out_seq_lens_ptr,
    const torch::Tensor& out_query_slot_mapping_ptr,
    const torch::Tensor& out_context_positions_ptr,
    const torch::Tensor& out_context_slot_mapping_ptr,
    const torch::Tensor& out_sample_indices_ptr, const torch::Tensor& out_sample_pos_ptr,
    const torch::Tensor& out_sample_idx_mapping_ptr, const torch::Tensor& out_temperature_ptr,
    const torch::Tensor& out_seeds_ptr, const torch::Tensor& target_positions_ptr,
    const torch::Tensor& target_query_start_loc_ptr, const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& last_sampled_ptr, const torch::Tensor& next_prefill_tokens_ptr,
    const torch::Tensor& num_sampled_ptr, const torch::Tensor& num_rejected_ptr,
    const torch::Tensor& temperature_ptr, const torch::Tensor& seeds_ptr,
    const torch::Tensor& block_table_ptr, int64_t block_table_stride,
    int64_t parallel_drafting_token_id, int64_t block_size, int64_t num_query_per_req,
    int64_t num_speculative_steps, int64_t max_num_reqs, int64_t max_num_tokens,
    int64_t max_model_len, int64_t cp_rank, bool SAMPLE_FROM_ANCHOR, int64_t PAD_SLOT_ID,
    int64_t CP_SIZE, int64_t CP_INTERLEAVE, int64_t BLOCK_SIZE) {
  TORCH_CHECK(CP_SIZE == 1, "MRV2 host prepare_dflash_inputs: context parallelism is not "
              "supported on Spyre");
  const Ints input_ids(out_input_ids_ptr), q_pos(out_query_positions_ptr),
      is_padding(out_is_padding_ptr), qsl_out(out_query_start_loc_ptr),
      seq_lens(out_seq_lens_ptr), q_slots(out_query_slot_mapping_ptr),
      ctx_pos_out(out_context_positions_ptr), ctx_slots(out_context_slot_mapping_ptr),
      sample_idx(out_sample_indices_ptr), sample_pos(out_sample_pos_ptr),
      sample_map(out_sample_idx_mapping_ptr), seeds_out(out_seeds_ptr),
      target_pos(target_positions_ptr), qsl(target_query_start_loc_ptr), idx(idx_mapping_ptr),
      last_sampled(last_sampled_ptr), next_prefill(next_prefill_tokens_ptr),
      num_sampled(num_sampled_ptr), num_rejected(num_rejected_ptr), seeds(seeds_ptr),
      table(block_table_ptr);
  float* temps_out = ptr<float>(out_temperature_ptr);
  const float* temps = ptr<float>(temperature_ptr);
  const int64_t num_reqs = grid[0];
  const int64_t span = grid[1] * BLOCK_SIZE;
  auto slot = [&](int64_t r, int64_t p, bool mask) -> int64_t {
    if (!mask) return PAD_SLOT_ID;
    const int64_t block_num = std::min(p / block_size, block_table_stride - 1);
    const int64_t block_id = table[r * block_table_stride + block_num];
    return block_id != 0 ? block_id * block_size + p % block_size : PAD_SLOT_ID;
  };
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t req = idx[r];
    const int64_t ctx_start = qsl[r], ctx_end = qsl[r + 1];
    const int64_t num_ctx = ctx_end - ctx_start;
    const int64_t valid_end = ctx_end - num_rejected[r];
    const int64_t num_valid = valid_end - ctx_start;
    const int64_t bonus = num_sampled[r] > 0 ? int32_t(last_sampled[req])
                                             : int32_t(next_prefill[req]);
    const int64_t last_valid_pos = target_pos[valid_end - 1];
    const int64_t query_base = r * num_query_per_req;
    for (int64_t j = 0; j < span; ++j) {
      const bool is_ctx = j < num_ctx, is_valid = j < num_valid;
      const bool is_query = j >= num_valid && j < num_valid + num_query_per_req;
      if (is_ctx) {
        const int64_t p = is_valid ? target_pos[ctx_start + j] : 0;
        ctx_pos_out.set(ctx_start + j, p);
        ctx_slots.set(ctx_start + j, slot(r, p, is_valid));
      }
      if (!is_query) continue;
      const int64_t off = j - num_valid;
      const int64_t p = last_valid_pos + 1 + off;
      const int64_t q = query_base + off;
      input_ids.set(q, off == 0 ? bonus : parallel_drafting_token_id);
      is_padding.set(q, 0);
      q_pos.set(q, std::min(p, max_model_len - 1));
      q_slots.set(q, slot(r, p, true));
      const int64_t sample_off = SAMPLE_FROM_ANCHOR ? 0 : 1;
      if (off >= sample_off) {
        const int64_t s = r * num_speculative_steps + (off - sample_off);
        sample_idx.set(s, q);
        sample_pos.set(s, SAMPLE_FROM_ANCHOR ? p + 1 : p);
        sample_map.set(s, req);
      }
    }
    qsl_out.set(r, query_base);
    seq_lens.set(r, std::min(last_valid_pos + 1 + num_query_per_req, max_model_len));
    temps_out[req] = temps[req];
    seeds_out.set(req, seeds[req]);
  }
  const int64_t last_query_end = num_reqs * num_query_per_req;
  for (int64_t i = num_reqs; i < max_num_reqs + 1; ++i) qsl_out.set(i, last_query_end);
  for (int64_t i = num_reqs; i < max_num_reqs; ++i) seq_lens.set(i, 0);
  for (int64_t i = num_reqs * num_speculative_steps; i < max_num_reqs * num_speculative_steps;
       ++i) {
    sample_idx.set(i, 0);
    sample_pos.set(i, 0);
    sample_map.set(i, -1);
  }
  for (int64_t i = last_query_end; i < max_num_tokens; ++i) {
    input_ids.set(i, 0);
    q_pos.set(i, 0);
    is_padding.set(i, 1);
    q_slots.set(i, PAD_SLOT_ID);
  }
}

TORCH_LIBRARY_FRAGMENT(_spyre_C, ops) {
  ops.def("mrv2_flatten_sampled(int[] grid, Tensor flat_sampled_ptr, Tensor sampled_ptr, int sampled_stride, Tensor num_sampled_ptr, Tensor cu_num_logits_ptr) -> ()");
  ops.impl("mrv2_flatten_sampled", torch::kCPU, &mrv2_flatten_sampled);
  ops.def("mrv2_gather_draft_sampled(int[] grid, Tensor draft_sampled_ptr, Tensor pos_ptr, Tensor input_ids_ptr, Tensor positions_ptr, Tensor logits_indices_ptr, Tensor expanded_idx_mapping_ptr, Tensor expanded_local_pos_ptr, Tensor prefill_len_ptr, int num_logits, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_gather_draft_sampled", torch::kCPU, &mrv2_gather_draft_sampled);
  ops.def("mrv2_compute_local_logits_stats(int[] grid, Tensor target_local_argmax_ptr, int target_local_argmax_stride, Tensor target_local_max_ptr, int target_local_max_stride, Tensor target_local_sumexp_ptr, int target_local_sumexp_stride, Tensor draft_local_max_ptr, int draft_local_max_stride, Tensor draft_local_sumexp_ptr, int draft_local_sumexp_stride, Tensor target_logits_ptr, int target_logits_stride, Tensor? draft_logits_ptr, int draft_logits_stride_0, int draft_logits_stride_1, Tensor expanded_idx_mapping_ptr, Tensor expanded_local_pos_ptr, Tensor temp_ptr, int vocab_size, int num_logits, int BLOCK_SIZE, bool HAS_DRAFT_LOGITS) -> ()");
  ops.impl("mrv2_compute_local_logits_stats", torch::kCPU, &mrv2_compute_local_logits_stats);
  ops.def("mrv2_rejection(int[] grid, Tensor sampled_ptr, int sampled_stride, Tensor rejected_steps_ptr, Tensor target_rejected_logsumexp_ptr, Tensor draft_rejected_logsumexp_ptr, Tensor target_logits_ptr, int target_logits_stride, Tensor target_local_argmax_ptr, int target_local_argmax_stride, Tensor target_local_max_ptr, int target_local_max_stride, Tensor target_local_sumexp_ptr, int target_local_sumexp_stride, Tensor draft_sampled_ptr, Tensor? draft_logits_ptr, int draft_logits_stride_0, int draft_logits_stride_1, Tensor draft_local_max_ptr, int draft_local_max_stride, Tensor draft_local_sumexp_ptr, int draft_local_sumexp_stride, Tensor cu_num_logits_ptr, Tensor idx_mapping_ptr, Tensor temp_ptr, Tensor seed_ptr, Tensor pos_ptr, Tensor? synthetic_conditional_rates_ptr, Tensor? cumulative_log_p_ptr, Tensor? local_residual_mass_ptr, int local_residual_mass_stride, Tensor? fly_entropy_ptr, float fly_entropy_threshold, int vocab_num_blocks, int PADDED_VOCAB_NUM_BLOCKS, bool HAS_DRAFT_LOGITS, bool SYNTHETIC_MODE, bool USE_BLOCK_VERIFICATION, int FLY_WINDOW_SIZE) -> ()");
  ops.impl("mrv2_rejection", torch::kCPU, &mrv2_rejection);
  ops.def("mrv2_resample(int[] grid, Tensor resampled_local_argmax_ptr, int resampled_local_argmax_stride, Tensor resampled_local_max_ptr, int resampled_local_max_stride, Tensor target_logits_ptr, int target_logits_stride, Tensor target_rejected_logsumexp_ptr, Tensor? draft_logits_ptr, int draft_logits_stride_0, int draft_logits_stride_1, Tensor draft_rejected_logsumexp_ptr, Tensor rejected_step_ptr, Tensor cu_num_logits_ptr, Tensor expanded_idx_mapping_ptr, Tensor draft_sampled_ptr, Tensor temp_ptr, Tensor seed_ptr, Tensor pos_ptr, Tensor? cumulative_log_p_ptr, Tensor? contexts_ptr, int contexts_stride, Tensor? watermarking_ptr, Tensor? watermarking_skip_mask_ptr, int watermark_key_0, int watermark_key_1, int vocab_size, int BLOCK_SIZE, bool HAS_DRAFT_LOGITS, bool USE_FP64, bool USE_BLOCK_VERIFICATION, int CONTEXT_WIDTH, bool WATERMARK, bool DEDUPLICATE_CONTEXTS) -> ()");
  ops.impl("mrv2_resample", torch::kCPU, &mrv2_resample);
  ops.def("mrv2_insert_resampled(int[] grid, Tensor sampled_ptr, int sampled_stride, Tensor num_sampled_ptr, Tensor resampled_local_argmax_ptr, int resampled_local_argmax_stride, Tensor resampled_local_max_ptr, int resampled_local_max_stride, int resample_num_blocks, Tensor cu_num_logits_ptr, Tensor expanded_idx_mapping_ptr, Tensor temp_ptr, int PADDED_RESAMPLE_NUM_BLOCKS) -> ()");
  ops.impl("mrv2_insert_resampled", torch::kCPU, &mrv2_insert_resampled);
  ops.def("mrv2_prepare_dflash_inputs(int[] grid, Tensor out_input_ids_ptr, Tensor out_query_positions_ptr, Tensor out_is_padding_ptr, Tensor out_query_start_loc_ptr, Tensor out_seq_lens_ptr, Tensor out_query_slot_mapping_ptr, Tensor out_context_positions_ptr, Tensor out_context_slot_mapping_ptr, Tensor out_sample_indices_ptr, Tensor out_sample_pos_ptr, Tensor out_sample_idx_mapping_ptr, Tensor out_temperature_ptr, Tensor out_seeds_ptr, Tensor target_positions_ptr, Tensor target_query_start_loc_ptr, Tensor idx_mapping_ptr, Tensor last_sampled_ptr, Tensor next_prefill_tokens_ptr, Tensor num_sampled_ptr, Tensor num_rejected_ptr, Tensor temperature_ptr, Tensor seeds_ptr, Tensor block_table_ptr, int block_table_stride, int parallel_drafting_token_id, int block_size, int num_query_per_req, int num_speculative_steps, int max_num_reqs, int max_num_tokens, int max_model_len, int cp_rank, bool SAMPLE_FROM_ANCHOR, int PAD_SLOT_ID, int CP_SIZE, int CP_INTERLEAVE, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_prepare_dflash_inputs", torch::kCPU, &mrv2_prepare_dflash_inputs);
}
