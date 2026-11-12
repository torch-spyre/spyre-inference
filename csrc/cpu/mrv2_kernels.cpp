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
// philox4x32_10 is Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as
// 1, 2, 3", SC'11); it and uint_to_uniform_float follow Triton's tl.rand
// (python/triton/language/random.py) so the outputs match it bit for bit.

// Host replacements for MRV2's Triton kernels (vllm/v1/worker/gpu). Each op
// takes the launch grid followed by the Triton kernel's own arguments, in
// order and by name, so kernels.py can bind it over the @triton.jit object.
// Each op also reproduces the kernel's semantics exactly, including the
// padding writes and the rows a kernel deliberately leaves alone.

#include "cpu_types.hpp"

#if defined(__AVX512F__)
  #include <immintrin.h>
  #include <sleef.h>
#endif

#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

void check_cpu(const torch::Tensor& t) {
  TORCH_CHECK(t.device().is_cpu(), "MRV2 host kernels need host tensors, got ",
              t.device());
}

template <typename T>
T* ptr(const torch::Tensor& t) {
  check_cpu(t);
  return static_cast<T*>(t.data_ptr());
}

// An int32 or int64 tensor read and written as int64, as Triton's typed
// loads and stores do. MRV2 mixes both widths for the same quantity.
class Ints {
 public:
  explicit Ints(const torch::Tensor& t)
      : data_(ptr<void>(t)), wide_(t.scalar_type() == at::kLong) {
    TORCH_CHECK(wide_ || t.scalar_type() == at::kInt,
                "expected an int32 or int64 tensor, got ", t.scalar_type());
  }
  int64_t operator[](int64_t i) const {
    return wide_ ? static_cast<const int64_t*>(data_)[i]
                 : static_cast<const int32_t*>(data_)[i];
  }
  void set(int64_t i, int64_t v) const {
    if (wide_)
      static_cast<int64_t*>(data_)[i] = v;
    else
      static_cast<int32_t*>(data_)[i] = static_cast<int32_t>(v);
  }

 private:
  void* data_;
  bool wide_;
};

// A table of raw host addresses (MRV2's uint64 pointer-to-pointer tensors).
template <typename T>
T* addr(const torch::Tensor& ptrs, int64_t i) {
  return reinterpret_cast<T*>(ptr<uint64_t>(ptrs)[i]);
}

// Calls f(scalar_t{}) for the logits dtype. A function rather than
// AT_DISPATCH_*, whose macro argument cannot hold the OpenMP pragmas.
template <typename F>
void dispatch_logits(const torch::Tensor& t, F&& f) {
  switch (t.scalar_type()) {
    case at::kFloat: return f(float{});
    case at::kDouble: return f(double{});
    case at::kHalf: return f(at::Half{});
    case at::kBFloat16: return f(at::BFloat16{});
    default: TORCH_CHECK(false, "unsupported logits dtype ", t.scalar_type());
  }
}

// ---------------------------------------------------------------- murmur3 RNG
// Mirrors sample/gumbel.py bit for bit: the noise for (seed, pos, token) is the
// same on every backend, which is what keeps seeded sampling reproducible.

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

// Everything but the token offset, mixed once per row.
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
  // Coefficients rounded from double, as Triton rounds the Python literals.
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

inline double murmur3_uniform64(uint32_t lo_prefix, uint32_t hi_prefix,
                                uint32_t offset) {
  const uint64_t lo = murmur3_finish(lo_prefix, offset);
  const uint64_t hi = murmur3_finish(hi_prefix, offset);
  const uint64_t random53 = ((hi << 32) | lo) >> 11;
  const double u = (static_cast<double>(random53) + 0.5) * 1.1102230246251565e-16;
  return u == 1.0 ? u - 1.1102230246251565e-16 : u;
}

// Upper bounds on the Gumbel noise either path can draw (u >= 2^-33 resp.
// 2^-54), so a token whose logit plus the bound cannot beat the best is
// skipped without hashing. Rounding is monotone, so the skip is exact.
constexpr float kGumbelMax32 = 24.0f;
constexpr double kGumbelMax64 = 38.0;
constexpr int64_t kDraftNoiseSalt = int64_t(1) << 30;

// Elementwise vocab kernels split each row into chunks of this many tokens, so a
// small batch over a 262k vocab still spreads across the cores.
constexpr int64_t kVocabChunk = 16384;

}  // namespace

// ------------------------------------------------------------ input batching

void mrv2_prepare_prefill_inputs(
    std::vector<int64_t> grid, const torch::Tensor& input_ids_ptr,
    const torch::Tensor& next_prefill_tokens_ptr, int64_t next_prefill_tokens_stride,
    int64_t num_lookahead, const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& query_start_loc_ptr,
    const torch::Tensor& all_token_ids_ptr, int64_t all_token_ids_stride,
    const torch::Tensor& prefill_lens_ptr,
    const torch::Tensor& num_computed_tokens_ptr, int64_t BLOCK_SIZE,
    int64_t LOOKAHEAD_BLOCK) {
  const Ints input_ids(input_ids_ptr), next(next_prefill_tokens_ptr),
      idx(idx_mapping_ptr), qsl(query_start_loc_ptr), tokens(all_token_ids_ptr),
      prefill_lens(prefill_lens_ptr), computed(num_computed_tokens_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    const int64_t prefill_len = prefill_lens[req];
    const int64_t num_computed = computed[req];
    if (num_computed >= prefill_len) continue;
    const int64_t start = qsl[b], query_len = qsl[b + 1] - start;
    const int64_t row = req * all_token_ids_stride;
    for (int64_t i = 0; i < query_len; ++i)
      input_ids.set(start + i, tokens[row + num_computed + i]);
    for (int64_t j = 0; j < num_lookahead; ++j) {
      const int64_t pos = num_computed + query_len + j;
      next.set(j * next_prefill_tokens_stride + req,
               pos < prefill_len ? tokens[row + pos] : 0);
    }
  }
}

void mrv2_prepare_pos_seq_lens(std::vector<int64_t> grid, const torch::Tensor& pos_ptr,
                               const torch::Tensor& seq_lens_ptr,
                               const torch::Tensor& idx_mapping_ptr,
                               const torch::Tensor& query_start_loc_ptr,
                               const torch::Tensor& num_computed_tokens_ptr,
                               int64_t max_num_reqs, int64_t BLOCK_SIZE) {
  const Ints pos(pos_ptr), seq_lens(seq_lens_ptr), idx(idx_mapping_ptr),
      qsl(query_start_loc_ptr), computed(num_computed_tokens_ptr);
  const int64_t num_reqs = grid[0] - 1;
  for (int64_t b = 0; b < num_reqs; ++b) {
    const int64_t num_computed = computed[idx[b]];
    const int64_t start = qsl[b], query_len = qsl[b + 1] - start;
    seq_lens.set(b, num_computed + query_len);
    for (int64_t i = 0; i < query_len; ++i) pos.set(start + i, num_computed + i);
  }
  for (int64_t b = num_reqs; b < max_num_reqs; ++b) seq_lens.set(b, 0);
}

void mrv2_combine_sampled_and_draft_tokens(
    std::vector<int64_t> grid, const torch::Tensor& input_ids_ptr,
    const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& last_sampled_tokens_ptr,
    const torch::Tensor& query_start_loc_ptr, const torch::Tensor& seq_lens_ptr,
    const torch::Tensor& prefill_len_ptr, const torch::Tensor& draft_tokens_ptr,
    int64_t draft_tokens_stride, const torch::Tensor& cu_num_logits_ptr,
    const torch::Tensor& logits_indices_ptr, int64_t BLOCK_SIZE,
    int64_t NUM_NEW_SAMPLED_TOKENS) {
  const Ints input_ids(input_ids_ptr), idx(idx_mapping_ptr),
      last_sampled(last_sampled_tokens_ptr), qsl(query_start_loc_ptr),
      seq_lens(seq_lens_ptr), prefill_lens(prefill_len_ptr),
      drafts(draft_tokens_ptr), cu(cu_num_logits_ptr),
      logits_indices(logits_indices_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    const int64_t cu_start = cu[b], num_logits = cu[b + 1] - cu_start;
    const int64_t num_drafts = num_logits - NUM_NEW_SAMPLED_TOKENS;
    const int64_t query_end = qsl[b + 1];
    const int64_t logits_start = query_end - num_logits;
    for (int64_t i = 0; i < std::min(num_logits, BLOCK_SIZE); ++i)
      logits_indices.set(cu_start + i, logits_start + i);

    const int64_t seq_len = seq_lens[b], prefill_len = prefill_lens[req];
    if (seq_len <= prefill_len) continue;
    if (NUM_NEW_SAMPLED_TOKENS > 0 && seq_len - num_logits >= prefill_len)
      input_ids.set(logits_start, last_sampled[req]);
    for (int64_t i = 0; i < std::min(num_drafts, BLOCK_SIZE); ++i)
      input_ids.set(query_end - num_drafts + i,
                    drafts[req * draft_tokens_stride + i]);
  }
}

void mrv2_get_num_sampled_and_rejected(
    std::vector<int64_t> grid, const torch::Tensor& num_sampled_ptr,
    const torch::Tensor& num_rejected_ptr, const torch::Tensor& seq_lens_ptr,
    const torch::Tensor& cu_num_logits_ptr, const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& prefill_len_ptr) {
  const Ints num_sampled(num_sampled_ptr), num_rejected(num_rejected_ptr),
      seq_lens(seq_lens_ptr), cu(cu_num_logits_ptr), idx(idx_mapping_ptr),
      prefill_lens(prefill_len_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const bool chunked = seq_lens[b] < prefill_lens[idx[b]];
    const int64_t sampled = chunked ? 0 : num_sampled[b];
    num_sampled.set(b, sampled);
    num_rejected.set(b, chunked ? 0 : cu[b + 1] - cu[b] - sampled);
  }
}

void mrv2_post_update(std::vector<int64_t> grid,
                      const torch::Tensor& idx_mapping_ptr,
                      const torch::Tensor& num_computed_tokens_ptr,
                      const torch::Tensor& last_sampled_tokens_ptr,
                      const std::optional<torch::Tensor>& output_bin_counts_ptr,
                      int64_t output_bin_counts_stride,
                      const torch::Tensor& sampled_tokens_ptr,
                      int64_t sampled_tokens_stride,
                      const torch::Tensor& num_sampled_ptr,
                      const torch::Tensor& num_rejected_ptr,
                      const std::optional<torch::Tensor>& query_start_loc_ptr,
                      const torch::Tensor& all_token_ids_ptr,
                      int64_t all_token_ids_stride, const torch::Tensor& total_len_ptr,
                      const std::optional<torch::Tensor>& draft_tokens_ptr,
                      const std::optional<torch::Tensor>& draft_tokens_out_ptr,
                      int64_t num_draft_tokens) {
  const Ints idx(idx_mapping_ptr), computed(num_computed_tokens_ptr),
      last_sampled(last_sampled_tokens_ptr), sampled_tokens(sampled_tokens_ptr),
      num_sampled_t(num_sampled_ptr), num_rejected_t(num_rejected_ptr),
      tokens(all_token_ids_ptr), total_lens(total_len_ptr);
  int32_t* counts = output_bin_counts_ptr ? ptr<int32_t>(*output_bin_counts_ptr)
                                          : nullptr;
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    if (req < 0) continue;
    if (draft_tokens_ptr) {
      const Ints drafts(*draft_tokens_ptr), drafts_out(*draft_tokens_out_ptr);
      for (int64_t i = 0; i < num_draft_tokens; ++i)
        drafts_out.set(req * num_draft_tokens + i, drafts[b * num_draft_tokens + i]);
    }
    const int64_t total_len = total_lens[req];
    const int64_t num_sampled = num_sampled_t[b];
    const int64_t sampled_row = b * sampled_tokens_stride;
    if (num_sampled > 0) {
      last_sampled.set(req, sampled_tokens[sampled_row + num_sampled - 1]);
      total_lens.set(req, total_len + num_sampled);
    }
    for (int64_t i = 0; i < num_sampled; ++i) {
      const int64_t token = sampled_tokens[sampled_row + i];
      tokens.set(req * all_token_ids_stride + total_len + i, token);
      if (counts) ++counts[req * output_bin_counts_stride + token];
    }
    int64_t query_len = 0;
    if (query_start_loc_ptr) {
      const Ints qsl(*query_start_loc_ptr);
      query_len = qsl[b + 1] - qsl[b];
    }
    const int64_t delta = query_len - num_rejected_t[b];
    if (delta != 0) computed.set(req, computed[req] + delta);
  }
}

void mrv2_post_update_num_computed_tokens(
    std::vector<int64_t> grid, const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& num_computed_tokens_ptr,
    const torch::Tensor& query_start_loc_ptr) {
  const Ints idx(idx_mapping_ptr), computed(num_computed_tokens_ptr),
      qsl(query_start_loc_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    computed.set(req, computed[req] + qsl[b + 1] - qsl[b]);
  }
}

void mrv2_expand_idx_mapping(std::vector<int64_t> grid,
                             const torch::Tensor& idx_mapping_ptr,
                             const torch::Tensor& expanded_idx_mapping_ptr,
                             const torch::Tensor& expanded_local_pos_ptr,
                             const torch::Tensor& cu_num_logits_ptr,
                             int64_t BLOCK_SIZE) {
  const Ints idx(idx_mapping_ptr), expanded(expanded_idx_mapping_ptr),
      local_pos(expanded_local_pos_ptr), cu(cu_num_logits_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t start = cu[b];
    const int64_t n = std::min(cu[b + 1] - start, BLOCK_SIZE);
    for (int64_t i = 0; i < n; ++i) {
      expanded.set(start + i, idx[b]);
      local_pos.set(start + i, i);
    }
  }
}

// ---------------------------------------------- block tables and staged writes

void mrv2_gather_block_tables(std::vector<int64_t> grid,
                              const torch::Tensor& batch_idx_to_req_idx,
                              const torch::Tensor& src_block_table_ptrs,
                              const torch::Tensor& dst_block_table_ptrs,
                              const torch::Tensor& block_table_strides,
                              const torch::Tensor& num_blocks_ptr,
                              int64_t num_blocks_stride, int64_t num_reqs,
                              const torch::Tensor& num_computed_tokens_ptr,
                              const torch::Tensor& sliding_window_reach_ptr,
                              const torch::Tensor& block_sizes_ptr,
                              bool MASK_SLIDING_WINDOW, int64_t BLOCK_SIZE) {
  const Ints idx(batch_idx_to_req_idx), strides(block_table_strides),
      num_blocks(num_blocks_ptr);
  std::optional<Ints> computed, reach, block_sizes;
  if (MASK_SLIDING_WINDOW) {
    computed.emplace(num_computed_tokens_ptr);
    reach.emplace(sliding_window_reach_ptr);
    block_sizes.emplace(block_sizes_ptr);
  }
  for (int64_t g = 0; g < grid[0]; ++g) {
    const int64_t stride = strides[g];
    int32_t* dst = addr<int32_t>(dst_block_table_ptrs, g);
    const int32_t* src = addr<int32_t>(src_block_table_ptrs, g);
    for (int64_t b = 0; b < grid[1]; ++b) {
      int32_t* dst_row = dst + b * stride;
      if (b >= num_reqs) {
        std::memset(dst_row, 0, stride * sizeof(int32_t));
        continue;
      }
      const int64_t req = idx[b];
      const int64_t n = num_blocks[g * num_blocks_stride + req];
      // Leading blocks no query of this step can attend are nulled.
      int64_t num_null = 0;
      if (MASK_SLIDING_WINDOW && (*reach)[g] > 0)
        num_null = std::min(
            std::max<int64_t>((*computed)[req] - (*reach)[g], 0) / (*block_sizes)[g], n);
      std::memset(dst_row, 0, num_null * sizeof(int32_t));
      std::memcpy(dst_row + num_null, src + req * stride + num_null,
                  (n - num_null) * sizeof(int32_t));
    }
  }
}

void mrv2_compute_slot_mappings(
    std::vector<int64_t> grid, int64_t max_num_tokens,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& pos, const torch::Tensor& block_table_ptrs,
    const torch::Tensor& block_table_strides, const torch::Tensor& block_sizes,
    const torch::Tensor& slot_mapping_enabled, const torch::Tensor& dcp_sharded,
    const torch::Tensor& slot_mappings_ptr, int64_t slot_mappings_stride,
    int64_t cp_rank, int64_t CP_SIZE, int64_t CP_INTERLEAVE, int64_t PAD_ID,
    int64_t TRITON_BLOCK_SIZE) {
  const Ints idx(idx_mapping), qsl(query_start_loc), positions(pos),
      strides(block_table_strides), kv_block_sizes(block_sizes),
      slots(slot_mappings_ptr);
  const bool* enabled_t = ptr<bool>(slot_mapping_enabled);
  const bool* sharded_t = ptr<bool>(dcp_sharded);
  const int64_t num_reqs = grid[1] - 1;
  for (int64_t g = 0; g < grid[0]; ++g) {
    const int64_t base = g * slot_mappings_stride;
    for (int64_t i = qsl[num_reqs]; i < max_num_tokens; ++i)
      slots.set(base + i, PAD_ID);

    const int32_t* table = addr<int32_t>(block_table_ptrs, g);
    const int64_t table_stride = strides[g];
    const int64_t kv_block_size = kv_block_sizes[g];
    const bool enabled = enabled_t[g];
    const bool sharded = CP_SIZE != 1 && sharded_t[g];
    for (int64_t b = 0; b < num_reqs; ++b) {
      const int64_t req = idx[b];
      const bool is_real = req >= 0;
      for (int64_t t = qsl[b]; t < qsl[b + 1]; ++t) {
        const int64_t p = positions[t];
        int64_t local = p;
        bool is_local = true;
        if (CP_SIZE != 1) {
          const int64_t virtual_block = kv_block_size * CP_SIZE;
          const int64_t offset = p % virtual_block;
          const int64_t local_offset = offset / (CP_INTERLEAVE * CP_SIZE) * CP_INTERLEAVE +
                                       offset % CP_INTERLEAVE;
          const bool owned = offset / CP_INTERLEAVE % CP_SIZE == cp_rank;
          local = sharded ? p / virtual_block * kv_block_size + local_offset : p;
          is_local = !sharded || owned;
        }
        const int64_t block_index = enabled ? local / kv_block_size : 0;
        const int64_t block_number =
            is_local && is_real ? table[req * table_stride + block_index] : 0;
        int64_t slot = block_number * kv_block_size + local % kv_block_size;
        if (!is_local || !(enabled && is_real)) slot = PAD_ID;
        slots.set(base + t, slot);
      }
    }
  }
}

void mrv2_apply_write(std::vector<int64_t> grid, const torch::Tensor& output_ptr,
                      const torch::Tensor& output_stride,
                      const torch::Tensor& write_indices_ptr,
                      const torch::Tensor& write_starts_ptr,
                      const torch::Tensor& write_contents_ptr,
                      const torch::Tensor& write_cu_lens_ptr,
                      const std::optional<torch::Tensor>& write_group_ids_ptr,
                      int64_t BLOCK_SIZE, bool MULTI_GROUP) {
  const Ints indices(write_indices_ptr), starts(write_starts_ptr),
      cu_lens(write_cu_lens_ptr);
  if (MULTI_GROUP) {
    // output_ptr/output_stride are per-group tables (one block table per KV
    // cache group); every group's table is int32.
    const Ints strides(output_stride), contents(write_contents_ptr),
        groups(*write_group_ids_ptr);
    for (int64_t w = 0; w < grid[0]; ++w) {
      const int64_t group = groups[w];
      int32_t* row = addr<int32_t>(output_ptr, group) +
                     indices[w] * strides[group] + starts[w];
      const int64_t cu_start = w > 0 ? cu_lens[w - 1] : 0;
      for (int64_t i = cu_start; i < cu_lens[w]; ++i)
        row[i - cu_start] = static_cast<int32_t>(contents[i]);
    }
    return;
  }
  TORCH_CHECK(output_ptr.scalar_type() == write_contents_ptr.scalar_type(),
              "staged write dtype mismatch");
  const int64_t stride = output_stride.item<int64_t>();
  const int64_t es = output_ptr.element_size();
  char* out = ptr<char>(output_ptr);
  const char* contents = ptr<const char>(write_contents_ptr);
  for (int64_t w = 0; w < grid[0]; ++w) {
    const int64_t cu_start = w > 0 ? cu_lens[w - 1] : 0;
    std::memcpy(out + (indices[w] * stride + starts[w]) * es,
                contents + cu_start * es, (cu_lens[w] - cu_start) * es);
  }
}

// ------------------------------------------------------------------- sampling

void mrv2_temperature(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                      int64_t logits_stride,
                      const torch::Tensor& expanded_idx_mapping_ptr,
                      const torch::Tensor& temperature_ptr, int64_t vocab_size,
                      int64_t BLOCK_SIZE) {
  const Ints idx(expanded_idx_mapping_ptr);
  const float* temperature = ptr<float>(temperature_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
    const int64_t chunks = (vocab_size + kVocabChunk - 1) / kVocabChunk;
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < grid[0]; ++t) {
      for (int64_t c = 0; c < chunks; ++c) {
        const int64_t req = idx[t];
        if (req < 0) continue;
        const float temp = temperature[req];
        if (temp == 0.0f || temp == 1.0f) continue;
        scalar_t* row = logits + t * logits_stride;
        for (int64_t v = c * kVocabChunk; v < std::min(vocab_size, (c + 1) * kVocabChunk); ++v)
          row[v] = static_cast<scalar_t>(static_cast<float>(row[v]) / temp);
      }
    }
  });
}

namespace {

// Argmax of one row under Gumbel-max sampling, or plain argmax at temp 0:
// the first index of the max, which is what upstream's per-block
// tie-break-left max followed by a first-max argmax over blocks returns.
#if defined(__AVX512F__)
// 16 tokens at a time with SLEEF's logf16_u10, the call triton-cpu's own
// _gumbel_sample_kernel lowers tl.log to, so the noise stays bit-identical to it.
inline __m512i murmur3_mix16(__m512i h, __m512i k) {
  k = _mm512_mullo_epi32(k, _mm512_set1_epi32(int(0xCC9E2D51u)));
  k = _mm512_rol_epi32(k, 15);
  k = _mm512_mullo_epi32(k, _mm512_set1_epi32(0x1B873593));
  h = _mm512_rol_epi32(_mm512_xor_si512(h, k), 13);
  return _mm512_add_epi32(_mm512_mullo_epi32(h, _mm512_set1_epi32(5)),
                          _mm512_set1_epi32(int(0xE6546B64u)));
}

inline __m512i murmur3_fmix16(__m512i h) {
  h = _mm512_xor_si512(h, _mm512_srli_epi32(h, 16));
  h = _mm512_mullo_epi32(h, _mm512_set1_epi32(int(0x85EBCA6Bu)));
  h = _mm512_xor_si512(h, _mm512_srli_epi32(h, 13));
  h = _mm512_mullo_epi32(h, _mm512_set1_epi32(int(0xC2B2AE35u)));
  return _mm512_xor_si512(h, _mm512_srli_epi32(h, 16));
}

// x + Gumbel noise for 16 consecutive tokens from v (fp32 path, as upstream).
inline __m512 gumbel_noised16(__m512 x, uint32_t prefix, int64_t v) {
  const __m512i offs = _mm512_add_epi32(_mm512_set1_epi32(int32_t(v)),
                                        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                          12, 13, 14, 15));
  const __m512i bits = murmur3_fmix16(_mm512_xor_si512(
      murmur3_mix16(_mm512_set1_epi32(int32_t(prefix)), offs), _mm512_set1_epi32(16)));
  const __m512 hi16 = _mm512_cvtepi32_ps(_mm512_srli_epi32(bits, 16));
  const __m512 lo16 = _mm512_cvtepi32_ps(_mm512_and_si512(bits, _mm512_set1_epi32(0xFFFF)));
  const __m512 u = _mm512_add_ps(
      _mm512_mul_ps(hi16, _mm512_set1_ps(1.52587890625e-05f)),
      _mm512_mul_ps(_mm512_add_ps(lo16, _mm512_set1_ps(0.5f)),
                    _mm512_set1_ps(2.3283064365386963e-10f)));
  // log1p_neg_stable(u), operation for operation.
  __m512 p = _mm512_set1_ps(float(1.0 / 8.0));
  for (const float c : {float(1.0 / 7.0), float(1.0 / 6.0), float(1.0 / 5.0), float(1.0 / 4.0),
                        float(1.0 / 3.0), float(1.0 / 2.0), 1.0f})
    p = _mm512_add_ps(_mm512_set1_ps(c), _mm512_mul_ps(u, p));
  const __m512 series = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), u), p);
  const __m512 direct = Sleef_logf16_u10(
      _mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), u), _mm512_set1_ps(5.960464477539063e-08f)));
  const __mmask16 small = _mm512_cmp_ps_mask(u, _mm512_set1_ps(0.25f), _CMP_LT_OQ);
  const __m512 l = _mm512_mask_blend_ps(small, direct, series);
  const __m512 g = _mm512_sub_ps(_mm512_setzero_ps(),
                                 Sleef_logf16_u10(_mm512_sub_ps(_mm512_setzero_ps(), l)));
  return _mm512_add_ps(x, g);
}
#endif

template <typename scalar_t, typename acc_t>
std::pair<int64_t, acc_t> gumbel_row(const scalar_t* row, int64_t begin, int64_t end,
                                     float temp, bool apply_temperature,
                                     int64_t seed, int64_t pos) {
  const bool noisy = temp != 0.0f;
  const bool divide = noisy && apply_temperature;
  constexpr bool fp64 = std::is_same_v<acc_t, double>;
  const uint32_t lo_prefix = murmur3_prefix(seed, pos, 0);
  const uint32_t hi_prefix = murmur3_prefix(seed, pos, 0x9E3779B9u);
  acc_t best = -std::numeric_limits<acc_t>::infinity();
  int64_t best_idx = begin;
  int64_t v = begin;
#if defined(__AVX512F__)
  if constexpr (!fp64 && std::is_same_v<scalar_t, float>) {
    if (noisy) {
      // Hash only blocks where some token could still win; the first-max over
      // blocks and then within a block keeps upstream's lowest-index tie-break.
      for (; v + 16 <= end; v += 16) {
        __m512 x = _mm512_loadu_ps(row + v);
        if (divide) x = _mm512_div_ps(x, _mm512_set1_ps(temp));
        if (_mm512_reduce_max_ps(x) + kGumbelMax32 <= best) continue;
        const __m512 value = gumbel_noised16(x, lo_prefix, v);
        const float m = _mm512_reduce_max_ps(value);
        if (m > best) {
          best = m;
          best_idx = v + __builtin_ctz(_mm512_cmp_ps_mask(value, _mm512_set1_ps(m), _CMP_EQ_OQ));
        }
      }
    } else {
      // Greedy: a running max, then one scan for its first index.
      __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
      int64_t w = v;
      for (; w + 16 <= end; w += 16) vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + w));
      const float m = _mm512_reduce_max_ps(vmax);
      if (m > best) {
        for (int64_t i = v; i + 16 <= end; i += 16) {
          const __mmask16 hit =
              _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i), _mm512_set1_ps(m), _CMP_EQ_OQ);
          if (hit) {
            best = m;
            best_idx = i + __builtin_ctz(hit);
            break;
          }
        }
      }
      v = w;
    }
  }
#endif
  for (; v < end; ++v) {
    float x32 = static_cast<float>(row[v]);
    if (divide) x32 = x32 / temp;
    const acc_t x = static_cast<acc_t>(x32);
    acc_t value = x;
    if (noisy) {
      if constexpr (fp64) {
        if (x + kGumbelMax64 <= best) continue;
        const double u = murmur3_uniform64(lo_prefix, hi_prefix, uint32_t(v));
        value = x + -std::log(-std::log(u));
      } else {
        if (x + kGumbelMax32 <= best) continue;
        const float u = murmur3_uniform32(murmur3_finish(lo_prefix, uint32_t(v)));
        value = x + -std::log(-log1p_neg_stable(u));
      }
    }
    if (value > best) {
      best = value;
      best_idx = v;
    }
  }
  return {best_idx, best};
}

}  // namespace

void mrv2_gumbel_sample(
    std::vector<int64_t> grid, const torch::Tensor& local_argmax_ptr,
    int64_t local_argmax_stride, const torch::Tensor& local_max_ptr,
    int64_t local_max_stride, const std::optional<torch::Tensor>& logits_cache_ptr,
    int64_t logits_cache_stride_0, int64_t logits_cache_stride_1,
    const std::optional<torch::Tensor>& logits_cache_col_ptr,
    const std::optional<torch::Tensor>& logits_cache_source_ptr,
    int64_t logits_cache_source_stride,
    const torch::Tensor& logits_ptr, int64_t logits_stride,
    const torch::Tensor& expanded_idx_mapping_ptr, const torch::Tensor& seeds_ptr,
    const torch::Tensor& pos_ptr, const torch::Tensor& temp_ptr,
    int64_t vocab_size, int64_t BLOCK_SIZE, bool IS_DRAFTING,
    bool APPLY_TEMPERATURE, bool USE_FP64, bool PER_TOKEN_COL) {
  const Ints idx(expanded_idx_mapping_ptr), seeds(seeds_ptr), positions(pos_ptr);
  // Draft logits cache (spec decode): the pre-temperature source row is stored at
  // [req, col], and col is mixed into the seed's high word (spec_step_seed).
  const bool cached = logits_cache_ptr.has_value();
  std::optional<Ints> cols;
  if (cached) cols.emplace(*logits_cache_col_ptr);
  auto col_of = [&](int64_t t) { return PER_TOKEN_COL ? (*cols)[t] : (*cols)[0]; };
  if (cached) {
    const at::ScalarType dtype = logits_cache_ptr->scalar_type();
    TORCH_CHECK(logits_cache_source_ptr->scalar_type() == dtype,
                "logits cache source and destination dtypes differ");
    const int64_t elem = logits_cache_ptr->element_size();
    char* cache = static_cast<char*>(ptr<void>(*logits_cache_ptr));
    const char* source = static_cast<const char*>(ptr<void>(*logits_cache_source_ptr));
    for (int64_t t = 0; t < grid[0]; ++t) {
      const int64_t req = idx[t];
      if (req < 0) continue;
      std::memcpy(
          cache + (req * logits_cache_stride_0 + col_of(t) * logits_cache_stride_1) * elem,
          source + t * logits_cache_source_stride * elem, vocab_size * elem);
    }
  }
  const float* temps = ptr<float>(temp_ptr);
  int64_t* local_argmax = ptr<int64_t>(local_argmax_ptr);
  const int64_t num_blocks = grid[1];
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
    auto run = [&](auto* local_max) {
      using acc_t = std::remove_pointer_t<decltype(local_max)>;
      // One task per (row, block), as Triton's grid: a batch of a few rows over a
      // 262k vocab would otherwise leave most cores idle. Each block keeps its own
      // first-max, and the wrapper's argmax over blocks keeps the first block's.
#pragma omp parallel for collapse(2) schedule(dynamic, 4) if (grid[0] * num_blocks > 1)
      for (int64_t t = 0; t < grid[0]; ++t) {
        for (int64_t blk = 0; blk < num_blocks; ++blk) {
          const int64_t req = idx[t];
          const float temp = req >= 0 ? temps[req] : 0.0f;
          int64_t seed = req >= 0 ? seeds[req] : 0;
          if (cached) seed ^= static_cast<int64_t>(static_cast<uint64_t>(col_of(t)) << 32);
          const int64_t pos = positions[t] + (IS_DRAFTING ? kDraftNoiseSalt : 0);
          const int64_t begin = blk * BLOCK_SIZE;
          const auto [token, value] = gumbel_row<scalar_t, acc_t>(
              logits + t * logits_stride, begin, std::min(begin + BLOCK_SIZE, vocab_size),
              temp, APPLY_TEMPERATURE, seed, pos);
          local_argmax[t * local_argmax_stride + blk] = token;
          local_max[t * local_max_stride + blk] = value;
        }
      }
    };
    if (USE_FP64)
      run(ptr<double>(local_max_ptr));
    else
      run(ptr<float>(local_max_ptr));
  });
}

void mrv2_min_p(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                int64_t logits_stride, const torch::Tensor& expanded_idx_mapping_ptr,
                const torch::Tensor& min_p_ptr, int64_t vocab_size,
                int64_t BLOCK_SIZE) {
  const Ints idx(expanded_idx_mapping_ptr);
  const float* min_ps = ptr<float>(min_p_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
    const int64_t chunks = (vocab_size + kVocabChunk - 1) / kVocabChunk;
    // Pass 1: per-(row, chunk) maxima; pass 2: mask against the row threshold.
    std::vector<float> chunk_max(grid[0] * chunks, kNegInf);
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < grid[0]; ++t) {
      for (int64_t c = 0; c < chunks; ++c) {
        if (min_ps[idx[t]] == 0.0f) continue;
        const scalar_t* row = logits + t * logits_stride;
        const int64_t hi = std::min(vocab_size, (c + 1) * kVocabChunk);
        float m = kNegInf;
        int64_t v = c * kVocabChunk;
#if defined(__AVX512F__)
        if constexpr (std::is_same_v<scalar_t, float>) {
          __m512 vm = _mm512_set1_ps(kNegInf);
          for (; v + 16 <= hi; v += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(row + v));
          m = _mm512_reduce_max_ps(vm);
        }
#endif
        for (; v < hi; ++v) m = std::max(m, static_cast<float>(row[v]));
        chunk_max[t * chunks + c] = m;
      }
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < grid[0]; ++t) {
      for (int64_t c = 0; c < chunks; ++c) {
        const float min_p = min_ps[idx[t]];
        if (min_p == 0.0f) continue;
        float max_val = kNegInf;
        for (int64_t i = 0; i < chunks; ++i) max_val = std::max(max_val, chunk_max[t * chunks + i]);
        const float threshold = max_val + std::log(min_p);
        scalar_t* row = logits + t * logits_stride;
        const int64_t hi = std::min(vocab_size, (c + 1) * kVocabChunk);
        int64_t v = c * kVocabChunk;
#if defined(__AVX512F__)
        if constexpr (std::is_same_v<scalar_t, float>) {
          const __m512 th = _mm512_set1_ps(threshold), ninf = _mm512_set1_ps(kNegInf);
          for (; v + 16 <= hi; v += 16) {
            const __m512 x = _mm512_loadu_ps(row + v);
            _mm512_storeu_ps(row + v,
                             _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, th, _CMP_LT_OQ), x, ninf));
          }
        }
#endif
        for (; v < hi; ++v)
          if (static_cast<float>(row[v]) < threshold) row[v] = static_cast<scalar_t>(kNegInf);
      }
    }
  });
}

void mrv2_penalties(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                    int64_t logits_stride,
                    const torch::Tensor& expanded_idx_mapping_ptr,
                    const torch::Tensor& token_ids_ptr,
                    const torch::Tensor& expanded_local_pos_ptr,
                    const torch::Tensor& repetition_penalty_ptr,
                    const torch::Tensor& frequency_penalty_ptr,
                    const torch::Tensor& presence_penalty_ptr,
                    const torch::Tensor& prompt_bin_mask_ptr,
                    int64_t prompt_bin_mask_stride,
                    const torch::Tensor& output_bin_counts_ptr,
                    int64_t output_bin_counts_stride, int64_t vocab_size,
                    int64_t BLOCK_SIZE) {
  const Ints idx(expanded_idx_mapping_ptr), token_ids(token_ids_ptr),
      local_pos(expanded_local_pos_ptr);
  const float* rep_t = ptr<float>(repetition_penalty_ptr);
  const float* freq_t = ptr<float>(frequency_penalty_ptr);
  const float* pres_t = ptr<float>(presence_penalty_ptr);
  const int32_t* prompt_mask = ptr<int32_t>(prompt_bin_mask_ptr);
  const int32_t* bin_counts = ptr<int32_t>(output_bin_counts_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
    const int64_t chunks = (vocab_size + kVocabChunk - 1) / kVocabChunk;
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < grid[0]; ++t) {
     for (int64_t c = 0; c < chunks; ++c) {
      const int64_t req = idx[t];
      const float rep = rep_t[req], freq = freq_t[req], pres = pres_t[req];
      if (rep == 1.0f && freq == 0.0f && pres == 0.0f) continue;
      const int64_t lo = c * kVocabChunk, hi = std::min(vocab_size, (c + 1) * kVocabChunk);
      const int32_t* base = bin_counts + req * output_bin_counts_stride;
      // Draft tokens at earlier positions of this request count as output.
      std::vector<int32_t> with_drafts;
      const int64_t p = local_pos[t];
      if (p > 0) {
        with_drafts.assign(base + lo, base + hi);
        for (int64_t prev = 0; prev < p; ++prev) {
          const int64_t tok = token_ids[t - p + prev + 1];
          if (tok >= lo && tok < hi) ++with_drafts[tok - lo];
        }
        base = with_drafts.data() - lo;
      }
      const int32_t* mask = prompt_mask + req * prompt_bin_mask_stride;
      scalar_t* row = logits + t * logits_stride;
      int64_t v = lo;
#if defined(__AVX512F__)
      // lo is a multiple of 32, so each 16-token step reads half of one mask word.
      if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vrep = _mm512_set1_ps(rep), vinv = _mm512_set1_ps(1.0f / rep);
        const __m512 vfreq = _mm512_set1_ps(freq), vpres = _mm512_set1_ps(pres);
        for (; v + 16 <= hi; v += 16) {
          const __m512i cnt = _mm512_loadu_si512(base + v);
          const __mmask16 nz = _mm512_test_epi32_mask(cnt, cnt);
          __m512 x = _mm512_loadu_ps(row + v);
          if (rep != 1.0f) {
            const __mmask16 seen = __mmask16(uint32_t(mask[v / 32]) >> (v % 32)) | nz;
            const __mmask16 pos = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GT_OQ);
            // 1.0f / scale for x > 0, scale otherwise; unseen tokens keep scale 1.
            const __m512 f = _mm512_mask_blend_ps(pos, vrep, vinv);
            x = _mm512_mask_mul_ps(x, seen, x, f);
          }
          x = _mm512_sub_ps(x, _mm512_mul_ps(vfreq, _mm512_cvtepi32_ps(cnt)));
          x = _mm512_sub_ps(x, _mm512_maskz_mov_ps(nz, vpres));
          _mm512_storeu_ps(row + v, x);
        }
      }
#endif
      for (; v < hi; ++v) {
        const int32_t count = base[v];
        float x = static_cast<float>(row[v]);
        if (rep != 1.0f) {
          const bool seen = ((mask[v / 32] >> (v % 32)) & 1) || count > 0;
          const float scale = seen ? rep : 1.0f;
          x *= x > 0 ? 1.0f / scale : scale;
        }
        x -= freq * static_cast<float>(count);
        x -= pres * static_cast<float>(count > 0);
        row[v] = static_cast<scalar_t>(x);
      }
     }
    }
  });
}

void mrv2_bincount(std::vector<int64_t> grid,
                   const torch::Tensor& expanded_idx_mapping_ptr,
                   const torch::Tensor& all_token_ids_ptr,
                   int64_t all_token_ids_stride, const torch::Tensor& prompt_len_ptr,
                   const torch::Tensor& prefill_len_ptr,
                   const torch::Tensor& prompt_bin_mask_ptr, int64_t prompt_bin_mask_stride,
                   const torch::Tensor& output_bin_counts_ptr,
                   int64_t output_bin_counts_stride, int64_t BLOCK_SIZE) {
  const Ints idx(expanded_idx_mapping_ptr), tokens(all_token_ids_ptr),
      prompt_lens(prompt_len_ptr), prefill_lens(prefill_len_ptr);
  int32_t* prompt_mask = ptr<int32_t>(prompt_bin_mask_ptr);
  int32_t* counts = ptr<int32_t>(output_bin_counts_ptr);
  for (int64_t t = 0; t < grid[0]; ++t) {
    const int64_t req = idx[t];
    const int64_t prompt_len = prompt_lens[req], prefill_len = prefill_lens[req];
    const int64_t row = req * all_token_ids_stride;
    for (int64_t i = 0; i < std::min(prompt_len, prefill_len); ++i) {
      const int64_t token = tokens[row + i];
      prompt_mask[req * prompt_bin_mask_stride + token / 32] |=
          int32_t(uint32_t(1) << (token % 32));
    }
    for (int64_t i = prompt_len; i < prefill_len; ++i)
      ++counts[req * output_bin_counts_stride + tokens[row + i]];
  }
}

void mrv2_logit_bias(
    std::vector<int64_t> grid, const torch::Tensor& logits_ptr, int64_t logits_stride,
    int64_t vocab_size, const torch::Tensor& expanded_idx_mapping_ptr,
    const torch::Tensor& num_allowed_token_ids_ptr,
    const torch::Tensor& allowed_token_ids_ptr, int64_t allowed_token_ids_stride,
    const torch::Tensor& num_logit_bias_ptr, const torch::Tensor& bias_token_ids_ptr,
    int64_t bias_token_ids_stride, const torch::Tensor& bias_ptr,
    int64_t bias_stride, const torch::Tensor& pos_ptr,
    const torch::Tensor& min_lens_ptr, const torch::Tensor& num_stop_token_ids_ptr,
    const torch::Tensor& restore_when_all_masked_ptr,
    const torch::Tensor& stop_token_ids_ptr, int64_t stop_token_ids_stride,
    int64_t BLOCK_SIZE, int64_t LOGITS_BLOCK_SIZE, bool CHECK_ALL_MASKED_ROWS) {
  const Ints idx(expanded_idx_mapping_ptr), num_allowed_t(num_allowed_token_ids_ptr),
      allowed(allowed_token_ids_ptr), num_bias_t(num_logit_bias_ptr),
      bias_ids(bias_token_ids_ptr), positions(pos_ptr), min_lens(min_lens_ptr),
      num_stop_t(num_stop_token_ids_ptr), restore(restore_when_all_masked_ptr),
      stop_ids(stop_token_ids_ptr);
  const float* bias = ptr<float>(bias_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
#pragma omp parallel for schedule(static) if (grid[0] > 1)
    for (int64_t t = 0; t < grid[0]; ++t) {
      const int64_t req = idx[t];
      scalar_t* row = logits + t * logits_stride;
      std::vector<float> saved;

      const int64_t num_allowed = std::min(num_allowed_t[req], BLOCK_SIZE);
      if (num_allowed > 0) {
        const int64_t base = req * allowed_token_ids_stride;
        saved.resize(num_allowed);
        for (int64_t j = 0; j < num_allowed; ++j)
          saved[j] = static_cast<float>(row[allowed[base + j]]);
        for (int64_t v = 0; v < vocab_size; ++v)
          row[v] = static_cast<scalar_t>(kNegInf);
        for (int64_t j = 0; j < num_allowed; ++j)
          row[allowed[base + j]] = static_cast<scalar_t>(saved[j]);
      }

      const int64_t num_bias = std::min(num_bias_t[req], BLOCK_SIZE);
      if (num_bias > 0) {
        saved.resize(num_bias);
        for (int64_t j = 0; j < num_bias; ++j)
          saved[j] = static_cast<float>(row[bias_ids[req * bias_token_ids_stride + j]]) +
                     bias[req * bias_stride + j];
        for (int64_t j = 0; j < num_bias; ++j)
          row[bias_ids[req * bias_token_ids_stride + j]] =
              static_cast<scalar_t>(saved[j]);
      }

      const int64_t num_stop = std::min(num_stop_t[req], BLOCK_SIZE);
      if (num_stop <= 0 || positions[t] + 1 >= min_lens[req]) continue;
      const int64_t base = req * stop_token_ids_stride;
      if (CHECK_ALL_MASKED_ROWS && restore[req]) {
        saved.resize(num_stop);
        for (int64_t j = 0; j < num_stop; ++j)
          saved[j] = static_cast<float>(row[stop_ids[base + j]]);
        for (int64_t j = 0; j < num_stop; ++j)
          row[stop_ids[base + j]] = static_cast<scalar_t>(kNegInf);
        float row_max = kNegInf;
        for (int64_t v = 0; v < vocab_size; ++v)
          row_max = std::max(row_max, static_cast<float>(row[v]));
        if (row_max == kNegInf)
          for (int64_t j = 0; j < num_stop; ++j)
            if (std::isfinite(saved[j]))
              row[stop_ids[base + j]] = static_cast<scalar_t>(saved[j]);
      } else {
        for (int64_t j = 0; j < num_stop; ++j)
          row[stop_ids[base + j]] = static_cast<scalar_t>(kNegInf);
      }
    }
  });
}

void mrv2_bad_words(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                    int64_t logits_stride,
                    const torch::Tensor& expanded_idx_mapping_ptr,
                    const torch::Tensor& bad_word_token_ids_ptr,
                    int64_t bad_word_token_ids_stride,
                    const torch::Tensor& bad_word_offsets_ptr,
                    int64_t bad_word_offsets_stride,
                    const torch::Tensor& num_bad_words_ptr,
                    const torch::Tensor& all_token_ids_ptr,
                    int64_t all_token_ids_stride, const torch::Tensor& prompt_len_ptr,
                    const torch::Tensor& total_len_ptr,
                    const torch::Tensor& input_ids_ptr,
                    const torch::Tensor& expanded_local_pos_ptr) {
  const Ints idx(expanded_idx_mapping_ptr), bw_tokens(bad_word_token_ids_ptr),
      bw_offsets(bad_word_offsets_ptr), num_bw_t(num_bad_words_ptr),
      tokens(all_token_ids_ptr), prompt_lens(prompt_len_ptr),
      total_lens(total_len_ptr), input_ids(input_ids_ptr),
      local_pos(expanded_local_pos_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
    for (int64_t t = 0; t < grid[0]; ++t) {
      const int64_t req = idx[t];
      const int64_t num_bw = std::min(num_bw_t[req], grid[1]);
      const int64_t pos = local_pos[t];
      const int64_t first = t - pos;
      const int64_t prompt_len = prompt_lens[req];
      const int64_t output_len = total_lens[req] - prompt_len;
      const int64_t effective_len = output_len + pos;
      const int64_t offsets = req * bad_word_offsets_stride;
      const int64_t bw_base = req * bad_word_token_ids_stride;
      const int64_t out_base = req * all_token_ids_stride + prompt_len;
      for (int64_t bw = 0; bw < num_bw; ++bw) {
        const int64_t start = bw_offsets[offsets + bw];
        const int64_t end = bw_offsets[offsets + bw + 1];
        const int64_t prefix_len = end - start - 1;
        if (prefix_len > effective_len) continue;
        bool match = true;
        for (int64_t i = 0; i < prefix_len && match; ++i) {
          const int64_t actual_pos = effective_len - prefix_len + i;
          const int64_t actual =
              actual_pos >= output_len
                  ? input_ids[first + actual_pos - output_len + 1]
                  : tokens[out_base + actual_pos];
          match = bw_tokens[bw_base + start + i] == actual;
        }
        if (match)
          logits[t * logits_stride + bw_tokens[bw_base + end - 1]] =
              static_cast<scalar_t>(kNegInf);
      }
    }
  });
}

// ------------------------------------------------------------------ logprobs

void mrv2_topk_log_softmax(std::vector<int64_t> grid, const torch::Tensor& output_ptr,
                           const torch::Tensor& logits_ptr, int64_t logits_stride,
                           const torch::Tensor& topk_ids_ptr, int64_t topk_ids_stride,
                           int64_t topk, int64_t vocab_size, int64_t BLOCK_SIZE,
                           int64_t TOPK_BLOCK_SIZE) {
  const Ints ids(topk_ids_ptr);
  float* output = ptr<float>(output_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
#pragma omp parallel for schedule(static) if (grid[0] > 1)
    for (int64_t r = 0; r < grid[0]; ++r) {
      const scalar_t* row = logits + r * logits_stride;
      float max_val = kNegInf;
      double se = 0.0;
      int64_t v = 0;
#if defined(__AVX512F__)
      if constexpr (std::is_same_v<scalar_t, float>) {
        __m512 vmax = _mm512_set1_ps(kNegInf);
        for (; v + 16 <= vocab_size; v += 16) vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(row + v));
        max_val = _mm512_reduce_max_ps(vmax);
        for (int64_t i = v; i < vocab_size; ++i) max_val = std::max(max_val, row[i]);
        // exp in fp32 lanes (SLEEF, as triton-cpu), summed in double as the scalar path.
        __m512d lo = _mm512_setzero_pd(), hi = _mm512_setzero_pd();
        const __m512 m = _mm512_set1_ps(max_val);
        for (v = 0; v + 16 <= vocab_size; v += 16) {
          const __m512 e = Sleef_expf16_u10(_mm512_sub_ps(_mm512_loadu_ps(row + v), m));
          lo = _mm512_add_pd(lo, _mm512_cvtps_pd(_mm512_castps512_ps256(e)));
          hi = _mm512_add_pd(hi, _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(e), 1))));
        }
        se = _mm512_reduce_add_pd(_mm512_add_pd(lo, hi));
      } else
#endif
      {
        for (int64_t i = 0; i < vocab_size; ++i)
          max_val = std::max(max_val, static_cast<float>(row[i]));
        v = 0;
      }
      for (; v < vocab_size; ++v) se += std::exp(static_cast<float>(row[v]) - max_val);
      const float lse = std::log(static_cast<float>(se));
      for (int64_t k = 0; k < topk; ++k)
        output[r * topk + k] =
            static_cast<float>(row[ids[r * topk_ids_stride + k]]) - max_val - lse;
    }
  });
}

void mrv2_ranks(std::vector<int64_t> grid, const torch::Tensor& output_ptr,
                const torch::Tensor& logits_ptr, int64_t logits_stride,
                const torch::Tensor& token_ids_ptr, int64_t vocab_size,
                int64_t BLOCK_SIZE) {
  const Ints token_ids(token_ids_ptr), output(output_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
#pragma omp parallel for schedule(static) if (grid[0] > 1)
    for (int64_t r = 0; r < grid[0]; ++r) {
      const scalar_t* row = logits + r * logits_stride;
      const float x = static_cast<float>(row[token_ids[r]]);
      int64_t n = 0;
      int64_t v = 0;
#if defined(__AVX512F__)
      if constexpr (std::is_same_v<scalar_t, float>) {
        const __m512 vx = _mm512_set1_ps(x);
        for (; v + 16 <= vocab_size; v += 16)
          n += __builtin_popcount(_mm512_cmp_ps_mask(_mm512_loadu_ps(row + v), vx, _CMP_GE_OQ));
      }
#endif
      for (; v < vocab_size; ++v) n += static_cast<float>(row[v]) >= x;
      output.set(r, n);
    }
  });
}

void mrv2_fill_logprob_token_ids(
    std::vector<int64_t> grid, const torch::Tensor& out_token_ids_ptr,
    int64_t out_token_ids_stride, const torch::Tensor& out_valid_mask_ptr,
    int64_t out_valid_mask_stride, const torch::Tensor& sampled_token_ids_ptr,
    const torch::Tensor& topk_indices_ptr, int64_t topk_indices_stride,
    const torch::Tensor& expanded_idx_mapping_ptr,
    const torch::Tensor& num_per_req_token_ids_ptr,
    const torch::Tensor& per_req_token_ids_ptr, int64_t per_req_token_ids_stride,
    int64_t NUM_TOPK, int64_t PADDED_COLS) {
  const Ints out(out_token_ids_ptr), sampled(sampled_token_ids_ptr),
      topk(topk_indices_ptr), idx(expanded_idx_mapping_ptr),
      num_custom_t(num_per_req_token_ids_ptr), custom(per_req_token_ids_ptr);
  bool* valid = ptr<bool>(out_valid_mask_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    out.set(b * out_token_ids_stride, sampled[b]);
    valid[b * out_valid_mask_stride] = true;
    const int64_t req = idx[b];
    const int64_t num_custom = num_custom_t[req];
    const bool use_custom = num_custom > 0;
    const int64_t n = std::min(use_custom ? num_custom : NUM_TOPK, PADDED_COLS);
    for (int64_t c = 0; c < n; ++c) {
      out.set(b * out_token_ids_stride + 1 + c,
              use_custom ? custom[req * per_req_token_ids_stride + c]
                         : topk[b * topk_indices_stride + c]);
      valid[b * out_valid_mask_stride + 1 + c] = true;
    }
  }
}

void mrv2_prompt_logprobs_token_ids(
    std::vector<int64_t> grid, const torch::Tensor& prompt_logprobs_token_ids_ptr,
    const torch::Tensor& query_start_loc_ptr, const torch::Tensor& idx_mapping_ptr,
    const torch::Tensor& num_computed_tokens_ptr,
    const torch::Tensor& all_token_ids_ptr, int64_t all_token_ids_stride,
    int64_t BLOCK_SIZE) {
  const Ints out(prompt_logprobs_token_ids_ptr), qsl(query_start_loc_ptr),
      idx(idx_mapping_ptr), computed(num_computed_tokens_ptr),
      tokens(all_token_ids_ptr);
  const int64_t numel = all_token_ids_ptr.numel();
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    const int64_t start = qsl[b], query_len = qsl[b + 1] - start;
    const int64_t first = req * all_token_ids_stride + computed[req] + 1;
    // The last position's target is past the prompt and its logprob is
    // discarded; Triton reads whatever follows, we read nothing past the end.
    for (int64_t i = 0; i < query_len; ++i)
      out.set(start + i, first + i < numel ? tokens[first + i] : 0);
  }
}

// ------------------------------------------------------- other logits kernels

void mrv2_apply_grammar_bitmask(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                                int64_t logits_stride,
                                const torch::Tensor& logits_indices_ptr,
                                const torch::Tensor& cu_num_logits_ptr,
                                const torch::Tensor& bitmask_ptr,
                                int64_t bitmask_stride, int64_t vocab_size,
                                int64_t MASK_STRIDE, int64_t BLOCK_SIZE) {
  const Ints mapping(logits_indices_ptr), cu(cu_num_logits_ptr);
  const int32_t* bitmask = ptr<int32_t>(bitmask_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
#pragma omp parallel for schedule(static) if (grid[0] > 1)
    for (int64_t m = 0; m < grid[0]; ++m) {
      const int64_t req = mapping[m] / MASK_STRIDE;
      const int64_t position = mapping[m] % MASK_STRIDE;
      if (position >= cu[req + 1] - cu[req]) continue;
      scalar_t* row = logits + (cu[req] + position) * logits_stride;
      const int32_t* words = bitmask + m * bitmask_stride;
      for (int64_t v = 0; v < vocab_size; ++v) {
        // Tokens past the bitmask's width (a padded vocab) are disallowed.
        const uint32_t word = v / 32 < bitmask_stride ? uint32_t(words[v / 32]) : 0;
        if (!((word >> (v % 32)) & 1u)) row[v] = static_cast<scalar_t>(kNegInf);
      }
    }
  });
}

void mrv2_num_nans(std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
                   int64_t logits_stride, const torch::Tensor& num_nans_ptr,
                   int64_t vocab_size, int64_t BLOCK_SIZE) {
  const Ints out(num_nans_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
    for (int64_t r = 0; r < grid[0]; ++r) {
      int64_t n = 0;
      for (int64_t v = 0; v < vocab_size; ++v)
        n += std::isnan(static_cast<float>(logits[r * logits_stride + v]));
      out.set(r, n);
    }
  });
}

void mrv2_compact_sampling_mask(
    std::vector<int64_t> grid, const torch::Tensor& logits_ptr,
    int64_t logits_row_stride, int64_t logits_col_stride,
    const torch::Tensor& cu_num_logits_ptr,
    const torch::Tensor& num_sampled_tokens_ptr, const torch::Tensor& token_ids_ptr,
    int64_t token_ids_row_stride, const torch::Tensor& packed_mask_ptr,
    int64_t packed_mask_row_stride, const torch::Tensor& counts_ptr, int64_t vocab_size,
    int64_t max_num_kept, int64_t ROWS_PER_REQUEST, int64_t BLOCK_SIZE) {
  const Ints cu_num_logits(cu_num_logits_ptr), num_sampled(num_sampled_tokens_ptr),
      token_ids(token_ids_ptr), counts(counts_ptr);
  uint8_t* packed = ptr<uint8_t>(packed_mask_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
#pragma omp parallel for schedule(static) if (grid[0] > 1)
    for (int64_t r = 0; r < grid[0]; ++r) {
      const int64_t req = r / ROWS_PER_REQUEST, slot = r % ROWS_PER_REQUEST;
      if (slot >= num_sampled[req]) {
        counts.set(r, 0);
        continue;
      }
      const int64_t source = cu_num_logits[req] + slot;
      uint8_t* bytes = packed + r * packed_mask_row_stride;
      std::memset(bytes, 0, (vocab_size + 7) / 8);
      int64_t count = 0;
      for (int64_t v = 0; v < vocab_size; ++v) {
        if (!std::isfinite(
                static_cast<float>(logits[source * logits_row_stride + v * logits_col_stride])))
          continue;
        if (count < max_num_kept) token_ids.set(r * token_ids_row_stride + count, v);
        ++count;
        bytes[v / 8] |= uint8_t(1u << (v % 8));
      }
      counts.set(r, count);
    }
  });
}

void mrv2_trace_replay(std::vector<int64_t> grid, const torch::Tensor& sampled_ptr,
                       const torch::Tensor& idx_mapping_ptr,
                       const torch::Tensor& trace_token_ids_ptr,
                       int64_t trace_token_ids_stride,
                       const torch::Tensor& trace_len_ptr,
                       const torch::Tensor& total_len_ptr,
                       const torch::Tensor& prompt_len_ptr) {
  const Ints sampled(sampled_ptr), idx(idx_mapping_ptr), trace(trace_token_ids_ptr),
      trace_lens(trace_len_ptr), total_lens(total_len_ptr), prompt_lens(prompt_len_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    if (req < 0 || trace_lens[req] <= 0) continue;
    const int64_t step = total_lens[req] - prompt_lens[req];
    if (step < 0 || step >= trace_lens[req]) continue;
    sampled.set(b, trace[req * trace_token_ids_stride + step]);
  }
}

namespace {

bool marker_at(const Ints& tokens, int64_t row, int64_t i, const Ints& marker,
               int64_t len, int64_t limit) {
  for (int64_t j = 0; j < len; ++j)
    if (i + j >= limit || tokens[row + i + j] != marker[j]) return false;
  return true;
}

}  // namespace

void mrv2_update_committed_marker_cache(
    std::vector<int64_t> grid, const torch::Tensor& req_ids_ptr,
    const torch::Tensor& thinking_token_budget_ptr,
    const torch::Tensor& all_token_ids_ptr, int64_t all_token_ids_stride,
    const torch::Tensor& total_len_ptr, const torch::Tensor& cached_last_start_ptr,
    const torch::Tensor& cached_last_end_ptr, const torch::Tensor& cached_scan_pos_ptr,
    const torch::Tensor& reasoning_start_token_ids_ptr,
    const torch::Tensor& natural_reasoning_end_token_ids_ptr, int64_t START_LEN,
    int64_t NATURAL_END_LEN, int64_t MAX_LEN, int64_t BLOCK) {
  const Ints req_ids(req_ids_ptr), budgets(thinking_token_budget_ptr),
      tokens(all_token_ids_ptr), total_lens(total_len_ptr),
      cached_start(cached_last_start_ptr), cached_end(cached_last_end_ptr),
      cached_scan(cached_scan_pos_ptr), start_ids(reasoning_start_token_ids_ptr),
      end_ids(natural_reasoning_end_token_ids_ptr);
  for (int64_t r = 0; r < grid[0]; ++r) {
    const int64_t req = req_ids[r];
    if (budgets[req] < 0) continue;
    const int64_t total_len = total_lens[req];
    const int64_t row = req * all_token_ids_stride;
    int64_t scan_pos = cached_scan[req];
    int64_t last_start = cached_start[req], last_end = cached_end[req];
    if (scan_pos > total_len) {
      scan_pos = 0;
      last_start = last_end = -1;
    }
    if (scan_pos == 0 && last_start < 0 && last_end < 0) {
      // Walk back a block at a time, stopping at the first with either marker.
      int64_t hi = total_len;
      while (hi > 0 && last_start < 0 && last_end < 0) {
        const int64_t lo = std::max<int64_t>(hi - BLOCK, 0);
        for (int64_t o = hi - 1; o >= lo; --o)
          if (last_start < 0 && marker_at(tokens, row, o, start_ids, START_LEN, total_len))
            last_start = o;
        for (int64_t o = hi - 1; o >= lo; --o)
          if (last_end < 0 &&
              marker_at(tokens, row, o, end_ids, NATURAL_END_LEN, total_len))
            last_end = o;
        hi = lo;
      }
    } else {
      for (int64_t i = scan_pos; i < total_len; ++i) {
        if (marker_at(tokens, row, i, start_ids, START_LEN, total_len)) last_start = i;
        if (marker_at(tokens, row, i, end_ids, NATURAL_END_LEN, total_len)) last_end = i;
      }
    }
    cached_start.set(req, last_start);
    cached_end.set(req, last_end);
    cached_scan.set(req, std::max<int64_t>(total_len - (MAX_LEN - 1), 0));
  }
}

void mrv2_thinking_budget(
    std::vector<int64_t> grid, const torch::Tensor& logits_ptr, int64_t logits_stride,
    const torch::Tensor& expanded_idx_mapping_ptr,
    const torch::Tensor& thinking_token_budget_ptr,
    const torch::Tensor& all_token_ids_ptr, int64_t all_token_ids_stride,
    const torch::Tensor& total_len_ptr, const torch::Tensor& input_ids_ptr,
    const torch::Tensor& expanded_local_pos_ptr,
    const torch::Tensor& cached_last_start_ptr,
    const torch::Tensor& cached_last_end_ptr,
    const torch::Tensor& reasoning_start_token_ids_ptr,
    const torch::Tensor& natural_reasoning_end_token_ids_ptr,
    const torch::Tensor& reasoning_end_token_ids_ptr, int64_t START_LEN,
    int64_t NATURAL_END_LEN, int64_t END_LEN) {
  const Ints idx(expanded_idx_mapping_ptr), budgets(thinking_token_budget_ptr),
      tokens(all_token_ids_ptr), total_lens(total_len_ptr), input_ids(input_ids_ptr),
      local_pos_t(expanded_local_pos_ptr), cached_start(cached_last_start_ptr),
      cached_end(cached_last_end_ptr), start_ids(reasoning_start_token_ids_ptr),
      natural_end_ids(natural_reasoning_end_token_ids_ptr),
      end_ids(reasoning_end_token_ids_ptr);
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    scalar_t* logits = ptr<scalar_t>(logits_ptr);
    for (int64_t t = 0; t < grid[0]; ++t) {
      const int64_t req = idx[t];
      const int64_t budget = budgets[req];
      if (budget < 0) continue;
      const int64_t local_pos = local_pos_t[t];
      const int64_t first = t - local_pos;
      const int64_t total_len = total_lens[req];
      const int64_t effective_len = total_len + local_pos;
      const int64_t row = req * all_token_ids_stride;
      // Committed tokens come from the history, draft-prefix ones from
      // input_ids (whose local position 0 is the last committed token).
      auto token_at = [&](int64_t pos) {
        return pos < total_len ? tokens[row + pos]
                               : input_ids[first + pos - total_len + 1];
      };
      auto matches = [&](int64_t i, const Ints& marker, int64_t len) {
        for (int64_t j = 0; j < len; ++j)
          if (token_at(i + j) != marker[j]) return false;
        return true;
      };
      int64_t last_start = cached_start[req], last_end = cached_end[req];
      for (int64_t i = std::max<int64_t>(total_len - START_LEN + 1, 0);
           i < effective_len - START_LEN + 1; ++i)
        if (matches(i, start_ids, START_LEN)) last_start = i;
      for (int64_t i = std::max<int64_t>(total_len - NATURAL_END_LEN + 1, 0);
           i < effective_len - NATURAL_END_LEN + 1; ++i)
        if (matches(i, natural_end_ids, NATURAL_END_LEN)) last_end = i;
      if (last_start < 0 || last_start <= last_end) continue;
      if (effective_len - (last_start + START_LEN) < budget) continue;

      int64_t end_prefix_len = 0;
      const int64_t max_prefix_len = std::min(END_LEN - 1, effective_len);
      for (int64_t prefix_len = 1; prefix_len <= max_prefix_len; ++prefix_len)
        if (matches(effective_len - prefix_len, end_ids, prefix_len))
          end_prefix_len = prefix_len;
      logits[t * logits_stride + end_ids[end_prefix_len]] = static_cast<scalar_t>(1.0e9f);
    }
  });
}

// ------------------------------------------- model-state and runner kernels

void mrv2_prepare_rope_positions(
    std::vector<int64_t> grid, const torch::Tensor& positions_ptr, int64_t positions_stride,
    const torch::Tensor& prefill_positions_ptr, int64_t prefill_positions_stride0,
    int64_t prefill_positions_stride1, const torch::Tensor& prefill_delta_ptr,
    const torch::Tensor& idx_mapping_ptr, const torch::Tensor& query_start_loc_ptr,
    const torch::Tensor& prefill_lens_ptr, const torch::Tensor& num_computed_tokens_ptr,
    int64_t BLOCK_SIZE, int64_t NUM_DIMS) {
  const Ints positions(positions_ptr), prefill_positions(prefill_positions_ptr),
      deltas(prefill_delta_ptr), idx(idx_mapping_ptr), qsl(query_start_loc_ptr),
      prefill_lens(prefill_lens_ptr), computed(num_computed_tokens_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    const int64_t num_computed = computed[req];
    const bool is_prefill = num_computed < prefill_lens[req];
    const int64_t start = qsl[b], query_len = qsl[b + 1] - start;
    for (int64_t i = 0; i < query_len; ++i) {
      const int64_t orig = num_computed + i;
      for (int64_t j = 0; j < NUM_DIMS; ++j)
        positions.set(j * positions_stride + start + i,
                      is_prefill ? prefill_positions[req * prefill_positions_stride0 +
                                                     j * prefill_positions_stride1 + orig]
                                 : orig + deltas[req]);
    }
  }
}

void mrv2_apply_prompt_embeds(
    std::vector<int64_t> grid, const torch::Tensor& inputs_embeds_ptr,
    int64_t inputs_embeds_stride, const torch::Tensor& embeds_ptrs_ptr,
    const torch::Tensor& mask_ptrs_ptr, const torch::Tensor& embeds_lens_ptr,
    const torch::Tensor& idx_mapping_ptr, const torch::Tensor& query_start_loc_ptr,
    const torch::Tensor& num_computed_tokens_ptr, int64_t hidden_size,
    int64_t TOKEN_BLOCK, int64_t BLOCK_SIZE) {
  const Ints lens(embeds_lens_ptr), idx(idx_mapping_ptr), qsl(query_start_loc_ptr),
      computed(num_computed_tokens_ptr);
  const int64_t es = inputs_embeds_ptr.element_size();
  char* dst = ptr<char>(inputs_embeds_ptr);
  for (int64_t b = 0; b < grid[0]; ++b) {
    const int64_t req = idx[b];
    const int64_t num_computed = computed[req];
    if (num_computed >= lens[req]) continue;
    const int64_t rows = std::min(qsl[b + 1] - qsl[b], lens[req] - num_computed);
    const char* src = addr<const char>(embeds_ptrs_ptr, req);
    const int8_t* mask = addr<const int8_t>(mask_ptrs_ptr, req);
    for (int64_t t = 0; t < rows; ++t) {
      const int64_t src_row = num_computed + t;
      if (mask && mask[src_row] != 0) continue;
      std::memcpy(dst + (qsl[b] + t) * inputs_embeds_stride * es,
                  src + src_row * hidden_size * es, hidden_size * es);
    }
  }
}

void mrv2_scatter_num_accepted(std::vector<int64_t> grid,
                               const torch::Tensor& idx_mapping_ptr,
                               const torch::Tensor& num_sampled_ptr,
                               const torch::Tensor& num_accepted_ptr) {
  const Ints idx(idx_mapping_ptr), sampled(num_sampled_ptr), accepted(num_accepted_ptr);
  for (int64_t r = 0; r < grid[0]; ++r)
    if (idx[r] >= 0) accepted.set(idx[r], std::max<int64_t>(sampled[r], 1));
}

void mrv2_fill_num_accepted(std::vector<int64_t> grid,
                            const torch::Tensor& idx_mapping_ptr,
                            const torch::Tensor& num_accepted_ptr, int64_t num_sampled) {
  const Ints idx(idx_mapping_ptr), accepted(num_accepted_ptr);
  for (int64_t r = 0; r < grid[0]; ++r)
    if (idx[r] >= 0) accepted.set(idx[r], num_sampled);
}


void mrv2_zero_kv_blocks(std::vector<int64_t> grid, const torch::Tensor& seg_addrs_ptr,
                         const torch::Tensor& seg_block_strides_ptr,
                         const torch::Tensor& seg_page_sizes_ptr,
                         const torch::Tensor& block_ids_ptr, int64_t BLOCK_SIZE) {
  const Ints strides(seg_block_strides_ptr), pages(seg_page_sizes_ptr),
      block_ids(block_ids_ptr);
  for (int64_t b = 0; b < grid[0]; ++b)
    for (int64_t s = 0; s < grid[1]; ++s)
      std::memset(addr<int32_t>(seg_addrs_ptr, s) + block_ids[b] * strides[s], 0,
                  pages[s] * sizeof(int32_t));
}

void mrv2_dcp_local_seq_lens(std::vector<int64_t> grid, const torch::Tensor& out_ptr,
                             const torch::Tensor& seq_lens_ptr, int64_t dcp_size,
                             int64_t dcp_rank, int64_t cp_interleave, int64_t num_reqs,
                             int64_t max_num_reqs, int64_t BLOCK_SIZE) {
  const Ints out(out_ptr), seq_lens(seq_lens_ptr);
  const int64_t round = dcp_size * cp_interleave;
  for (int64_t r = 0; r < max_num_reqs; ++r) {
    if (r >= num_reqs) {
      out.set(r, 0);
      continue;
    }
    const int64_t remainder = std::min(
        std::max<int64_t>(seq_lens[r] % round - dcp_rank * cp_interleave, 0),
        cp_interleave);
    out.set(r, seq_lens[r] / round * cp_interleave + remainder);
  }
}

// ------------------------------------------------------- watermark (Philox)
// Mirrors sample/watermark.py, so a detector keyed on the same PRF scores our
// output exactly as it would a GPU's.

namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u, kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u, kPhiloxW1 = 0xBB67AE85u;
constexpr uint32_t kContextDomain = 0x574D4358u, kTokenDomain = 0x574D544Bu;

void philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
  for (int round = 0; round < 10; ++round) {
    const uint64_t p0 = uint64_t(kPhiloxM0) * c[0];
    const uint64_t p1 = uint64_t(kPhiloxM1) * c[2];
    const uint32_t n0 = uint32_t(p1 >> 32) ^ c[1] ^ k0;
    const uint32_t n2 = uint32_t(p0 >> 32) ^ c[3] ^ k1;
    c[0] = n0;
    c[1] = uint32_t(p1);
    c[2] = n2;
    c[3] = uint32_t(p0);
    if (round != 9) {
      k0 += kPhiloxW0;
      k1 += kPhiloxW1;
    }
  }
}

inline float philox_uniform(uint32_t value) {
  const float scaled = float((value >> 8) + 1) * 0x1.0p-24f;
  uint32_t bits;
  std::memcpy(&bits, &scaled, sizeof(bits));
  --bits;
  float u;
  std::memcpy(&u, &bits, sizeof(u));
  return u;
}

}  // namespace

void mrv2_philox_gumbel(
    std::vector<int64_t> grid, const torch::Tensor& local_argmax_ptr,
    int64_t local_argmax_stride, const torch::Tensor& local_max_ptr,
    int64_t local_max_stride, const torch::Tensor& logits_ptr, int64_t logits_stride,
    const torch::Tensor& contexts_ptr, int64_t context_stride,
    const std::optional<torch::Tensor>& skip_mask_ptr,
    const std::optional<torch::Tensor>& expanded_idx_mapping_ptr,
    const std::optional<torch::Tensor>& seeds_ptr,
    const std::optional<torch::Tensor>& pos_ptr,
    const std::optional<torch::Tensor>& temp_ptr,
    const std::optional<torch::Tensor>& logits_cache_ptr, int64_t logits_cache_stride_0,
    int64_t logits_cache_stride_1,
    const std::optional<torch::Tensor>& logits_cache_col_ptr,
    const std::optional<torch::Tensor>& logits_cache_source_ptr,
    int64_t logits_cache_source_stride, int64_t key_0_value,
    int64_t key_1_value, int64_t vocab_size, int64_t CONTEXT_WIDTH,
    int64_t BLOCK_SIZE, bool USE_FP64, bool IS_DRAFTING, bool PER_TOKEN_COL,
    bool APPLY_TEMPERATURE) {
  const Ints contexts(contexts_ptr);
  const bool cached = logits_cache_ptr.has_value();
  std::optional<Ints> cols;
  if (cached) cols.emplace(*logits_cache_col_ptr);
  auto col_of = [&](int64_t r) { return PER_TOKEN_COL ? (*cols)[r] : (*cols)[0]; };
  int64_t* local_argmax = ptr<int64_t>(local_argmax_ptr);
  const int64_t num_blocks = grid[1];
  // Rows in skip_mask take ordinary Gumbel sampling from the request's state.
  const bool* skip = skip_mask_ptr ? ptr<bool>(*skip_mask_ptr) : nullptr;
  std::optional<Ints> idx, seeds, positions;
  const float* temps = nullptr;
  if (skip) {
    seeds.emplace(*seeds_ptr);
    positions.emplace(*pos_ptr);
  }
  if (expanded_idx_mapping_ptr) idx.emplace(*expanded_idx_mapping_ptr);
  if (temp_ptr) temps = ptr<float>(*temp_ptr);
  auto req_of = [&](int64_t r) { return idx ? (*idx)[r] : r; };
  if (cached) {
    TORCH_CHECK(logits_cache_source_ptr->scalar_type() == logits_cache_ptr->scalar_type(),
                "logits cache source and destination dtypes differ");
    const int64_t elem = logits_cache_ptr->element_size();
    char* cache = static_cast<char*>(ptr<void>(*logits_cache_ptr));
    const char* source = static_cast<const char*>(ptr<void>(*logits_cache_source_ptr));
    for (int64_t r = 0; r < grid[0]; ++r) {
      const int64_t req = req_of(r);
      if (req < 0) continue;
      std::memcpy(
          cache + (req * logits_cache_stride_0 + col_of(r) * logits_cache_stride_1) * elem,
          source + r * logits_cache_source_stride * elem, vocab_size * elem);
    }
  }
  dispatch_logits(logits_ptr, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* logits = ptr<scalar_t>(logits_ptr);
    auto run = [&](auto* local_max) {
      using acc_t = std::remove_pointer_t<decltype(local_max)>;
#pragma omp parallel for schedule(dynamic) if (grid[0] > 1)
      for (int64_t r = 0; r < grid[0]; ++r) {
        const scalar_t* row = logits + r * logits_stride;
        const int64_t req = req_of(r);
        // APPLY_TEMPERATURE divides by the row's temperature (0 read as 1) first.
        float row_temp = 1.0f;
        if (APPLY_TEMPERATURE) {
          row_temp = req >= 0 ? temps[req] : 0.0f;
          if (row_temp == 0.0f) row_temp = 1.0f;
        }
        int64_t token = 0;
        acc_t best = -std::numeric_limits<acc_t>::infinity();
        if (skip && skip[r]) {
          const float temp = req >= 0 ? temps[req] : 0.0f;
          int64_t seed = req >= 0 ? (*seeds)[req] : 0;
          if (cached) seed ^= static_cast<int64_t>(static_cast<uint64_t>(col_of(r)) << 32);
          const int64_t pos = (*positions)[r] + (IS_DRAFTING ? kDraftNoiseSalt : 0);
          if (APPLY_TEMPERATURE) {
            // gumbel_row's own divide skips temp 0, but this one divides by row_temp.
            std::vector<float> scaled(vocab_size);
            for (int64_t v = 0; v < vocab_size; ++v)
              scaled[v] = static_cast<float>(row[v]) / row_temp;
            std::tie(token, best) = gumbel_row<float, acc_t>(scaled.data(), 0, vocab_size,
                                                             temp, false, seed, pos);
          } else {
            std::tie(token, best) =
                gumbel_row<scalar_t, acc_t>(row, 0, vocab_size, temp, false, seed, pos);
          }
        } else {
          const uint32_t k0 = uint32_t(key_0_value), k1 = uint32_t(key_1_value);
          uint32_t state[4] = {kContextDomain, uint32_t(CONTEXT_WIDTH), 0, 0};
          for (int64_t off = 0; off < CONTEXT_WIDTH; off += 4) {
            for (int64_t i = 0; i < 4; ++i)
              state[i] ^= off + i < CONTEXT_WIDTH
                              ? uint32_t(contexts[r * context_stride + off + i])
                              : 0xFFFFFFFFu - uint32_t(i);
            philox4x32_10(state, k0 ^ uint32_t(off), k1);
          }
          float best32 = kNegInf;
          for (int64_t g = 0; g * 4 < vocab_size; ++g) {
            uint32_t words[4] = {uint32_t(g), state[0], state[1], state[2]};
            philox4x32_10(words, k0 ^ state[3], k1 ^ kTokenDomain);
            for (int64_t i = 0; i < 4 && g * 4 + i < vocab_size; ++i) {
              float x = static_cast<float>(row[g * 4 + i]);
              if (APPLY_TEMPERATURE) x = x / row_temp;
              if (std::isnan(x)) x = kNegInf;
              const float value = x - std::log(-std::log(philox_uniform(words[i])));
              if (value > best32) {
                best32 = value;
                token = g * 4 + i;
              }
            }
          }
          best = static_cast<acc_t>(best32);
        }
        local_argmax[r * local_argmax_stride] = token;
        local_max[r * local_max_stride] = best;
        for (int64_t blk = 1; blk < num_blocks; ++blk) {
          local_argmax[r * local_argmax_stride + blk] = token;
          local_max[r * local_max_stride + blk] = -std::numeric_limits<acc_t>::infinity();
        }
      }
    };
    if (USE_FP64)
      run(ptr<double>(local_max_ptr));
    else
      run(ptr<float>(local_max_ptr));
  });
}

// Finishes draft step `step` (SAMPLE) and builds the next step's skip mask
// (PREPARE_NEXT), with _repeated_context_mask_kernel's DRAFT_CONTEXTS body
// inlined for DEDUPLICATE. Rows are independent, as in the Triton grid.
void mrv2_draft_step(
    std::vector<int64_t> grid, const torch::Tensor& sampled_ptr, int64_t sampled_stride,
    int64_t step, const torch::Tensor& local_argmax_ptr, const torch::Tensor& local_max_ptr,
    int64_t num_blocks, const torch::Tensor& contexts_ptr, int64_t context_stride,
    const torch::Tensor& skip_mask_ptr, const torch::Tensor& next_req_indices_ptr,
    const torch::Tensor& next_steps_ptr, const torch::Tensor& temp_ptr,
    const torch::Tensor& enabled_ptr, const std::optional<torch::Tensor>& all_token_ids_ptr,
    int64_t all_token_ids_stride, const torch::Tensor& prompt_lens_ptr,
    const std::optional<torch::Tensor>& total_lens_ptr,
    const torch::Tensor& prior_contexts_ptr, int64_t prior_contexts_stride_0,
    int64_t prior_contexts_stride_1, int64_t CONTEXT_WIDTH, int64_t NUM_SPECULATIVE_STEPS,
    int64_t MAX_HISTORY, bool INCLUDE_PROMPT, bool DEDUPLICATE, bool SAMPLE,
    bool PREPARE_NEXT, int64_t BLOCKS_POW2) {
  const Ints sampled(sampled_ptr), local_argmax(local_argmax_ptr), contexts(contexts_ptr),
      next_req(next_req_indices_ptr), next_steps(next_steps_ptr),
      prompt_lens(prompt_lens_ptr), prior(prior_contexts_ptr);
  bool* skip = ptr<bool>(skip_mask_ptr);
  const bool* enabled = ptr<bool>(enabled_ptr);
  const float* temps = ptr<float>(temp_ptr);
  std::optional<Ints> tokens, total_lens;
  if (DEDUPLICATE) {
    tokens.emplace(*all_token_ids_ptr);
    total_lens.emplace(*total_lens_ptr);
  }
  auto block_max = [&](int64_t row) -> int64_t {
    // local_max.argmax(dim=-1): the first NaN, else the first max.
    auto pick = [&](auto* values) -> int64_t {
      using acc_t = std::remove_const_t<std::remove_pointer_t<decltype(values)>>;
      acc_t best = -std::numeric_limits<acc_t>::infinity();
      for (int64_t b = 0; b < num_blocks; ++b) {
        const acc_t v = values[row * num_blocks + b];
        if (std::isnan(v)) return b;
        if (v > best) best = v;
      }
      for (int64_t b = 0; b < num_blocks; ++b)
        if (values[row * num_blocks + b] == best) return b;
      return 0;
    };
    return local_max_ptr.scalar_type() == at::kDouble ? pick(ptr<double>(local_max_ptr))
                                                      : pick(ptr<float>(local_max_ptr));
  };
  for (int64_t row = 0; row < grid[0]; ++row) {
    const int64_t ctx = row * context_stride;
    if (SAMPLE) {
      const int64_t token = local_argmax[row * num_blocks + block_max(row)];
      sampled.set(row * sampled_stride + step, token);
      for (int64_t i = 0; i + 1 < CONTEXT_WIDTH; ++i) contexts.set(ctx + i, contexts[ctx + i + 1]);
      contexts.set(ctx + CONTEXT_WIDTH - 1, token);
    }
    if (!PREPARE_NEXT) continue;
    const int64_t req = next_req[row];
    const float temperature = req >= 0 ? temps[req] : 0.0f;
    bool active = enabled[row] && temperature != 0.0f;
    if (DEDUPLICATE) {
      if (req < 0) {
        active = false;
      } else {
        const int64_t total_len = (*total_lens)[req];
        const int64_t seq_start = INCLUDE_PROMPT ? 0 : prompt_lens[req];
        const int64_t history_len = total_len - seq_start;
        const int64_t draft_step = next_steps[row];
        int64_t scan_start = 0;
        if (MAX_HISTORY > 0)
          scan_start = std::max<int64_t>(
              history_len - std::max<int64_t>(MAX_HISTORY - draft_step, 0), 0);
        bool repeated = false;
        const int64_t base = req * all_token_ids_stride + seq_start;
        for (int64_t p = scan_start; p < history_len && !repeated; ++p) {
          bool match = true;
          for (int64_t o = 0; o < CONTEXT_WIDTH && match; ++o) {
            const int64_t h = p + o - CONTEXT_WIDTH;
            match = (h >= 0 ? (*tokens)[base + h] : -1) == contexts[ctx + o];
          }
          repeated = match;
        }
        bool partial = false;
        for (int64_t o = 0; o < CONTEXT_WIDTH; ++o) partial |= contexts[ctx + o] < 0;
        if (INCLUDE_PROMPT) repeated |= partial;
        const int64_t prow = row * prior_contexts_stride_0;
        for (int64_t ps = 0; ps < NUM_SPECULATIVE_STEPS && !repeated; ++ps) {
          if (ps >= draft_step || (MAX_HISTORY > 0 && ps < draft_step - MAX_HISTORY)) continue;
          bool match = true;
          for (int64_t o = 0; o < CONTEXT_WIDTH && match; ++o)
            match = contexts[ctx + o] == prior[prow + ps * prior_contexts_stride_1 + o];
          repeated = match;
        }
        for (int64_t o = 0; o < CONTEXT_WIDTH; ++o)
          prior.set(prow + draft_step * prior_contexts_stride_1 + o, contexts[ctx + o]);
        active = active && !repeated;
      }
    }
    skip[row] = !active;
  }
}

TORCH_LIBRARY_FRAGMENT(_spyre_C, ops) {
  ops.def("mrv2_draft_step(int[] grid, Tensor sampled_ptr, int sampled_stride, int step, Tensor local_argmax_ptr, Tensor local_max_ptr, int num_blocks, Tensor contexts_ptr, int context_stride, Tensor skip_mask_ptr, Tensor next_req_indices_ptr, Tensor next_steps_ptr, Tensor temp_ptr, Tensor enabled_ptr, Tensor? all_token_ids_ptr, int all_token_ids_stride, Tensor prompt_lens_ptr, Tensor? total_lens_ptr, Tensor prior_contexts_ptr, int prior_contexts_stride_0, int prior_contexts_stride_1, int CONTEXT_WIDTH, int NUM_SPECULATIVE_STEPS, int MAX_HISTORY, bool INCLUDE_PROMPT, bool DEDUPLICATE, bool SAMPLE, bool PREPARE_NEXT, int BLOCKS_POW2) -> ()");
  ops.impl("mrv2_draft_step", torch::kCPU, &mrv2_draft_step);
  ops.def("mrv2_prepare_prefill_inputs(int[] grid, Tensor input_ids_ptr, Tensor next_prefill_tokens_ptr, int next_prefill_tokens_stride, int num_lookahead, Tensor idx_mapping_ptr, Tensor query_start_loc_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor prefill_lens_ptr, Tensor num_computed_tokens_ptr, int BLOCK_SIZE, int LOOKAHEAD_BLOCK) -> ()");
  ops.impl("mrv2_prepare_prefill_inputs", torch::kCPU, &mrv2_prepare_prefill_inputs);
  ops.def("mrv2_prepare_pos_seq_lens(int[] grid, Tensor pos_ptr, Tensor seq_lens_ptr, Tensor idx_mapping_ptr, Tensor query_start_loc_ptr, Tensor num_computed_tokens_ptr, int max_num_reqs, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_prepare_pos_seq_lens", torch::kCPU, &mrv2_prepare_pos_seq_lens);
  ops.def("mrv2_combine_sampled_and_draft_tokens(int[] grid, Tensor input_ids_ptr, Tensor idx_mapping_ptr, Tensor last_sampled_tokens_ptr, Tensor query_start_loc_ptr, Tensor seq_lens_ptr, Tensor prefill_len_ptr, Tensor draft_tokens_ptr, int draft_tokens_stride, Tensor cu_num_logits_ptr, Tensor logits_indices_ptr, int BLOCK_SIZE, int NUM_NEW_SAMPLED_TOKENS) -> ()");
  ops.impl("mrv2_combine_sampled_and_draft_tokens", torch::kCPU, &mrv2_combine_sampled_and_draft_tokens);
  ops.def("mrv2_get_num_sampled_and_rejected(int[] grid, Tensor num_sampled_ptr, Tensor num_rejected_ptr, Tensor seq_lens_ptr, Tensor cu_num_logits_ptr, Tensor idx_mapping_ptr, Tensor prefill_len_ptr) -> ()");
  ops.impl("mrv2_get_num_sampled_and_rejected", torch::kCPU, &mrv2_get_num_sampled_and_rejected);
  ops.def("mrv2_post_update(int[] grid, Tensor idx_mapping_ptr, Tensor num_computed_tokens_ptr, Tensor last_sampled_tokens_ptr, Tensor? output_bin_counts_ptr, int output_bin_counts_stride, Tensor sampled_tokens_ptr, int sampled_tokens_stride, Tensor num_sampled_ptr, Tensor num_rejected_ptr, Tensor? query_start_loc_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor total_len_ptr, Tensor? draft_tokens_ptr, Tensor? draft_tokens_out_ptr, int num_draft_tokens) -> ()");
  ops.impl("mrv2_post_update", torch::kCPU, &mrv2_post_update);
  ops.def("mrv2_post_update_num_computed_tokens(int[] grid, Tensor idx_mapping_ptr, Tensor num_computed_tokens_ptr, Tensor query_start_loc_ptr) -> ()");
  ops.impl("mrv2_post_update_num_computed_tokens", torch::kCPU, &mrv2_post_update_num_computed_tokens);
  ops.def("mrv2_expand_idx_mapping(int[] grid, Tensor idx_mapping_ptr, Tensor expanded_idx_mapping_ptr, Tensor expanded_local_pos_ptr, Tensor cu_num_logits_ptr, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_expand_idx_mapping", torch::kCPU, &mrv2_expand_idx_mapping);
  ops.def("mrv2_gather_block_tables(int[] grid, Tensor batch_idx_to_req_idx, Tensor src_block_table_ptrs, Tensor dst_block_table_ptrs, Tensor block_table_strides, Tensor num_blocks_ptr, int num_blocks_stride, int num_reqs, Tensor num_computed_tokens_ptr, Tensor sliding_window_reach_ptr, Tensor block_sizes_ptr, bool MASK_SLIDING_WINDOW, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_gather_block_tables", torch::kCPU, &mrv2_gather_block_tables);
  ops.def("mrv2_compute_slot_mappings(int[] grid, int max_num_tokens, Tensor idx_mapping, Tensor query_start_loc, Tensor pos, Tensor block_table_ptrs, Tensor block_table_strides, Tensor block_sizes, Tensor slot_mapping_enabled, Tensor dcp_sharded, Tensor slot_mappings_ptr, int slot_mappings_stride, int cp_rank, int CP_SIZE, int CP_INTERLEAVE, int PAD_ID, int TRITON_BLOCK_SIZE) -> ()");
  ops.impl("mrv2_compute_slot_mappings", torch::kCPU, &mrv2_compute_slot_mappings);
  ops.def("mrv2_apply_write(int[] grid, Tensor output_ptr, Tensor output_stride, Tensor write_indices_ptr, Tensor write_starts_ptr, Tensor write_contents_ptr, Tensor write_cu_lens_ptr, Tensor? write_group_ids_ptr, int BLOCK_SIZE, bool MULTI_GROUP) -> ()");
  ops.impl("mrv2_apply_write", torch::kCPU, &mrv2_apply_write);
  ops.def("mrv2_temperature(int[] grid, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor temperature_ptr, int vocab_size, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_temperature", torch::kCPU, &mrv2_temperature);
  ops.def("mrv2_gumbel_sample(int[] grid, Tensor local_argmax_ptr, int local_argmax_stride, Tensor local_max_ptr, int local_max_stride, Tensor? logits_cache_ptr, int logits_cache_stride_0, int logits_cache_stride_1, Tensor? logits_cache_col_ptr, Tensor? logits_cache_source_ptr, int logits_cache_source_stride, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor seeds_ptr, Tensor pos_ptr, Tensor temp_ptr, int vocab_size, int BLOCK_SIZE, bool IS_DRAFTING, bool APPLY_TEMPERATURE, bool USE_FP64, bool PER_TOKEN_COL) -> ()");
  ops.impl("mrv2_gumbel_sample", torch::kCPU, &mrv2_gumbel_sample);
  ops.def("mrv2_min_p(int[] grid, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor min_p_ptr, int vocab_size, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_min_p", torch::kCPU, &mrv2_min_p);
  ops.def("mrv2_penalties(int[] grid, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor token_ids_ptr, Tensor expanded_local_pos_ptr, Tensor repetition_penalty_ptr, Tensor frequency_penalty_ptr, Tensor presence_penalty_ptr, Tensor prompt_bin_mask_ptr, int prompt_bin_mask_stride, Tensor output_bin_counts_ptr, int output_bin_counts_stride, int vocab_size, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_penalties", torch::kCPU, &mrv2_penalties);
  ops.def("mrv2_bincount(int[] grid, Tensor expanded_idx_mapping_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor prompt_len_ptr, Tensor prefill_len_ptr, Tensor prompt_bin_mask_ptr, int prompt_bin_mask_stride, Tensor output_bin_counts_ptr, int output_bin_counts_stride, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_bincount", torch::kCPU, &mrv2_bincount);
  ops.def("mrv2_logit_bias(int[] grid, Tensor logits_ptr, int logits_stride, int vocab_size, Tensor expanded_idx_mapping_ptr, Tensor num_allowed_token_ids_ptr, Tensor allowed_token_ids_ptr, int allowed_token_ids_stride, Tensor num_logit_bias_ptr, Tensor bias_token_ids_ptr, int bias_token_ids_stride, Tensor bias_ptr, int bias_stride, Tensor pos_ptr, Tensor min_lens_ptr, Tensor num_stop_token_ids_ptr, Tensor restore_when_all_masked_ptr, Tensor stop_token_ids_ptr, int stop_token_ids_stride, int BLOCK_SIZE, int LOGITS_BLOCK_SIZE, bool CHECK_ALL_MASKED_ROWS) -> ()");
  ops.impl("mrv2_logit_bias", torch::kCPU, &mrv2_logit_bias);
  ops.def("mrv2_bad_words(int[] grid, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor bad_word_token_ids_ptr, int bad_word_token_ids_stride, Tensor bad_word_offsets_ptr, int bad_word_offsets_stride, Tensor num_bad_words_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor prompt_len_ptr, Tensor total_len_ptr, Tensor input_ids_ptr, Tensor expanded_local_pos_ptr) -> ()");
  ops.impl("mrv2_bad_words", torch::kCPU, &mrv2_bad_words);
  ops.def("mrv2_topk_log_softmax(int[] grid, Tensor output_ptr, Tensor logits_ptr, int logits_stride, Tensor topk_ids_ptr, int topk_ids_stride, int topk, int vocab_size, int BLOCK_SIZE, int TOPK_BLOCK_SIZE) -> ()");
  ops.impl("mrv2_topk_log_softmax", torch::kCPU, &mrv2_topk_log_softmax);
  ops.def("mrv2_ranks(int[] grid, Tensor output_ptr, Tensor logits_ptr, int logits_stride, Tensor token_ids_ptr, int vocab_size, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_ranks", torch::kCPU, &mrv2_ranks);
  ops.def("mrv2_fill_logprob_token_ids(int[] grid, Tensor out_token_ids_ptr, int out_token_ids_stride, Tensor out_valid_mask_ptr, int out_valid_mask_stride, Tensor sampled_token_ids_ptr, Tensor topk_indices_ptr, int topk_indices_stride, Tensor expanded_idx_mapping_ptr, Tensor num_per_req_token_ids_ptr, Tensor per_req_token_ids_ptr, int per_req_token_ids_stride, int NUM_TOPK, int PADDED_COLS) -> ()");
  ops.impl("mrv2_fill_logprob_token_ids", torch::kCPU, &mrv2_fill_logprob_token_ids);
  ops.def("mrv2_prompt_logprobs_token_ids(int[] grid, Tensor prompt_logprobs_token_ids_ptr, Tensor query_start_loc_ptr, Tensor idx_mapping_ptr, Tensor num_computed_tokens_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_prompt_logprobs_token_ids", torch::kCPU, &mrv2_prompt_logprobs_token_ids);
  ops.def("mrv2_apply_grammar_bitmask(int[] grid, Tensor logits_ptr, int logits_stride, Tensor logits_indices_ptr, Tensor cu_num_logits_ptr, Tensor bitmask_ptr, int bitmask_stride, int vocab_size, int MASK_STRIDE, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_apply_grammar_bitmask", torch::kCPU, &mrv2_apply_grammar_bitmask);
  ops.def("mrv2_num_nans(int[] grid, Tensor logits_ptr, int logits_stride, Tensor num_nans_ptr, int vocab_size, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_num_nans", torch::kCPU, &mrv2_num_nans);
  ops.def("mrv2_compact_sampling_mask(int[] grid, Tensor logits_ptr, int logits_row_stride, int logits_col_stride, Tensor cu_num_logits_ptr, Tensor num_sampled_tokens_ptr, Tensor token_ids_ptr, int token_ids_row_stride, Tensor packed_mask_ptr, int packed_mask_row_stride, Tensor counts_ptr, int vocab_size, int max_num_kept, int ROWS_PER_REQUEST, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_compact_sampling_mask", torch::kCPU, &mrv2_compact_sampling_mask);
  ops.def("mrv2_trace_replay(int[] grid, Tensor sampled_ptr, Tensor idx_mapping_ptr, Tensor trace_token_ids_ptr, int trace_token_ids_stride, Tensor trace_len_ptr, Tensor total_len_ptr, Tensor prompt_len_ptr) -> ()");
  ops.impl("mrv2_trace_replay", torch::kCPU, &mrv2_trace_replay);
  ops.def("mrv2_update_committed_marker_cache(int[] grid, Tensor req_ids_ptr, Tensor thinking_token_budget_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor total_len_ptr, Tensor cached_last_start_ptr, Tensor cached_last_end_ptr, Tensor cached_scan_pos_ptr, Tensor reasoning_start_token_ids_ptr, Tensor natural_reasoning_end_token_ids_ptr, int START_LEN, int NATURAL_END_LEN, int MAX_LEN, int BLOCK) -> ()");
  ops.impl("mrv2_update_committed_marker_cache", torch::kCPU, &mrv2_update_committed_marker_cache);
  ops.def("mrv2_thinking_budget(int[] grid, Tensor logits_ptr, int logits_stride, Tensor expanded_idx_mapping_ptr, Tensor thinking_token_budget_ptr, Tensor all_token_ids_ptr, int all_token_ids_stride, Tensor total_len_ptr, Tensor input_ids_ptr, Tensor expanded_local_pos_ptr, Tensor cached_last_start_ptr, Tensor cached_last_end_ptr, Tensor reasoning_start_token_ids_ptr, Tensor natural_reasoning_end_token_ids_ptr, Tensor reasoning_end_token_ids_ptr, int START_LEN, int NATURAL_END_LEN, int END_LEN) -> ()");
  ops.impl("mrv2_thinking_budget", torch::kCPU, &mrv2_thinking_budget);
  ops.def("mrv2_prepare_rope_positions(int[] grid, Tensor positions_ptr, int positions_stride, Tensor prefill_positions_ptr, int prefill_positions_stride0, int prefill_positions_stride1, Tensor prefill_delta_ptr, Tensor idx_mapping_ptr, Tensor query_start_loc_ptr, Tensor prefill_lens_ptr, Tensor num_computed_tokens_ptr, int BLOCK_SIZE, int NUM_DIMS) -> ()");
  ops.impl("mrv2_prepare_rope_positions", torch::kCPU, &mrv2_prepare_rope_positions);
  ops.def("mrv2_apply_prompt_embeds(int[] grid, Tensor inputs_embeds_ptr, int inputs_embeds_stride, Tensor embeds_ptrs_ptr, Tensor mask_ptrs_ptr, Tensor embeds_lens_ptr, Tensor idx_mapping_ptr, Tensor query_start_loc_ptr, Tensor num_computed_tokens_ptr, int hidden_size, int TOKEN_BLOCK, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_apply_prompt_embeds", torch::kCPU, &mrv2_apply_prompt_embeds);
  ops.def("mrv2_scatter_num_accepted(int[] grid, Tensor idx_mapping_ptr, Tensor num_sampled_ptr, Tensor num_accepted_ptr) -> ()");
  ops.impl("mrv2_scatter_num_accepted", torch::kCPU, &mrv2_scatter_num_accepted);
  ops.def("mrv2_fill_num_accepted(int[] grid, Tensor idx_mapping_ptr, Tensor num_accepted_ptr, int num_sampled) -> ()");
  ops.impl("mrv2_fill_num_accepted", torch::kCPU, &mrv2_fill_num_accepted);
  ops.def("mrv2_zero_kv_blocks(int[] grid, Tensor seg_addrs_ptr, Tensor seg_block_strides_ptr, Tensor seg_page_sizes_ptr, Tensor block_ids_ptr, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_zero_kv_blocks", torch::kCPU, &mrv2_zero_kv_blocks);
  ops.def("mrv2_dcp_local_seq_lens(int[] grid, Tensor out_ptr, Tensor seq_lens_ptr, int dcp_size, int dcp_rank, int cp_interleave, int num_reqs, int max_num_reqs, int BLOCK_SIZE) -> ()");
  ops.impl("mrv2_dcp_local_seq_lens", torch::kCPU, &mrv2_dcp_local_seq_lens);
  ops.def("mrv2_philox_gumbel(int[] grid, Tensor local_argmax_ptr, int local_argmax_stride, Tensor local_max_ptr, int local_max_stride, Tensor logits_ptr, int logits_stride, Tensor contexts_ptr, int context_stride, Tensor? skip_mask_ptr, Tensor? expanded_idx_mapping_ptr, Tensor? seeds_ptr, Tensor? pos_ptr, Tensor? temp_ptr, Tensor? logits_cache_ptr, int logits_cache_stride_0, int logits_cache_stride_1, Tensor? logits_cache_col_ptr, Tensor? logits_cache_source_ptr, int logits_cache_source_stride, int key_0_value, int key_1_value, int vocab_size, int CONTEXT_WIDTH, int BLOCK_SIZE, bool USE_FP64, bool IS_DRAFTING, bool PER_TOKEN_COL, bool APPLY_TEMPERATURE) -> ()");
  ops.impl("mrv2_philox_gumbel", torch::kCPU, &mrv2_philox_gumbel);
}
