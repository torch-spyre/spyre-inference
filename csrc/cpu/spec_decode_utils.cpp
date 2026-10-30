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
// VENDORED ADAPTATION: vLLM v0.32.0, commit
// 5df63a7203cdb9328d824a27b682e77c7b9928be. Source:
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/csrc/cpu/spec_decode_utils.cpp
// Copyright contributors to the vLLM project, Apache-2.0.
// Kept: the four rejection-sampling kernels. Adaptations preserve the Triton
// interface's mixed integer dtypes, FP64 noise, synthetic acceptance, negative
// placeholders and zero-probability rejection; inputs are checked before
// access. Recovery fuses scoring and argmax, parallelizing token rows without
// score buffers. Re-sync with vLLM using
// docs/contributing/vllm-upgrade-procedure.md.

#include <torch/all.h>

#include "gumbel_noise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace cpu_utils {
namespace {

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// The dispatcher mixes int32 outputs/metadata with int64 IDs. Avoid a copy for
// each input and an output round trip merely to satisfy the upstream int64 ABI.
struct IndexTensor {
  void* data;
  bool wide;

  IndexTensor(const torch::Tensor& tensor, const char* name) {
    check_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32 ||
                    tensor.scalar_type() == torch::kInt64,
                name, " must be int32 or int64");
    data = tensor.data_ptr();
    wide = tensor.scalar_type() == torch::kInt64;
  }

  int64_t operator[](int64_t index) const {
    return wide ? static_cast<const int64_t*>(data)[index]
                : static_cast<const int32_t*>(data)[index];
  }

  void store(int64_t index, int64_t value) const {
    if (wide) {
      static_cast<int64_t*>(data)[index] = value;
    } else {
      static_cast<int32_t*>(data)[index] = static_cast<int32_t>(value);
    }
  }

  int64_t request_for(int64_t token, int64_t batch_size) const {
    if (wide) {
      const auto* begin = static_cast<const int64_t*>(data);
      return std::upper_bound(begin, begin + batch_size, token) - begin;
    }
    const auto* begin = static_cast<const int32_t*>(data);
    return std::upper_bound(begin, begin + batch_size, token) - begin;
  }
};

struct RealTensor {
  const void* data;
  bool wide;

  RealTensor(const torch::Tensor& tensor, const char* name) {
    check_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 ||
                    tensor.scalar_type() == torch::kFloat64,
                name, " must be float32 or float64");
    data = tensor.const_data_ptr();
    wide = tensor.scalar_type() == torch::kFloat64;
  }

  double operator[](int64_t index) const {
    return wide ? static_cast<const double*>(data)[index]
                : static_cast<const float*>(data)[index];
  }
};

void check_spans(const torch::Tensor& cumulative, const IndexTensor& ends,
                 int64_t num_tokens, int64_t max_spec_len) {
  TORCH_CHECK(cumulative.dim() == 1, "cumulative must be 1-D");
  int64_t previous = 0;
  for (int64_t request = 0; request < cumulative.numel(); ++request) {
    const int64_t end = ends[request];
    TORCH_CHECK(end >= previous && end <= num_tokens,
                "cumulative must be nondecreasing and within the token count");
    TORCH_CHECK(end - previous <= max_spec_len,
                "draft length exceeds max_spec_len");
    previous = end;
  }
  TORCH_CHECK(previous == num_tokens, "cumulative must end at the token count");
}

const bool* check_output(const torch::Tensor& output,
                         const torch::Tensor& bonus,
                         const std::optional<torch::Tensor>& is_greedy,
                         int64_t batch_size, int64_t max_spec_len) {
  TORCH_CHECK(max_spec_len >= 0 && output.dim() == 2 &&
                  output.size(0) == batch_size && output.size(1) > max_spec_len,
              "output must have shape [batch, max_spec_len + 1] or wider");
  TORCH_CHECK(bonus.numel() == batch_size,
              "bonus must contain one ID per request");
  if (!is_greedy.has_value()) return nullptr;
  check_tensor(*is_greedy, "is_greedy");
  TORCH_CHECK(is_greedy->scalar_type() == torch::kBool &&
                  is_greedy->dim() == 1 && is_greedy->numel() == batch_size,
              "is_greedy must be bool [batch]");
  return is_greedy->const_data_ptr<bool>();
}

const float* check_probs(const torch::Tensor& tensor, const char* name,
                         int64_t num_tokens, int64_t vocab_size) {
  check_tensor(tensor, name);
  TORCH_CHECK(vocab_size > 0 && tensor.dim() == 2 &&
                  tensor.size(0) == num_tokens &&
                  tensor.size(1) == vocab_size &&
                  tensor.scalar_type() == torch::kFloat32,
              name, " must be float32 [num_tokens, vocab_size]");
  return tensor.const_data_ptr<float>();
}

void check_ids(const IndexTensor& ids, int64_t num_tokens, int64_t vocab_size) {
  for (int64_t i = 0; i < num_tokens; ++i) {
    TORCH_CHECK(ids[i] < vocab_size, "draft ID exceeds vocab_size");
  }
}

}  // namespace

void rejection_greedy_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids, const torch::Tensor& target_argmax,
    const torch::Tensor& bonus_token_ids,
    const std::optional<torch::Tensor>& is_greedy, const int64_t max_spec_len,
    const std::optional<torch::Tensor>& uniform_probs,
    const std::optional<torch::Tensor>& conditional_rates,
    const bool synthetic_mode) {
  const int64_t batch_size = cu_num_draft_tokens.numel();
  const int64_t num_tokens = draft_token_ids.numel();
  const IndexTensor output(output_token_ids, "output"),
      ends(cu_num_draft_tokens, "cumulative"),
      draft(draft_token_ids, "draft IDs"), target(target_argmax, "target IDs"),
      bonus(bonus_token_ids, "bonus IDs");
  TORCH_CHECK(draft_token_ids.dim() == 1 && target_argmax.dim() == 1 &&
                  target_argmax.numel() == num_tokens,
              "draft and target IDs must be 1-D with matching lengths");
  check_spans(cu_num_draft_tokens, ends, num_tokens, max_spec_len);
  const bool* greedy = check_output(output_token_ids, bonus_token_ids,
                                    is_greedy, batch_size, max_spec_len);
  std::optional<RealTensor> uniform, rates;
  if (synthetic_mode) {
    TORCH_CHECK(uniform_probs && conditional_rates,
                "synthetic mode requires uniforms and conditional rates");
    uniform.emplace(*uniform_probs, "uniforms");
    rates.emplace(*conditional_rates, "conditional rates");
    TORCH_CHECK(uniform_probs->numel() == num_tokens &&
                    conditional_rates->numel() >= max_spec_len,
                "synthetic uniforms or rates have incorrect lengths");
  }
  const int64_t stride = output_token_ids.size(1);

#pragma omp parallel for if (batch_size >= 256) schedule(static)
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    if (greedy && !greedy[req_idx]) continue;
    const int64_t start = req_idx == 0 ? 0 : ends[req_idx - 1];
    const int64_t count = ends[req_idx] - start;
    bool rejected = false;
    for (int64_t pos = 0; pos < count; ++pos) {
      const int64_t index = start + pos;
      const bool accepted =
          synthetic_mode
              ? draft[index] >= 0 && (*uniform)[index] < (*rates)[pos]
              : draft[index] == target[index];
      output.store(req_idx * stride + pos,
                   synthetic_mode && accepted ? draft[index] : target[index]);
      if (!accepted) {
        rejected = true;
        break;
      }
    }
    if (!rejected) output.store(req_idx * stride + count, bonus[req_idx]);
  }
}

void rejection_random_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& bonus_token_ids,
    const torch::Tensor& recovered_token_ids,
    const torch::Tensor& uniform_probs,
    const std::optional<torch::Tensor>& is_greedy, const int64_t max_spec_len,
    const int64_t vocab_size,
    const std::optional<torch::Tensor>& conditional_rates,
    const bool no_draft_probs, const bool synthetic_mode) {
  const int64_t batch_size = cu_num_draft_tokens.numel();
  const int64_t num_tokens = draft_token_ids.numel();
  const IndexTensor output(output_token_ids, "output"),
      ends(cu_num_draft_tokens, "cumulative"),
      draft(draft_token_ids, "draft IDs"),
      recovered(recovered_token_ids, "recovered IDs"),
      bonus(bonus_token_ids, "bonus IDs");
  TORCH_CHECK(
      draft_token_ids.dim() == 1 && recovered_token_ids.numel() == num_tokens,
      "draft and recovered IDs must have matching lengths");
  check_spans(cu_num_draft_tokens, ends, num_tokens, max_spec_len);
  check_ids(draft, num_tokens, vocab_size);
  const bool* greedy = check_output(output_token_ids, bonus_token_ids,
                                    is_greedy, batch_size, max_spec_len);
  const RealTensor uniform(uniform_probs, "uniforms");
  TORCH_CHECK(uniform_probs.numel() == num_tokens,
              "uniforms have incorrect length");
  const float* target =
      check_probs(target_probs, "target probabilities", num_tokens, vocab_size);
  const float* proposal = nullptr;
  std::optional<RealTensor> rates;
  if (synthetic_mode) {
    TORCH_CHECK(conditional_rates && conditional_rates->numel() >= max_spec_len,
                "synthetic mode requires a conditional rate per position");
    rates.emplace(*conditional_rates, "conditional rates");
  } else if (!no_draft_probs) {
    TORCH_CHECK(draft_probs, "draft probabilities are required");
    proposal = check_probs(*draft_probs, "draft probabilities", num_tokens,
                           vocab_size);
  }
  const int64_t stride = output_token_ids.size(1);

#pragma omp parallel for if (batch_size >= 256) schedule(static)
  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    if (greedy && greedy[req_idx]) continue;
    const int64_t start = req_idx == 0 ? 0 : ends[req_idx - 1];
    const int64_t count = ends[req_idx] - start;
    bool rejected = false;
    for (int64_t pos = 0; pos < count; ++pos) {
      const int64_t index = start + pos;
      const int64_t draft_id = draft[index];
      bool accepted = false;
      if (draft_id >= 0) {
        if (synthetic_mode) {
          accepted = uniform[index] < (*rates)[pos];
        } else {
          const int64_t offset = index * vocab_size + draft_id;
          const float q = no_draft_probs ? 1.0f : proposal[offset];
          // Divide in FP32, then compare against the original uniform
          // precision.
          accepted = q > 0.0f && target[offset] / q >= uniform[index];
        }
      }
      output.store(req_idx * stride + pos,
                   accepted ? draft_id : recovered[index]);
      if (!accepted) {
        rejected = true;
        break;
      }
    }
    if (!rejected) output.store(req_idx * stride + count, bonus[req_idx]);
  }
}

void expand_kernel_impl(torch::Tensor& output, const torch::Tensor& input,
                        const torch::Tensor& cu_num_tokens,
                        const at::Scalar& replace_from,
                        const at::Scalar& replace_to) {
  check_tensor(output, "output");
  check_tensor(input, "input");
  const IndexTensor ends(cu_num_tokens, "cumulative");
  const int64_t batch_size = cu_num_tokens.numel();
  TORCH_CHECK(
      output.dim() == 1 && input.dim() == 1 && input.numel() == batch_size &&
          output.scalar_type() == input.scalar_type(),
      "expand requires matching output/input dtypes and one input per request");
  check_spans(cu_num_tokens, ends, output.numel(), output.numel());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kInt, at::kLong, input.scalar_type(), "expand", [&] {
        const auto* values = input.const_data_ptr<scalar_t>();
        auto* result = output.data_ptr<scalar_t>();
        const scalar_t from = replace_from.to<scalar_t>();
        const scalar_t to = replace_to.to<scalar_t>();
        _Pragma("omp parallel for if(batch_size >= 256) schedule(static)") for (
            int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
          const int64_t start = req_idx == 0 ? 0 : ends[req_idx - 1];
          const scalar_t value = values[req_idx] == from ? to : values[req_idx];
          for (int64_t i = start; i < ends[req_idx]; ++i) result[i] = value;
        }
      });
}

void sample_recovered_tokens_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& inv_q,
    const int64_t vocab_size, const bool no_draft_probs) {
  const int64_t batch_size = cu_num_draft_tokens.numel();
  const int64_t num_tokens = draft_token_ids.numel();
  const IndexTensor output(output_token_ids, "output"),
      ends(cu_num_draft_tokens, "cumulative"),
      draft(draft_token_ids, "draft IDs");
  TORCH_CHECK(output_token_ids.dim() == 1 && draft_token_ids.dim() == 1 &&
                  output_token_ids.numel() == num_tokens,
              "output and draft IDs must be 1-D with matching lengths");
  check_spans(cu_num_draft_tokens, ends, num_tokens, num_tokens);
  check_ids(draft, num_tokens, vocab_size);
  const float* target =
      check_probs(target_probs, "target probabilities", num_tokens, vocab_size);
  const float* proposal = nullptr;
  if (!no_draft_probs) {
    TORCH_CHECK(draft_probs, "draft probabilities are required");
    proposal = check_probs(*draft_probs, "draft probabilities", num_tokens,
                           vocab_size);
  }
  check_tensor(inv_q, "inverse noise");
  TORCH_CHECK(inv_q.dim() == 2 && inv_q.size(0) == batch_size &&
                  inv_q.size(1) == vocab_size,
              "inverse noise must have shape [batch, vocab_size]");
  AT_DISPATCH_FLOATING_TYPES(inv_q.scalar_type(), "recovery", [&] {
    const auto* noise = inv_q.const_data_ptr<scalar_t>();
    // Flatten positions so even one request can use multiple CPU threads.
    _Pragma("omp parallel for if(num_tokens > 1) schedule(static)") for (
        int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
      const int64_t req_idx = ends.request_for(token_idx, batch_size);
      const int64_t draft_id = draft[token_idx];
      const float* token_target = target + token_idx * vocab_size;
      const float* token_draft =
          no_draft_probs ? nullptr : proposal + token_idx * vocab_size;
      const scalar_t* req_noise = noise + req_idx * vocab_size;
      int64_t best_id = 0;
      scalar_t best_val = -std::numeric_limits<scalar_t>::infinity();
      for (int64_t v = 0; v < vocab_size; ++v) {
        float prob = token_target[v];
        if (no_draft_probs) {
          if (v == draft_id) prob = 0.0f;
        } else {
          prob = std::max(prob - token_draft[v], 0.0f);
        }
        const scalar_t val = static_cast<scalar_t>(prob) * req_noise[v];
        // Match torch.argmax's first-index ties and first NaN, including 0 *
        // inf.
        if (val > best_val || (std::isnan(val) && !std::isnan(best_val))) {
          best_val = val;
          best_id = v;
        }
      }
      output.store(token_idx, best_id);
    }
  });
}

// Not vendored: recovery with the noise drawn in-kernel, as fused_gumbel_argmax
// draws it, instead of from a [batch, vocab] inv_q the caller fills with
// exponential_. Gumbel-max over log(residual) has the distribution of
// upstream's argmax(residual / q), and every draft position of a request reuses
// that request's noise, as upstream's one q row per request does.
void sample_recovered_tokens_seeded_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& seeds,
    const int64_t vocab_size, const bool no_draft_probs) {
  using gumbel_noise::GUMBEL_MAX;
  const int64_t batch_size = cu_num_draft_tokens.numel();
  const int64_t num_tokens = draft_token_ids.numel();
  const IndexTensor output(output_token_ids, "output"),
      ends(cu_num_draft_tokens, "cumulative"),
      draft(draft_token_ids, "draft IDs");
  TORCH_CHECK(output_token_ids.dim() == 1 && draft_token_ids.dim() == 1 &&
                  output_token_ids.numel() == num_tokens,
              "output and draft IDs must be 1-D with matching lengths");
  check_spans(cu_num_draft_tokens, ends, num_tokens, num_tokens);
  check_ids(draft, num_tokens, vocab_size);
  const float* target =
      check_probs(target_probs, "target probabilities", num_tokens, vocab_size);
  const float* proposal = nullptr;
  if (!no_draft_probs) {
    TORCH_CHECK(draft_probs, "draft probabilities are required");
    proposal = check_probs(*draft_probs, "draft probabilities", num_tokens,
                           vocab_size);
  }
  check_tensor(seeds, "seeds");
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64 && seeds.dim() == 1 &&
                  seeds.numel() == batch_size,
              "seeds must be int64 [batch]");
  const int64_t* req_seeds = seeds.const_data_ptr<int64_t>();
  _Pragma("omp parallel for if(num_tokens > 1) schedule(static)") for (
      int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t req_idx = ends.request_for(token_idx, batch_size);
    const int64_t draft_id = draft[token_idx];
    const float* token_target = target + token_idx * vocab_size;
    const float* token_draft =
        no_draft_probs ? nullptr : proposal + token_idx * vocab_size;
    const uint64_t key =
        gumbel_noise::splitmix64(static_cast<uint64_t>(req_seeds[req_idx]));
    // Scores are log(prob) + g, kept in probability space until a token can
    // win: prob <= prune cannot (g < GUMBEL_MAX), and prob * inv_best bounds
    // -log u from below, so most tokens cost no log at all. prune starts at 0,
    // so a zero (or NaN) residual is never drawn; an all-zero row gives 0.
    int64_t best_id = -1;
    double best_score = 0.0, prune = 0.0, inv_best = 0.0;
    for (int64_t v = 0; v < vocab_size; ++v) {
      float prob = token_target[v];
      if (no_draft_probs) {
        if (v == draft_id) prob = 0.0f;
      } else {
        prob = std::max(prob - token_draft[v], 0.0f);
      }
      if (!(prob > prune)) continue;
      const uint64_t bits = gumbel_noise::noise_bits(key, v);
      if (best_id >= 0 &&
          gumbel_noise::one_minus_u_from_bits(bits) >= prob * inv_best)
        continue;
      const double score =
          std::log(double(prob)) + gumbel_noise::gumbel_from_bits(bits);
      if (best_id < 0 || score > best_score) {
        best_id = v;
        best_score = score;
        prune = std::exp(best_score - GUMBEL_MAX);
        inv_best = std::exp(-best_score);
      }
    }
    output.store(token_idx, best_id < 0 ? 0 : best_id);
  }
}

}  // namespace cpu_utils
