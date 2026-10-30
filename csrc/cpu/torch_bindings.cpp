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

#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds);

torch::Tensor greedy_argmax(const torch::Tensor& logits);

namespace cpu_utils {
void rejection_greedy_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids, const torch::Tensor& target_argmax,
    const torch::Tensor& bonus_token_ids,
    const std::optional<torch::Tensor>& is_greedy, int64_t max_spec_len,
    const std::optional<torch::Tensor>& uniform_probs,
    const std::optional<torch::Tensor>& conditional_rates, bool synthetic_mode);
void rejection_random_sample_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& bonus_token_ids,
    const torch::Tensor& recovered_token_ids,
    const torch::Tensor& uniform_probs,
    const std::optional<torch::Tensor>& is_greedy, int64_t max_spec_len,
    int64_t vocab_size, const std::optional<torch::Tensor>& conditional_rates,
    bool no_draft_probs, bool synthetic_mode);
void expand_kernel_impl(torch::Tensor& output, const torch::Tensor& input,
                        const torch::Tensor& cu_num_tokens,
                        const at::Scalar& replace_from,
                        const at::Scalar& replace_to);
void sample_recovered_tokens_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& inv_q,
    int64_t vocab_size, bool no_draft_probs);
void sample_recovered_tokens_seeded_kernel_impl(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& seeds,
    int64_t vocab_size, bool no_draft_probs);
}  // namespace cpu_utils

// A fixed namespace rather than TORCH_EXTENSION_NAME: every ISA variant
// registers the same ops, and `_spyre_C` cannot collide with a vLLM `_C` build.
TORCH_LIBRARY(_spyre_C, ops) {
  ops.def("fused_gumbel_argmax(Tensor logits, Tensor seeds) -> Tensor");
  ops.impl("fused_gumbel_argmax", torch::kCPU, &fused_gumbel_argmax);

  ops.def("greedy_argmax(Tensor logits) -> Tensor");
  ops.impl("greedy_argmax", torch::kCPU, &greedy_argmax);

  ops.def(
      "rejection_greedy_sample_kernel_impl("
      "Tensor(a!) output_token_ids, Tensor cu_num_draft_tokens, Tensor "
      "draft_token_ids, "
      "Tensor target_argmax, Tensor bonus_token_ids, Tensor? is_greedy, int "
      "max_spec_len, "
      "Tensor? uniform_probs, Tensor? conditional_rates, bool synthetic_mode) "
      "-> ()");
  ops.impl("rejection_greedy_sample_kernel_impl", torch::kCPU,
           &cpu_utils::rejection_greedy_sample_kernel_impl);
  ops.def(
      "rejection_random_sample_kernel_impl("
      "Tensor(a!) output_token_ids, Tensor cu_num_draft_tokens, Tensor "
      "draft_token_ids, "
      "Tensor? draft_probs, Tensor target_probs, Tensor bonus_token_ids, "
      "Tensor recovered_token_ids, Tensor uniform_probs, Tensor? is_greedy, "
      "int max_spec_len, "
      "int vocab_size, Tensor? conditional_rates, bool no_draft_probs, "
      "bool synthetic_mode) -> ()");
  ops.impl("rejection_random_sample_kernel_impl", torch::kCPU,
           &cpu_utils::rejection_random_sample_kernel_impl);
  ops.def(
      "expand_kernel_impl(Tensor(a!) output, Tensor input, Tensor "
      "cu_num_tokens, "
      "Scalar replace_from, Scalar replace_to) -> ()");
  ops.impl("expand_kernel_impl", torch::kCPU, &cpu_utils::expand_kernel_impl);
  ops.def(
      "sample_recovered_tokens_kernel_impl("
      "Tensor(a!) output_token_ids, Tensor cu_num_draft_tokens, Tensor "
      "draft_token_ids, "
      "Tensor? draft_probs, Tensor target_probs, Tensor inv_q, int vocab_size, "
      "bool no_draft_probs) -> ()");
  ops.impl("sample_recovered_tokens_kernel_impl", torch::kCPU,
           &cpu_utils::sample_recovered_tokens_kernel_impl);
  ops.def(
      "sample_recovered_tokens_seeded_kernel_impl("
      "Tensor(a!) output_token_ids, Tensor cu_num_draft_tokens, Tensor "
      "draft_token_ids, "
      "Tensor? draft_probs, Tensor target_probs, Tensor seeds, int vocab_size, "
      "bool no_draft_probs) -> ()");
  ops.impl("sample_recovered_tokens_seeded_kernel_impl", torch::kCPU,
           &cpu_utils::sample_recovered_tokens_seeded_kernel_impl);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
