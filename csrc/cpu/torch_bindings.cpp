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

void cpu_topp_sampling(torch::Tensor& logits, const torch::Tensor& p);
void cpu_topk_sampling(torch::Tensor& logits, const torch::Tensor& k);
void cpu_topk_topp_sampling(torch::Tensor& logits, const torch::Tensor& k,
                            const torch::Tensor& p);

// A fixed namespace rather than TORCH_EXTENSION_NAME: every ISA variant
// registers the same ops, and `_spyre_C` cannot collide with a vLLM `_C` build.
TORCH_LIBRARY(_spyre_C, ops) {
  // Top-k / top-p logit filters; mask in place to -inf.
  ops.def("cpu_topp_sampling(Tensor(a!) logits, Tensor p) -> ()");
  ops.impl("cpu_topp_sampling", torch::kCPU, &cpu_topp_sampling);
  ops.def("cpu_topk_sampling(Tensor(a!) logits, Tensor k) -> ()");
  ops.impl("cpu_topk_sampling", torch::kCPU, &cpu_topk_sampling);
  ops.def(
      "cpu_topk_topp_sampling(Tensor(a!) logits, Tensor k, Tensor p) -> ()");
  ops.impl("cpu_topk_topp_sampling", torch::kCPU, &cpu_topk_topp_sampling);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
