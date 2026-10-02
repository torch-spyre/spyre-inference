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

#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds);

torch::Tensor greedy_argmax(const torch::Tensor& logits);

// A fixed namespace rather than TORCH_EXTENSION_NAME: every ISA variant
// registers the same ops, and `_spyre_C` cannot collide with a vLLM `_C` build.
TORCH_LIBRARY(_spyre_C, ops) {
  ops.def("fused_gumbel_argmax(Tensor logits, Tensor seeds) -> Tensor");
  ops.impl("fused_gumbel_argmax", torch::kCPU, &fused_gumbel_argmax);

  ops.def("greedy_argmax(Tensor logits) -> Tensor");
  ops.impl("greedy_argmax", torch::kCPU, &greedy_argmax);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
