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

// Schemas copied from vLLM's csrc/cpu/torch_bindings.cpp, which binds the whole
// CPU backend; cmake/check_schemas.py fails the build if they drift.

#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds);

torch::Tensor greedy_argmax(const torch::Tensor& logits);

void compute_slot_mapping_kernel_impl(const torch::Tensor query_start_loc,
                                      const torch::Tensor positions,
                                      const torch::Tensor block_table,
                                      torch::Tensor slot_mapping,
                                      const int64_t block_size);

// vLLM's namespace, so vLLM's Python finds the ops. Only one ISA variant is
// ever loaded, and never next to a CPU-built vllm._C.
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("fused_gumbel_argmax(Tensor logits, Tensor seeds) -> Tensor");
  ops.impl("fused_gumbel_argmax", torch::kCPU, &fused_gumbel_argmax);

  ops.def("greedy_argmax(Tensor logits) -> Tensor");
  ops.impl("greedy_argmax", torch::kCPU, &greedy_argmax);

  ops.def(
      "compute_slot_mapping_kernel_impl(Tensor query_start_loc, Tensor "
      "positions, Tensor block_table, Tensor(a3!) slot_mapping, SymInt "
      "block_size) -> ()",
      &compute_slot_mapping_kernel_impl);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
