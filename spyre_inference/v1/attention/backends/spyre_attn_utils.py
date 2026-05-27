# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for the Spyre attention backends (spyre_attn, spyre_attn_exp)."""

import torch

from spyre_inference.custom_ops.utils import convert


def _maybe_compile(fn):
    """Compile fn unless vLLM's compilation config disables it.

    Mirrors the gating in CustomOp.maybe_compile without requiring CustomOp
    inheritance: returns fn unchanged when compilation mode is NONE or the
    backend is "eager", otherwise wraps it with torch.compile.
    """
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    cfg = get_cached_compilation_config()
    if cfg.mode == CompilationMode.NONE:
        return fn
    if cfg.backend == "eager":
        return fn
    return torch.compile(fn)


def _attn_4d(q, k, v, scale, mask):
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale
    scores = scores + mask
    p = scores.softmax(dim=-1)
    return p @ v


def copy_attn_output(attn_output, num_actual_tokens, output):
    """Place `attn_output[:num_actual_tokens]` into the caller-provided `output` buffer."""
    if output.device.type == "spyre":
        # Workaround for torch-spyre bug: a CPU->Spyre `.copy_()` into a
        # destination whose rank differs from its allocation rank (vLLM
        # allocates `output` 2-D and views it 3-D) crashes in
        # spyre::get_device_stride_infos. Build the full result on CPU,
        # upload as a fresh Spyre tensor, then Spyre->Spyre copy (which
        # dispatches through torch.ops.spyre.copy_from_d2d and works).
        full_cpu = torch.zeros(output.shape, dtype=output.dtype, device="cpu")
        full_cpu[:num_actual_tokens] = attn_output
        full_spyre = convert(full_cpu, "spyre", output.dtype)
        output.copy_(full_spyre)
    else:
        output[:num_actual_tokens].copy_(attn_output)
    return output
