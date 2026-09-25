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

"""BLIP-2 / Q-Former workarounds for Spyre.

Blip2QFormerMultiHeadAttention.forward contains an explicit
torch.matmul / torch.softmax chain whose permute/matmul/softmax layout
Spyre's restickify and bmm_padding passes cannot reconcile, so the entire
module is run on CPU.

vllm-project/vllm@a1541f5 replaces that chain with a single
F.scaled_dot_product_attention call, which SpyreMMEncoderAttention already
handles natively.  Once we upgrade, the Q-Former attention runs on-card and
this entire workaround becomes dead code.

When that vLLM version is in use, test_vllm_blip2_qformer_uses_sdpa in
tests/probes/test_spyre_fallback_probes.py will flip to XPASS, signalling that
this file and its call site in apply() can be removed.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)


def patch_blip2_qformer_attention() -> None:
    """Run Blip2QFormerMultiHeadAttention.forward on CPU.

    The full forward contains permute/matmul/softmax chains that produce
    non-contiguous layouts Spyre's restickify and bmm_padding passes cannot reconcile.
    Run entirely on CPU. Weights (query/key/value linears) live on Spyre, so we move
    the whole module to CPU for the call and restore it after.
    """
    try:
        from vllm.model_executor.models.blip2 import Blip2QFormerMultiHeadAttention
    except ImportError:
        return

    if getattr(Blip2QFormerMultiHeadAttention.forward, "_spyre_patched", False):
        return

    _orig_blip2_attn_forward = Blip2QFormerMultiHeadAttention.forward

    def _blip2_attn_forward_cpu(self, hidden_states, encoder_hidden_states=None):
        target_device = next(self.parameters()).device
        hidden_states = convert(hidden_states, device="cpu")
        if encoder_hidden_states is not None:
            encoder_hidden_states = convert(encoder_hidden_states, device="cpu")
        self.to("cpu")
        try:
            out = _orig_blip2_attn_forward(self, hidden_states, encoder_hidden_states)
        finally:
            self.to(target_device)
        return convert(out, device=target_device)

    _blip2_attn_forward_cpu._spyre_patched = True  # type: ignore[attr-defined]
    Blip2QFormerMultiHeadAttention.forward = _blip2_attn_forward_cpu  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched Blip2QFormerMultiHeadAttention.forward to run on CPU "
        "(permute/matmul chains not restickifiable on Spyre)."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply BLIP-2 Q-Former workarounds."""
    patch_blip2_qformer_attention()
