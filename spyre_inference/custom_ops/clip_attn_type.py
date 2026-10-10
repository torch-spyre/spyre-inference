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

"""Route CLIP's text tower onto Spyre's causal encoder-attention path.

vLLM builds CLIP's text ``Attention`` as plain ``DECODER`` by default, even though a
pooling model never reuses a KV cache -- so Spyre pays for an unused per-sequence KV
path. This builds it ``ENCODER_ONLY`` instead and marks the impl ``causal``, before the
model is constructed (``Attention.__init__`` picks the backend from ``attn_type``).
Only the module-local ``Attention`` name in vLLM's ``clip`` module is patched; the
vision tower uses ``MMEncoderAttention`` and is unaffected.
"""

from __future__ import annotations


def register() -> None:
    import vllm.model_executor.models.clip as clip_mod
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    if getattr(clip_mod.Attention, "_spyre_causal_encoder", False):
        return

    class _SpyreClipTextAttention(Attention):
        _spyre_causal_encoder = True

        def __init__(self, *args, **kwargs) -> None:
            kwargs.setdefault("attn_type", AttentionType.ENCODER_ONLY)
            super().__init__(*args, **kwargs)
            if kwargs["attn_type"] == AttentionType.ENCODER_ONLY:
                self.impl.causal = True

    clip_mod.Attention = _SpyreClipTextAttention  # ty: ignore[invalid-assignment]
