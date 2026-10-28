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

"""Spyre ``_merge_multimodal_embeddings`` replacement (custom_ops/multimodal_embeddings.py).

Upstream merges image embeddings into the text embedding tensor with a boolean-mask
index_put (``inputs_embeds[is_multimodal] = mm``), whose ``aten::_index_put_impl_`` has
no Spyre kernel. The Spyre replacement scatters on CPU and ``torch.where``-s the result
back onto the device. These tests pin:

  - the monkeypatch onto vLLM's symbol,
  - the CPU-input passthrough (non-Spyre callers keep upstream behaviour verbatim),
  - the empty-embeddings short-circuit,
  - on-card agreement with the upstream CPU merge, including that image positions get
    the image rows and text positions keep the text rows, and
  - the placeholder/token count-mismatch ValueError.

The oracle is the upstream merge itself, run on CPU, so the device path is held to the
exact placement upstream would have produced rather than a re-derivation of it.
"""

import sys

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.models import utils as vllm_utils

from spyre_inference.custom_ops import multimodal_embeddings as me

pytestmark = pytest.mark.multimodal

# Hidden size is a whole number of fp16 sticks (64 elems), so the merge's final
# row-outermost relayout takes its intended path rather than the None fallback.
_HIDDEN = 64


def _skip_without_card():
    if not spyre_available():
        pytest.skip("Spyre device not available")


def test_register_patches_upstream_symbol():
    """``register()`` must point vLLM's ``_merge_multimodal_embeddings`` at ours."""
    me.register()
    assert vllm_utils._merge_multimodal_embeddings is me._spyre_merge_multimodal_embeddings


def test_cpu_input_delegates_to_upstream(monkeypatch):
    """A non-Spyre ``inputs_embeds`` must bypass the workaround and call upstream verbatim.

    The device branch (CPU scatter + ``torch.where``) must never run on a CPU-only caller;
    proven by swapping the captured upstream callable for a sentinel and asserting it,
    not the workaround, produced the result.
    """
    sentinel = object()
    seen = {}

    def _fake_orig(inputs_embeds, multimodal_embeddings, is_multimodal):
        seen["args"] = (inputs_embeds, multimodal_embeddings, is_multimodal)
        return sentinel

    monkeypatch.setattr(me, "_orig_merge_multimodal_embeddings", _fake_orig)

    inputs_embeds = torch.randn(4, _HIDDEN, dtype=torch.float16)
    mm = [torch.randn(2, _HIDDEN, dtype=torch.float16)]
    is_multimodal = torch.tensor([True, False, True, False])

    out = me._spyre_merge_multimodal_embeddings(inputs_embeds, mm, is_multimodal)

    assert out is sentinel
    assert seen["args"] == (inputs_embeds, mm, is_multimodal)


def test_empty_multimodal_embeddings_returns_input_unchanged():
    """On device, no image embeddings must short-circuit to the input tensor itself."""
    _skip_without_card()

    inputs_embeds = torch.randn(4, _HIDDEN, dtype=torch.float16).to("spyre")
    is_multimodal = torch.zeros(4, dtype=torch.bool).to("spyre")

    out = me._spyre_merge_multimodal_embeddings(inputs_embeds, [], is_multimodal)

    assert out is inputs_embeds


@pytest.mark.parametrize("num_tokens,image_positions", [(8, [3, 5]), (6, [0, 1, 2])])
def test_device_merge_matches_upstream_cpu_reference(num_tokens, image_positions):
    """The on-card merge must place image rows exactly where upstream's index_put would."""
    _skip_without_card()

    torch.manual_seed(0)
    num_img = len(image_positions)
    text = torch.randn(num_tokens, _HIDDEN, dtype=torch.float16)
    mm = torch.randn(num_img, _HIDDEN, dtype=torch.float16)
    is_multimodal = torch.zeros(num_tokens, dtype=torch.bool)
    for p in image_positions:
        is_multimodal[p] = True

    # Oracle: upstream's own in-place index_put on CPU copies.
    expected = text.clone()
    expected[is_multimodal] = mm

    actual = me._spyre_merge_multimodal_embeddings(
        text.clone().to("spyre"), [mm.clone().to("spyre")], is_multimodal.to("spyre")
    ).cpu()

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)

    # The two halves, stated directly: image positions carry the image rows, and the
    # text positions are untouched — guards against a merge that agreed on the whole
    # tensor only because the random text and image rows happened to be close.
    torch.testing.assert_close(actual[is_multimodal].float(), mm.float(), atol=1e-3, rtol=1e-3)
    text_mask = ~is_multimodal
    torch.testing.assert_close(
        actual[text_mask].float(), text[text_mask].float(), atol=1e-3, rtol=1e-3
    )


def test_device_count_mismatch_raises_valueerror():
    """More image rows than placeholders must raise the counted ValueError, not a crash.

    The CPU scatter still raises on the mismatch; the workaround must surface it as the
    same diagnostic upstream gives rather than letting the bool->int64 ``.sum()`` detour
    (unlowerable on Spyre) swallow it. A row count *above* the placeholder count is used
    on purpose: a single extra row would broadcast silently, so only an over-count makes
    the index_put raise the shape error the branch is there to translate.
    """
    _skip_without_card()

    text = torch.randn(8, _HIDDEN, dtype=torch.float16)
    mm = torch.randn(3, _HIDDEN, dtype=torch.float16)  # three rows for two placeholders
    is_multimodal = torch.zeros(8, dtype=torch.bool)
    is_multimodal[2] = True
    is_multimodal[5] = True

    with pytest.raises(ValueError) as excinfo:
        me._spyre_merge_multimodal_embeddings(
            text.to("spyre"), [mm.to("spyre")], is_multimodal.to("spyre")
        )

    msg = str(excinfo.value)
    assert "3" in msg and "2" in msg, f"ValueError must report 3 tokens vs 2 placeholders: {msg!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
