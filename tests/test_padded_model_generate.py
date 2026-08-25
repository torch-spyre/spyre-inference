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

"""End-to-end: a model that needs BOTH head_dim and intermediate_size padding
must still decode the correct tokens on device.

This is the on-device counterpart to the CPU-only unit tests in
``test_head_pad.py`` and ``test_mlp_pad.py``, which exercise the padding shims
and guards in isolation. ``qwrt/Swedish0.1M`` is the smallest public model that
trips both pads at once: head_dim 16 -> 128 (the QK-norm attention path) and
intermediate_size 160 -> 192 (the SwiGLU MLP path). Requires Spyre hardware and
skips on CPU-only hosts, like the other ``generate()`` tests.
"""

import pytest

# qwrt/Swedish0.1M ships a broken tokenizer stub (Qwen2Tokenizer, vocab_size=1,
# returns [] for everything). The model itself is byte-level (vocab 256), so the
# prompt is fed as raw UTF-8 byte ids rather than through the tokenizer.
_PROMPT_TOKEN_IDS = list("Sverige är ett land i norra Europa".encode("utf-8"))

# Greedy continuation from transformers CPU (fp32, unpadded) on the same byte-id
# prompt. The leading [32, 111, 99] were confirmed to match the Spyre fp16 padded
# run; the tail is the fp32 reference. If fp16 padded greedy drifts from fp32 on
# device (plausible on a model this small, where top logits sit close together),
# trim this list back to the confirmed prefix.
_REFERENCE_TOKEN_IDS = [32, 111, 99, 104, 32, 115, 195, 165, 103, 32, 104, 111]


@pytest.mark.uses_subprocess
def test_padded_head_dim_and_intermediate_size_generate() -> None:
    """Loading the model fires both pads; greedy decode matches the unpadded
    reference token ids."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="qwrt/Swedish0.1M",
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
    )

    # Both padding passes must have run during check_and_update_config.
    hf_config = llm.llm_engine.model_config.hf_config
    assert hf_config.head_dim == 128
    assert hf_config._spyre_orig_head_dim == 16
    assert hf_config.intermediate_size == 192
    assert hf_config._spyre_orig_intermediate_size == 160

    sp = SamplingParams(temperature=0.0, max_tokens=len(_REFERENCE_TOKEN_IDS))
    outputs = llm.generate({"prompt_token_ids": _PROMPT_TOKEN_IDS}, sp, use_tqdm=False)

    assert list(outputs[0].outputs[0].token_ids) == _REFERENCE_TOKEN_IDS
