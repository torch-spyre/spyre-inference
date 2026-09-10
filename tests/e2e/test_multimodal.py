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

"""End-to-end multimodal (Pixtral vision + Ministral-3 decoder) tests.

No upstream VLM generation test runs on Spyre (see the test_pixtral.py entry in
`upstream_tests.yaml`), so the end-to-end guarantee lives here. A synthetic image has
no meaningful reference string, so this only asserts the path runs and decodes text.
"""

import io
import sys

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

# Pixtral vision encoder + multimodal projector + Ministral-3 text decoder.
# The 14B is what this branch was brought up against — no smaller stand-in.
MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"

MAX_MODEL_LEN = 4096
MAX_TOKENS = 16


def _synthetic_image_data_uri(size: int = 176, seed: int = 0) -> str:
    """A deterministic image built in-process — no network, no binary asset.

    Content does not matter; size does. 176x176 gives an 11x11 = 121 patch grid,
    the coprime-with-64 case the vision SDPA and conv patches exist for.
    """
    import base64

    from PIL import Image

    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 7 + seed) % 256,
                (y * 5 + seed) % 256,
                ((x + y) * 3 + seed) % 256,
            )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _conversation(*uris: str):
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image_url", "image_url": {"url": u}} for u in uris),
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]


def _generate(conversations, enforce_eager: bool, images_per_prompt: int = 1):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        # Explicit, not `auto`: auto's mistral probe is a live Hub call, so the
        # offline CI runner silently loads the unpatched HF tower instead.
        config_format="mistral",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        dtype="float16",
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": images_per_prompt},
    )
    outputs = llm.chat(
        conversations,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.multimodal
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.uses_subprocess
def test_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """Smoke: the whole vision path (conv patch embed -> vision rope -> padded
    SDPA -> patch merger -> projector norm -> decoder) runs and decodes text.

    Both modes: `CompileOutermost` samples the mode at construction, so under
    `enforce_eager` every `compile_when_outermost` kernel falls through to eager and the
    compiled path — the repo default — goes uncovered.
    """
    # Not `spyre_available()`: it allocates on the card, opening /dev/vfio here, and
    # the `LLM` worker subprocess then cannot ("Device or resource busy").
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # Building the graphs for a 14B two-tower model outruns the default timeout.
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate([_conversation(uri)], enforce_eager=enforce_eager)

    assert text.strip(), "empty generation from the multimodal path"


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
def test_two_image_prompt_produces_output():
    """Two images make the vision mask non-trivial: a pair of strided sub-block writes
    rather than one full-range write, which is what `patch_block_attention_mask` is for.

    Eager only: the mask is built on CPU either way, so a compiled twin would cost a
    graph build for no new coverage.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    uris = (_synthetic_image_data_uri(seed=0), _synthetic_image_data_uri(seed=97))
    (text,) = _generate([_conversation(*uris)], enforce_eager=True, images_per_prompt=2)

    assert text.strip(), "empty generation from the two-image multimodal path"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
