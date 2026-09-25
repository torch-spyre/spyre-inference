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

"""End-to-end Granite Vision 4.1 tests.

Mirrors the structure of test_multimodal.py (Pixtral/Ministral) for the
Granite Vision path.  A synthetic image carries no meaningful semantics, so
the tests only assert that the full pipeline (SigLIP encoder → BLIP-2
Q-Former projector → Granite decoder) runs and produces non-empty text.

Both `enforce_eager` modes are covered:
- eager: every `compile_when_outermost` kernel falls through to eager;
  the three CPU-offload patches (InterpolateDownsampler,
  _pack_and_unpad_image_features, Blip2QFormerMultiHeadAttention) are the
  only non-trivial code paths exercised.
- compiled: the compiled graph is built once per shape and the patches
  must survive the compilation boundary intact.
"""

import io
import sys

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

# Granite Vision 4.1-4B is the model this branch was developed against.
# There is no smaller stand-in that exercises the same three-module vision
# path (SigLIP + BLIP-2 projector + Granite decoder).
MODEL = "ibm-granite/granite-vision-4.1-4b"

MAX_MODEL_LEN = 4096
MAX_TOKENS = 16


def _synthetic_image_data_uri(size: int = 336, seed: int = 0) -> str:
    """A deterministic RGB image built in-process — no network, no binary asset.

    The default 336×336 matches Granite Vision 4.1's native SigLIP input size
    (patch=14, giving a 24×24 token grid → 144 tokens after the ½-rate
    downsampler) and produces exactly one tile.  Pass size=672 to force
    multi-tile expansion (2–4 tiles) and exercise the multi-patch branch.
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
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        dtype="float16",
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": images_per_prompt},
        trust_remote_code=True,
    )
    outputs = llm.chat(
        conversations,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.granite_vision_e2e
@pytest.mark.multimodal
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.uses_subprocess
def test_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """Smoke: the whole Granite Vision path (SigLIP encoder → InterpolateDownsampler
    → BLIP-2 Q-Former → _pack_and_unpad → Granite decoder) runs and decodes text.

    Both modes are covered: under `enforce_eager` the three CPU-offload patches
    run in eager mode; under the compiled path the patches must survive the
    torch.compile boundary.
    """
    # Not `spyre_available()`: it allocates on the card, opening /dev/vfio here, and
    # the `LLM` worker subprocess then cannot ("Device or resource busy").
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # Graph building for a vision+decoder model exceeds the default timeout.
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate([_conversation(uri)], enforce_eager=enforce_eager)

    assert text.strip(), "empty generation from the Granite Vision path"


@pytest.mark.granite_vision_e2e
@pytest.mark.multimodal
@pytest.mark.uses_subprocess
def test_multi_tile_image_prompt_produces_output(monkeypatch):
    """A 672px image forces multi-tile expansion and exercises the multi-patch
    branch of _pack_and_unpad_image_features.

    The single-patch branch (image_feature.shape[0] == 1) needs only
    `self.image_newline`; the multi-patch branch additionally reads
    `self.config` and `self._downsample_rate` and is the only place the
    5-D permute that the patch exists for is actually reached.  A 336px
    image always produces exactly one tile and takes the single-patch branch.
    Using 672px forces multi-tile expansion (2–4 tiles), so image_feature
    shape[0] > 1 and the patched 5-D permute path is exercised end-to-end.

    Eager only: the patch runs on CPU either way, so a compiled twin
    would cost a full graph build for no additional patch coverage.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # Graph building for a vision+decoder model exceeds the default timeout.
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    # 672px triggers multi-tile expansion; 336px would stay in the single-patch branch.
    uri = _synthetic_image_data_uri(size=672, seed=0)
    (text,) = _generate([_conversation(uri)], enforce_eager=True, images_per_prompt=1)

    assert text.strip(), "empty generation from the multi-tile Granite Vision path"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
