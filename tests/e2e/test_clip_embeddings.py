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

"""CLIP text and image embeddings on Spyre vs Hugging Face on CPU.

The only on-card check of either CLIP tower: the unit tests cover the patches one at a
time, so nothing else would notice an embedding that is finite but wrong.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_device_count

MODEL = "openai/clip-vit-base-patch32"
# Keep in sync with .github/cache_config/hf_models_and_datasets.yaml.
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MAX_MODEL_LEN = 77

PROMPTS = [
    "a photo of a cat",
    "a photo of a dog sitting on a red couch near the window",
    "a photo of a bowl of fruit",
    "a photo of the city skyline at night",
]

# Match upstream check_embeddings_close(tol=1e-2), as the encoder embedding gates do.
COSINE_MIN = 0.99


def _synthetic_image(size: int = 224, seed: int = 0):
    """Deterministic, in-process, and already at CLIP's input size: 7x7 patches plus
    CLS is 50 rows, the length the vision SDPA padding exists for."""
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
    return image


def _hf_text_embeddings(prompts: list[str]) -> list[torch.Tensor]:
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(MODEL, revision=REVISION).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, revision=REVISION)
    out = []
    with torch.inference_mode():
        # One prompt at a time: no padding, so the EOS pick cannot depend on it.
        for prompt in prompts:
            features = model.get_text_features(**tokenizer(prompt, return_tensors="pt"))
            out.append(F.normalize(features.pooler_output.float(), dim=-1)[0])
    return out


def _hf_image_embedding(image) -> torch.Tensor:
    from transformers import CLIPImageProcessor, CLIPModel

    model = CLIPModel.from_pretrained(MODEL, revision=REVISION).eval()
    processor = CLIPImageProcessor.from_pretrained(MODEL, revision=REVISION)
    with torch.inference_mode():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        features = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(features.pooler_output.float(), dim=-1)[0]


def _llm(enforce_eager: bool, max_num_seqs: int):
    from vllm import LLM

    return LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        runner="pooling",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": 1},
    )


def _assert_close(label: str, emb: list[float], ref: torch.Tensor) -> None:
    assert len(emb) == ref.numel(), f"{label}: dim {len(emb)} vs HF {ref.numel()}"
    assert all(math.isfinite(x) for x in emb), f"{label}: non-finite embedding"
    sim = F.cosine_similarity(torch.tensor(emb, dtype=torch.float32), ref, dim=0).item()
    assert sim >= COSINE_MIN, f"{label}: cosine {sim:.4f} < {COSINE_MIN} vs HF"


# Compiled cases join the encoder gates' compiled cases in the model_quality job.
_EAGER_AND_COMPILED = pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(True, id="eager"),
        pytest.param(False, id="compiled", marks=pytest.mark.model_quality),
    ],
)


def _skip_without_spyre() -> None:
    # Not `spyre_available()`: it allocates on the card, opening /dev/vfio here, and
    # the `LLM` worker subprocess then cannot ("Device or resource busy").
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
@_EAGER_AND_COMPILED
def test_clip_text_embeddings_match_hf_one_request_per_step(enforce_eager: bool) -> None:
    _skip_without_spyre()
    refs = _hf_text_embeddings(PROMPTS)
    llm = _llm(enforce_eager, max_num_seqs=1)
    outputs = llm.embed(PROMPTS)
    assert len(outputs) == len(PROMPTS)
    for prompt, out, ref in zip(PROMPTS, outputs, refs):
        _assert_close(f"text {prompt!r}", out.outputs.embedding, ref)


@pytest.mark.multimodal
@pytest.mark.model_quality
@pytest.mark.uses_subprocess
@pytest.mark.xfail(
    reason="CLIP text embeddings are wrong when 2+ requests share a step "
    "(https://github.com/torch-spyre/spyre-inference/issues/1140)",
    strict=True,
)
def test_clip_text_embeddings_match_hf_batched() -> None:
    """Every prompt in one step. Strict: once #1140 is fixed this passes, and the
    marker has to go."""
    _skip_without_spyre()
    refs = _hf_text_embeddings(PROMPTS)
    llm = _llm(enforce_eager=False, max_num_seqs=len(PROMPTS))
    outputs = llm.embed(PROMPTS)
    assert len(outputs) == len(PROMPTS)
    for prompt, out, ref in zip(PROMPTS, outputs, refs):
        _assert_close(f"batched text {prompt!r}", out.outputs.embedding, ref)


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "enforce_eager",
    [
        pytest.param(
            True,
            id="eager",
            marks=pytest.mark.xfail(
                reason="Eager CLIP vision blocks crash until the tower is compiled "
                "(https://github.com/torch-spyre/spyre-inference/pull/1185)",
                strict=True,
            ),
        ),
        pytest.param(False, id="compiled", marks=pytest.mark.model_quality),
    ],
)
def test_clip_image_embedding_matches_hf(enforce_eager: bool) -> None:
    _skip_without_spyre()
    image = _synthetic_image()
    ref = _hf_image_embedding(image)
    llm = _llm(enforce_eager, max_num_seqs=1)
    # CLIP's image-only prompt: the text must be empty.
    (out,) = llm.embed([{"prompt": "", "multi_modal_data": {"image": image}}])
    _assert_close("image", out.outputs.embedding, ref)
