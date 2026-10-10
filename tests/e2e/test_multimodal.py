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

"""End-to-end multimodal tests: Pixtral vision + Ministral-3, and the Gemma 4 tower.

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

# Stock transformers vision tower + Gemma-4 MoE decoder, the second vision path here.
# `-it`: the base checkpoint ships no chat template.
GEMMA4_MODEL = "google/gemma-4-26B-A4B-it"
# The CI cache config pins this repo by commit sha, and a sha-pinned `snapshot_download`
# writes no `refs/main` — so the offline runner can only resolve it by that same sha.
# Keep in sync with .github/cache_config/hf_models_and_datasets.yaml.
GEMMA4_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

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


def _generate(
    conversations,
    enforce_eager: bool,
    images_per_prompt: int = 1,
    model: str = MODEL,
    config_format: str = "mistral",
    dtype: str = "float16",
    revision: str | None = None,
):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        # Explicit, not `auto`: auto's mistral probe is a live Hub call, so the
        # offline CI runner silently loads the unpatched HF tower instead.
        config_format=config_format,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        dtype=dtype,
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
    `enforce_eager` the `maybe_compile` kernels fall through to eager — except the
    norms, which pass `force=True` and compile in either mode — and the
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
def test_warmup_covers_text_only_token_embedding_on_multimodal_model(monkeypatch):
    """Serve with ``SPYRE_COMPILE_GUARD=error`` so a late embedding compile is fatal.

    A text-only request on a multimodal model still reaches ``embed_input_ids`` while
    avoiding unrelated first-use compiles in the vision tower.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    (text,) = _generate([_conversation()], enforce_eager=False)

    assert text.strip(), "empty text-only generation from the multimodal model"


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


@pytest.mark.multimodal
@pytest.mark.gemma4_vision
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.uses_subprocess
def test_gemma4_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """The Gemma 4 tower on card, which the unit tests cannot reach: they check the
    rewrite on CPU, this checks that what it rewrites to actually lowers.

    Both modes, as for Pixtral above: compiled is the repo default, and the tower's
    rewrites (permutation-matmul rope, padded SDPA, the pooling matmul) have to lower
    inside a graph, not just run eagerly.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")

    # The 26B weights are prefetched only by the perf job, which never runs on a PR, so
    # the shared cache can legitimately not have them yet.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(GEMMA4_MODEL, revision=GEMMA4_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{GEMMA4_MODEL}@{GEMMA4_REVISION[:7]} not in the local HF cache")

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    uri = _synthetic_image_data_uri()
    (text,) = _generate(
        [_conversation(uri)],
        enforce_eager=enforce_eager,
        model=GEMMA4_MODEL,
        config_format="hf",
        revision=GEMMA4_REVISION,
    )

    assert text.strip(), "empty generation from the Gemma 4 vision path"


CLIP_MODEL = "openai/clip-vit-base-patch32"


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
def test_clip_mixed_image_text_step_matches_each_request_alone():
    """One image and two texts of different lengths in one step embed as they do alone.

    vLLM 0.28 treated a step holding any image as image-only, so its texts skipped the
    text tower (the ``multimodal/clip.py`` workaround). One image only: batched images
    take a different conv path.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")
    import torch
    from huggingface_hub import try_to_load_from_cache
    from PIL import Image
    from vllm import LLM

    if not isinstance(try_to_load_from_cache(CLIP_MODEL, "config.json"), str):
        pytest.skip(f"{CLIP_MODEL} not in the local HF cache")

    image = Image.effect_mandelbrot((224, 224), (-2.0, -1.5, 1.0, 1.5), 64).convert("RGB")
    short_text = "a photo of a cat"
    long_text = "a long description of a busy city street at night with neon signs, rain and cars"
    image_prompt = {"prompt": "", "multi_modal_data": {"image": image}}

    llm = LLM(model=CLIP_MODEL, runner="pooling", max_model_len=77, max_num_seqs=4)
    mixed = [o.outputs.embedding for o in llm.embed([short_text, image_prompt, long_text])]
    alone = [llm.embed([p])[0].outputs.embedding for p in (short_text, image_prompt, long_text)]

    for name, got, ref in zip(("short text", "image", "long text"), mixed, alone):
        got_t, ref_t = torch.tensor(got), torch.tensor(ref)
        cos = torch.nn.functional.cosine_similarity(got_t, ref_t, dim=0).item()
        assert cos > 0.999, f"{name}: mixed-step embedding differs from alone (cosine {cos:.4f})"


@pytest.mark.multimodal
@pytest.mark.uses_subprocess
def test_clip_batched_images_match_hf():
    """Several images in one step each embed to their own HF reference.

    Past one image, the class token was a strided slice the on-card post-norm misread,
    so every image after the first came back wrong.
    """
    if spyre_device_count() == 0:
        pytest.skip("Spyre device not available")
    import numpy as np
    import torch
    from huggingface_hub import try_to_load_from_cache
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from vllm import LLM

    if not isinstance(try_to_load_from_cache(CLIP_MODEL, "config.json"), str):
        pytest.skip(f"{CLIP_MODEL} not in the local HF cache")

    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)).resize((224, 224))
        for _ in range(4)
    ]
    hf = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    pixels = CLIPProcessor.from_pretrained(CLIP_MODEL)(images=images, return_tensors="pt")
    with torch.no_grad():
        ref = hf.get_image_features(**pixels)
    ref = getattr(ref, "pooler_output", ref)

    llm = LLM(model=CLIP_MODEL, runner="pooling", max_model_len=77, max_num_seqs=4)
    outputs = llm.embed([{"prompt": "", "multi_modal_data": {"image": im}} for im in images])

    for i, (out, r) in enumerate(zip(outputs, ref)):
        cos = torch.nn.functional.cosine_similarity(torch.tensor(out.outputs.embedding), r, dim=0)
        assert cos.item() > 0.999, f"image {i}: cosine {cos.item():.4f} vs HF"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
