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

# Short, so warmup records attention and token buckets only up to it: these prompts carry
# at most two 176x176 images (Pixtral) or one image of at most 280 tokens (Gemma 4).
MAX_MODEL_LEN = 512
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


def _llm(
    enforce_eager: bool,
    max_num_seqs: int = 1,
    images_per_prompt: int = 1,
    model: str = MODEL,
    config_format: str = "mistral",
    dtype: str = "float16",
    revision: str | None = None,
    max_model_len: int = MAX_MODEL_LEN,
    encoder_budgets: tuple[int, ...] = (),
):
    from vllm import LLM

    return LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        # Explicit, not `auto`: auto's mistral probe is a live Hub call, so the
        # offline CI runner silently loads the unpatched HF tower instead.
        config_format=config_format,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        # One decoding sequence, and a prefill chunk of the whole prompt up to the platform's 512.
        compilation_config={
            "compile_sizes": [1, min(max_model_len, 512)],
            "encoder_cudagraph_token_budgets": list(encoder_budgets),
        },
        dtype=dtype,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": images_per_prompt},
    )


def _generate(conversations, enforce_eager: bool, **llm_kwargs):
    from vllm import SamplingParams

    llm = _llm(enforce_eager, max_num_seqs=len(conversations), **llm_kwargs)
    outputs = llm.chat(
        conversations,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.multimodal
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.uses_subprocess
@pytest.mark.timeout(3600)
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
@pytest.mark.timeout(3600)
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
    """Two images make the vision mask non-trivial: the `cu_seqlens`-derived
    block-diagonal mask must actually block cross-image attention, not collapse to a
    single full-range block.

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
@pytest.mark.timeout(3600)
def test_gemma4_single_image_prompt_produces_output(enforce_eager, monkeypatch):
    """The Gemma 4 tower on card, which the unit tests cannot reach: they check the
    rewrite on CPU, this checks that what it rewrites to actually lowers.

    Both modes, as for Pixtral above: compiled is the repo default, where the tower's
    rewrites (gathered position embeddings and rope, permutation-matmul rope, the
    stick-aligned SDPA, the offset-group pool) have to lower inside per-budget graphs,
    not just run eagerly. Compiled, an image above every budget is rejected up front.
    """
    _skip_without_gemma4()
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # A tower compile at request time is a missed warmup shape: fail, don't log.
    if not enforce_eager:
        monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    llm = _llm(enforce_eager, model=GEMMA4_MODEL, config_format="hf", revision=GEMMA4_REVISION)
    conversation = [_conversation(_synthetic_image_data_uri())]
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    (output,) = llm.chat(conversation, params)

    assert output.outputs[0].text.strip(), "empty generation from the Gemma 4 vision path"
    if not enforce_eager:
        with pytest.raises(VLLMValidationError, match="encoder_cudagraph_token_budgets"):
            llm.chat(conversation, params, mm_processor_kwargs={"max_soft_tokens": 560})


@pytest.mark.multimodal
@pytest.mark.gemma4_vision
@pytest.mark.uses_subprocess
@pytest.mark.timeout(3600)
def test_gemma4_request_max_soft_tokens_replays_warmed_budgets(monkeypatch):
    """Every opted-in ``max_soft_tokens``, and a default and a 1120 image sharing one encoder
    call, replay a budget compiled at warmup."""
    _skip_without_gemma4()
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    monkeypatch.setenv("SPYRE_COMPILE_GUARD", "error")
    # collective_rpc ships the probes below to the engine process with cloudpickle.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # An image is encoded with its request's prefill, and by default one request prefills
    # at a time, so two requests' images only meet in one encoder call with a second slot.
    monkeypatch.setenv("SPYRE_MAX_NUM_PARTIAL_PREFILLS", "2")
    from vllm import SamplingParams

    def record_encoder_calls(worker):
        manager = worker.model_runner.encoder_cudagraph_manager
        manager.calls = []
        execute = manager._execute_local

        def recording(mm_kwargs):
            manager.calls.append(manager._get_per_item_out_tokens(mm_kwargs))
            return execute(mm_kwargs)

        manager._execute_local = recording
        return manager.graph_hits, manager.graph_misses

    def encoder_calls(worker):
        manager = worker.model_runner.encoder_cudagraph_manager
        return manager.calls, manager._pack_tokens, manager.graph_hits, manager.graph_misses

    llm = _llm(
        enforce_eager=False,
        max_num_seqs=2,
        model=GEMMA4_MODEL,
        config_format="hf",
        revision=GEMMA4_REVISION,
        max_model_len=2048,  # room for the 1120-token image
        # 576 and 1152 hold 560 and 1120.
        encoder_budgets=(64, 128, 256, 280, 576, 1152),
    )
    ((hits, misses),) = llm.collective_rpc(record_encoder_calls)
    larger = (560, 1120)
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    for seed, tokens in enumerate(larger):
        (output,) = llm.chat(
            [_conversation(_synthetic_image_data_uri(seed=seed))],
            params,
            mm_processor_kwargs={"max_soft_tokens": tokens},
        )
        assert output.outputs[0].text.strip(), f"empty generation at max_soft_tokens={tokens}"

    # Paused, so both requests are queued before the step that encodes them is scheduled.
    llm.sleep(level=0)
    mixed = (None, {"max_soft_tokens": max(larger)})
    for seed, kwargs in enumerate(mixed, start=len(larger)):
        llm.enqueue_chat(
            [_conversation(_synthetic_image_data_uri(seed=seed))],
            params,
            mm_processor_kwargs=kwargs,
        )
    llm.wake_up(tags=["scheduling"])
    for output in llm.wait_for_completion():
        assert output.outputs[0].text.strip(), "empty generation from the mixed batch"

    ((calls, pack, *counts),) = llm.collective_rpc(encoder_calls)
    assert any(len(c) == 2 and min(c) <= pack < max(c) for c in calls), calls
    assert counts == [hits + len(larger) + 2, misses]


def _skip_without_gemma4() -> None:
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
