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

"""Ministral 3 FP8 loaded in HF format keeps its vision tower and projector unquantized (#1697).

The repo also ships Mistral-format files, and with them ``config_format="auto"`` takes the
Pixtral path, whose tower gets no quant config. So this links the cached snapshot without
them, as a copy holding only the HF-format files looks.
"""

from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_available

MODEL = "mistralai/Ministral-3-14B-Instruct-2512"
MISTRAL_FORMAT_FILES = {"consolidated.safetensors", "params.json", "tekken.json"}


def _hf_format_checkpoint(dst: Path) -> Path:
    from huggingface_hub import snapshot_download

    try:
        src = Path(snapshot_download(MODEL, local_files_only=True))
    except Exception:
        pytest.skip(f"{MODEL} not in the local HF cache")
    for f in src.iterdir():
        if f.name not in MISTRAL_FORMAT_FILES:
            (dst / f.name).symlink_to(f.resolve())
    return dst


@pytest.mark.fp8
@pytest.mark.multimodal
@pytest.mark.uses_subprocess
@pytest.mark.skipif(not spyre_available(), reason="needs a Spyre card")
def test_hf_format_tower_stays_unquantized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from vllm import LLM

    model = _hf_format_checkpoint(tmp_path)
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    # apply_model ships the probe below to the engine process with cloudpickle.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def linears(model):
        layers = {
            "tower": model.vision_tower.transformer.layers[0].attention.qkv_proj,
            "projector": model.multi_modal_projector.linear_1,
            "decoder": model.language_model.model.layers[0].self_attn.qkv_proj,
        }
        return {
            name: (type(layer.quant_method).__name__, str(layer.weight.dtype))
            for name, layer in layers.items()
        }

    llm = LLM(
        model=str(model),
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
    )
    try:
        (methods,) = llm.apply_model(linears)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)

    assert methods == {
        "tower": ("UnquantizedLinearMethod", "torch.float16"),
        "projector": ("UnquantizedLinearMethod", "torch.float16"),
        "decoder": ("SpyreFp8LinearMethod", "torch.float8_e4m3fn"),
    }
