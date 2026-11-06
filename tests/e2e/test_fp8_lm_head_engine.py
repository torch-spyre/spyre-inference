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

"""The FP8 LM head through a real engine: load, warmup, logits row padding, compile guard.

Every cached FP8 checkpoint lists ``lm_head`` in its ``ignore``, so this builds one that
ships its head in FP8: the cached micro Granite FP8 checkpoint with its FP16 head
quantized per vocab row and a config group targeting ``lm_head``, as llm-compressor
writes it. The engine loads it through the real compressed-tensors path. Like
``test_transformers_matches_native``, it is ordered with the ``uses_subprocess`` tests,
ahead of the session ``tp_group`` that would already hold the TP group the engine
initializes, and tears the engine's distributed state down after.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8"


def _fp8_head_checkpoint(dst: Path) -> Path:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    try:
        src = Path(snapshot_download(MODEL, local_files_only=True))
    except Exception:
        pytest.skip(f"{MODEL} not in the local HF cache")

    for f in src.iterdir():
        if f.name not in ("config.json", "model.safetensors"):
            shutil.copy(f, dst / f.name)

    tensors = load_file(src / "model.safetensors")
    head = tensors["lm_head.weight"].float()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (head.abs().amax(dim=1, keepdim=True) / fp8_max).clamp(min=1e-12)
    tensors["lm_head.weight"] = (head / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    tensors["lm_head.weight_scale"] = scale
    save_file(tensors, dst / "model.safetensors")

    config = json.loads((src / "config.json").read_text())
    quant = config["quantization_config"]
    head_group = {**quant["config_groups"]["group_0"], "targets": ["lm_head"]}
    quant["config_groups"]["group_1"] = head_group
    quant["ignore"] = []
    (dst / "config.json").write_text(json.dumps(config))
    return dst


@pytest.mark.fp8
@pytest.mark.uses_subprocess
def test_fp8_lm_head_serves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig
    from vllm.distributed import cleanup_dist_env_and_memory

    from spyre_inference.custom_ops import parallel_lm_head

    model = _fp8_head_checkpoint(tmp_path)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    loads = []
    load = parallel_lm_head.SpyreFp8LMHeadMethod.process_weights_after_loading

    def counting_load(self, layer):
        loads.append(layer)
        load(self, layer)

    monkeypatch.setattr(
        parallel_lm_head.SpyreFp8LMHeadMethod, "process_weights_after_loading", counting_load
    )

    prompt = "What are IBMs main businesses?"
    engine = None
    try:
        engine = LLM(
            model=str(model),
            enforce_eager=False,
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=8,
            compilation_config=CompilationConfig(compile_sizes=[1, 8]),
        )
        output = engine.generate(
            prompt, SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False
        )
    finally:
        # A half-built engine still holds the TP group the next test would initialize.
        del engine
        cleanup_dist_env_and_memory()

    assert loads, "the checkpoint's FP8 head was not loaded as FP8"
    # The FP16-head reference: FP8 logits keep the same greedy tokens on this prompt.
    expected = "\n\nIBMs are a set of standards that are used to measure the performance"
    assert output[0].outputs[0].text == expected
