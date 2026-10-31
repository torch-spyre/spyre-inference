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

"""Shared helpers for distributed e2e tests."""

from __future__ import annotations

import gc
import os


def generate(
    model: str,
    tp: int,
    enforce_eager: bool,
    compilation_config: dict | None = None,
    hf_overrides=None,
) -> list[list[int]]:
    """Run vllm.LLM.generate and return per-prompt token-id lists.

    Shuts down the engine and waits for Spyre device release before returning.
    The post-shutdown assertion sits outside the ``finally`` block so a
    generate failure does not mask the device-release failure's own traceback.
    """
    from spyre_testing_plugin.vfio_reaper import wait_until_card_free
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=enforce_eager,
        max_model_len=128,
        max_num_seqs=2,
        **({"compilation_config": compilation_config} if compilation_config is not None else {}),
        **({"hf_overrides": hf_overrides} if hf_overrides is not None else {}),
    )
    try:
        outs = llm.generate(
            ["Hello, world!", "The capital of France is"],
            SamplingParams(max_tokens=8, temperature=0.0),
        )
        result = [list(o.outputs[0].token_ids) for o in outs]
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=60)
        del llm
        gc.collect()
        freed = wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)
    # Outside the finally, where a generate failure would mask this check's own.
    assert freed, "Spyre devices were not released after LLM shutdown"
    return result
