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

"""Run a script in a fresh interpreter: vllm.triton_utils binds HAS_TRITON once."""

import os
import subprocess
import sys
import textwrap


def run_script(script: str, **env: str) -> str:
    """The script's stdout. Drops a VLLM_USE_V2_MODEL_RUNNER an earlier in-process
    VllmConfig pinned, so ``env`` alone decides it."""
    base = {k: v for k, v in os.environ.items() if k != "VLLM_USE_V2_MODEL_RUNNER"}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**base, **env},
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout
