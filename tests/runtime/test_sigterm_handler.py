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

"""Loading senlib must not take SIGTERM away from vLLM's shutdown handler."""

import signal
import subprocess
import sys
import textwrap

import pytest

from spyre_inference import keep_sigterm_handler


def test_restores_the_handler_replaced_inside():
    def ours(signum, frame):
        pass

    previous = signal.signal(signal.SIGTERM, ours)
    try:
        with keep_sigterm_handler():
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        assert signal.getsignal(signal.SIGTERM) is ours
    finally:
        signal.signal(signal.SIGTERM, previous)


# senlib installs its handler once, when it is first loaded, so each path runs in a
# fresh interpreter. SIGTERM must reach the handler as vLLM's does: raise SystemExit.
_SCRIPT = """
import os, signal, sys
{env}
try:
    import torch  # noqa: F401
    import torch_spyre  # noqa: F401
except Exception:
    print("SKIP")
    sys.exit(0)

def vllm_handler(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, vllm_handler)
{load}
try:
    os.kill(os.getpid(), signal.SIGTERM)
except SystemExit:
    print("GRACEFUL")
"""


_AUTOLOAD_OFF = 'os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"'
_AUTOLOAD_ON = 'os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)'


@pytest.mark.parametrize(
    "env",
    [_AUTOLOAD_OFF, _AUTOLOAD_ON],
    # Autoload off is where the guard does the work: senlib first loads inside it.
    ids=["autoload_off", "autoload_on"],
)
@pytest.mark.parametrize(
    "load",
    [
        "import spyre_inference; spyre_inference.register_ops()",
        "from spyre_inference import keep_sigterm_handler\n"
        "with keep_sigterm_handler():\n"
        "    torch_spyre._autoload()",
    ],
    ids=["register_ops", "worker_autoload"],
)
def test_sigterm_reaches_vllms_handler_after_loading_senlib(load, env):
    script = _SCRIPT.format(env=env, load=textwrap.dedent(load))
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    ).stdout
    if "SKIP" in out:
        pytest.skip("torch_spyre is not importable")
    assert "GRACEFUL" in out, "senlib's SIGTERM handler replaced vLLM's"
