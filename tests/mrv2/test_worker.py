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

"""The copied init_device body hasn't drifted from TorchSpyreWorker's, and
installs the device shims and binds the kernels before building the runner.

Both checked on source rather than by running init_device: the real thing
needs a card and a non-stub SpyreModelRunnerV2, and init_device is
straight-line code with no branching around any of this, so source order
is runtime order. An execution-based test is more honest once worker.py's
callees stop being NotImplementedError stubs.
"""

import difflib
import inspect
import re

from spyre_inference.v1.worker.spyre.worker import SpyreWorkerV2
from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker

_TY_IGNORE_RE = re.compile(r"\s*#\s*ty:\s*ignore(\[[\w,-]+\])?\s*$")


def _normalized_body(source: str, *, strip_marker: str | None = None) -> list[str]:
    """init_device's statements, one per line, blank lines and the `def`
    line dropped. strip_marker drops any line carrying that comment too.
    A trailing `# ty: ignore[...]` is stripped from every line: V2 needs
    one where it builds a nominally mismatched runner type (worker.py);
    V1 doesn't, so comparing the suppression itself would be noise."""
    lines = source.splitlines()[1:]
    out = []
    for line in lines:
        if strip_marker and strip_marker in line:
            continue
        stripped = _TY_IGNORE_RE.sub("", line).strip()
        if stripped:
            out.append(stripped)
    return out


def test_init_device_has_not_drifted_from_v1() -> None:
    """The copy in worker.py must match TorchSpyreWorker's line for line,
    apart from the MRV2-only additions and the runner class it builds."""
    v1_body = _normalized_body(inspect.getsource(TorchSpyreWorker.init_device))
    v2_body = _normalized_body(
        inspect.getsource(SpyreWorkerV2.init_device), strip_marker="MRV2-only"
    )
    v2_body = [line.replace("SpyreModelRunnerV2", "TorchSpyreModelRunner") for line in v2_body]

    diff = list(difflib.unified_diff(v1_body, v2_body, lineterm=""))
    assert not diff, (
        "SpyreWorkerV2.init_device has drifted from TorchSpyreWorker.init_device "
        "(apart from the lines marked '# MRV2-only'):\n" + "\n".join(diff)
    )


def test_init_device_installs_shims_and_binds_kernels_before_building_the_runner() -> None:
    """GPUModelRunner.__init__ constructs a real torch.cuda.Stream, so
    device_shims.install() has to run before the runner is built, and
    kernels.bind_or_raise() before anything the runner constructs needs a
    bound kernel."""
    source = inspect.getsource(SpyreWorkerV2.init_device)
    markers = ["device_shims.install()", "kernels.bind_or_raise()", "SpyreModelRunnerV2("]
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions), (
        f"expected {markers} in that order in SpyreWorkerV2.init_device"
    )
