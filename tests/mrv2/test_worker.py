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

"""init_device installs the device shims and binds the kernels before building the runner.

Checked on source rather than by running init_device: the real thing needs a card,
and init_device is straight-line code with no branching around any of this, so
source order is runtime order.
"""

import inspect

from spyre_inference.v1.worker.spyre.worker import SpyreWorkerV2


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
