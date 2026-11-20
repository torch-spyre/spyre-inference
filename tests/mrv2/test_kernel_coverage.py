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

"""kernels.bind() covers every Triton launch site MRV2 can reach on the pinned vLLM,
with each host op taking its kernel's arguments, so a new, renamed or re-signed
kernel fails here on the bump, not mid-serving."""

import pytest

from spyre_inference.v1.worker.spyre import kernels

from .kernels import parity_cases, spec_decode_cases  # noqa: F401  (registers cases)


def test_every_launch_site_is_bound() -> None:
    bound = set(kernels.HOST_KERNELS) | set(kernels.UNSUPPORTED_KERNELS)
    sites = kernels.launch_sites()
    assert sites == bound, f"unbound: {sorted(sites - bound)}, gone: {sorted(bound - sites)}"


@pytest.mark.parametrize("name", sorted(kernels.HOST_KERNELS))
def test_host_op_takes_the_kernel_arguments(name: str) -> None:
    kernel = kernels.original(name)
    assert kernels.op_params(kernels.HOST_KERNELS[name]) == kernels.kernel_params(kernel)


def test_every_host_kernel_has_a_parity_case() -> None:
    covered = {k for case in parity_cases.CASES.values() for k in case.kernels}
    assert set(kernels.HOST_KERNELS) <= covered, sorted(set(kernels.HOST_KERNELS) - covered)
