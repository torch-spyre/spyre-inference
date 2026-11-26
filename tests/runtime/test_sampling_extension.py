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

"""Which csrc/ extension build a host may load."""

import pytest

from spyre_inference.v1.sample.sampling_kernels import (
    isa_from_capabilities,
    sampling_extension_name,
)


def test_avx2_without_fma_does_not_load_the_fma_build() -> None:
    assert sampling_extension_name("x86_64", avx512=True, avx2_fma=True) == "_C"
    assert sampling_extension_name("x86_64", avx512=False, avx2_fma=True) == "_C_AVX2"
    with pytest.raises(ImportError):
        sampling_extension_name("x86_64", avx512=False, avx2_fma=False)
    assert sampling_extension_name("ppc64le", avx512=False, avx2_fma=False) == "_C"
    assert sampling_extension_name("s390x", avx512=False, avx2_fma=False) == "_C"


def test_avx512_build_needs_vl_bw_and_dq() -> None:
    """AVX512F alone is not the _C build. Knights Landing has F and not VL/BW/DQ."""
    full = {flag: True for flag in ("avx512_f", "avx512_vl", "avx512_bw", "avx512_dq")}
    assert isa_from_capabilities({**full, "avx2": True, "fma3": True}) == (True, True)
    knl = {**full, "avx512_vl": False, "avx512_bw": False, "avx512_dq": False}
    avx512, avx2_fma = isa_from_capabilities({**knl, "avx2": True, "fma3": True})
    assert (avx512, avx2_fma) == (False, True)
    assert sampling_extension_name("x86_64", avx512=avx512, avx2_fma=avx2_fma) == "_C_AVX2"
    assert isa_from_capabilities({"avx2": True, "fma3": False}) == (False, False)
