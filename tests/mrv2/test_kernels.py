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

"""kernels.py's seam: HostKernel launches like a Triton kernel, bind_or_raise()
enforces the full list, and unsupported features fail loudly."""

import pytest
import torch

from spyre_inference.v1.worker.spyre import kernels


def test_host_kernel_launches_by_position_and_keyword_and_drops_launch_options() -> None:
    calls = []
    kernel = kernels.HostKernel("k", lambda *args: calls.append(args), ["a", "b", "C"])
    kernel[(3, 2)](1, b=2, C=3, num_warps=4, num_stages=1)
    assert calls == [([3, 2], 1, 2, 3)]


def test_unsupported_kernel_names_the_feature() -> None:
    kernel = kernels.UnsupportedKernel("gpu.x._foo_kernel", "frobnication")
    with pytest.raises(NotImplementedError, match="does not support frobnication"):
        kernel[(1,)](0)


def test_bind_or_raise_names_a_kernel_bind_left_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "use_triton", lambda: False)
    monkeypatch.setattr(kernels, "HOST_KERNELS", {"a": "op"})
    monkeypatch.setattr(kernels, "UNSUPPORTED_KERNELS", {"b": "feature"})
    monkeypatch.setattr(
        kernels,
        "bind",
        lambda: {"a", *(f"{s}.apply_top_k_top_p" for s in kernels._TOP_K_TOP_P_SITES)},
    )

    with pytest.raises(RuntimeError, match=r"\['b'\]"):
        kernels.bind_or_raise()


def test_bind_then_unbind_restores_upstream() -> None:
    from vllm.v1.worker.gpu import input_batch

    original = input_batch._post_update_kernel
    try:
        kernels.bind(force_host=True)
        assert isinstance(input_batch._post_update_kernel, kernels.HostKernel)
    finally:
        kernels.unbind()
    assert input_batch._post_update_kernel is original


def test_placeholder_next_power_of_2() -> None:
    from vllm.triton_utils import triton

    try:
        kernels.bind(force_host=True)
    finally:
        kernels.unbind()
    assert [triton.next_power_of_2(n) for n in (0, 1, 2, 3, 1000, 1024)] == [1, 1, 2, 4, 1024, 1024]


@pytest.mark.parametrize("use_k", [True, False])
@pytest.mark.parametrize("use_p", [True, False])
def test_sort_free_top_k_top_p_keeps_the_same_tokens(use_k: bool, use_p: bool) -> None:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    # Not torch.manual_seed: seeding every device's generator initializes Spyre.
    logits = torch.randn(4, 1000, generator=torch.Generator().manual_seed(0))
    k = torch.tensor([1, 5, 50, 1000]) if use_k else None
    p = torch.tensor([0.5, 0.9, 0.99, 1.0]) if use_p else None
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = kernels.spyre_apply_top_k_top_p(logits.clone(), k, p)
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    assert torch.equal(actual[torch.isfinite(actual)], expected[torch.isfinite(expected)])
