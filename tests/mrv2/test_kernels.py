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

"""kernels.bind_or_raise() actually enforces the diff it claims to."""

import pytest

from spyre_inference.v1.worker.spyre import kernels


def test_bind_or_raise_names_a_kernel_bind_left_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "DEFAULT_PATH_KERNELS", frozenset({"a", "b"}))
    monkeypatch.setattr(kernels, "bind", lambda: {"a"})

    with pytest.raises(RuntimeError, match=r"\['b'\]"):
        kernels.bind_or_raise()


def test_bind_or_raise_is_a_noop_once_everything_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "DEFAULT_PATH_KERNELS", frozenset({"a", "b"}))
    monkeypatch.setattr(kernels, "bind", lambda: {"a", "b"})

    kernels.bind_or_raise()
