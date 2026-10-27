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

"""MRV2: the single kernel seam for MRV2's default-path Triton kernels.

bind() is the one place that picks each kernel's source: triton-cpu on x86
until upstream's non-Triton replacements exist, then those, then a local
override only if measurement on Power/Z calls for one. DEFAULT_PATH_KERNELS
is the full list a coverage test checks bind() against.
"""

DEFAULT_PATH_KERNELS: frozenset[str] = frozenset()


def bind() -> set[str]:
    """Bind every DEFAULT_PATH_KERNELS entry to its active source.

    Returns the kernel names actually bound, so callers (and the coverage
    test) can diff against DEFAULT_PATH_KERNELS.
    """
    # Placeholder: with DEFAULT_PATH_KERNELS empty, bind_or_raise() checks nothing yet (#906).
    return set()


def bind_or_raise() -> None:
    """Call bind() and raise if any DEFAULT_PATH_KERNELS entry is still
    unbound, rather than letting it fall through to upstream's Triton path
    or fail at its first launch mid-serving."""
    missing = DEFAULT_PATH_KERNELS - bind()
    if missing:
        raise RuntimeError(f"kernels.bind() left {sorted(missing)} unbound")
