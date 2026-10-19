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

"""MRV2: process-wide torch.cuda Stream/Event/stream/current_stream placeholders.

MRV2's GPUModelRunner constructs real torch.cuda.Stream/Event objects
unconditionally (main_stream, the pooling copy_stream/copy_event), and
calls torch.cuda.current_stream() (main_stream) and enters
torch.cuda.stream(...) (async_utils, structured_outputs, spec_decode/utils)
— all of which raise on CPU-only torch. install() swaps in no-op
placeholders for all four instead.
"""


def install() -> None:
    """Install the torch.cuda placeholders (Stream, Event, stream,
    current_stream). Idempotent."""
    raise NotImplementedError


def uninstall() -> None:
    """Undo install(), restoring the original torch.cuda attributes."""
    raise NotImplementedError


def is_installed() -> bool:
    """Whether install() has run and uninstall() hasn't undone it."""
    raise NotImplementedError


def spyre_fence() -> None:
    """Replace the CUDA event wait on the pooling output path.

    Pooling's AsyncPoolingOutput copies results to host behind a
    torch.cuda.Event; on Spyre this synchronizes the device directly instead.
    """
    raise NotImplementedError
