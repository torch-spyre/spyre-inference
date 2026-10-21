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

"""device_shims: reversible torch.cuda placeholders, and the Spyre fence."""

import subprocess
import sys
import textwrap
from collections.abc import Iterator

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncPoolingOutput

from spyre_inference.v1.worker.spyre import device_shims

_SHIMMED = ("Stream", "Event", "stream", "current_stream", "set_stream")


@pytest.fixture
def shims() -> Iterator[None]:
    device_shims.install()
    yield
    device_shims.uninstall()


def test_install_is_idempotent_and_uninstall_restores_torch_cuda() -> None:
    originals = {name: getattr(torch.cuda, name) for name in _SHIMMED}

    device_shims.install()
    device_shims.install()
    assert device_shims.is_installed()
    assert all(getattr(torch.cuda, name) is not originals[name] for name in _SHIMMED)

    device_shims.uninstall()
    assert not device_shims.is_installed()
    assert {name: getattr(torch.cuda, name) for name in _SHIMMED} == originals


@pytest.mark.usefixtures("shims")
def test_mrv2_pooling_output_fences_on_the_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """AsyncPoolingOutput exercises all five placeholders, and its copy-event wait fences."""
    fences = []
    monkeypatch.setattr(device_shims, "spyre_fence", lambda: fences.append(None))

    output = AsyncPoolingOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}),
        pooler_output=torch.arange(4.0).reshape(2, 2),
        finished_mask=[True, True],
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        result = output.get_output()

    assert [t.tolist() for t in result.pooler_output] == [[0.0, 1.0], [2.0, 3.0]]
    assert fences


@pytest.mark.spyre_device
@pytest.mark.usefixtures("shims")
def test_placeholder_event_wait_synchronizes_spyre(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not through AsyncPoolingOutput: its non_blocking Spyre-to-host copy needs a pinned
    allocator torch-spyre doesn't implement, which pool/output.py replaces."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    calls = []
    synchronize = torch.spyre.synchronize
    monkeypatch.setattr(
        torch.spyre, "synchronize", lambda *args: calls.append(args) or synchronize(*args)
    )
    host = torch.arange(128, dtype=torch.float16).reshape(2, 64)

    on_device = host.to("spyre") * 2
    torch.cuda.Event(blocking=True).synchronize()

    assert calls
    assert torch.equal(on_device.cpu(), host * 2)


@pytest.mark.subprocess_isolated
def test_fence_skips_the_sync_before_the_runtime_starts() -> None:
    """A placeholder wait before init_device must not start torch-spyre's runtime."""
    script = textwrap.dedent("""
        import torch
        import torch_spyre

        from spyre_inference.v1.worker.spyre import device_shims

        torch_spyre._autoload()
        calls = []
        torch.spyre.synchronize = lambda *args: calls.append(args)
        device_shims.install()
        torch.cuda.Event().synchronize()
        torch.cuda.current_stream().synchronize()
        print(torch.spyre.is_initialized(), len(calls))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["False", "0"]
