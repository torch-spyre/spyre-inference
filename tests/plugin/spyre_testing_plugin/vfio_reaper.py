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

"""Free the Spyre accelerator (VFIO) card when a test leaves it claimed.

When a vLLM engine dies abnormally (e.g. an execute_model RPC timeout during a
long Spyre compile), the worker holding the card is orphaned — reparented to
init but still holding the VFIO container (``/dev/vfio/vfio``) and device inode
(``anon_inode:[vfio-device]``) open. The next test's ``torch.spyre.set_device()``
then fails with ``RAS::VFIO::DeviceOpenFail ... "Device or resource busy"``,
cascading through the shard.

So after a failed test we find the holder by fd (a ``/proc/*/fd`` scan, not the
process tree — the holder may be reparented) and SIGKILL it, which frees the
card. The pytest process never opens the device itself, so it is excluded.
"""

from __future__ import annotations

import contextlib
import glob
import os
import signal
import time
from collections.abc import Callable


def spyre_hardware_present() -> bool:
    """True only on a real Spyre host (has /dev/vfio and AIU_WORLD_SIZE set)."""
    if not os.path.isdir("/dev/vfio"):
        return False
    try:
        return int(os.environ.get("AIU_WORLD_SIZE", "0") or 0) > 0
    except ValueError:
        return False


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip() or "<unknown>"
    except OSError:
        return "<unknown>"


def _pids_holding_vfio(exclude_pids: set[int]) -> list[tuple[int, str, str]]:
    """(pid, device, cmdline) for every process holding the Spyre card open,
    found by scanning `/proc/*/fd` for `/dev/vfio/*` or `anon_inode:[vfio-device]`."""
    holders: dict[int, tuple[int, str, str]] = {}
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        pid = int(fd_path.split("/")[2])
        if pid in exclude_pids or pid in holders:
            continue
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue  # fd or process vanished mid-scan — expected race
        if target.startswith("/dev/vfio/") or target == "anon_inode:[vfio-device]":
            holders[pid] = (pid, target, _read_cmdline(pid))
    return list(holders.values())


def reap_vfio_holders(
    exclude_pids: set[int],
    log: Callable[[str], None] = print,
    timeout: float = 5.0,
    poll: float = 0.1,
) -> None:
    """SIGKILL every process holding a Spyre card fd, then poll until the card is
    free. If it can't be freed, warn and keep going: a best-effort cleanup should
    not abort the whole session (the holder may be an unrelated VFIO device or a
    process we can't kill), and any genuinely card-blocked test still fails loudly
    on its own."""
    holders = _pids_holding_vfio(exclude_pids)
    if not holders:
        return

    start = time.monotonic()
    for pid, device, cmdline in holders:
        log(f"[vfio-reaper] orphan pid={pid} holding {device} cmd={cmdline!r}; sending SIGKILL")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)

    while True:
        survivors = _pids_holding_vfio(exclude_pids)
        if not survivors:
            log(f"[vfio-reaper] card freed in {time.monotonic() - start:.2f}s")
            return
        if time.monotonic() - start >= timeout:
            break
        time.sleep(poll)

    detail = ", ".join(f"pid={p} {dev} ({cmd!r})" for p, dev, cmd in survivors)
    log(
        f"[vfio-reaper] WARNING: Spyre card still held after {timeout}s by {detail}; "
        f"later card tests may fail with DeviceOpenFail until it is freed."
    )
