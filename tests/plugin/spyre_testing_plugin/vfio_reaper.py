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

"""Free the Spyre accelerator (VFIO) card between tests.

A test leaves the card claimed two ways:

1. A crashed engine orphans its worker, which keeps the VFIO container
   (``/dev/vfio/vfio``) and device (``anon_inode:[vfio-device]``) open. We find
   it by fd (``/proc/*/fd`` scan — it may be reparented) and SIGKILL it.
2. vLLM force-kills its worker at normal shutdown. The fd leaves ``/proc`` at
   once, but the kernel's device reset is async: ``/dev/vfio/<grp>`` stays EBUSY
   on ``open()`` for a short window (longer under CI load) after the holder is
   gone. So "no fd-holder" != "openable" — the next ``start_runtime()`` still
   loses the race unless we probe openability. (The window only appears after a
   real container+device attach whose holder exits; a bare open()/close() of the
   group node never triggers the reset, so it can't be reproduced standalone.)

pytest opens the device for in-process tests, so it is always excluded from scans.
"""

from __future__ import annotations

import contextlib
import errno
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


def _aiu_group_nodes(log: Callable[[str], None] = print) -> list[str]:
    """`/dev/vfio/<grp>` node(s) for the AIU card(s) in ``PCIDEVICE_IBM_COM_AIU_PF``
    (BDF -> sysfs iommu_group). Targeting the specific card(s) avoids tripping the
    openability probe on an unrelated VFIO device. Falls back to all
    ``/dev/vfio/<n>`` nodes when the env var is unset or a BDF won't resolve."""
    nodes: list[str] = []
    bdfs = os.environ.get("PCIDEVICE_IBM_COM_AIU_PF", "")
    for bdf in (b.strip() for b in bdfs.split(",") if b.strip()):
        try:
            grp = os.path.basename(os.readlink(f"/sys/bus/pci/devices/{bdf}/iommu_group"))
        except OSError:
            continue
        node = f"/dev/vfio/{grp}"
        if os.path.exists(node):
            nodes.append(node)
    if not nodes:
        nodes = sorted(glob.glob("/dev/vfio/[0-9]*"))
        log(
            f"[vfio-reaper] no AIU card node resolved from PCIDEVICE_IBM_COM_AIU_PF="
            f"{bdfs!r}; probing all VFIO group nodes instead: {nodes}"
        )
    return nodes


def _cards_openable(nodes: list[str], log: Callable[[str], None] = print) -> bool:
    """True if every AIU group node is ``open()``able now.

    A node returns EBUSY while the kernel resets the device after its holder
    exited — the async window the ``/proc`` scan misses. open()+close() of the
    group node is side-effect-free (no container attach, no device fd). Non-EBUSY
    errors (perms, missing node) can't prove busy, so they don't block (but are
    logged, since they usually mean a perms/config problem, not a free card).

    With no nodes to probe there is nothing that can be busy, so this reports
    openable — a genuinely absent card fails loudly in the test that needs it."""
    if not nodes:
        return True
    for node in nodes:
        try:
            os.close(os.open(node, os.O_RDWR))
        except OSError as e:
            if e.errno == errno.EBUSY:
                return False
            log(f"[vfio-reaper] unexpected error probing {node}: {e}; not treating as busy")
    return True


def _self_holds_device(pids: set[int]) -> bool:
    """True if a pid in `pids` (the pytest process) holds a live device fd
    (`anon_inode:[vfio-device]`) — i.e. the card is in-process-held for reuse, so
    callers skip the openability probe (it would only see our own EBUSY).

    Deliberately narrower than `_pids_holding_vfio`, which also matches the
    `/dev/vfio/` container fd: a container fd alone doesn't mean the device is
    open and warm, so it isn't the in-process-reuse signal we want here."""
    for pid in pids:
        for fd_path in glob.glob(f"/proc/{pid}/fd/*"):
            try:
                if os.readlink(fd_path) == "anon_inode:[vfio-device]":
                    return True
            except OSError:
                continue
    return False


def wait_until_card_free(
    exclude_pids: set[int],
    log: Callable[[str], None] = print,
    timeout: float = 10.0,
    poll: float = 0.1,
) -> bool:
    """Poll until the card is free to open, or `timeout` elapses (returns False).

    Free = no holder outside `exclude_pids` **and** openable again (the second
    clause rides out the async reset — see module docstring). The probe is
    skipped when pytest itself holds the device (in-process reuse).

    Kills nothing — a barrier for when the previous test's engine is on its way
    down. Timeout is non-fatal: warn and proceed so a stuck card fails loudly in
    the test that needs it, not here."""
    nodes = _aiu_group_nodes(log=log)
    start = time.monotonic()
    waited = False
    while True:
        holders = _pids_holding_vfio(exclude_pids)
        if holders:
            reason = ", ".join(f"pid={p} {dev} ({cmd!r})" for p, dev, cmd in holders)
        elif _self_holds_device(exclude_pids) or _cards_openable(nodes, log=log):
            if waited:
                log(f"[vfio-reaper] card freed in {time.monotonic() - start:.2f}s")
            return True
        else:
            reason = f"device still resetting (EBUSY on open of {nodes})"
        if time.monotonic() - start >= timeout:
            log(
                f"[vfio-reaper] WARNING: Spyre card still busy after {timeout}s: {reason}; "
                f"later card tests may fail with DeviceOpenFail until it is freed."
            )
            return False
        waited = True
        time.sleep(poll)


def reap_vfio_holders(
    exclude_pids: set[int],
    log: Callable[[str], None] = print,
    timeout: float = 10.0,
    poll: float = 0.1,
) -> None:
    """SIGKILL every process holding a card fd, then wait for the card. Best
    effort: if it won't free (unrelated VFIO device, unkillable process), warn
    and continue — any truly blocked test fails loudly on its own."""
    holders = _pids_holding_vfio(exclude_pids)
    if not holders:
        return

    for pid, device, cmdline in holders:
        log(f"[vfio-reaper] orphan pid={pid} holding {device} cmd={cmdline!r}; sending SIGKILL")
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)

    wait_until_card_free(exclude_pids, log=log, timeout=timeout, poll=poll)
