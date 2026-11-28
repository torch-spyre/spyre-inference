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

"""TEMPORARY (PR validation only): compare serial vs PARALLEL JUnit results."""

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

serial_dir, parallel_dir, serial_s, parallel_s = sys.argv[1:5]
targets = sys.argv[5:]


def cases(d, t):
    p = Path(d) / f"junit-{t}" / f"junit-{t}.xml"
    if not p.is_file():
        return None, Counter()
    ids, outcome = [], Counter()
    for tc in ET.parse(p).getroot().iter("testcase"):
        ids.append(f"{tc.get('classname')}::{tc.get('name')}")
        kids = [c.tag for c in tc]
        outcome[
            "failed"
            if {"failure", "error"} & set(kids)
            else "skipped"
            if "skipped" in kids
            else "passed"
        ] += 1
    return ids, outcome


rc = 0
tot = {"serial": 0, "parallel": 0}
print(f"{'target':28} {'serial':>7} {'parallel':>8}  same-ids  serial(p/s/f)  parallel(p/s/f)")
for t in targets:
    s_ids, s_out = cases(serial_dir, t)
    p_ids, p_out = cases(parallel_dir, t)
    same = s_ids is not None and p_ids is not None and sorted(s_ids) == sorted(p_ids)
    rc |= not same
    tot["serial"] += len(s_ids or [])
    tot["parallel"] += len(p_ids or [])
    fmt = lambda o: f"{o['passed']}/{o['skipped']}/{o['failed']}"  # noqa: E731
    print(
        f"{t:28} {len(s_ids or []):>7} {len(p_ids or []):>8}  {str(same):8}  "
        f"{fmt(s_out):13}  {fmt(p_out)}"
    )
print(f"TOTAL test IDs: serial={tot['serial']} parallel={tot['parallel']}")
print(
    f"WALL: serial={serial_s}s parallel={parallel_s}s "
    f"speedup={int(serial_s) / max(1, int(parallel_s)):.2f}x"
)
print("AB-PARITY:", "PASS" if rc == 0 else "FAIL")
sys.exit(rc)
