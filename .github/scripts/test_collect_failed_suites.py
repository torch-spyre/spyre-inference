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

"""collect_failed_suites.py: artifact name is attacker-controllable, so a
traversing name must not steer the extraction path off the runner workspace."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import collect_failed_suites as cfs  # noqa: E402


def test_safe_segment_accepts_real_artifact_names():
    # The names the workflow actually uploads: failed-suite-<attempt>-<job-index>.
    assert cfs.safe_segment("failed-suite-1-0")
    assert cfs.safe_segment("failed-suite-2-11")


def test_safe_segment_rejects_traversal_and_separators():
    # Each of these keeps the startswith("failed-suite-<attempt>-") filter but
    # escapes FAILED_SUITES_DIR when joined as a path component.
    assert not cfs.safe_segment("failed-suite-1-../../../../tmp/evil")
    assert not cfs.safe_segment("failed-suite-1-sub/evil")
    assert not cfs.safe_segment("..")
    assert not cfs.safe_segment(".")
    assert not cfs.safe_segment("")
    assert not cfs.safe_segment(".hidden")
    assert not cfs.safe_segment("a\\b")


def test_download_skips_traversing_name(monkeypatch, tmp_path, capsys):
    # A traversing artifact name must be refused before any download/extract,
    # so nothing is written outside FAILED_SUITES_DIR.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cfs, "FAILED_SUITES_DIR", pathlib.Path("failed_suites"))

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):  # pragma: no cover - should not run
        calls.append(cmd)
        raise AssertionError(f"unexpected subprocess call for unsafe name: {cmd}")

    monkeypatch.setattr(cfs.subprocess, "run", fake_run)

    evil = "failed-suite-1-" + "../" * 6 + "tmp/evil"
    cfs.download_and_extract("org/repo", [{"id": 1, "name": evil}])

    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "Downloaded 0" in out
    assert calls == []
    # Nothing escaped the (empty, maybe-absent) base dir.
    assert not (tmp_path / "tmp" / "evil").exists()
