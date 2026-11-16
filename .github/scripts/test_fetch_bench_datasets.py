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

"""The dataset resolver against a fake store: no network, no credentials."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tarfile
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).with_name("fetch_bench_datasets.py")
_spec = importlib.util.spec_from_file_location("fetch_bench_datasets", _SCRIPT)
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

PAYLOAD = b'{"prompt": "x"}\n' * 50
SHA = hashlib.sha256(PAYLOAD).hexdigest()
SPEC = {"path": "grp/trace.jsonl", "sha256": SHA, "name": "demo"}
ARCHIVE_SPEC = {"path": "grp/set.tar.gz", "sha256": "ab" * 32, "name": "demo-archive"}


class FakeResponse(io.BytesIO):
    """A store response; `fail_after` bytes then a dropped connection."""

    def __init__(self, data, fail_after=None):
        super().__init__(data)
        self.fail_after = fail_after

    def read(self, n=-1):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("reset")
        if self.fail_after is not None:
            n = min(n if n > 0 else self.fail_after, self.fail_after - self.tell())
        return super().read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "https://store.invalid")
    monkeypatch.setenv("ARTIFACTORY_DATASETS_PATH", "repo/datasets")
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "tok-secret")
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ARTIFACTORY_DATA_PATH", raising=False)


def opener_for(data, seen=None, **kw):
    def opener(url, token):
        if seen is not None:
            seen.append((url, token))
        return FakeResponse(data, **kw)

    return opener


def leftovers(tmp_path):
    return sorted(p.name for p in (tmp_path / "out").rglob("*") if p.is_file())


def test_definitions_cover_every_dataset_with_a_real_digest():
    assert len(fb.ARTIFACTS) == 13 and list(fb.ARCHIVES) == ["SPYRE_VISION_DATASET"]
    for var, spec in {**fb.ARTIFACTS, **fb.ARCHIVES}.items():
        assert var.startswith("SPYRE_") and var.endswith("_DATASET")
        assert len(spec["sha256"]) == 64 and int(spec["sha256"], 16) >= 0
    assert fb.ARTIFACTS["SPYRE_CICS_DATASET"]["sha256"].startswith("e0efa895")
    assert fb.ARCHIVES["SPYRE_VISION_DATASET"]["media_var"] == "SPYRE_VISION_IMAGES_DATASET"
    assert fb.STORE == {
        "base": "ARTIFACTORY_BASE_URL",
        "root": "ARTIFACTORY_DATASETS_PATH",
        "token": "ARTIFACTORY_TOKEN",
        "allow_insecure": False,
    }


def test_download_verifies_and_publishes(tmp_path):
    seen = []
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    fb.download(SPEC["path"], dest, SHA, opener=opener_for(PAYLOAD, seen))
    assert dest.read_bytes() == PAYLOAD
    assert seen == [
        (
            "https://store.invalid/artifactory/repo/datasets/grp/trace.jsonl",
            "tok-secret",
        )
    ]
    assert leftovers(tmp_path) == ["trace.jsonl"]


def test_hash_mismatch_publishes_nothing(tmp_path):
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="does not match"):
        fb.download(SPEC["path"], dest, "0" * 64, opener=opener_for(PAYLOAD))
    assert leftovers(tmp_path) == []


def test_missing_dataset_names_the_path_not_the_url(tmp_path):
    def opener(url, token):
        raise urllib.error.HTTPError(url, 404, "nf", None, None)

    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable) as err:
        fb.download(SPEC["path"], dest, SHA, opener=opener)
    assert "HTTP 404" in str(err.value) and "store.invalid" not in str(err.value)
    assert leftovers(tmp_path) == []


def test_partial_download_leaves_no_part_file(tmp_path):
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="ConnectionResetError"):
        fb.download(
            SPEC["path"],
            dest,
            SHA,
            opener=opener_for(PAYLOAD, fail_after=100),
            sleep=lambda s: None,
        )
    assert leftovers(tmp_path) == []


def test_truncated_but_clean_transfer_fails_the_digest(tmp_path):
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="does not match"):
        fb.download(SPEC["path"], dest, SHA, opener=opener_for(PAYLOAD[:100]))
    assert leftovers(tmp_path) == []


def test_there_is_no_fallback_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ARTIFACTORY_DATASETS_PATH")
    monkeypatch.setenv("ARTIFACTORY_DATA_PATH", "some/other/root")
    with pytest.raises(fb.FetchUnavailable, match="ARTIFACTORY_DATASETS_PATH"):
        fb.download(SPEC["path"], tmp_path / "x", SHA, opener=opener_for(PAYLOAD))


def test_ensure_cached_is_cache_first_and_force_refetches(tmp_path):
    dest = tmp_path / "cache"
    calls = []
    opener = opener_for(PAYLOAD, calls)
    assert fb.ensure_cached(SPEC, dest, opener=opener) == "downloaded"
    assert fb.ensure_cached(SPEC, dest, opener=opener) == "cached"
    assert fb.ensure_cached(SPEC, dest, force=True, opener=opener) == "downloaded"
    assert len(calls) == 2


def test_a_corrupt_cached_copy_is_replaced(tmp_path):
    dest = tmp_path / "cache"
    (dest / "grp").mkdir(parents=True)
    (dest / SPEC["path"]).write_bytes(b"corrupt")
    assert fb.ensure_cached(SPEC, dest, opener=opener_for(PAYLOAD)) == "downloaded"
    assert (dest / SPEC["path"]).read_bytes() == PAYLOAD


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    "member",
    [("../escape.txt", b"x", "file"), ("/abs.txt", b"x", "file"), ("link", b"", "link")],
)
def test_archive_extraction_refuses_unsafe_members(tmp_path, member):
    archive = tmp_path / "bad.tar.gz"
    _tar(archive, [("prompts.jsonl", b"{}\n", "file"), member])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="unsafe member"):
        fb.safe_extract(archive, out)
    assert not (tmp_path / "escape.txt").exists() and list(out.iterdir()) == []


def test_archive_materializes_with_absolute_image_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "cache" / "grp" / "set.tar.gz"
    archive.parent.mkdir(parents=True)
    row = b'{"prompt": "p", "image_files": ["img/a.png"]}\n'
    _tar(archive, [("prompts.jsonl", row, "file"), ("img/a.png", b"png", "file")])
    dataset, media = fb.materialize(archive, ARCHIVE_SPEC)
    assert str(media / "img/a.png") in dataset.read_text()


def test_status_reports_state_and_readers(monkeypatch, tmp_path, capsys):
    cached = tmp_path / "cache" / fb.ARTIFACTS["SPYRE_CICS_DATASET"]["path"]
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not the right bytes")
    preset = tmp_path / "mine.jsonl"
    preset.write_text("{}")
    monkeypatch.setenv("SPYRE_AIOPS_DATASET", str(preset))
    fb.status()
    lines = {
        line.split("\t")[0]: line for line in capsys.readouterr().out.splitlines() if "\t" in line
    }
    assert (
        "\tpreset\t" in lines["SPYRE_AIOPS_DATASET"]
        and "serve_granite33-8b-fp8_tp4_aiops" in lines["SPYRE_AIOPS_DATASET"]
    )
    assert "\tcorrupt\t" in lines["SPYRE_CICS_DATASET"]
    assert "\tabsent\t" in lines["SPYRE_TLS_DATASET"]


def test_redirect_drops_the_token_on_another_host_and_refuses_http():
    import urllib.request

    handler = fb._SameHostRedirect()
    req = urllib.request.Request("https://store.invalid/a", headers={"Authorization": "Bearer t"})

    def follow(url):
        return handler.redirect_request(req, None, 302, "Found", {}, url)

    same = follow("https://store.invalid/b")
    assert same is not None and same.get_header("Authorization") == "Bearer t"
    other = follow("https://blob.invalid/b")
    assert other is not None and other.get_header("Authorization") is None
    assert follow("http://store.invalid/b") is None


def test_transient_failures_are_retried_and_permanent_ones_are_not(tmp_path):
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    attempts, sleeps = [], []

    def flaky(url, token):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 503, "busy", None, None)
        return FakeResponse(PAYLOAD)

    fb.download(SPEC["path"], dest, SHA, opener=flaky, sleep=sleeps.append)
    assert len(attempts) == 3 and len(sleeps) == 2 and dest.read_bytes() == PAYLOAD

    attempts.clear()

    def forbidden(url, token):
        attempts.append(1)
        raise urllib.error.HTTPError(url, 403, "no", None, None)

    with pytest.raises(fb.FetchUnavailable, match="HTTP 403"):
        fb.download(
            SPEC["path"], tmp_path / "out" / "x", SHA, opener=forbidden, sleep=sleeps.append
        )
    assert len(attempts) == 1


def test_an_incomplete_read_is_a_fetch_failure_not_a_crash(tmp_path):
    import http.client

    def opener(url, token):
        raise http.client.IncompleteRead(b"ab")

    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="IncompleteRead"):
        fb.download(SPEC["path"], dest, SHA, opener=opener, sleep=lambda s: None)
    assert leftovers(tmp_path) == []


def test_archive_extraction_refuses_hardlinks_and_devices(tmp_path):
    for kind in (tarfile.LNKTYPE, tarfile.CHRTYPE):
        archive = tmp_path / f"bad{kind.decode()}.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("member")
            info.type = kind
            info.linkname = "prompts.jsonl"
            tar.addfile(info)
        out = tmp_path / f"out{kind.decode()}"
        out.mkdir()
        with pytest.raises(fb.FetchUnavailable, match="unsafe member"):
            fb.safe_extract(archive, out)
        assert list(out.iterdir()) == []


def test_cli_fetch_without_a_secret_warns_and_exits_zero_but_verify_fails(monkeypatch, capsys):
    monkeypatch.delenv("ARTIFACTORY_TOKEN")
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py"])
    fb.main()
    captured = capsys.readouterr()
    assert (
        "WARNING: could not fetch benchmark artifacts" in captured.err
        and "export" not in captured.out
    )
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", "verify"])
    with pytest.raises(SystemExit) as exit_info:
        fb.main()
    assert exit_info.value.code == 1


def test_archive_rewrites_prompts_with_absolute_image_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "cache" / "grp" / "set.tar.gz"
    archive.parent.mkdir(parents=True)
    _tar(
        archive,
        [
            (
                "prompts.jsonl",
                b'{"prompt": "p", "image_files": ["img/a.png", "img/b.png"]}\n',
                "file",
            ),
            ("img/a.png", b"a", "file"),
            ("img/b.png", b"b", "file"),
        ],
    )
    dataset, media = fb.materialize(archive, ARCHIVE_SPEC)
    import json

    (row,) = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert (
        row["image_files"] == [str(media / "img/a.png"), str(media / "img/b.png")]
        and row["prompt"] == "p"
    )


# --- credentials -------------------------------------------------------------------------------


def test_the_token_is_sent_as_a_bearer_header_with_a_timeout(monkeypatch):
    seen = {}

    class Opener:
        def open(self, request, timeout=None):
            seen["auth"] = request.get_header("Authorization")
            seen["timeout"] = timeout
            return FakeResponse(b"")

    monkeypatch.setattr(fb.urllib.request, "build_opener", lambda *handlers: Opener())
    fb._open_url("https://store.invalid/x", "tok")
    assert (
        seen == {"auth": "Bearer tok", "timeout": fb.DOWNLOAD_TIMEOUT_S}
        and fb.DOWNLOAD_TIMEOUT_S > 0
    )


def test_a_token_with_a_trailing_newline_is_stripped(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "tok-secret\n")
    tokens = []
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    fb.download(
        SPEC["path"],
        dest,
        SHA,
        opener=lambda url, token: tokens.append(token) or FakeResponse(PAYLOAD),
    )
    assert tokens == ["tok-secret"]


def test_a_control_character_in_the_token_is_refused_without_echoing_it(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("ARTIFACTORY_TOKEN", "tok-sec\nret")
    with pytest.raises(fb.FetchUnavailable) as err:
        fb.download(SPEC["path"], tmp_path / "x", SHA, opener=opener_for(PAYLOAD))
    assert "ARTIFACTORY_TOKEN contains a control character" in str(
        err.value
    ) and "tok-sec" not in str(err.value)


def test_a_header_value_error_from_the_stack_is_not_echoed(tmp_path):
    def opener(url, token):
        raise ValueError("Invalid header value b'Bearer tok-secret'")

    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable) as err:
        fb.download(SPEC["path"], dest, SHA, opener=opener, sleep=lambda s: None)
    assert "tok-secret" not in str(err.value) and "ValueError" in str(err.value)


def test_a_plain_http_endpoint_is_refused_unless_the_store_allows_it(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "http://store.invalid")
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    with pytest.raises(fb.FetchUnavailable, match="not an https endpoint"):
        fb.download(SPEC["path"], dest, SHA, opener=opener_for(PAYLOAD))
    monkeypatch.setitem(fb.STORE, "allow_insecure", True)
    fb.download(SPEC["path"], dest, SHA, opener=opener_for(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_a_trailing_slash_on_the_base_url_makes_no_double_slash(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTORY_BASE_URL", "https://store.invalid///")
    seen = []
    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    fb.download(SPEC["path"], dest, SHA, opener=opener_for(PAYLOAD, seen))
    assert seen[0][0] == "https://store.invalid/artifactory/repo/datasets/grp/trace.jsonl"


def test_part_files_are_unique_per_download(tmp_path):
    import re

    dest = tmp_path / "out" / "trace.jsonl"
    dest.parent.mkdir()
    names = []

    class Spying(FakeResponse):
        def read(self, n=-1):
            names.extend(p.name for p in dest.parent.glob("*.part"))
            return super().read(n)

    for _ in range(2):
        fb.download(SPEC["path"], dest, SHA, opener=lambda url, token: Spying(PAYLOAD))
        dest.unlink()
    names = sorted(set(names))
    assert len(names) == 2
    assert all(re.fullmatch(rf"trace\.jsonl\.{os.getpid()}-[0-9a-f]{{8}}\.part", n) for n in names)


# --- locking and the cache ---------------------------------------------------------------------


def _locked_elsewhere(path):
    import fcntl

    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def test_file_lock_really_holds_the_lock(tmp_path):
    target = tmp_path / "cache" / "f.jsonl"
    assert not _locked_elsewhere(target) if target.parent.exists() else True
    with fb.file_lock(target):
        assert _locked_elsewhere(target)
    assert not _locked_elsewhere(target)


def test_extraction_and_download_hold_the_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "cache" / "grp" / "set.tar.gz"
    archive.parent.mkdir(parents=True)
    _tar(archive, [("prompts.jsonl", b'{"image_files": []}\n', "file")])
    held = []
    real = fb._build_extraction

    def spy(*args, **kwargs):
        held.append(_locked_elsewhere(archive))
        return real(*args, **kwargs)

    monkeypatch.setattr(fb, "_build_extraction", spy)
    fb.materialize(archive, ARCHIVE_SPEC)
    assert held == [True]

    during = []

    def opener(url, token):
        during.append(_locked_elsewhere(tmp_path / "cache" / SPEC["path"]))
        return FakeResponse(PAYLOAD)

    fb.ensure_cached(SPEC, tmp_path / "cache", opener=opener)
    assert during == [True]


def test_a_digest_note_with_the_wrong_size_is_not_trusted(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_bytes(b"corrupt")
    fb._digest_note(path).write_text(
        json.dumps({"size": 999, "mtime_ns": path.stat().st_mtime_ns, "sha256": SHA})
    )
    assert fb.cache_state(path, SHA) == "corrupt"
    path.write_bytes(PAYLOAD)
    fb._digest_note(path).write_text(
        json.dumps(
            {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": "0" * 64}
        )
    )
    assert fb.cache_state(path, SHA) == "corrupt"


def test_read_only_commands_write_nothing(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    for var, spec in fb.ARTIFACTS.items():
        path = cache / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    before = sorted(p.name for p in cache.rglob("*"))
    for command in (["env"], ["verify"], ["status"]):
        monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", *command])
        with contextlib.suppress(SystemExit):
            fb.main()
    assert sorted(p.name for p in cache.rglob("*")) == before


def test_verify_extracts_only_when_asked(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    spec = fb.ARCHIVES["SPYRE_VISION_DATASET"]
    archive = cache / spec["path"]
    archive.parent.mkdir(parents=True)
    _tar(archive, [("prompts.jsonl", b'{"prompt": "p", "image_files": []}\n', "file")])
    monkeypatch.setitem(spec, "sha256", fb.sha256_of(archive))
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(cache))
    for var, other in fb.ARTIFACTS.items():
        monkeypatch.setenv(var, str(archive))  # presets: only the archive is under test
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", "verify"])
    fb.main()
    out = fb.extract_dir(cache, spec)
    assert not out.exists() and "SPYRE_VISION_DATASET" not in capsys.readouterr().out
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", "verify", "--materialize"])
    fb.main()
    assert (out / "dataset.jsonl").is_file() and "SPYRE_VISION_DATASET" in capsys.readouterr().out


def test_exported_values_are_shell_quoted(tmp_path, monkeypatch, capsys):
    odd = tmp_path / "my dir; echo hi" / "mine.jsonl"
    odd.parent.mkdir()
    odd.write_text("{}")
    monkeypatch.setenv("SPYRE_AIOPS_DATASET", str(odd))
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", "env"])
    fb.main()
    line = next(x for x in capsys.readouterr().out.splitlines() if "SPYRE_AIOPS_DATASET" in x)
    import shlex

    assert shlex.split(line) == ["export", f"SPYRE_AIOPS_DATASET={odd}"]


def test_a_preset_that_is_not_a_file_is_ignored(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPYRE_AIOPS_DATASET", str(tmp_path / "does-not-exist.jsonl"))
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py", "env"])
    fb.main()
    line = next(x for x in capsys.readouterr().out.splitlines() if "SPYRE_AIOPS_DATASET" in x)
    assert "does-not-exist" not in line and fb.ARTIFACTS["SPYRE_AIOPS_DATASET"]["path"] in line


def test_an_os_error_while_fetching_is_a_warning_not_a_crash(tmp_path, monkeypatch, capsys):
    from contextlib import contextmanager

    @contextmanager
    def broken(path):
        raise PermissionError("read-only cache")
        yield

    monkeypatch.setattr(fb, "file_lock", broken)
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py"])
    fb.main()
    err = capsys.readouterr().err
    assert "WARNING: could not fetch benchmark artifacts" in err and "read-only cache" in err


# --- extraction is all or nothing --------------------------------------------------------------


def _vision_archive(tmp_path, monkeypatch, members=None):
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(tmp_path / "cache"))
    archive = tmp_path / "cache" / "grp" / "set.tar.gz"
    archive.parent.mkdir(parents=True)
    row = b'{"prompt": "p", "image_files": ["img/a.png"]}\n'
    _tar(archive, members or [("prompts.jsonl", row, "file"), ("img/a.png", b"png", "file")])
    return archive


def test_a_failure_during_extraction_publishes_nothing_and_a_retry_succeeds(tmp_path, monkeypatch):
    archive = _vision_archive(tmp_path, monkeypatch)
    out = fb.extract_dir(tmp_path / "cache", ARCHIVE_SPEC)
    real = fb.safe_extract

    def disk_full(arc, dest):
        real(arc, dest)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fb, "safe_extract", disk_full)
    with pytest.raises(OSError):
        fb.materialize(archive, ARCHIVE_SPEC)
    assert not out.exists() and not list(out.parent.glob(".*tmp"))
    monkeypatch.setattr(fb, "safe_extract", real)
    dataset, media = fb.materialize(archive, ARCHIVE_SPEC)
    assert dataset.is_file() and (media / "img" / "a.png").is_file()


def test_a_killed_extraction_leftover_is_rebuilt_not_accepted(tmp_path, monkeypatch):
    archive = _vision_archive(tmp_path, monkeypatch)
    out = fb.extract_dir(tmp_path / "cache", ARCHIVE_SPEC)
    out.mkdir(parents=True)
    (out / "prompts.jsonl").write_text('{"prompt": "trunc')  # truncated, no marker, no images
    (out / "dataset.jsonl").write_text("{}")
    dataset, media = fb.materialize(archive, ARCHIVE_SPEC)
    assert (media / "img" / "a.png").is_file() and (media / fb._COMPLETE).is_file()
    assert json.loads(dataset.read_text().splitlines()[0])["prompt"] == "p"


def test_a_marker_for_another_archive_is_rebuilt(tmp_path, monkeypatch):
    archive = _vision_archive(tmp_path, monkeypatch)
    fb.materialize(archive, ARCHIVE_SPEC)
    newer = dict(ARCHIVE_SPEC, sha256="cd" * 32)
    out = fb.extract_dir(tmp_path / "cache", newer)
    (out / "img" / "a.png").unlink()
    fb.materialize(archive, newer)
    assert (out / "img" / "a.png").is_file()


def test_a_missing_image_fails_the_extraction(tmp_path, monkeypatch):
    archive = _vision_archive(
        tmp_path,
        monkeypatch,
        [("prompts.jsonl", b'{"prompt": "p", "image_files": ["img/gone.png"]}\n', "file")],
    )
    with pytest.raises(fb.FetchUnavailable, match="image gone.png is missing"):
        fb.materialize(archive, ARCHIVE_SPEC)
    assert not fb.extract_dir(tmp_path / "cache", ARCHIVE_SPEC).exists()


def test_a_broken_dataset_only_affects_its_own_variable(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    spec = fb.ARCHIVES["SPYRE_VISION_DATASET"]
    archive = cache / spec["path"]
    archive.parent.mkdir(parents=True)
    _tar(archive, [("prompts.jsonl", b"not json at all\n", "file")])
    monkeypatch.setitem(spec, "sha256", fb.sha256_of(archive))
    monkeypatch.setenv("SPYRE_BENCH_DATA_DIR", str(cache))
    trace = cache / fb.ARTIFACTS["SPYRE_CICS_DATASET"]["path"]
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_bytes(PAYLOAD)
    monkeypatch.setitem(fb.ARTIFACTS["SPYRE_CICS_DATASET"], "sha256", SHA)
    monkeypatch.delenv("ARTIFACTORY_TOKEN")  # the other datasets fail fast instead of dialing out
    monkeypatch.setattr(fb.sys, "argv", ["fetch_bench_datasets.py"])
    fb.main()
    captured = capsys.readouterr()
    assert "SPYRE_CICS_DATASET" in captured.out and "SPYRE_VISION_DATASET" not in captured.out
    assert "SPYRE_VISION_DATASET:" in captured.err


# --- definitions -------------------------------------------------------------------------------


STORE_DOC = """\
apiVersion: bench/v1alpha1
kind: Store
metadata: {name: NAME}
spec:
  type: core/artifactory
  endpoint: {env: B}
  root: {env: R}
  auth: {mode: token, refs: [{env: T}]}
  visibility: internal
"""
ASSET_DOC = """\
apiVersion: bench/v1alpha1
kind: Asset
metadata:
  name: NAME
  annotations:
    consumer.env: V_NAME
spec:
  type: fileset
  format: jsonl
  visibility: internal
  source: {store: STORE, path: p}
  integrity:
    files:
      - {path: f.jsonl, sha256: SHA}
"""
DATASET_DOC = """\
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: ds-NAME}
spec: {asset: NAME, role: trace}
"""


def _defs(tmp_path, assets, datasets, store_names=("datasets",)):
    d = tmp_path / "defs"
    d.mkdir(exist_ok=True)
    docs = [STORE_DOC.replace("NAME", name) for name in store_names]
    docs += [
        ASSET_DOC.replace("NAME", name).replace("STORE", store).replace("SHA", "a" * 64)
        for name, store in assets
    ]
    docs += [DATASET_DOC.replace("NAME", name) for name in datasets]
    (d / "all.yaml").write_text("---\n".join(docs))
    return d


def test_definitions_load_from_a_directory(tmp_path):
    d = _defs(tmp_path, [("one", "datasets")], ["one"])
    artifacts, archives, store = fb.load_definitions(d)
    assert list(artifacts) == ["V_one"] and archives == {} and store["token"] == "T"


def test_an_asset_no_dataset_reads_is_an_error(tmp_path):
    d = _defs(tmp_path, [("one", "datasets"), ("two", "datasets")], ["one"])
    with pytest.raises(RuntimeError, match="not read by any Dataset"):
        fb.load_definitions(d)


def test_assets_on_different_stores_are_an_error(tmp_path):
    d = _defs(
        tmp_path, [("one", "datasets"), ("two", "other")], ["one", "two"], ("datasets", "other")
    )
    with pytest.raises(RuntimeError, match="share one Store"):
        fb.load_definitions(d)


def test_no_datasets_is_an_error(tmp_path):
    d = _defs(tmp_path, [], [])
    with pytest.raises(RuntimeError, match="defines no datasets"):
        fb.load_definitions(d)
