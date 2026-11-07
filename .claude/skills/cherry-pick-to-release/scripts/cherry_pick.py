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

"""Helpers for the cherry-pick-to-release skill. Run from the repository root.

message  compose the release commit message and leave it in MERGE_MSG for `git commit -F`
verify   check that the staged change is the original commit's change and nothing else
env      check that the test interpreter runs this checkout on the stack the branch locks
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

TRAILER = re.compile(r"^[A-Za-z][A-Za-z0-9-]*: \S")
PR_SUFFIX = re.compile(r"\(#(\d+)\)$")
MARKER = re.compile(r"^(<{7}|>{7}|\|{7})( |$)|^={7}$")
MAX_LINES_PER_FILE = 20


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def resolve(commit: str) -> str:
    return git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()


def git_path(name: str) -> Path:
    return Path(git("rev-parse", "--git-path", name).strip())


# --------------------------------------------------------------------------- message


def split_trailers(body: str) -> tuple[str, str]:
    """Split off the last paragraph when every line in it is a `Key: value` trailer."""
    body = body.strip("\n")
    head, _, last = body.rpartition("\n\n")
    lines = last.splitlines()
    if lines and all(TRAILER.match(line) for line in lines):
        return head.rstrip("\n"), last
    return body, ""


def cmd_message(args: argparse.Namespace) -> int:
    sha = resolve(args.commit)
    subject = git("log", "-1", "--format=%s", sha).strip()
    body, trailers = split_trailers(git("log", "-1", "--format=%b", sha))
    repo = f"https://github.com/{args.repo}"
    pr = PR_SUFFIX.search(subject)

    ref = f"#{pr.group(1)} ({sha[:7]})" if pr else sha[:7]
    intro = [f"Cherry-pick of {ref} from main onto `{args.into}`."]
    if pr:
        intro.append(f"Original PR: {repo}/pull/{pr.group(1)}")
    intro.append(f"Original commit: {repo}/commit/{sha}")
    if args.note:
        notes = f"Cherry-pick notes for {args.into}:\n" + "\n".join(f"- {n}" for n in args.note)
    else:
        notes = f"Applied cleanly; the staged change is identical to {sha[:7]}."

    parts = [
        f"[{args.into}] {subject}",
        "\n".join(intro),
        body,
        notes,
        f"(cherry picked from commit {sha})",
        trailers,
    ]
    message = "\n\n".join(p for p in parts if p) + "\n"

    merge_msg = git_path("MERGE_MSG")
    backup = git_path(f"cherry-pick-{args.into.replace('/', '-')}-{sha[:7]}.msg")
    for path in (merge_msg, backup):
        path.write_text(message)
    print(message, end="")
    print(f"\n# written to {merge_msg} (backup: {backup})", file=sys.stderr)
    print(f'# commit later with: git commit -s -F "{merge_msg}"', file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- verify


class Entry(NamedTuple):
    status: str
    old_mode: str
    new_mode: str
    old_blob: str
    new_blob: str


def raw_diff(*rev_args: str) -> dict[str, Entry]:
    out = git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", *rev_args)
    fields = out.split("\0")
    entries: dict[str, Entry] = {}
    for meta, path in zip(fields[::2], fields[1::2]):
        old_mode, new_mode, old_blob, new_blob, status = meta.lstrip(":").split(" ")
        entries[path] = Entry(status[0], old_mode, new_mode, old_blob, new_blob)
    return entries


def changed_lines(path: str, *rev_args: str) -> tuple[Counter[str], Counter[str]]:
    """Added and removed lines of `path`, ignoring context, position and hunk headers."""
    out = git("diff", "-U0", "--no-renames", "--no-color", "--no-ext-diff", *rev_args, "--", path)
    added: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    in_hunk = False
    for line in out.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line[:1] == "+":
            added[line[1:]] += 1
        elif in_hunk and line[:1] == "-":
            removed[line[1:]] += 1
    return added, removed


def show_lines(label: str, sign: str, lines: Counter[str]) -> list[str]:
    flat = list(lines.elements())
    shown = [f"{label} {sign} {line[:110]}" for line in flat[:MAX_LINES_PER_FILE]]
    if len(flat) > MAX_LINES_PER_FILE:
        shown.append(f"{label}   ... {len(flat) - MAX_LINES_PER_FILE} more")
    return shown


def cmd_verify(args: argparse.Namespace) -> int:
    sha = resolve(args.commit)
    parent = resolve(f"{sha}^{args.parent}")
    base = resolve(args.base)
    commit_revs = (parent, sha)
    staged_revs = ("--cached", base)

    blocking: list[str] = []
    unmerged = git("diff", "--name-only", "--diff-filter=U").split()
    if unmerged:
        blocking.append("unmerged (conflicts not resolved and added): " + ", ".join(unmerged))
    unstaged = sorted(set(git("diff", "--name-only").split()) - set(unmerged))
    if unstaged:
        blocking.append("tracked files with unstaged changes: " + ", ".join(unstaged))

    commit = raw_diff(*commit_revs)
    staged = raw_diff(*staged_revs)
    rows: list[tuple[str, str, list[str]]] = []
    for path in sorted(commit.keys() | staged.keys()):
        c, s = commit.get(path), staged.get(path)
        c_added, c_removed = changed_lines(path, *commit_revs) if c else (Counter(), Counter())
        s_added, s_removed = changed_lines(path, *staged_revs) if s else (Counter(), Counter())
        if any(MARKER.match(line) for line in s_added - c_added):
            blocking.append(f"conflict markers staged in {path}")
        if s is None:
            rows.append(("MISSING", path, ["the commit changes this file; nothing is staged"]))
            continue
        if c is None:
            rows.append(("EXTRA", path, ["staged, but the commit does not touch this file"]))
            continue

        details = []
        if c.status != s.status:
            details.append(f"status {c.status} in the commit, {s.status} staged")
        if c.new_mode != s.new_mode:
            details.append(f"mode {c.new_mode} in the commit, {s.new_mode} staged")
        if not details and c.old_blob == s.old_blob and c.new_blob == s.new_blob:
            rows.append(("identical", path, []))
            continue
        same_lines = (c_added, c_removed) == (s_added, s_removed)
        if not details and same_lines and (c_added or c_removed):
            if c.old_blob != s.old_blob:
                rows.append(("same change", path, ["release's text differs elsewhere in the file"]))
                continue
            details.append("same lines as the commit, but placed differently")
        if not (c_added or c_removed or s_added or s_removed):
            details.append("no textual diff to compare (binary?): check by hand")
        details += show_lines("commit only", "+", c_added - s_added)
        details += show_lines("staged only", "+", s_added - c_added)
        details += show_lines("commit only", "-", c_removed - s_removed)
        details += show_lines("staged only", "-", s_removed - c_removed)
        rows.append(("DIFFERS", path, details))

    head = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    print(f"{sha[:8]} ({len(commit)} files) vs staged on {base[:8]} [{head}] ({len(staged)} files)")
    print()
    width = max((len(r[0]) for r in rows), default=0)
    for status, path, details in rows:
        print(f"  {status:<{width}}  {path}")
        for detail in details:
            print(f"      {detail}")
    print()
    merge_msg = git_path("MERGE_MSG")
    print(f"commit message: {merge_msg} {'present' if merge_msg.exists() else 'MISSING'}")
    for problem in blocking:
        print(f"BLOCKING: {problem}")
    differing = [r for r in rows if r[0] in ("MISSING", "EXTRA", "DIFFERS")]
    if blocking:
        return 2
    if differing:
        print(
            f"{len(differing)} file(s) differ from the commit: fix each one, or explain it in a "
            "cherry-pick note"
        )
        return 1
    print("The staged change is the commit's change and nothing else.")
    return 0


# --------------------------------------------------------------------------- env


def read_file(root: Path, ref: str | None, name: str) -> str:
    return git("show", f"{ref}:{name}") if ref else (root / name).read_text()


def locked_versions(lock: dict, name: str) -> list[tuple[str, str | None, str | None]]:
    """(version, git commit, git rev) of every uv.lock entry for `name`."""
    entries = []
    for pkg in lock.get("package", []):
        if pkg["name"] == name:
            url, _, commit = pkg.get("source", {}).get("git", "").partition("#")
            entries.append(
                (pkg.get("version", "?"), commit or None, url.partition("?rev=")[2] or None)
            )
    return entries


def cmd_env(args: argparse.Namespace) -> int:
    import importlib.metadata as md
    import importlib.util
    import tomllib

    root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
    pyproject = tomllib.loads(read_file(root, args.ref, "pyproject.toml"))
    lock = tomllib.loads(read_file(root, args.ref, "uv.lock"))
    project = pyproject["project"]
    results: list[tuple[str, bool, str]] = []

    spec = importlib.util.find_spec(project["name"].replace("-", "_"))
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    results.append(
        (
            "import",
            origin is not None and root in origin.parents,
            f"{project['name']} -> {origin} (must be inside {root})",
        )
    )

    spec = importlib.util.find_spec("tests")
    shadow = spec.origin if spec and spec.origin else None
    if shadow and root in Path(shadow).resolve().parents:
        shadow = None
    results.append(
        (
            "tests package",
            shadow is None,
            f"another top-level `tests` package shadows ours: {shadow}"
            if shadow
            else "not shadowed",
        )
    )

    groups = dict(project.get("entry-points", {}))
    groups["console_scripts"] = project.get("scripts", {})
    declared = {(g, n, v) for g, eps in groups.items() for n, v in eps.items()}
    try:
        installed = {
            (e.group, e.name, e.value) for e in md.distribution(project["name"]).entry_points
        }
    except md.PackageNotFoundError:
        installed = set()
    stale = [
        f"{'missing' if ep in declared else 'stale'} {ep[0]}:{ep[1]}"
        for ep in sorted(declared ^ installed)
    ]
    results.append(
        (
            "entry points",
            not stale,
            "; ".join(stale) + " (reinstall: uv pip install --no-deps -e .)"
            if stale
            else "installed metadata matches pyproject.toml",
        )
    )

    for name in args.packages:
        locked = locked_versions(lock, name)
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            results.append((name, False, "not installed"))
            continue
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        ok = any((commit == c) if (commit and c) else (dist.version == v) for v, c, _ in locked)
        want = (
            ", ".join(
                f"{v}" + (f" @ {c[:7]}" if c else "") + (f" ({r})" if r else "")
                for v, c, r in locked
            )
            or "nothing (not in uv.lock)"
        )
        have = dist.version + (f" @ {commit[:7]}" if commit else "")
        results.append((name, ok, f"installed {have}; uv.lock locks {want}"))

    source = f"{args.ref}'s" if args.ref else "the working tree's"
    print(f"{sys.executable} vs {source} pyproject.toml / uv.lock\n")
    for check, ok, detail in results:
        print(f"  {'OK' if ok else 'MISMATCH':8}  {check:14}  {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("message", help="compose the commit message into MERGE_MSG")
    p.add_argument("commit")
    p.add_argument("--into", required=True, help="the release branch to pick into")
    p.add_argument("--repo", default="torch-spyre/spyre-inference")
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("--clean", action="store_true", help="the pick applied with no changes")
    how.add_argument("--note", action="append", help="one cherry-pick note; repeatable")
    p.set_defaults(func=cmd_message)

    p = sub.add_parser("verify", help="compare the staged change with the commit")
    p.add_argument("commit")
    p.add_argument("--base", default="HEAD", help="what the change is staged on (default HEAD)")
    p.add_argument("--parent", default="1", help="parent to diff a merge commit against")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("env", help="check the test interpreter; run it with that interpreter")
    p.add_argument("--ref", help="read pyproject.toml / uv.lock from this ref, not the tree")
    p.add_argument("--packages", nargs="+", default=["torch-spyre", "vllm", "torch"])
    p.set_defaults(func=cmd_env)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
