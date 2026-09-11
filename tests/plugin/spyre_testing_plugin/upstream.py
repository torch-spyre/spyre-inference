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

"""Materialize the upstream vLLM ``tests/`` tree at the pinned commit.

Two consumers:

* ``pytest_plugin``, which collects upstream tests out of the tree (opt-in, see the
  marker gate there).
* Local tests, which import upstream test helpers -- ``check_logprobs_close``,
  ``check_embeddings_close``, ``HfRunner``, ``VllmRunner`` -- rather than reimplementing
  them. Those live in vLLM's ``tests/`` tree, not the installed wheel, so reaching them
  needs the clone even when no upstream test is collected.

Deliberately free of pytest and vllm imports so a plain script can call it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

Log = Callable[[str], None]


def _stderr_log(msg: str) -> None:
    print(msg, file=sys.stderr)


def cache_root() -> Path:
    """
    Cache directory for cloned tests (persists across runs)
    """
    # Respect XDG if present, fallback to ~/.cache
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "vllm-upstream-tests"


def repo_root() -> Path:
    """This repo's root, derived from the installed-editable plugin at
    ``<root>/tests/plugin/spyre_testing_plugin/``."""
    return Path(__file__).resolve().parents[3]


def _extract_vllm_commit_from_pyproject(repo_root_dir: Path) -> str:
    """
    Extract the vLLM git reference from pyproject.toml [tool.uv.sources] section.
    Raises FileNotFoundError if pyproject.toml is missing, or KeyError
    if the expected source entry is not found.
    """
    pyproject_path = repo_root_dir / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found in {repo_root_dir}")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        vllm_source = data["tool"]["uv"]["sources"]["vllm"]
    except KeyError as e:
        raise KeyError(
            "Ensure vllm is specified with 'rev' in pyproject.toml"
            f" [tool.uv.sources]: missing key {e}"
        ) from e

    # Handle both a single source dict and a list of sources (e.g. index + git fallback)
    if isinstance(vllm_source, list):
        for source in vllm_source:
            if isinstance(source, dict) and "git" in source and "rev" in source:
                return source["rev"]
    elif isinstance(vllm_source, dict) and "git" in vllm_source and "rev" in vllm_source:
        return vllm_source["rev"]

    raise KeyError("Ensure vllm is specified with 'rev' in pyproject.toml [tool.uv.sources]")


def resolve_vllm_commit(repo_root_dir: Path) -> str:
    """
    Resolve the vLLM git reference to use for cloning upstream tests.
    Priority: VLLM_COMMIT env var > pyproject.toml > error
    """
    # Allow env var override for testing/CI
    env_commit = os.environ.get("VLLM_COMMIT", "").strip()
    if env_commit:
        if not re.match(r"^(?:[0-9a-f]{7,40}|v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)$", env_commit):
            raise ValueError(f"Invalid VLLM_COMMIT format: {env_commit}")
        return env_commit

    # Extract from pyproject.toml
    return _extract_vllm_commit_from_pyproject(repo_root_dir)


def _run(cmd: list[str], cwd: Path | None = None, max_retries: int = 3) -> None:
    """Run command with optional retries for network operations."""
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                raise


def ensure_repo_at_commit(
    repo_dir: Path,
    url: str,
    commit: str,
    sparse_paths: list[str],
    log: Log = _stderr_log,
) -> Path:
    """
    Ensure repo cloned at 'repo_dir/commit' with sparse checkout of 'sparse_paths'.
    Returns the path to the working tree at that commit.
    """
    # We create a separate worktree per commit to allow co-existence of different commits
    base_dir = repo_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    git_dir = base_dir / "repo.git"

    if not git_dir.exists():
        _run(["git", "init", "--bare", str(git_dir)])

    # Prepare a worktree dir per commit
    wt_dir = base_dir / f"worktree-{commit[:12]}"
    if wt_dir.exists():
        log(f"[vllm-upstream] Using cached worktree at {wt_dir}")
        return wt_dir

    # Create temp dir to set up the sparse worktree then move into place atomically
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as td:
        td_path = Path(td)

        # Ensure origin remote exists and points to the correct URL
        result = subprocess.run(
            ["git", "--git-dir", str(git_dir), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Origin doesn't exist - add it
            _run(["git", "--git-dir", str(git_dir), "remote", "add", "origin", url])
        elif result.stdout.strip() != url:
            # Origin exists but points to different URL - update it
            log(f"[vllm-upstream] Updating origin URL: {result.stdout.strip()} -> {url}")
            _run(["git", "--git-dir", str(git_dir), "remote", "set-url", "origin", url])

        # Determine if commit is a tag (starts with 'v' and matches semver pattern) or a SHA
        is_tag = re.match(r"^v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$", commit)

        if is_tag:
            log(f"[vllm-upstream] Fetching tag {commit} from {url}")
            # For tags, fetch the tag reference
            _run(
                [
                    "git",
                    "--git-dir",
                    str(git_dir),
                    "fetch",
                    "--depth=1",
                    "origin",
                    f"refs/tags/{commit}:refs/tags/{commit}",
                ]
            )
        else:
            log(f"[vllm-upstream] Fetching commit {commit[:12]} from {url}")
            # For commit SHAs, fetch the commit directly
            _run(["git", "--git-dir", str(git_dir), "fetch", "--depth=1", "origin", commit])

        # Create a new worktree at temp
        # For tags, use the full tag reference; for commits, use the commit SHA directly
        worktree_ref = f"refs/tags/{commit}" if is_tag else commit
        _run(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "worktree",
                "add",
                "--detach",
                str(td_path),
                worktree_ref,
            ]
        )

        # Enable sparse checkout at the worktree
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=td_path)
        _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=td_path)

        # Ensure we're exactly at the commit (detached HEAD)
        _run(["git", "checkout", "--detach", commit], cwd=td_path)

        # Atomically move into place
        td_path.rename(wt_dir)

    return wt_dir


def prepare_upstream_tests_dir(repo_root_dir: Path, log: Log = _stderr_log) -> Path:
    """Clone vLLM to cache and return path to tests directory."""
    commit = resolve_vllm_commit(repo_root_dir)
    wt_dir = ensure_repo_at_commit(
        repo_dir=cache_root(),
        url=os.environ.get("VLLM_REPO_URL", "https://github.com/vllm-project/vllm"),
        commit=commit,
        sparse_paths=["tests"],
        log=log,
    )
    tests_dir = wt_dir / "tests"
    if not tests_dir.is_dir():
        raise RuntimeError(f"Upstream tests directory not found at {tests_dir}")
    return tests_dir


def apply_temp_upstream_code_edits(upstream_tests_dir: Path) -> None:
    """Apply small code edits to the upstream tests directory before importing.

    These should be _temporary_ edits to source code for vllm tests while we work to make them more
    portable. This should only be used where mocking is not possible or too cumbersome.
    """

    # Mocking out torch.device seems impossible to do (at least multiple rounds of Bob and Claude
    # were unsuccessful). So we patch the source code to change the hardcoded
    # `torch.device("cuda:0")` to `torch.device("cpu")`.
    hardcoded_cuda_test_path = (
        upstream_tests_dir / "v1" / "attention" / "test_attention_backends.py"
    )
    with open(hardcoded_cuda_test_path) as f:
        content = f.read()
    content = content.replace('torch.device("cuda:0")', 'torch.device("cpu")')
    with open(hardcoded_cuda_test_path, "w") as f:
        f.write(content)


def ensure_upstream_tests_importable(log: Log = _stderr_log) -> Path:
    """Make the pinned vLLM ``tests`` tree importable, returning its path.

    Upstream's test modules import each other absolutely (``from tests.models.utils
    import ...``), so the tree has to own the top-level name ``tests``. It does: upstream's
    ``tests/`` is a regular package, and a regular package wins over this repo's
    ``__init__.py``-less namespace directory of the same name whatever the sys.path order --
    so nothing local may import ``tests.*`` (sibling test modules import each other as
    top-level modules instead, since none of those directories is a package either).
    """
    tests_dir = prepare_upstream_tests_dir(repo_root(), log=log)
    upstream_root = str(tests_dir.parent)
    if upstream_root not in sys.path:
        sys.path.append(upstream_root)
    return tests_dir
