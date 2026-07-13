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

"""
pytest11 plugin for spyre-inference.

This plugin integrates upstream vLLM tests with spyre-inference
and filtering via a declarative YAML config (upstream_tests.yaml).

Hook Execution Order
---------------------
1. pytest_configure (tryfirst)
    - Loads Spyre plugins (custom ops, platform)
    - Detects local vLLM repo OR clones to ~/.cache/vllm-upstream-tests/
    - Injects test paths into pytest collection

2. pytest_generate_tests (tryfirst)
    - Overrides test parameters from YAML (e.g., num_tokens: [1, 16])
    - Must run during collection before parametrization finalizes

3. pytest_collection_modifyitems
    - Applies skip/xfail markers based on YAML allow_list/block_list
    - Applies tag markers for filtering (e.g., pytest -m rmsnorm)

4. pytest_fixture_setup (tryfirst)
    - Overrides default_vllm_config fixture with Spyre-specific config

Environment Variables
---------------------
SKIP_UPSTREAM_TESTS     Set to 1/true/yes to skip upstream test cloning
UPSTREAM_TESTS_PATHS    Comma-separated paths (default: auto from YAML)
VLLM_COMMIT             Override vLLM commit (default: from pyproject.toml)
VLLM_REPO_URL           Override vLLM repo URL
XDG_CACHE_HOME          Base cache directory (default: ~/.cache)
"""

from __future__ import annotations

import fnmatch
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import torch.distributed as dist
import torch.testing

from pathlib import Path

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum
import yaml

from spyre_testing_plugin.models import (
    AllowEntry,
    BlockEntry,
    FileConfig,
    ParamAllow,
    ParamOverride,
    ParamSkip,
    Tolerances,
    UpstreamTestConfig,
)

_YAML_FILENAME = "upstream_tests.yaml"
_YAML_PATH = Path(__file__).parent / _YAML_FILENAME

# Global terminal reporter for pytest-aware logging
_terminal_reporter = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str):
    """Log message to pytest terminal reporter if available.
    This allows logs to be printed from this file even when pytest is capturing output.
    """
    if _terminal_reporter:
        _terminal_reporter.write_line(msg)
    else:
        # Fallback to stderr when terminal reporter not available
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# YAML Config Loading
# ---------------------------------------------------------------------------


def _load_upstream_config() -> UpstreamTestConfig:
    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)
    if not raw or "tests" not in raw or "files" not in raw["tests"]:
        raise RuntimeError(
            f'Invalid YAML in {_YAML_PATH}: missing "tests" or "tests.files" sections'
        )
    return _parse_config(raw["tests"])


def _parse_config(raw_tests: dict) -> UpstreamTestConfig:
    files: list[FileConfig] = []
    for file_entry in raw_tests.get("files", []):
        allow_list: list[AllowEntry] = []
        for allow in file_entry.get("allow_list", []):
            params_section = allow.get("params", {})
            param_skips = [
                ParamSkip(param_name=k, values=frozenset(v))
                for k, v in params_section.get("skip", {}).items()
            ]
            param_allows = [
                ParamAllow(param_name=k, values=frozenset(v))
                for k, v in params_section.get("allow", {}).items()
            ]
            param_overrides = [
                ParamOverride(param_name=k, values=tuple(v))
                for k, v in params_section.get("override", {}).items()
            ]
            tolerances_section = allow.get("tolerances")
            tolerances = None
            if tolerances_section:
                tolerances = Tolerances(
                    atol=float(tolerances_section.get("atol", 1e-3)),
                    rtol=float(tolerances_section.get("rtol", 1e-3)),
                )
            allow_list.append(
                AllowEntry(
                    test=allow["test"],
                    mode=allow.get("mode", "mandatory_pass"),
                    tags=tuple(allow.get("tags", [])),
                    param_skips=tuple(param_skips),
                    param_allows=tuple(param_allows),
                    param_overrides=tuple(param_overrides),
                    tolerances=tolerances,
                    fixture_names=tuple(allow.get("fixture_names", ())),
                )
            )
        block_list = [BlockEntry(test=b["test"]) for b in file_entry.get("block_list", [])]
        files.append(
            FileConfig(
                rel_path=file_entry["rel_path"],
                allow_list=tuple(allow_list),
                block_list=tuple(block_list),
            )
        )
    return UpstreamTestConfig(files=tuple(files))


_UPSTREAM_CONFIG: UpstreamTestConfig = _load_upstream_config()


def _get_paths_from_yaml() -> str:
    """Extract test paths from upstream_tests.yaml rel_path entries.

    Uses exact rel_path (file or folder), e.g.:
      "tests/kernels/core/test_layernorm.py" -> "kernels/core/test_layernorm.py"
      "tests/models/" -> "models/"
    """
    paths = []
    for fc in _UPSTREAM_CONFIG.files:
        p = Path(fc.rel_path)
        # Strip "tests/" prefix if present
        if p.parts and p.parts[0] == "tests":
            paths.append(str(Path(*p.parts[1:])))
        else:
            paths.append(str(p))
    return ",".join(paths)


# ---------------------------------------------------------------------------
# vLLM Repository Cloning
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    """
    Cache directory for cloned tests (persists across runs)
    """
    # Respect XDG if present, fallback to ~/.cache
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "vllm-upstream-tests"


def _extract_vllm_commit_from_pyproject() -> str:
    """
    Extract the vLLM git reference from pyproject.toml [tool.uv.sources] section.
    Raises FileNotFoundError if pyproject.toml is missing, or KeyError
    if the expected source entry is not found.
    """
    repo_root_dir = Path(__file__).parent.parent.parent.parent
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


def _resolve_vllm_commit() -> str:
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
    return _extract_vllm_commit_from_pyproject()


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


def _ensure_repo_at_commit(repo_dir: Path, url: str, commit: str, sparse_paths: list[str]) -> Path:
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
        _log(f"[vllm-upstream] Using cached worktree at {wt_dir}")
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
            _log(f"[vllm-upstream] Updating origin URL: {result.stdout.strip()} -> {url}")
            _run(["git", "--git-dir", str(git_dir), "remote", "set-url", "origin", url])

        # Determine if commit is a tag (starts with 'v' and matches semver pattern) or a SHA
        is_tag = re.match(r"^v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$", commit)

        if is_tag:
            _log(f"[vllm-upstream] Fetching tag {commit} from {url}")
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
            _log(f"[vllm-upstream] Fetching commit {commit[:12]} from {url}")
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


def _prepare_upstream_tests_dir() -> Path:
    """Clone vLLM to cache and return path to tests directory."""
    commit = _resolve_vllm_commit()
    cache_root = _cache_root()
    wt_dir = _ensure_repo_at_commit(
        repo_dir=cache_root,
        url=os.environ.get("VLLM_REPO_URL", "https://github.com/vllm-project/vllm"),
        commit=commit,
        sparse_paths=["tests"],
    )
    tests_dir = wt_dir / "tests"
    if not tests_dir.is_dir():
        raise RuntimeError(f"Upstream tests directory not found at {tests_dir}")
    return tests_dir


def _temp_upstream_code_edits(upstream_tests_dir: Path):
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


# ---------------------------------------------------------------------------
# Pytest Hooks
# ---------------------------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Register Spyre plugins and detect/clone vLLM repo."""
    global _terminal_reporter
    _terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")

    # Set env vars BEFORE any vllm imports
    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops,spyre_inference_hf_adaptor"
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"

    # Load plugins early to register custom ops before test modules import RMSNorm
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    # Detect local vLLM repo or clone it
    rootdir = Path(config.rootdir)
    tests_dir = rootdir / "tests"
    vllm_pkg = rootdir / "vllm"

    if tests_dir.is_dir() and vllm_pkg.is_dir():
        # Running from vLLM repo itself
        config._upstream_tests_base = tests_dir
        _log("[vllm-upstream] Using local vLLM tests")
    else:
        # Not in vLLM repo - check if we should clone
        skip_upstream = os.environ.get("SKIP_UPSTREAM_TESTS", "").lower() in ("1", "true", "yes")
        if skip_upstream:
            _log("[vllm-upstream] SKIP_UPSTREAM_TESTS is set, skipping upstream test collection")
            config._upstream_tests_base = None
            return

        try:
            # Clone vLLM to cache
            upstream_tests_base = _prepare_upstream_tests_dir()
            _temp_upstream_code_edits(upstream_tests_base)
            config._upstream_tests_base = upstream_tests_base

            # Determine which test paths to inject
            paths_env = _get_paths_from_yaml()
            _log(f"[vllm-upstream] Auto-derived test paths from YAML: {paths_env}")

            if not paths_env:
                _log("[vllm-upstream] No test paths configured, skipping upstream tests")
                config._upstream_tests_base = None
                return

            # Inject test paths into pytest collection
            for rel_path in paths_env.split(","):
                rel_path = rel_path.strip()
                if not rel_path:
                    continue
                test_dir = upstream_tests_base / rel_path
                if test_dir.exists():
                    _log(f"[vllm-upstream] Including tests from: {rel_path}")
                    config.args.append(str(test_dir))
                else:
                    _log(f"[vllm-upstream] Warning: Path not found: {test_dir}")

        except Exception as e:
            raise SystemExit(f"[vllm-upstream] Failed to prepare upstream tests: {e}") from e


# ---------------------------------------------------------------------------
# YAML Filtering Helpers
# ---------------------------------------------------------------------------


def _find_file_config(test_path: Path, file_configs: dict[Path, FileConfig]) -> FileConfig | None:
    if test_path in file_configs:
        return file_configs[test_path]
    for config_path, fc in file_configs.items():
        if test_path.is_relative_to(config_path):
            return fc
    return None


def _matches_block_list(test_name: str, block_list: tuple[BlockEntry, ...]) -> bool:
    return any(fnmatch.fnmatch(test_name, e.test) for e in block_list)


def _find_allow_entry(test_name: str, allow_list: tuple[AllowEntry, ...]) -> AllowEntry | None:
    for entry in allow_list:
        if fnmatch.fnmatch(test_name, entry.test):
            return entry
    return None


def _should_skip_params(item: pytest.Item, allow_entry: AllowEntry) -> bool:
    """Check if test should be skipped based on param_skips or param_allows.

    If param_allows is specified for a parameter, only those values are allowed.
    Otherwise, param_skips is used to exclude specific values.
    """
    callspec = getattr(item, "callspec", None)
    if not callspec:
        return False

    # Check param_allows first (whitelist takes precedence)
    for pa in allow_entry.param_allows:
        if pa.param_name in callspec.params and callspec.params[pa.param_name] not in pa.values:
            # If allowlist exists for this param, skip if value is NOT in allowlist
            return True

    # Check param_skips (blacklist)
    for ps in allow_entry.param_skips:
        if ps.param_name in callspec.params and callspec.params[ps.param_name] in ps.values:
            return True

    return False


# ---------------------------------------------------------------------------
# Collection Modification
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply YAML-based filtering to upstream tests and reorder tests."""
    upstream_tests_base = getattr(config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        # Still reorder tests even if not running upstream tests
        _reorder_tests_by_name(items)
        return

    upstream_tests_base = Path(upstream_tests_base).resolve()
    upstream_repo_root = upstream_tests_base.parent
    file_configs = {
        (upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files
    }

    upstream_marker = pytest.mark.upstream

    for item in items:
        test_path = Path(item.fspath).resolve()
        if not test_path.is_relative_to(upstream_tests_base):
            continue

        item.add_marker(upstream_marker)

        fc = _find_file_config(test_path, file_configs)
        if fc is None:
            item.add_marker(pytest.mark.skip(reason=f"not in {_YAML_FILENAME}"))
            continue

        test_name = item.originalname or item.name
        if _matches_block_list(test_name, fc.block_list):
            item.add_marker(pytest.mark.skip(reason=f"blocked by {_YAML_FILENAME}"))
            continue

        allow_entry = _find_allow_entry(test_name, fc.allow_list)

        if allow_entry:
            for tag in allow_entry.tags:
                item.add_marker(getattr(pytest.mark, tag))

        if allow_entry is None:
            item.add_marker(pytest.mark.skip(reason="not in allow_list"))
            continue

        if _should_skip_params(item, allow_entry):
            item.add_marker(pytest.mark.skip(reason="param skipped"))
            continue

        if allow_entry.mode == "skip":
            item.add_marker(pytest.mark.skip(reason=f"skipped by {_YAML_FILENAME}"))
            continue

        if allow_entry.mode == "xfail":
            item.add_marker(pytest.mark.xfail(strict=False))
        elif allow_entry.mode == "xfail_strict":
            item.add_marker(pytest.mark.xfail(strict=True))

        # Store tolerances on item for fixture access
        if allow_entry.tolerances:
            item._spyre_tolerances = allow_entry.tolerances
        # Inject fixtures for tests that have fixture_names defined
        for fixture_name in allow_entry.fixture_names:
            item.fixturenames.append(fixture_name)

    # Reorder tests so that tests with "uses_subprocess" marker run first
    _reorder_tests_by_name(items)


def _reorder_tests_by_name(items: list[pytest.Item]) -> None:
    """Reorder tests so that tests with 'uses_subprocess' marker run first.

    This modifies the items list in-place using a stable sort, so tests marked
    with 'uses_subprocess' will run first while preserving the relative order
    within each group.
    """
    stable_map = {item: idx for idx, item in enumerate(items)}

    def sort_key(item: pytest.Item) -> tuple[int, int]:
        # Check if the test has the 'uses_subprocess' marker
        has_subprocess_marker = any(
            marker.name == "uses_subprocess" for marker in item.iter_markers()
        )

        # Priority 0: tests with uses_subprocess marker run first
        # Priority 1: all other tests
        priority = 0 if has_subprocess_marker else 1

        return (priority, stable_map[item])

    items.sort(key=sort_key)


def _convert_yaml_value(value):
    """Convert YAML string values to Python objects where appropriate.

    Handles torch dtype strings like "torch.half" -> torch.half.
    """
    if isinstance(value, str):
        import torch

        dtype_map = {
            "torch.half": torch.half,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float": torch.float,
            "torch.float64": torch.float64,
            "torch.double": torch.double,
        }
        if value in dtype_map:
            return dtype_map[value]
    return value


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Apply parameter overrides from YAML config."""
    upstream_tests_base = getattr(metafunc.config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    upstream_repo_root = Path(upstream_tests_base).resolve().parent
    test_path = Path(metafunc.definition.fspath).resolve()
    file_configs = {
        (upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files
    }

    fc = _find_file_config(test_path, file_configs)
    if not fc:
        return

    test_name = metafunc.definition.originalname or metafunc.definition.name
    allow_entry = _find_allow_entry(test_name, fc.allow_list)
    if not allow_entry or not allow_entry.param_overrides:
        return

    for po in allow_entry.param_overrides:
        # Support both single-name keys and comma-joined tuple-parametrize keys
        # (e.g. `@pytest.mark.parametrize("a, b, c", [(1, 2, 3), ...])`).
        sub_names = [n.strip() for n in po.param_name.split(",")]
        if any(n not in metafunc.fixturenames for n in sub_names):
            continue
        for i, marker in enumerate(metafunc.definition.own_markers):
            if marker.name != "parametrize":
                continue
            marker_names = [n.strip() for n in marker.args[0].split(",")]
            if marker_names != sub_names:
                continue
            if len(sub_names) > 1:
                new_values = [tuple(_convert_yaml_value(v) for v in row) for row in po.values]
            else:
                new_values = [_convert_yaml_value(v) for v in po.values]
            metafunc.definition.own_markers[i] = pytest.mark.parametrize(
                po.param_name, new_values
            ).mark
            break


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _spyre_session_config():
    """Session-wide plugin setup and default VllmConfig context.

    Runs once per test session to:
    1. Set env vars and load plugins
    2. Register custom ops
    3. Patch platform to OOT
    4. Enter set_current_vllm_config + set_forward_context for the session

    Individual tests no longer need to request default_vllm_config just for
    plugin initialization. Tests that need a different config can still enter
    their own set_current_vllm_config context (nesting is safe).
    """
    os.environ["VLLM_PLUGINS"] = "spyre_inference,spyre_inference_ops,spyre_inference_hf_adaptor"
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    from spyre_inference.custom_ops import register_all

    register_all()

    from vllm.platforms import PlatformEnum, current_platform

    type(current_platform)._enum = PlatformEnum.OOT

    from vllm.config import DeviceConfig, VllmConfig, ModelConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.forward_context import set_forward_context

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield


def _spyre_default_vllm_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, ModelConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.forward_context import set_forward_context

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)

    # Explicitly register custom ops
    from spyre_inference.custom_ops import register_all

    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        model_config=ModelConfig(dtype=torch.float16),
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        yield


@pytest.fixture()
def default_vllm_config(monkeypatch):
    yield from _spyre_default_vllm_config(monkeypatch)


@pytest.fixture(scope="session")
def _distributed_init(_spyre_session_config):
    """Initialize torch.distributed with gloo backend once per test session.

    Uses a FileStore so no network port is required. The process group is
    destroyed at the end of the session.
    """
    fd, store_path = tempfile.mkstemp(suffix=".store")
    os.close(fd)
    try:
        store = dist.FileStore(store_path, 1)
        dist.init_process_group(
            backend="gloo",
            store=store,
            world_size=1,
            rank=0,
        )
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(store_path):
            os.unlink(store_path)


@pytest.fixture(scope="session")
def tp_group(_distributed_init):
    """Set up a real TP=1 GroupCoordinator inside vLLM's parallel_state.

    The session-scoped _spyre_session_config fixture (autouse) ensures the
    VllmConfig context and OOT platform patch are already active.
    After the test the previous _TP value is restored.

    Tests that create vLLM linear layers should use this fixture instead of
    (or in addition to) `default_vllm_config`.
    """
    from vllm.distributed.parallel_state import GroupCoordinator
    import vllm.distributed.parallel_state as ps

    group = GroupCoordinator(
        group_ranks=[[0]],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="tp",
    )
    original_tp = ps._TP
    ps._TP = group
    yield group
    ps._TP = original_tp


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def run_tp_probe(pytestconfig):
    """Spawn one subprocess per rank running `tp_probe.py --probe NAME`.

    The probe body lives in `tests/probes/tp_probe.py`. Tests pass the
    probe name and world size; this fixture handles port allocation,
    env-rendezvous setup, and process collection. If any rank exits
    non-zero (or times out), the test fails with stdout/stderr tails
    from the failing ranks.
    """
    probe_script = pytestconfig.rootpath / "tests" / "probes" / "tp_probe.py"

    def _run(probe_name: str, *, world_size: int, timeout: float = 300.0) -> None:
        port = _free_tcp_port()
        procs = []
        for rank in range(world_size):
            env = {
                **os.environ,
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "PYTHONUNBUFFERED": "1",
            }
            procs.append(
                subprocess.Popen(  # noqa: S603
                    [sys.executable, str(probe_script), "--probe", probe_name],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        results: list[tuple[int, str, str]] = []
        for p in procs:
            try:
                out, err = p.communicate(timeout=timeout)
                results.append((p.returncode, out or "", err or ""))
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
                results.append((-1, out or "", err or ""))

        failed = [(i, rc, out, err) for i, (rc, out, err) in enumerate(results) if rc != 0]
        if failed:
            msg = "\n".join(
                f"--- rank {i} (rc={rc}) ---\n"
                f"stdout (tail):\n{out[-2000:]}\n"
                f"stderr (tail):\n{err[-2000:]}"
                for i, rc, out, err in failed
            )
            pytest.fail(f"probe {probe_name!r} ranks failed:\n{msg}")

    return _run


@pytest.fixture()
def should_do_global_cleanup_after_test():
    """Skip global cleanup for Spyre - torch.accelerator.empty_cache() doesn't work yet."""
    return False


@pytest.fixture(autouse=True)
def relax_torch_tolerances(request, monkeypatch):
    """Relax torch.testing.assert_close tolerances for upstream tests.

    Only applies to tests with explicit tolerances configured in upstream_tests.yaml:

        - test: "test_foo"
          tolerances:
            atol: 1e-3
            rtol: 1e-3
    """
    tolerances = getattr(request.node, "_spyre_tolerances", None)
    if tolerances is None:
        return

    _original = torch.testing.assert_close

    def relaxed_assert_close(*args, **kwargs):
        kwargs["atol"] = tolerances.atol
        kwargs["rtol"] = tolerances.rtol
        return _original(*args, **kwargs)

    monkeypatch.setattr(torch.testing, "assert_close", relaxed_assert_close)


@pytest.fixture()
def patch_backend_list(request, monkeypatch):
    """This fixture patches things for tests/v1/attention/test_attention_backends.py"""

    # The BACKENDS_TO_TEST list has to be patched with only our backend
    our_backend_list = [
        AttentionBackendEnum.CUSTOM,
    ]
    test_module = request.node.module
    monkeypatch.setattr(test_module, "BACKENDS_TO_TEST", our_backend_list)

    # _test_backend_correctness may be called with a hardcoded AttentionBackendEnum.FLEX_ATTENTION,
    # which we want to ignore
    orig_tbc = test_module._test_backend_correctness

    def tbc_wrapper(
        batch_spec, model, backend_to_test: list[AttentionBackendEnum | str], *args, **kwargs
    ):
        if "AttentionBackendEnum.FLEX_ATTENTION" in str(backend_to_test):
            return
        # Force block_size=64 for list-based attention
        # This overrides the test's default block_size=16
        kwargs["block_size"] = 64
        return orig_tbc(batch_spec, model, backend_to_test, *args, **kwargs)

    monkeypatch.setattr(test_module, "_test_backend_correctness", tbc_wrapper)

    # Patch the KV cache layout for CUSTOM backend. The upstream test allocates
    # kv_cache as a single tensor [2, num_blocks, block_size, num_kv_heads, head_size];
    # SpyreAttentionImpl.forward expects (k_pages, v_pages) where each is a
    # per-block list of [num_kv_heads, block_size, head_size] tensors.
    orig_run_attention_backend = test_module.run_attention_backend

    def patched_run_attention_backend(
        backend,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        common_attn_metadata,
        query,
        key,
        value,
        kv_cache,
        attn_type=None,
        sliding_window=None,
    ):
        if backend == AttentionBackendEnum.CUSTOM:
            # [num_blocks, 2, block_size, num_kv_heads, head_size]
            #   -> per-side [num_blocks, num_kv_heads, block_size, head_size]
            #   -> list of num_blocks tensors of [num_kv_heads, block_size, head_size]
            k_blocks = kv_cache[:, 0].transpose(1, 2).contiguous()
            v_blocks = kv_cache[:, 1].transpose(1, 2).contiguous()
            kv_cache = (list(k_blocks.unbind(0)), list(v_blocks.unbind(0)))
        return orig_run_attention_backend(
            backend,
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            common_attn_metadata,
            query,
            key,
            value,
            kv_cache,
            attn_type,
            sliding_window,
        )

    monkeypatch.setattr(test_module, "run_attention_backend", patched_run_attention_backend)

    yield


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    """Override fixtures when running upstream vLLM tests."""
    upstream_tests_base = getattr(request.config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    if fixturedef.argname == "default_vllm_config":
        fixturedef.func = _spyre_default_vllm_config
        fixturedef.argnames = ("monkeypatch",)
    elif fixturedef.argname == "should_do_global_cleanup_after_test":
        fixturedef.func = lambda: False
        fixturedef.argnames = ()
