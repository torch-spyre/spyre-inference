"""
Sync upstream test dependencies with vLLM test dependencies.

Run this whenever the vLLM version is updated to keep test dependencies in sync.

Usage:
    python -m spyre_testing_plugin.sync_upstream_tests
    # or
    uv run sync-upstream-tests
"""

import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# Plugin package root - syncs the plugin's pyproject.toml
PLUGIN_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = PLUGIN_ROOT / "pyproject.toml"


def extract_vllm_commit(pyproject_path: Path) -> str:
    """
    Extract the vLLM git commit/tag from pyproject.toml.

    Returns the commit hash or tag specified in [tool.uv.sources].
    """
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        vllm_source = data["tool"]["uv"]["sources"]["vllm"]

        # Handle both single source and list of sources
        if isinstance(vllm_source, list):
            for source in vllm_source:
                if isinstance(source, dict) and "git" in source and "rev" in source:
                    return source["rev"]
            raise ValueError("No git source with rev found in vllm sources list")
        elif isinstance(vllm_source, dict):
            if "git" in vllm_source and "rev" in vllm_source:
                return vllm_source["rev"]
            raise ValueError("vLLM source does not have both 'git' and 'rev' fields")
        else:
            raise ValueError(f"Unexpected vllm source type: {type(vllm_source)}")

    except KeyError as e:
        raise ValueError(
            f"Could not find vLLM git rev in pyproject.toml [tool.uv.sources]: missing key {e}"
        ) from e


def download_test_requirements(commit: str, cache_dir: Path) -> Path:
    """
    Download the test.in file from vLLM repository at the specified commit.

    Returns the path to the downloaded file.
    """
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{commit}/requirements/test.in"
    cache_file = cache_dir / f"vllm-{commit[:8]}-test.in"

    print(f"Downloading test requirements from vLLM commit {commit[:8]}...")

    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(cache_file, "wb") as f:
            f.write(content)

        print(f"Downloaded to: {cache_file}")
        return cache_file

    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download test.in from vLLM commit {commit}: {e}\n"
            f"URL: {url}\n"
            "Please verify the commit exists in the vLLM repository."
        ) from e


def clear_dependencies(pyproject_path: Path) -> None:
    """
    Clear the [project].dependencies section, keeping only pytest and pyyaml.
    """
    with open(pyproject_path) as f:
        lines = f.readlines()

    result, inside, depth = [], False, 0
    skip_line = False
    for i, line in enumerate(lines):
        # Detect start of dependencies array
        if not inside and re.match(r'^dependencies\s*=\s*\[', line):
            inside = True
            depth = line.count("[") - line.count("]")
            # Start fresh dependencies with minimal base
            result.append('dependencies = [\n    "pytest",\n    "pyyaml",\n')
            if depth <= 0 and ']' in line:
                # Single-line array, skip to end
                inside = False
            continue
        if inside:
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                inside = False
                # Close the array
                result.append(']\n')
            continue
        result.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(result)


def main():
    if len(sys.argv) > 1:
        print(f"Usage: python -m spyre_testing_plugin.sync_upstream_tests", file=sys.stderr)
        return 1

    if not PYPROJECT_PATH.exists():
        print(f"Error: {PYPROJECT_PATH} not found", file=sys.stderr)
        return 1

    try:
        # Extract vLLM commit from the ROOT pyproject.toml (workspace root)
        root_pyproject = PLUGIN_ROOT.parent.parent / "pyproject.toml"
        if not root_pyproject.exists():
            print(f"Error: Root pyproject.toml not found at {root_pyproject}", file=sys.stderr)
            return 1

        vllm_commit = extract_vllm_commit(root_pyproject)
        print(f"Found vLLM commit: {vllm_commit}")

        # Create cache directory for downloaded files
        cache_dir = PLUGIN_ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)

        # Download test.in from the vLLM repository
        test_in = download_test_requirements(vllm_commit, cache_dir)

        # Clear existing dependencies (keep pytest, pyyaml)
        print("Clearing existing dependencies...")
        clear_dependencies(PYPROJECT_PATH)

        # Add dependencies using uv
        print(f"Adding dependencies from {test_in}...")
        result = subprocess.run(
            ["uv", "add", "--no-sync", "-r", test_in],
            cwd=PLUGIN_ROOT,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error: uv command failed with exit code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return 1

        print("Done.")
        print("Review changes to tests/plugin/pyproject.toml before committing.")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
