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
Sync upstream test dependencies with vLLM test dependencies.

Run this whenever the vLLM version is updated to keep test dependencies in sync.

Usage:
    python -m spyre_testing_plugin.sync_upstream_test_deps
    # or
    uv run sync-upstream-test-deps
"""

import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from spyre_testing_plugin import pytest_plugin

# Plugin package root - syncs the plugin's pyproject.toml
PLUGIN_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = PLUGIN_ROOT / "pyproject.toml"

# Libraries to exclude from upstream test dependencies.
# torchcodec: ST ≥5 imports it for A/V; Spyre CI is CPU-only (no libnvrtc).
# Text-only SentenceTransformer HF compare uses a stub in pytest_plugin instead.
FILTERED_LIBRARIES = {"terratorch", "transformers", "torchcodec"}


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
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{commit}/requirements/test/cuda.in"
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


def filter_requirements(test_in: Path, filtered_libs: set[str]) -> Path:
    """
    Filter out specified libraries from the requirements file.

    Returns path to the filtered requirements file.
    """
    with open(test_in) as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith("#"):
            filtered_lines.append(line)
            continue

        # Extract package name (handle various formats: pkg, pkg==ver, pkg>=ver, etc.)
        pkg_name = re.split(r"[=<>!~\[]", line.strip())[0].strip()

        # Keep line if package is not in filtered list
        if pkg_name.lower() not in {lib.lower() for lib in filtered_libs}:
            filtered_lines.append(line)

    # Write filtered content to new file
    filtered_path = test_in.parent / f"{test_in.stem}-filtered{test_in.suffix}"
    with open(filtered_path, "w") as f:
        f.writelines(filtered_lines)

    return filtered_path


def clear_dependencies(pyproject_path: Path) -> None:
    """
    Clear the [project].dependencies section, keeping only pytest and pyyaml.
    """
    with open(pyproject_path) as f:
        lines = f.readlines()

    result, inside, depth = [], False, 0
    for i, line in enumerate(lines):
        # Detect start of dependencies array
        if not inside and re.match(r"^dependencies\s*=\s*\[", line):
            inside = True
            depth = line.count("[") - line.count("]")
            # Start fresh dependencies with minimal base
            result.append('dependencies = [\n    "pytest",\n    "pyyaml",\n')
            if depth <= 0 and "]" in line:
                # Single-line array, skip to end
                inside = False
            continue
        if inside:
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                inside = False
                # Close the array
                result.append("]\n")
            continue
        result.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(result)


def reorder_dependencies(pyproject_path: Path) -> None:
    """
    Reorder dependencies so pytest and pyyaml come first, followed by a comment
    separating them from the upstream vLLM test dependencies.
    """
    with open(pyproject_path) as f:
        content = f.read()

    # Extract the dependencies array
    match = re.search(r"^(dependencies\s*=\s*\[)\s*\n(.*?)(^\])", content, re.MULTILINE | re.DOTALL)
    if not match:
        return

    prefix = match.group(1)
    body = match.group(2)
    suffix = match.group(3)

    # Parse individual dependency lines (ignore comments)
    deps = []
    for line in body.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            deps.append(stripped.rstrip(","))

    # Separate our deps from upstream deps
    our_deps = []
    upstream_deps = []
    for dep in deps:
        # Remove quotes for comparison
        name = dep.strip('"').split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("!")[0]
        if name in ("pytest", "pyyaml"):
            our_deps.append(dep)
        else:
            upstream_deps.append(dep)

    # Rebuild the dependencies section
    lines = [f"{prefix}\n"]
    for dep in our_deps:
        lines.append(f"    {dep},\n")
    lines.append("    # upstream vLLM test dependencies: see sync_upstream_test_deps\n")
    for dep in upstream_deps:
        lines.append(f"    {dep},\n")
    lines.append(f"{suffix}\n")

    new_section = "".join(lines)
    new_content = content[: match.start()] + new_section + content[match.end() + 1 :]

    with open(pyproject_path, "w") as f:
        f.write(new_content)


def main():
    if len(sys.argv) > 1:
        print("Usage: python -m spyre_testing_plugin.sync_upstream_test_deps", file=sys.stderr)
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
        cache_dir = pytest_plugin._cache_root() / ".cache"
        cache_dir.mkdir(exist_ok=True)

        # Download test.in from the vLLM repository
        test_in = download_test_requirements(vllm_commit, cache_dir)

        # Filter out excluded libraries
        if FILTERED_LIBRARIES:
            print(f"Filtering out libraries: {', '.join(FILTERED_LIBRARIES)}")
            test_in = filter_requirements(test_in, FILTERED_LIBRARIES)

        # Clear existing upstream-tests section
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
            print(
                f"Error: uv command failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return 1

        # Reorder so pytest/pyyaml come first with a separator comment
        reorder_dependencies(PYPROJECT_PATH)

        print("Done.")
        print("Review changes to tests/plugin/pyproject.toml before committing.")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
