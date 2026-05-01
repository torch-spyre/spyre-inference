"""
Sync upstream-tests dependency group with vLLM test dependencies.

Run this whenever the vLLM version is updated to keep test dependencies in sync.

Usage:
    python scripts/sync_upstream_tests.py
"""

import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Libraries to exclude from upstream test dependencies
FILTERED_LIBRARIES = {"terratorch"}


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
            # If it's a list, find the one with git and rev
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


def clear_upstream_tests_section(pyproject_path: Path) -> None:
    """
    Clear the upstream-tests dependency group, leaving an empty group
    so uv doesn't error on it being referenced in default-groups.
    """
    with open(pyproject_path) as f:
        lines = f.readlines()

    result, inside, depth = [], False, 0
    for line in lines:
        if not inside and re.match(r"^upstream-tests\s*=\s*\[", line):
            inside = True
            depth = line.count("[") - line.count("]")
            result.append("upstream-tests = []\n")
            if depth <= 0:
                inside = False
            continue
        if inside:
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                inside = False
            continue
        result.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(result)


def main():
    if len(sys.argv) > 1:
        print(f"Usage: {sys.argv[0]}", file=sys.stderr)
        return 1

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        return 1

    try:
        # Extract vLLM commit from pyproject.toml
        vllm_commit = extract_vllm_commit(pyproject_path)
        print(f"Found vLLM commit: {vllm_commit}")

        # Create cache directory for downloaded files
        cache_dir = PROJECT_ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)

        # Download test.in from the vLLM repository
        test_in = download_test_requirements(vllm_commit, cache_dir)

        # Filter out excluded libraries
        if FILTERED_LIBRARIES:
            print(f"Filtering out libraries: {', '.join(FILTERED_LIBRARIES)}")
            test_in = filter_requirements(test_in, FILTERED_LIBRARIES)

        # Clear existing upstream-tests section
        print("Clearing existing upstream-tests section...")
        clear_upstream_tests_section(pyproject_path)

        # Add dependencies using uv
        print(f"Adding dependencies from {test_in}...")
        result = subprocess.run(
            ["uv", "add", "--group", "upstream-tests", "--no-sync", "-r", test_in],
            cwd=PROJECT_ROOT,
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

        print("Done.")
        print("Review changes to pyproject.toml and uv.lock before committing.")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
