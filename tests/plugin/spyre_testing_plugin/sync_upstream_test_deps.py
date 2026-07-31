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
Sync the plugin's upstream vLLM test dependencies.

Run this whenever the vLLM version is updated. Unlike a wholesale mirror of
vLLM's `requirements/test/cuda.in`, this enforces a curated ALLOW_LIST: only
deps that (a) a test we collect/run actually needs AND (b) vLLM's own runtime
closure does not already provide. Everything else in cuda.in is intentionally
omitted (see issue #447). The `[project].dependencies` array is regenerated
from the allow-list on every run, so it must not be hand-edited — add names to
ALLOW_LIST below and re-run instead. Versions and extras are taken verbatim
from cuda.in; platform markers come from the allow-list value.

After running, review the diff, then `uv lock` (from the workspace root) to
refresh the lockfile with any new versions.

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
ROOT_PYPROJECT = PLUGIN_ROOT.parent.parent / "pyproject.toml"

# Curated allow-list: upstream test deps we actually need that vLLM's runtime
# closure does not already provide. The value is an optional PEP 508 environment
# marker (platform gating etc.); None means unconditional. Names must match a
# package in cuda.in — the version and extras are pulled from there.
#
# To add a dep: confirm a test we collect/run needs it AND that it is not
# already in vLLM's runtime closure (`uv export --no-dev` from the root), then
# add its name here. The sync run reports both checks for any cuda.in dep not
# yet listed.
ALLOW_LIST: dict[str, str | None] = {
    "buildkite-test-collector": None,
    # ppl_utils.py (imported by the collected test_qwen.py, even while its test
    # is skipped) does a module-level `from datasets import load_dataset`, so
    # collection fails without it. Pulls pyarrow+pandas — heavy on ppc64le.
    "datasets": None,
    "pytest-asyncio": None,
    "pytest-cov": None,
    "pytest-forked": None,
    "pytest-rerunfailures": None,
    "pytest-shard": None,
    "pytest-timeout": None,
    "sentence-transformers": None,
    "tblib": None,
}

# Base deps always present regardless of upstream; not part of the sync.
BASE_DEPS = {"pytest", "pyyaml"}

# cuda.in deps we have reviewed and deliberately do NOT take, so the sync report
# can flag only genuinely-new upstream additions. These back model families,
# eval harnesses, and tooling that none of the tests we collect/run exercise.
EXCLUDED = {
    # Vision-model test deps
    "albumentations",
    "decord",
    "imagehash",
    "instanttensor",
    "open-clip-torch",
    "perceptron",
    "segmentation-models-pytorch",
    "timm",
    # Audio-model test deps
    "av",  # audio_in_video decode; no collected/run test uses it
    "soundfile",  # audio / speech-to-text tests; no collected/run test uses it
    "kaldi-native-fbank",
    "librosa",
    "transformers-stream-generator",
    "vector-quantize-pytorch",
    "vocos",
    # A/V decode: SentenceTransformer>=5 imports torchcodec; Spyre CI is CPU-only
    "torchcodec",
    "terratorch",
    # Eval harnesses / metrics
    "jiwer",
    "lm-eval",
    "mteb",
    "num2words",
    # Quantization / alternate model backends
    "bitsandbytes",
    "arctic-inference",
    "cohere-melody",
    "gpt-oss",
    "peft",
    # torch extras excluded elsewhere (torchaudio/torchvision via the root
    # override; numba/ray are pure test deps for paths we don't run)
    "numba",
    "ray",
    "torchaudio",
    "torchvision",
    # Alternate model-loader / weight-format test deps (each pulls native
    # wheels painful on ppc64le); only their own *_loader tests use them, none
    # of which we collect.
    "runai-model-streamer",  # runai_streamer load format
    "tensorizer",  # tensorizer load format
    "fastsafetensors",  # fastsafetensors load format
    # Entrypoints / OpenAI-server test deps (no entrypoints test collected)
    "grpcio-reflection",  # grpc server reflection
    "schemathesis",  # openai schema fuzz test
    # Misc tooling not exercised by our tests
    "backoff",
    "blobfile",
    "datamodel-code-generator",
    "genai-perf",
    "matplotlib",
    "openpyxl",
    "plotly",
    "pqdm",
    "tritonclient",
}

# Header emitted inside the generated dependencies array.
_GENERATED_HEADER = (
    "    # Net-new upstream vLLM test deps — GENERATED by sync_upstream_test_deps\n"
    "    # from ALLOW_LIST. Do not hand-edit; edit the allow-list and re-run.\n"
    "    # Versions/extras track vLLM cuda.in.\n"
)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def extract_vllm_commit(pyproject_path: Path) -> str:
    """Extract the vLLM git commit/tag from [tool.uv.sources] in pyproject.toml."""
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    try:
        vllm_source = data["tool"]["uv"]["sources"]["vllm"]
        sources = vllm_source if isinstance(vllm_source, list) else [vllm_source]
        for source in sources:
            if isinstance(source, dict) and "git" in source and "rev" in source:
                return source["rev"]
        raise ValueError("No git source with 'rev' found in vllm sources")
    except KeyError as e:
        raise ValueError(
            f"Could not find vLLM git rev in {pyproject_path} [tool.uv.sources]: missing {e}"
        ) from e


def download_test_requirements(commit: str, cache_dir: Path) -> Path:
    """Download cuda.in from the vLLM repository at the given commit."""
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{commit}/requirements/test/cuda.in"
    cache_file = cache_dir / f"vllm-{commit[:8]}-cuda.in"

    print(f"Downloading test requirements from vLLM commit {commit[:8]}...")
    try:
        with urllib.request.urlopen(url) as response:
            cache_file.write_bytes(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download cuda.in from vLLM commit {commit}: {e}\nURL: {url}"
        ) from e

    print(f"Downloaded to: {cache_file}")
    return cache_file


def parse_requirements(path: Path) -> dict[str, tuple[str, str | None]]:
    """
    Parse a pip-style requirements file.

    Returns {normalized_name: (spec, inline_comment)} where `spec` is the
    name+extras+version as declared upstream (any environment marker stripped,
    since we overlay our own) and `inline_comment` is upstream's trailing
    ``# ...`` note if any.
    """
    reqs: dict[str, tuple[str, str | None]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # Skip blanks, comments, and `-r`/`-c` includes. The only include here is
        # `-r ../common.txt` (vLLM's runtime/common deps) — those are already
        # provided by the installed vllm package, so we never allow-list them; the
        # runtime-closure check confirms membership independently.
        if not line or line.startswith(("#", "-")):
            continue
        spec, _, comment = line.partition("#")
        spec = spec.split(";", 1)[0].strip()
        if not spec:
            continue
        name = re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0]
        reqs[_norm(name)] = (spec, comment.strip() or None)
    return reqs


def runtime_closure(root_dir: Path) -> set[str]:
    """Package names in vLLM's non-dev runtime closure, from the current lock.

    Best-effort: returns an empty set if the export fails, in which case the
    report simply omits the "already provided by runtime" annotation.
    """
    result = subprocess.run(
        ["uv", "export", "--no-dev", "--frozen", "--no-hashes", "--no-emit-project"],
        cwd=root_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Warning: `uv export --no-dev` failed; skipping runtime-closure report.")
        return set()
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        names.add(_norm(re.split(r"[<>=!~\[ ;]", line, maxsplit=1)[0]))
    return names


def build_dependency_lines(
    upstream: dict[str, tuple[str, str | None]],
) -> tuple[list[str], list[str]]:
    """Generate the pyproject dependency lines for the allow-list.

    Returns (lines, missing) where `missing` are allow-list names absent from cuda.in.
    """
    lines, missing = [], []
    for name in sorted(ALLOW_LIST):
        if name not in upstream:
            missing.append(name)
            continue
        spec, comment = upstream[name]
        marker = ALLOW_LIST[name]
        entry = f"{spec} ; {marker}" if marker else spec
        line = f'    "{entry}",'
        if comment:
            line += f"  # {comment}"
        lines.append(line)
    return lines, missing


def write_dependencies(pyproject_path: Path, dep_lines: list[str]) -> None:
    """Regenerate the [project].dependencies array in place."""
    content = pyproject_path.read_text()
    block = (
        "dependencies = [\n"
        '    "pytest",\n'
        '    "pyyaml",\n'
        f"{_GENERATED_HEADER}"
        f"{chr(10).join(dep_lines)}\n"
        "]"
    )
    new_content, n = re.subn(r"(?ms)^dependencies\s*=\s*\[.*?^\]", block, content, count=1)
    if n != 1:
        raise ValueError(f"Could not locate a [project].dependencies array in {pyproject_path}")
    pyproject_path.write_text(new_content)


def report_unlisted(upstream: dict[str, tuple[str, str | None]], runtime: set[str]) -> None:
    """Report cuda.in deps not in the allow-list, classified for review."""
    already, excluded, new = [], [], []
    for name in sorted(upstream):
        if name in ALLOW_LIST or name in BASE_DEPS:
            continue
        if name in EXCLUDED:
            excluded.append(name)
        elif name in runtime:
            already.append(name)
        else:
            new.append(name)

    print(f"\ncuda.in deps not in the allow-list ({len(already) + len(excluded) + len(new)}):")
    print(f"  already provided by vLLM runtime ({len(already)}): {', '.join(already) or '-'}")
    print(f"  excluded by policy ({len(excluded)}): {', '.join(excluded) or '-'}")
    if new:
        print(f"  NEW — review whether any collected/run test needs these ({len(new)}):")
        for name in new:
            print(f"      {name}")
    else:
        print("  NEW: none")


def main():
    if len(sys.argv) > 1:
        print("Usage: python -m spyre_testing_plugin.sync_upstream_test_deps", file=sys.stderr)
        return 1
    if not PYPROJECT_PATH.exists():
        print(f"Error: {PYPROJECT_PATH} not found", file=sys.stderr)
        return 1
    if not ROOT_PYPROJECT.exists():
        print(f"Error: root pyproject.toml not found at {ROOT_PYPROJECT}", file=sys.stderr)
        return 1

    try:
        vllm_commit = extract_vllm_commit(ROOT_PYPROJECT)
        print(f"Found vLLM commit: {vllm_commit}")

        cache_dir = pytest_plugin._cache_root() / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cuda_in = download_test_requirements(vllm_commit, cache_dir)

        upstream = parse_requirements(cuda_in)

        dep_lines, missing = build_dependency_lines(upstream)
        if missing:
            print(
                "Error: these ALLOW_LIST names are no longer in vLLM cuda.in — "
                f"reconcile the allow-list: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1

        # Read the runtime closure before rewriting the plugin pyproject: the
        # frozen `uv export` needs the lock to still match on-disk pyprojects
        # (true right after the `uv sync` that precedes this script). Rewriting
        # first would stale the lock and collapse the runtime bucket.
        runtime = runtime_closure(ROOT_PYPROJECT.parent)

        write_dependencies(PYPROJECT_PATH, dep_lines)
        print(f"Regenerated {len(dep_lines)} allow-listed deps in {PYPROJECT_PATH.name}.")

        report_unlisted(upstream, runtime)

        print("\nDone. Review the diff, then run `uv lock` from the workspace root.")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
