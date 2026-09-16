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

"""The GSM8K accuracy evals live in the upstream vLLM tree, where the same test is
parametrized over a large *GPU* config list. The plugin redirects that parametrization
onto our own Spyre config list (`config_list` on the file's `upstream_tests.yaml` entry
-> `pytest_generate_tests` points the upstream conftest's `--config-list-file` at it).

This drives real collection to pin that redirect down: nothing else asserts that the
steering actually swaps the config list. If it silently no-ops -- the upstream conftest
renames the option, or our `tryfirst` ordering breaks -- collection falls back to the
GPU defaults and still comes up "green" with the wrong (and hardware-unrunnable) configs.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_LIST = _REPO_ROOT / "tests/plugin/spyre_testing_plugin/gsm8k_configs/models-spyre.txt"


def _expected_config_ids() -> set[str]:
    """The parametrization ids collection should produce: one per Spyre config, named
    by its yaml stem. Read from the list itself so this tracks edits to it."""
    ids = set()
    for line in _CONFIG_LIST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line.removesuffix(".yaml"))
    return ids


def _collect_gsm8k_ids(markexpr: str) -> tuple[set[str], str]:
    """Drive `--collect-only` for `markexpr` in a subprocess and return the collected
    `test_gsm8k_correctness[...]` parametrization ids plus the combined output.

    timeout: collection can trigger the vLLM clone on a cold cache; a hung fetch must fail
    this test rather than sit until the whole job's timeout.
    """
    # --continue-on-collection-errors: on a CPU-only host an *unrelated* upstream file
    # (multi_gpu_test probing torch.spyre) errors at collection; it must not mask the
    # gsm8k ids, which collect fine. Harmless on hardware, where nothing errors.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            "-m",
            markexpr,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        # Collection only imports the test modules; the backend autoload would fail the
        # import on a CPU host. This never runs an eval, so disabling it is safe on CI too.
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    output = result.stdout + result.stderr
    return set(re.findall(r"test_gsm8k_correctness\[([^\]]+)\]", output)), output


@pytest.mark.upstream
def test_gsm8k_parametrization_is_steered_onto_spyre_configs():
    expected = _expected_config_ids()
    assert expected, f"no configs parsed from {_CONFIG_LIST}"

    collected, output = _collect_gsm8k_ids("gsm8k and upstream")

    if not collected:
        # An empty selection is ambiguous: the upstream clone may be unavailable (a legit
        # skip), or the `gsm8k`/`upstream` tags may have been dropped from the entry / the
        # file may have left the pinned tree -- a real regression this skip must not hide.
        # Every cloned item gets the `upstream` marker unconditionally, so the file collects
        # under `-m upstream` iff the clone is present, regardless of its yaml tags.
        under_upstream, up_output = _collect_gsm8k_ids("upstream")
        if not under_upstream:
            pytest.skip(
                "upstream gsm8k eval not collectable in this environment "
                f"(clone unavailable?):\n{up_output[-1500:]}"
            )
        pytest.fail(
            "gsm8k eval is collectable under `-m upstream` but not under "
            "`-m 'gsm8k and upstream'`: the `gsm8k` tag on its upstream_tests.yaml entry "
            f"regressed (or the tag-to-marker application broke).\n{output[-1500:]}"
        )

    assert collected == expected, (
        "gsm8k config_filename was not steered onto the Spyre config list. "
        f"expected {sorted(expected)}, collected {sorted(collected)}. A large or "
        "GPU-named set means the redirect no-opped and collection fell back to the "
        "upstream default config list (see this file's module docstring)."
    )
