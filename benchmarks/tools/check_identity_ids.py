#!/usr/bin/env python3
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

"""Check that every catalog case reproduces the benchmark_id the ingest already stores.

The stored id is derived from the test *name*: `run_mode` is the first `_`-separated token and the
`tp<N>_in<N>_out<N>_c<N>...` tokens become discriminators. So a renamed case, a mistyped
discriminator or a reordered key list silently starts a NEW trend line rather than failing. This
recomputes each id two ways and refuses to agree quietly:

  catalog side  `spyre-bench ids`, from the Benchmark's own `identity` block
  ingest side   `spyre_clickhouse_ingest.vllm`'s own name parser over the test names in
                vllm-benchmarks/, which is what actually ran and actually landed rows

A case the configs do not define is reported too: its command would fail at run time with
`--tests` finding no entry, which is better caught here.

    python3 benchmarks/tools/check_identity_ids.py [--catalog benchmarks] [--configs-dir DIR]
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH_TYPES = ("latency", "throughput", "serve")


def config_test_names(configs_dir: Path) -> set[str]:
    """Every `test_name` the CI configs define, as spelled there."""
    names = set()
    files = [configs_dir / f"{b}-tests.yaml" for b in BENCH_TYPES]
    files.append(configs_dir / "multimodal-serve-tests.yaml")
    for path in files:
        if not path.is_file():
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        tests = raw.get("tests") if isinstance(raw, dict) else raw
        for entry in tests or []:
            if isinstance(entry, dict) and entry.get("test_name"):
                names.add(str(entry["test_name"]))
    return names


def ingest_ids(names: set[str], component: str) -> dict[str, str]:
    """benchmark_id per test name, derived by the ingest's own code."""
    from spyre_clickhouse_ingest.identity import BenchmarkId
    from spyre_clickhouse_ingest.vllm import BENCH_ID_KEYS, SWEEP_ID_KEYS, parse_input_shapes

    out = {}
    for name in names:
        props = {"record_type": "model", "run_mode": name.split("_")[0]}
        props.update(parse_input_shapes(name))
        # The ingest widens the key list exactly when a sweep token is present.
        keys = SWEEP_ID_KEYS if set(props) - set(BENCH_ID_KEYS) - {"model"} else BENCH_ID_KEYS
        out[name.lower()] = BenchmarkId.derive(component, name, [], props, keys)
    return out


def catalog_ids(bench: str, catalog: Path, central: str) -> dict[str, str]:
    cmd = [bench, "ids", "--catalog", str(catalog), "--json"]
    if central:
        cmd += ["--central", central]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout + done.stderr, file=sys.stderr)
        raise SystemExit(f"spyre-bench ids failed with exit {done.returncode}")
    return {r["case"]: r.get("benchmark_id", "") for r in json.loads(done.stdout)}


def shared_catalog(bench: str) -> str:
    done = subprocess.run([bench, "catalogs", "--path", "llm"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def main() -> int:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=REPO / "benchmarks")
    ap.add_argument("--configs-dir", type=Path, default=REPO / "vllm-benchmarks/benchmarks/spyre")
    ap.add_argument("--bench", default="spyre-bench", help="the engine executable")
    ap.add_argument("--component", default="spyre-inference")
    ap.add_argument("--central", default="", help="shared catalog dir (default: the engine's own)")
    a = ap.parse_args()

    central = a.central or shared_catalog(a.bench)
    cases = catalog_ids(a.bench, a.catalog, central)
    names = config_test_names(a.configs_dir)
    expected = ingest_ids(names, a.component)

    bad, missing = [], []
    for case, got in sorted(cases.items()):
        want = expected.get(case)
        if want is None:
            missing.append(case)
        elif want != got:
            bad.append((case, got, want))

    for case in missing:
        print(
            f"NO CONFIG  {case}: no test_name in {a.configs_dir} (its --tests would match nothing)"
        )
    for case, got, want in bad:
        print(f"MISMATCH   {case}\n  catalog {got}\n  ingest  {want}")

    unclaimed = sorted(n for n in names if n.lower() not in cases)
    for name in unclaimed:
        print(f"note: config test {name} has no catalog case, so no Selection can schedule it")

    print(
        f"{'FAIL' if bad or missing else 'PASS'}: {len(cases)} cases checked, "
        f"{len(bad)} mismatched, {len(missing)} without a config, "
        f"{len(unclaimed)} unclaimed configs"
    )
    return 2 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
