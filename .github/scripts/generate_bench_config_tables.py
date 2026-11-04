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

"""Write the benchmark config tables into vllm-benchmarks/README.md.

Reads every *-runs.yaml under vllm-benchmarks/benchmarks and replaces the part
of the README between the BEGIN/END markers with:

- one matrix per bench family (decoder serve, pooling, latency/throughput),
  showing each entry's labels; the decoder matrix groups its concurrency
  columns under each dataset,
- one list per suite (minimal, base, reference) with each entry's config.

Fails with exit 2 if two entries share a test_name, since the name is the
result file and the benchmark's identity downstream, or if an entry has no
label, since it would run in no suite.

    python3 .github/scripts/generate_bench_config_tables.py          # rewrite
    python3 .github/scripts/generate_bench_config_tables.py --check  # verify only
"""

import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
CONFIGS_DIR = _REPO / "vllm-benchmarks" / "benchmarks"
README = _REPO / "vllm-benchmarks" / "README.md"
BEGIN = "<!-- BEGIN GENERATED: bench-config-tables -->"
END = "<!-- END GENERATED: bench-config-tables -->"

# The scripts are not a package, so load the runner by path: the tables read
# the configs the same way the run does.
_RUNNER_SCRIPT = Path(__file__).resolve().parent / "run_vllm_benchmarks.py"
_runner_spec = importlib.util.spec_from_file_location("run_vllm_benchmarks", _RUNNER_SCRIPT)
assert _runner_spec and _runner_spec.loader
runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)

SHORT_LABEL = {"minimal": "min", "base": "base", "reference": "ref"}
TRACE_DATASETS = ("aiops", "cics", "all")


def load_entries() -> list[dict]:
    """Every entry of every config file, with its file and bench type.

    Exits on an entry without labels, and on a test_name used more than once,
    listing every collision.
    """
    entries = []
    for config_file in runner.config_files(CONFIGS_DIR):
        configs = runner._load_configs(config_file)
        if configs is None:
            sys.exit(2)
        for config in configs:
            entries.append(
                {
                    "config": config,
                    "file": config_file.relative_to(README.parent).as_posix(),
                    "bench_type": runner._bench_type(config_file),
                    "labels": runner._labels(config),
                }
            )

    files_by_name: dict[str, list[str]] = {}
    for entry in entries:
        files_by_name.setdefault(entry["config"]["test_name"], []).append(entry["file"])
    collisions = {name: files for name, files in files_by_name.items() if len(files) > 1}
    if collisions:
        for name, files in sorted(collisions.items()):
            print(f"test_name {name} is used more than once: {', '.join(files)}", file=sys.stderr)
        sys.exit(2)
    return entries


def _model(entry: dict) -> str:
    return runner._config_model(entry["config"]) or "?"


def _short_model(entry: dict) -> str:
    return _model(entry).split("/")[-1]


def _tp(entry: dict) -> int:
    return runner._config_tp(entry["config"]) or 0


def _name_tail(entry: dict) -> str:
    """The part of the test_name after `_tp<N>_`, e.g. `conc5_aiops`."""
    return entry["config"]["test_name"].split(f"_tp{_tp(entry)}_", 1)[-1]


def _cell(entries: list[dict]) -> str:
    if not entries:
        return ""
    cells = []
    for entry in entries:
        cells.append(" ".join(SHORT_LABEL[label] for label in entry["labels"]))
    # Latency and throughput share a cell; show both only when they differ.
    return " / ".join(dict.fromkeys(cells))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _row_keys(entries: list[dict]) -> list[tuple[str, int]]:
    """(model, tp) pairs in file order, TP4 before TP1 within a model."""
    keys = list(dict.fromkeys((_short_model(e), _tp(e)) for e in entries))
    order = {model: i for i, model in enumerate(dict.fromkeys(m for m, _ in keys))}
    return sorted(keys, key=lambda k: (order[k[0]], -k[1]))


def _matrix(entries: list[dict], columns: list[str], column_of) -> list[str]:
    by_cell: dict[tuple, list[dict]] = {}
    for entry in entries:
        key = (_short_model(entry), _tp(entry), column_of(entry))
        by_cell.setdefault(key, []).append(entry)
    rows = [
        [f"{model} · tp{tp}"] + [_cell(by_cell.get((model, tp, c), [])) for c in columns]
        for model, tp in _row_keys(entries)
    ]
    return _table(["model · TP", *columns], rows)


def _grouped_matrix(entries: list[dict], group_of, column_of, groups: list[str]) -> list[str]:
    """An HTML table whose columns are grouped under a second header row,
    e.g. each dataset over its concurrencies. Markdown has no column groups."""
    by_cell: dict[tuple, list[dict]] = {}
    for entry in entries:
        key = (_short_model(entry), _tp(entry), group_of(entry), column_of(entry))
        by_cell.setdefault(key, []).append(entry)
    columns = {
        g: _sorted_columns(column_of(e) for e in entries if group_of(e) == g) for g in groups
    }
    columns = {g: c for g, c in columns.items() if c}
    lines = [
        "<table>",
        "<thead>",
        '<tr><th rowspan="2">model · TP</th>'
        + "".join(f'<th colspan="{len(c)}">{g}</th>' for g, c in columns.items())
        + "</tr>",
        "<tr>" + "".join(f"<th>{c}</th>" for cs in columns.values() for c in cs) + "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for model, tp in _row_keys(entries):
        cells = [_cell(by_cell.get((model, tp, g, c), [])) for g, cs in columns.items() for c in cs]
        lines.append(
            f"<tr><td>{model} · tp{tp}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        )
    return [*lines, "</tbody>", "</table>"]


def _sorted_columns(values) -> list[str]:
    """Column names in numeric order, e.g. conc4 before conc16."""

    def key(value: str):
        digits = "".join(ch if ch.isdigit() else " " for ch in value).split()
        return ([int(d) for d in digits], value)

    return sorted(set(values), key=key)


def _dataset(entry: dict) -> str:
    parameters = entry["config"].get("parameters", {})
    path = str(parameters.get("dataset-path", ""))
    if path.startswith("${SPYRE_") and path.endswith("_DATASET}"):
        return path[len("${SPYRE_") : -len("_DATASET}")].lower()
    if entry["bench_type"] != "serve":
        input_key = "input-len" if entry["bench_type"] == "latency" else "random-input-len"
        output_key = "output-len" if entry["bench_type"] == "latency" else "random-output-len"
        return f"in{parameters.get(input_key)} / out{parameters.get(output_key)}"
    name = parameters.get("dataset-name", "?")
    if "random-input-len" in parameters:
        return f"{name} {parameters['random-input-len']}"
    return name


def _setting(entry: dict, key: str) -> str:
    config = entry["config"]
    for section in ("server_parameters", "parameters"):
        if key in config.get(section, {}):
            return str(config[section][key])
    return "–"


def _suite_list(entries: list[dict], suite: str) -> list[str]:
    rows = [
        [
            f"`{e['config']['test_name']}`",
            _model(e),
            str(_tp(e)),
            _setting(e, "max-model-len"),
            _setting(e, "max-num-seqs"),
            _setting(e, "max-concurrency"),
            _dataset(e),
            _setting(e, "num-prompts"),
            " ".join(SHORT_LABEL[label] for label in e["labels"]),
            f"[{Path(e['file']).name}]({e['file']})",
        ]
        for e in entries
        if suite in e["labels"]
    ]
    header = [
        "test_name",
        "model",
        "TP",
        "max-model-len",
        "max-num-seqs",
        "concurrency",
        "dataset",
        "prompts",
        "labels",
        "file",
    ]
    return [f"### `{suite}` ({len(rows)} entries)", "", *_table(header, rows), ""]


def render(entries: list[dict]) -> str:
    serve = [e for e in entries if e["bench_type"] == "serve"]
    decoder = [e for e in serve if e["file"].startswith("benchmarks/decoder/")]
    encoder = [e for e in serve if e not in decoder]
    offline = [e for e in entries if e["bench_type"] != "serve"]

    counts = {suite: sum(suite in e["labels"] for e in entries) for suite in SHORT_LABEL}
    lines = [
        BEGIN,
        "",
        "<!-- Generated by .github/scripts/generate_bench_config_tables.py; do not edit. -->",
        "",
        "## Config tables",
        "",
        f"{len(entries)} entries: {counts['minimal']} `minimal`, {counts['base']} `base`, "
        f"{counts['reference']} `reference`. Each cell gives the labels of one entry: "
        "`min` = minimal, `base` = base, `ref` = reference. An empty cell has no entry.",
        "",
    ]
    if decoder:
        # A decoder tail is `conc<C>_<dataset>`: datasets head the columns, each
        # over its concurrencies. Traces come first, in TRACE_DATASETS order.
        datasets = list(TRACE_DATASETS) + sorted(
            {_name_tail(e).split("_", 1)[1] for e in decoder} - set(TRACE_DATASETS)
        )
        lines += ["### Decoder serve", ""]
        lines += _grouped_matrix(
            decoder,
            lambda e: _name_tail(e).split("_", 1)[1],
            lambda e: _name_tail(e).split("_", 1)[0],
            datasets,
        )
        lines += [""]
    if encoder:
        columns = _sorted_columns(_name_tail(e) for e in encoder)
        lines += ["### Pooling serve", ""]
        lines += _matrix(encoder, columns, _name_tail) + [""]
    if offline:
        columns = _sorted_columns(_name_tail(e) for e in offline)
        lines += [
            "### Latency / throughput",
            "",
            "Each cell is one latency and one throughput entry; it shows both label sets "
            "only when they differ.",
            "",
        ]
        lines += _matrix(offline, columns, _name_tail) + [""]
    for suite in SHORT_LABEL:
        lines += _suite_list(entries, suite)
    lines += [END]
    return "\n".join(lines) + "\n"


def updated_readme(current: str, section: str) -> str:
    if BEGIN in current and END in current:
        head, rest = current.split(BEGIN, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
        return head + section + (("\n" + tail) if tail else "")
    return current.rstrip("\n") + "\n\n" + section


def main() -> None:
    parser = ArgumentParser(description="Write the benchmark config tables into the README")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 instead of writing if the README is not up to date",
    )
    args = parser.parse_args()

    section = render(load_entries())
    current = README.read_text()
    expected = updated_readme(current, section)
    if args.check:
        if current != expected:
            print(
                f"{README.relative_to(_REPO)} is out of date; run "
                "`python3 .github/scripts/generate_bench_config_tables.py` and commit the result.",
                file=sys.stderr,
            )
            sys.exit(1)
        return
    README.write_text(expected)


if __name__ == "__main__":
    main()
