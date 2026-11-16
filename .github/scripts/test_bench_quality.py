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

"""The quality cases: definitions against the gate configs, the recorder, the architecture profiles.

The id comparison needs the ingest package (and, for the engine id, the engine); the rest needs
only PyYAML.
"""

import importlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks"
QUALITY = BENCH / "quality"
GATE_CONFIGS = REPO / "tests" / "plugin" / "spyre_testing_plugin" / "gsm8k_configs"

_spec = importlib.util.spec_from_file_location(
    "bench_quality_gate", Path(__file__).with_name("bench_quality_gate.py")
)
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)

LOG = """\
Starting ACC evaluation for model: org/model-a
ACC Results for org/model-a:
  Measured metric: 0.5000
  Expected metric: 0.9000
  Tolerance: 0.0800
  Questions: 64
  Invalid rate: 0.047
  Latency: 12.0s
PASSED
ACC Results for org/model-b:
  Measured metric: 0.2500
  Questions: 64
  Invalid rate: 0.000
"""


def _benchmarks():
    out = []
    for path in sorted(QUALITY.glob("*.yaml")):
        out += [
            d for d in yaml.safe_load_all(path.read_text()) if d and d.get("kind") == "Benchmark"
        ]
    return out


def _config_of(bench):
    argv = bench["spec"]["run"]["command"]
    return REPO / argv[argv.index("--config") + 1]


def _need(module):
    if os.environ.get("BENCH_NIGHTLY_REQUIRE_ENGINE"):
        return importlib.import_module(module)
    return pytest.importorskip(module)


def test_log_parse_keeps_complete_blocks_and_ignores_the_threshold():
    got = rec.parse_gate_log(LOG)
    assert got["org/model-a"] == {"accuracy": 0.5, "questions": 64.0, "invalid_rate": 0.047}
    assert "org/model-b" in got and set(got) == {"org/model-a", "org/model-b"}
    assert rec.parse_gate_log("Measured metric: 0.5\n") == {}


def test_a_later_block_of_the_same_model_wins():
    text = LOG + LOG.replace("0.5000", "0.7500")
    assert rec.parse_gate_log(text)["org/model-a"]["accuracy"] == 0.75


def test_every_benchmark_has_a_gate_config_and_a_matching_identity():
    benches = _benchmarks()
    assert len(benches) == 3
    for b in benches:
        cfg = yaml.safe_load(_config_of(b).read_text())
        blk = b["spec"]["identity"]
        assert blk["profile"] == "generic" and not blk.get("tags")
        props = rec.identity_props(cfg)
        keys = [d["key"] for d in blk["discriminators"]]
        assert keys == sorted(props), b["metadata"]["name"]
        assert {d["key"]: str(d["value"]) for d in blk["discriminators"]} == props
        (variant,) = b["spec"]["variants"]
        assert variant["name"] == blk["name"]
        assert b["spec"]["workload"]["inputs"][0]["expect"]["records"] == cfg["num_questions"]
        assert b["spec"]["workload"]["eval"]["num_fewshot"] == cfg["num_fewshot"]
        assert b["spec"]["workload"]["eval"]["temperature"] == 0
        assert "seed" not in b["spec"]["workload"]["eval"]


def test_definitions_carry_no_threshold_or_score():
    for path in QUALITY.glob("*.yaml"):
        text = path.read_text()
        assert "accuracy_threshold" not in text and "threshold:" not in text, path.name


def test_identity_matches_the_generic_ingest_entries():
    bundle = _need("spyre_clickhouse_ingest.bundle")
    for b in _benchmarks():
        cfg = yaml.safe_load(_config_of(b).read_text())
        blk = b["spec"]["identity"]
        row = {"name": blk["name"], "props": rec.identity_props(cfg), "measurements": {"m": [1.0]}}
        (entry,) = bundle.generic_entries([row])
        assert list(entry["disc_keys"]) == [d["key"] for d in blk["discriminators"]]
        assert entry["tags"] == []


def test_engine_id_equals_the_ingest_id():
    engine = _need("spyre_bench.engine")
    ident = _need("spyre_clickhouse_ingest.identity")
    bundle = _need("spyre_clickhouse_ingest.bundle")
    cat = engine.Catalog([BENCH], [], [])
    assert cat.errors == []
    for b in _benchmarks():
        cfg = yaml.safe_load(_config_of(b).read_text())
        blk = b["spec"]["identity"]
        (entry,) = bundle.generic_entries(
            [{"name": blk["name"], "props": rec.identity_props(cfg), "measurements": {"m": [1.0]}}]
        )
        want = ident.BenchmarkId.derive(
            "spyre-inference", entry["name"], entry["tags"], entry["disc"], entry["disc_keys"]
        )
        case = next(c for c in cat.cases if c["name"] == blk["name"])
        assert str(cat.ids(case)["benchmark_id"]) == str(want)


def _args(tmp_path, **kw):
    ns = {
        "hf_home": str(tmp_path / "hf"),
        "dataset_file": str(tmp_path / "test.jsonl"),
        "harness_version": "v-test",
    }
    return type("A", (), {**ns, **kw})()


def test_build_writes_the_sidecar_and_a_generic_file(tmp_path):
    cfg = yaml.safe_load((GATE_CONFIGS / "qwen2.5-0.5b-instruct.yaml").read_text())
    (tmp_path / "test.jsonl").write_text(
        "# comment\n" + "\n".join(f'{{"q": {i}}}' for i in range(100)) + "\n"
    )
    ref = tmp_path / "hf" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc123\n")
    res = {"accuracy": 0.5, "questions": 64.0, "invalid_rate": 0.0625}
    metrics, bench = rec.build("case-1", cfg, res, _args(tmp_path))
    s, p = metrics["samples"], metrics["props"]
    assert s["llm.eval.n_samples"] == [64] and s["llm.eval.exact_match"] == [0.5]
    assert s["llm.eval.exact_match_stderr"][0] == pytest.approx((0.25 / 64) ** 0.5)
    assert "llm.eval.error_frac" not in s  # the gate has no request error counter
    assert p["run.quality.model_revision"] == "abc123"
    assert p["run.quality.dataset_digest"].startswith("sha256:")
    assert p["run.quality.protocol"].startswith("sha256:") and p["run.quality.temperature"] == "0.0"
    (row,) = bench["benchmarks"]
    assert row["name"] == "case-1" and row["backend"] == "spyre" and row["iterations"] == 64
    assert row["props"] == rec.identity_props(cfg) and set(row["measurements"]) == set(s)


def test_a_later_partial_block_does_not_erase_a_complete_one():
    partial = "ACC Results for org/model-a:\n  Measured metric: 0.1000\n"
    assert rec.parse_gate_log(LOG + partial)["org/model-a"]["accuracy"] == 0.5
    assert rec.parse_gate_log(LOG + "PASSED ACC Results for org/model-a:\n  Measured metric: 0.1\n")


def test_non_finite_values_are_ignored():
    text = "ACC Results for m:\n  Measured metric: nan\n  Questions: 64\n  Invalid rate: 0.0\n"
    assert rec.parse_gate_log(text) == {}


def test_a_header_with_a_prefix_on_the_same_line_is_found():
    body = "  Measured metric: 0.4\n  Questions: 8\n  Invalid rate: 0.0\n"
    text = "tests/x.py::t ACC Results for m:\n" + body
    assert rec.parse_gate_log(text)["m"]["accuracy"] == 0.4


def test_missing_cache_and_dataset_leave_the_pins_out(tmp_path):
    cfg = yaml.safe_load((GATE_CONFIGS / "granite-4.1-3b.yaml").read_text())
    res = {"accuracy": 0.5, "questions": 64.0, "invalid_rate": 0.0}
    _, bench = rec.build("c", cfg, res, _args(tmp_path))
    props = bench["benchmarks"][0]["run_props"]
    assert "run.quality.model_revision" not in props and "run.quality.dataset_digest" not in props


def test_protocol_digest_changes_with_the_protocol_only():
    cfg = yaml.safe_load((GATE_CONFIGS / "granite-4.1-3b.yaml").read_text())
    base = rec.protocol_digest(cfg, "v1")
    assert rec.protocol_digest({**cfg, "accuracy_threshold": 0.1, "tolerance": 0.5}, "v1") == base
    assert rec.protocol_digest({**cfg, "num_fewshot": 8}, "v1") != base
    assert rec.protocol_digest(cfg, "v2") != base


def test_record_cli(tmp_path, capsys, monkeypatch):
    log = tmp_path / "gate.log"
    log.write_text(LOG.replace("org/model-a", "ibm-granite/granite-4.1-3b"))
    cfg = GATE_CONFIGS / "granite-4.1-3b.yaml"
    argv = ["record", "--case", "c1", "--config", str(cfg), "--log", str(log),
            "--repeat-dir", str(tmp_path / "r"),
            "--raw-dir", str(tmp_path / "raw"), "--hf-home", str(tmp_path / "hf"),
            "--dataset-file", str(tmp_path / "none.jsonl")]  # fmt: skip
    monkeypatch.setenv("BENCH_QUIET", "1")
    assert rec.main(argv) == 0
    assert capsys.readouterr().out.strip() == "recorded c1"  # quiet: no score in the output
    assert json.loads((tmp_path / "r" / "metrics.json").read_text())["samples"][
        "llm.eval.n_samples"
    ] == [64]
    assert (tmp_path / "raw" / "c1.json").is_file()
    monkeypatch.delenv("BENCH_QUIET")
    assert rec.main(argv) == 0 and "0.5000 over 64" in capsys.readouterr().out


def test_record_cli_fails_loudly_without_a_block_or_log(tmp_path, capsys):
    cfg = GATE_CONFIGS / "mistral-7b-instruct-v0.3.yaml"
    base = ["record", "--case", "c", "--config", str(cfg), "--repeat-dir", str(tmp_path / "r"),
            "--raw-dir", str(tmp_path / "raw")]  # fmt: skip
    assert rec.main([*base, "--log", str(tmp_path / "nope.log")]) == 2
    (tmp_path / "gate.log").write_text(LOG)
    assert rec.main([*base, "--log", str(tmp_path / "gate.log")]) == 2
    assert "no results block" in capsys.readouterr().err
    assert not (tmp_path / "r" / "metrics.json").exists()


def test_profiles_enable_x86_now_and_keep_p_and_z_as_disabled_stubs(capsys):
    profiles = str(QUALITY / "profiles")
    assert rec.main(["list", "--arch", "x86_64", "--profiles", profiles]) == 0
    names = capsys.readouterr().out.strip().split(",")
    assert sorted(names) == sorted(b["spec"]["identity"]["name"] for b in _benchmarks())
    for arch in ("ppc64le", "s390x"):
        prof = yaml.safe_load((QUALITY / "profiles" / f"{arch}.yml").read_text())
        assert prof["enabled"] is False and prof["cases"] == [] and prof["arch"] == arch
        assert rec.main(["list", "--arch", arch, "--profiles", profiles]) == 2


def test_the_variant_system_matches_the_identity():
    for b in _benchmarks():
        cfg = yaml.safe_load(_config_of(b).read_text())
        props = rec.identity_props(cfg)
        (variant,) = b["spec"]["variants"]
        sys_ = variant["system"]
        assert str(sys_["tp"]) == props["tp"] and sys_["precision"] == props["precision"]
        stem = _config_of(b).stem
        assert sys_["model"] == sys_["token"] == stem


def test_an_architecture_without_a_profile_is_a_clean_error(capsys):
    assert rec.main(["list", "--arch", "nonesuch", "--profiles", str(QUALITY / "profiles")]) == 2
    assert "no quality profile for nonesuch" in capsys.readouterr().err


def test_profile_cases_use_the_models_of_their_benchmarks():
    prof = yaml.safe_load((QUALITY / "profiles" / "x86_64.yml").read_text())
    models = {
        b["spec"]["identity"]["name"]: yaml.safe_load(_config_of(b).read_text())["model_name"]
        for b in _benchmarks()
    }
    assert {c["case"]: c["model"] for c in prof["cases"]} == models


def test_skip_models_are_dropped_from_the_list(tmp_path, capsys):
    prof = yaml.safe_load((QUALITY / "profiles" / "x86_64.yml").read_text())
    prof["skip_models"] = [{"model": prof["cases"][0]["model"], "reason": "gated_model"}]
    (tmp_path / "x86_64.yml").write_text(yaml.safe_dump(prof))
    assert rec.main(["list", "--arch", "x86_64", "--profiles", str(tmp_path)]) == 0
    assert len(capsys.readouterr().out.strip().split(",")) == 2


def test_architecture_names_appear_only_in_the_profile_files():
    for path in list(QUALITY.glob("*.yaml")) + [Path(rec.__file__)]:
        text = path.read_text()
        assert "x86_64" not in text and "ppc64le" not in text and "s390x" not in text, path.name


def test_list_with_a_log_keeps_only_the_models_the_log_covers(tmp_path, capsys):
    profiles = str(QUALITY / "profiles")
    log = tmp_path / "gate.log"
    log.write_text(LOG.replace("org/model-a", "ibm-granite/granite-4.1-3b"))
    assert rec.main(["list", "--arch", "x86_64", "--profiles", profiles, "--log", str(log)]) == 0
    assert capsys.readouterr().out.strip() == "eval_slot-12_granite-4.1-3b_tp1_l64"
    log.write_text("no results here\n")
    assert rec.main(["list", "--arch", "x86_64", "--profiles", profiles, "--log", str(log)]) == 2
    assert (
        rec.main(
            ["list", "--arch", "x86_64", "--profiles", profiles, "--log", str(tmp_path / "absent")]
        )
        == 2
    )


def test_the_seed_is_recorded_only_when_the_gate_config_has_one(tmp_path):
    cfg = yaml.safe_load((GATE_CONFIGS / "granite-4.1-3b.yaml").read_text())
    res = {"accuracy": 0.5, "questions": 64.0, "invalid_rate": 0.0}
    props = rec.build("c", cfg, res, _args(tmp_path))[0]["props"]
    assert "run.quality.seed" not in props
    props = rec.build("c", {**cfg, "seed": 7}, res, _args(tmp_path))[0]["props"]
    assert props["run.quality.seed"] == "7"


def test_the_make_fragment_has_no_stray_lines():
    import re

    ok = re.compile(r"^(#|\t|\.PHONY:|export |[A-Za-z_]+ *[?:]?= |[A-Za-z_-]+:|$)")
    lines = (REPO / "mk" / "bench-quality.mk").read_text().splitlines()
    bad = [ln for ln in lines if not ok.match(ln)]
    assert bad == [], bad
