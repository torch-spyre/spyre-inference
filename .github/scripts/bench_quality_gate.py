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

"""Record the score the accuracy gate already measures as a Bench quality result.

The gate (the upstream accuracy test, run with `pytest -s`) prints one results block per
model and asserts its threshold. Nothing here judges or gates: `record` reads that output and writes
what the engine and the ingest read.

  list    the case names a per-architecture profile enables (comma separated); with --log only the
          cases whose model the log has a results block for, since a sharded gate run covers some
          models per job
  record  one case: the gate log's block for the case's model -> `metrics.json` (engine sidecar) and
          `<raw>/<case>.json` (a generic `benchmarks` file for the ingest, test type quality)

The gate has no per-request error counter (a failed request scores as an invalid answer), so no
error rate is recorded; the invalid share is. The per-question vector is not in its output either.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

HARNESS = "accuracy-gate"
TASK = "slot-12"
DATASET_FILE = "test.jsonl"
BLOCK = re.compile(r"(?:^|\s)\S+ Results for (.+):$")  # "<TASK> Results for <model>:"
FIELDS = {
    "Measured metric:": "accuracy",
    "Questions:": "questions",
    "Invalid rate:": "invalid_rate",
}
PRECISION = "fp16"  # the gate configs run the checkpoints unquantized
QUIET = "BENCH_QUIET"


def parse_gate_log(text: str) -> dict[str, dict]:
    """{model: results} from the gate's printed blocks. A block counts once it has all its fields,
    so a later partial block (a retry, a log cut off mid-block) never erases a complete one."""
    out: dict[str, dict] = {}
    model, cur = None, {}
    for raw in text.splitlines():
        line = raw.strip()
        head = BLOCK.search(line)
        if head:
            model, cur = head.group(1), {}
        elif model is not None:
            for label, key in FIELDS.items():
                if line.startswith(label):
                    with contextlib.suppress(ValueError):
                        value = float(line[len(label) :])
                        if math.isfinite(value):
                            cur[key] = value
            if set(cur) == set(FIELDS.values()):
                out[model] = dict(cur)
    return out


def server_arg(args: str, flag: str, default: str) -> str:
    parts = args.split()
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
        if p.startswith(flag + "="):
            return p.split("=", 1)[1]
    return default


def identity_props(cfg: dict) -> dict[str, str]:
    """The identity of the case, as the generic ingest hashes it: every prop is a discriminator."""
    return {
        "dataset": TASK,
        "harness": HARNESS,
        "model": cfg["model_name"],
        "num_fewshot": str(cfg["num_fewshot"]),
        "num_questions": str(cfg["num_questions"]),
        "precision": PRECISION,
        "record_type": "quality",
        "task": TASK,
        "tp": server_arg(cfg.get("server_args", ""), "--tensor-parallel-size", "1"),
    }


def protocol_digest(cfg: dict, harness_version: str) -> str:
    """Digest of everything that changes what the score means, as the gate's config states it."""
    body = {
        "harness": HARNESS,
        "version": harness_version,
        "num_fewshot": cfg["num_fewshot"],
        "num_questions": cfg["num_questions"],
        "max_tokens": cfg.get("max_tokens", 256),
        "temperature": cfg.get("temperature", 0.0),
        "use_chat_completions": cfg.get("use_chat_completions", False),
        "gen_prefix": cfg.get("gen_prefix", ""),
    }
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def question_digest(path: Path, n: int) -> str | None:
    """Digest of the first n questions of the gate's cached test file, in file order."""
    if not path.is_file():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")][
        :n
    ]
    return "sha256:" + hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def model_revision(model: str, hf_home: str | None) -> str | None:
    """Commit of the cached snapshot the gate served, from the HF cache refs; None if not cached."""
    home = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    ref = home / "hub" / ("models--" + model.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def stderr_binomial(p: float, n: int) -> float:
    return (max(p * (1 - p), 0.0) / n) ** 0.5 if n else 0.0


def build(case: str, cfg: dict, res: dict, args) -> tuple[dict, dict]:
    """(metrics.json document, benchmarks file document) for one case."""
    n = int(res["questions"])
    acc = res["accuracy"]
    revision = model_revision(cfg["model_name"], args.hf_home)
    digest = question_digest(Path(args.dataset_file), n)
    props = {
        "run.quality.harness": HARNESS,
        "run.quality.harness_version": args.harness_version,
        "run.quality.protocol": protocol_digest(cfg, args.harness_version),
        "run.quality.temperature": str(cfg.get("temperature", 0.0)),
        **({"run.quality.seed": str(cfg["seed"])} if "seed" in cfg else {}),
        **({"run.quality.model_revision": revision} if revision else {}),
        **({"run.quality.dataset_digest": digest} if digest else {}),
    }
    samples = {
        "llm.eval.exact_match": [acc],
        "llm.eval.exact_match_stderr": [stderr_binomial(acc, n)],
        "llm.eval.n_samples": [n],
        "llm.eval.invalid_frac": [res["invalid_rate"]],
    }
    row = {
        "name": case,
        "tags": [],
        "backend": "spyre",
        "props": identity_props(cfg),
        "measurements": samples,
        "iterations": n,
        "run_props": props,
    }
    return {"samples": samples, "props": props}, {"benchmarks": [row]}


def load_profile(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def cmd_list(args) -> int:
    path = Path(args.profiles) / f"{args.arch}.yml"
    if not path.is_file():
        print(f"no quality profile for {args.arch} in {args.profiles}", file=sys.stderr)
        return 2
    prof = load_profile(path)
    if not prof.get("enabled"):
        print(f"quality profile {args.arch} is disabled", file=sys.stderr)
        return 2
    skipped = {s["model"] for s in prof.get("skip_models", [])}
    models = {c["case"]: c["model"] for c in prof["cases"]}
    names = [c for c, m in models.items() if m not in skipped]
    if args.log:
        have = (
            parse_gate_log(Path(args.log).read_text(errors="replace"))
            if Path(args.log).is_file()
            else {}
        )
        names = [c for c in names if models[c] in have]
    if not names:
        print(f"quality profile {args.arch} lists no runnable case", file=sys.stderr)
        return 2
    print(",".join(names))
    return 0


def cmd_record(args) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text())
    log = Path(args.log)
    if not log.is_file():
        print(
            f"gate log {log} not found: run the quality gate with pytest -s and tee it there",
            file=sys.stderr,
        )
        return 2
    blocks = parse_gate_log(log.read_text(errors="replace"))
    res = blocks.get(cfg["model_name"])
    if res is None:
        print(f"{args.case}: no results block for {cfg['model_name']} in {log}", file=sys.stderr)
        return 2
    metrics, bench = build(args.case, cfg, res, args)
    Path(args.repeat_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.repeat_dir) / "metrics.json").write_text(json.dumps(metrics))
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{args.case}.json").write_text(json.dumps(bench))
    quiet = bool(os.environ.get(QUIET))
    print(
        f"recorded {args.case}"
        + ("" if quiet else f": {res['accuracy']:.4f} over {int(res['questions'])}")
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="case names enabled by a per-architecture profile")
    ls.add_argument("--arch", required=True)
    ls.add_argument("--profiles", default="benchmarks/quality/profiles")
    ls.add_argument(
        "--log", default="", help="keep only the cases this gate log has a results block for"
    )
    ls.set_defaults(fn=cmd_list)
    rec = sub.add_parser(
        "record", help="write one case's sidecar and benchmarks file from a gate log"
    )
    rec.add_argument("--case", required=True)
    rec.add_argument("--config", required=True, help="the gate's config yaml for this case's model")
    rec.add_argument("--log", default=os.environ.get("BENCH_QUALITY_LOG", ""))
    rec.add_argument("--repeat-dir", required=True)
    rec.add_argument("--raw-dir", required=True)
    rec.add_argument(
        "--harness-version", default=os.environ.get("BENCH_QUALITY_HARNESS_VERSION", "unknown")
    )
    rec.add_argument("--hf-home", default=None)
    rec.add_argument("--dataset-file", default=str(Path(tempfile.gettempdir()) / DATASET_FILE))
    rec.set_defaults(fn=cmd_record)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
