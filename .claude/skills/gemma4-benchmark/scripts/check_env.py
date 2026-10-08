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

"""Check an environment against the spyre-inference#1102 recipe. Changes nothing.

Imports only torch and torch_spyre (to prove the runtime libraries load); every other
package is located with find_spec and read through its metadata. Checkouts are only read:
`git rev-parse` / `status` / `log` / `rev-list` / `remote` / `merge-base --is-ancestor`.

Each finding is ERROR (cannot be benchmarked), WARN (deviates from the issue's recipe; the
user must accept it) or INFO (recorded only).

The main arm runs spyre-inference at --main-ref (default: <the torch-spyre/spyre-inference
remote>/main), exported from the checkout with `git archive`; this script resolves that ref
and compares it with GitHub's main via `git ls-remote` (a read).

Usage: python -I check_env.py --out env.json --arms hf,main,branch --model PATH
           [--main-ref REF] [--allow-profiler]
Exit status: 0 = no ERROR, 1 = at least one ERROR.
"""

import argparse
import importlib.metadata as md
import importlib.util as iu
import json
import os
import re
import subprocess
import sys

# The issue's pins: hf-adapters #620 and the torch-spyre its hf-adapters env was built with.
HF_SHA = "9b075e46bc9689671fb7ba546ef12b3e509bbdc5"
UPSTREAM = "torch-spyre/spyre-inference"
TS_SHA = "e9d31328345f55ead94d1a65736380e6e07513bc"
PACKAGES = {
    "torch-spyre": "torch_spyre",
    "transformers": "transformers",
    "vllm": "vllm",
    "spyre-inference": "spyre_inference",
    "hf-adapters-spyre": "hf_adapters",
}
SPYRE_LIBS = ("libflex", "libsenlib", "libspyre_comms", "libsendnn", "libaiupti", "libdeeptools")
# /opt/ibm/spyre/runtime/lib/libflex.so -> /opt/ibm/spyre;
# <tree>/sentient/runtime/lib/... -> <tree>/sentient
SPYRE_ROOT_RE = re.compile(
    r"(/\S*?)/(?:runtime|senlib|deeptools|spyre-comms|spyre_comms|libaiupti)/"
)
# Host settings that do not change what is measured.
BENIGN_ENV = {
    "OMP_NUM_THREADS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_SENDNN_CACHE_DIR",
    "TORCH_SENDNN_CACHE_ENABLE",
    "TORCH_SENDNN_LOG",
    "VLLM_PLUGINS",
    "VLLM_CACHE_ROOT",
    "SPYRE_COMMS_INSTALL_DIR",
    "SPYRE_DEVICES",
}
KNOB_RE = re.compile(
    r"^(SPYRE_|VLLM_|TORCH_LOGS$|TORCH_COMPILE|TORCHINDUCTOR_|TORCH_SENDNN|"
    r"FRONTEND_POOL_ALLOCATION$|SENCORES$|CO_OPTIMIZING|DXP_)"
)

findings: list[dict] = []


def flag(level: str, arm: str, message: str) -> None:
    findings.append({"level": level, "arm": arm, "message": message})


def _run(*cmd: str, env: dict | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _package(dist: str, module: str) -> dict:
    try:
        info: dict = {"version": md.version(dist)}
    except md.PackageNotFoundError:
        return {}
    spec = iu.find_spec(module)
    if spec is not None and spec.origin:
        info["location"] = os.path.dirname(os.path.realpath(spec.origin))
        top = _run("git", "-C", info["location"], "rev-parse", "--show-toplevel")
        if top:
            info["checkout"] = top
            info["commit"] = _run("git", "-C", top, "rev-parse", "HEAD")
            status = _run("git", "-C", top, "status", "--porcelain", "--untracked-files=no")
            info["dirty"] = len(status.splitlines()) if status else 0
    if not info.get("commit"):
        try:
            url = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
            info["commit"] = (url.get("vcs_info") or {}).get("commit_id")
        except (OSError, ValueError):
            pass
    # setuptools-scm versions carry the commit they were built from: 0.6.0.dev87+head.g2ab31d08
    if m := re.search(r"\+.*?g([0-9a-f]{7,})", info["version"]):
        info["built_from"] = m.group(1)
    return info


def _requirement_met(requirement: str, dist: str, installed: str) -> bool | None:
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return None
    req = Requirement(requirement)
    if req.name != dist or (req.marker and not req.marker.evaluate({"extra": ""})):
        return None
    return req.specifier.contains(installed.split("+")[0], prereleases=True)


def _requirements(env: dict, owner: str, dist: str, arm: str) -> None:
    try:
        reqs = md.requires(owner) or []
    except md.PackageNotFoundError:
        return
    installed = env.get(dist, {}).get("version")
    for r in reqs:
        ok = _requirement_met(r, dist, installed) if installed else None
        if ok is False:
            flag("WARN", arm, f"{owner} requires `{r}`, the env has {dist} {installed}")


def _check_torch_spyre(env: dict, arms: list[str], allow_profiler: bool) -> None:
    ts = env["torch-spyre"]
    so = ts["c_so"]
    # libc10/libtorch live in torch/lib, which `import torch` loads; ldd needs it on the path.
    ld_path = os.pathsep.join(
        p for p in (env["torch"]["lib"], os.environ.get("LD_LIBRARY_PATH")) if p
    )
    ldd = _run("ldd", so, env={**os.environ, "LD_LIBRARY_PATH": ld_path}) or ""
    if "not found" in ldd:
        missing = [line.split()[0] for line in ldd.splitlines() if "not found" in line]
        flag("ERROR", "both", f"{so} has unresolved libraries: {', '.join(missing)}")
    if "libaiupti" in ldd:
        flag(
            "WARN" if allow_profiler else "ERROR",
            "both",
            "torch_spyre/_C.so links libaiupti (a USE_SPYRE_PROFILER=1 build): latency is inflated",
        )
    roots = set()
    for line in ldd.splitlines():
        lib, _, target = line.partition("=>")
        if any(name in lib for name in SPYRE_LIBS) and (m := SPYRE_ROOT_RE.match(target.strip())):
            roots.add(m.group(1))
    if len(roots) > 1:
        flag(
            "WARN",
            "both",
            f"Spyre runtime libraries resolve from more than one install: {sorted(roots)}",
        )
    env["torch-spyre"]["spyre_lib_roots"] = sorted(roots)

    commit, checkout = ts.get("commit"), ts.get("checkout")
    if ts.get("built_from") and commit and not commit.startswith(ts["built_from"]):
        flag(
            "WARN",
            "both",
            f"torch-spyre ({checkout or ts['version']}) is at {commit[:12]} but the installed "
            f"build is from {ts['built_from']}: _C.so is older than the Python sources",
        )
    if ts.get("dirty"):
        flag(
            "WARN",
            "both",
            f"torch-spyre checkout {checkout} has {ts['dirty']} modified tracked file(s)",
        )
    if "hf" in arms and commit != TS_SHA:
        if (
            checkout
            and _run("git", "-C", checkout, "merge-base", "--is-ancestor", TS_SHA, "HEAD")
            is not None
        ):
            ahead = _run("git", "-C", checkout, "rev-list", "--count", f"{TS_SHA}..HEAD")
            flag(
                "WARN",
                "hf",
                f"torch-spyre is {commit[:12]}, not the issue's {TS_SHA[:8]} "
                f"(it contains it, {ahead} commits newer)",
            )
        else:
            flag("WARN", "hf", f"torch-spyre is {str(commit)[:12]}, not the issue's {TS_SHA[:8]}")


def _check_hf(env: dict) -> None:
    hf = env.get("hf-adapters-spyre")
    if not hf:
        flag("ERROR", "hf", "hf-adapters (hf-adapters-spyre) is not installed")
        return
    commit = hf.get("commit") or ""
    if commit != HF_SHA and not (hf.get("built_from") and HF_SHA.startswith(hf["built_from"])):
        flag(
            "WARN",
            "hf",
            f"hf-adapters is at {commit[:12] or hf['version']}, not the issue's "
            f"{HF_SHA[:7]} (PR #620)",
        )
    if hf.get("dirty"):
        flag(
            "WARN",
            "hf",
            f"hf-adapters checkout {hf.get('checkout')} has {hf['dirty']} modified tracked file(s)",
        )
    _requirements(env, "hf-adapters-spyre", "transformers", "hf")
    _requirements(env, "hf-adapters-spyre", "torch", "hf")


def _check_vllm(env: dict) -> None:
    for dist in ("vllm", "spyre-inference"):
        if not env.get(dist):
            flag("ERROR", "vllm", f"{dist} is not installed")
    si = env.get("spyre-inference") or {}
    if not si:
        return
    if si.get("dirty"):
        flag(
            "WARN",
            "vllm",
            f"spyre-inference checkout {si.get('checkout')} has "
            f"{si['dirty']} modified tracked file(s)",
        )
    if si.get("built_from") and si.get("commit") and not si["commit"].startswith(si["built_from"]):
        flag(
            "INFO",
            "vllm",
            f"spyre-inference metadata says {si['built_from']}, checkout is at "
            f"{si['commit'][:12]} (editable: the checkout's code runs)",
        )
    if not si.get("checkout"):
        _requirements(env, "spyre-inference", "vllm", "vllm")
        return
    # The checkout's pyproject is the source of truth: installers may strip requirements
    # from the installed metadata so they are not re-resolved.
    try:
        import tomllib

        with open(os.path.join(si["checkout"], "pyproject.toml"), "rb") as f:
            project = tomllib.load(f)
    except (OSError, ValueError, ImportError):
        return
    vllm_version = (env.get("vllm") or {}).get("version")
    for r in project.get("project", {}).get("dependencies", []):
        if vllm_version and _requirement_met(r, "vllm", vllm_version) is False:
            flag("WARN", "vllm", f"spyre-inference requires `{r}`, the env has vllm {vllm_version}")
    pin = (project.get("tool", {}).get("uv", {}).get("sources", {}) or {}).get("torch-spyre") or {}
    pin = pin.get("rev") if isinstance(pin, dict) else None
    installed = env["torch-spyre"].get("commit") or ""
    if pin and not installed.startswith(pin[:12]):
        flag(
            "INFO",
            "vllm",
            f"spyre-inference pins torch-spyre {pin[:12]}, the env runs {installed[:12]}",
        )


def _check_main(env: dict, ref: str) -> None:
    checkout = (env.get("spyre-inference") or {}).get("checkout")
    if not checkout:
        flag("ERROR", "main", "the main arm needs spyre-inference installed from a git checkout")
        return
    git = ("git", "-C", checkout)
    if ref == "auto":
        remotes = (_run(*git, "remote", "-v") or "").splitlines()
        names = sorted(
            {r.split()[0] for r in remotes if re.search(rf"[/:]{UPSTREAM}(\.git)?\s", r)}
        )
        if not names:
            flag("ERROR", "main", f"no remote of {checkout} points at {UPSTREAM}; pass --main-ref")
            return
        ref = f"{names[0]}/main"
    sha = _run(*git, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if not sha:
        flag("ERROR", "main", f"--main-ref {ref} does not resolve in {checkout}")
        return
    main = {"ref": ref, "commit": sha, "subject": _run(*git, "log", "-1", "--format=%s", sha)}
    env["spyre-inference-main"] = main
    remote_head = _run(
        "timeout", "20", "git", "ls-remote", f"https://github.com/{UPSTREAM}.git", "refs/heads/main"
    )
    if remote_head:
        main["github_main"] = remote_head.split()[0]
        if main["github_main"] != sha:
            flag(
                "WARN",
                "main",
                f"{ref} is {sha[:12]} but GitHub main is {main['github_main'][:12]}: "
                "the main arm would measure a stale main (fetch it yourself)",
            )
    else:
        flag("INFO", "main", f"could not reach GitHub to confirm {ref} is current")
    head = env["spyre-inference"].get("commit")
    if head and _run(*git, "merge-base", "--is-ancestor", sha, head) is None:
        missing = _run(*git, "rev-list", "--count", f"{head}..{sha}")
        flag(
            "WARN",
            "main",
            f"the branch under test lacks {missing} commit(s) of {ref}: "
            "main vs branch also measures those",
        )
    elif head:
        main["branch_ahead"] = int(_run(*git, "rev-list", "--count", f"{sha}..{head}") or 0)


def _check_host(args: argparse.Namespace) -> None:
    for key in sorted(os.environ):
        if KNOB_RE.match(key):
            level = "INFO" if key in BENIGN_ENV else "WARN"
            flag(
                level,
                "both",
                f"environment sets {key}={os.environ[key]}"
                + ("" if level == "INFO" else " (the issue ran without it)"),
            )
    if os.path.isabs(args.model):
        try:
            with open(os.path.join(args.model, "config.json")) as f:
                model_type = json.load(f).get("model_type", "")
        except (OSError, ValueError):
            flag("ERROR", "both", f"cannot read {args.model}/config.json")
            return
        if "gemma4" not in model_type:
            flag("WARN", "both", f"{args.model} has model_type={model_type!r}, not Gemma-4")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="hf,main,branch")
    p.add_argument("--main-ref", default="auto")
    p.add_argument("--model", required=True)
    p.add_argument("--allow-profiler", action="store_true")
    args = p.parse_args()
    arms = args.arms.split(",")

    env: dict = {"python": sys.executable}
    try:
        import torch
        import torch_spyre
    except Exception as exc:  # noqa: BLE001 - any import failure is the finding
        flag("ERROR", "both", f"import torch / torch_spyre failed: {type(exc).__name__}: {exc}")
    else:
        env["torch"] = {
            "version": torch.__version__,
            "lib": os.path.join(os.path.dirname(torch.__file__), "lib"),
        }
        for dist, module in PACKAGES.items():
            env[dist] = _package(dist, module)
        env["torch-spyre"]["c_so"] = os.path.join(os.path.dirname(torch_spyre.__file__), "_C.so")
        _check_torch_spyre(env, arms, args.allow_profiler)
        _requirements(env, "torch-spyre", "torch", "both")
        if "hf" in arms:
            _check_hf(env)
        si = env.get("spyre-inference") or {}
        if si.get("checkout"):
            git = ("git", "-C", si["checkout"])
            si["branch"] = _run(*git, "rev-parse", "--abbrev-ref", "HEAD")
            si["subject"] = _run(*git, "log", "-1", "--format=%s")
        if {"main", "branch"} & set(arms):
            _check_vllm(env)
        if "main" in arms:
            _check_main(env, args.main_ref)
    _check_host(args)
    env["findings"] = findings
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
    return 1 if any(f["level"] == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
