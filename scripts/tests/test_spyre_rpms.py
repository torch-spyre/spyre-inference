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

"""Offline tests for scripts/spyre_rpms.py.

These live beside the script, not under tests/, because tests/conftest.py
imports spyre_inference (which needs Spyre hardware to build) while this script
is a standalone lint/resolve tool that must run anywhere, including a plain
ubuntu-latest GHA runner. Run them with:

    uv run --isolated --no-project --with pydantic --with requests --with pytest \
        pytest scripts/tests/test_spyre_rpms.py

The plan fixture is built by `_plan()` below rather than checked in: an
orchestrator build plan is internal, so committing one here would publish the
component graph. The fixture is written to the same SHAPE instead, covering every
case the parser handles: a built node, a dropped node with no `identity_inputs`,
subpackage dep aliases, an image `base=<identity>` edge, a `dep_pins` `base` key,
and a checkout name that is not a plan component.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
from pydantic import ValidationError

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent  # <repo>/scripts/tests -> <repo>
# Placeholder host only; see the module docstring.
GIT = "https://git.example.com"


def _load_module():
    path = HERE.parent / "spyre_rpms.py"
    spec = importlib.util.spec_from_file_location("spyre_rpms", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spyre_rpms"] = mod
    spec.loader.exec_module(mod)
    return mod


sr = _load_module()


# Arbitrary but internally consistent: every dep edge points at the source sha
# its producer declares, so a healthy plan must show zero mismatches.
SHA = {
    "deeptools": "092f4e799b2523c4d66a40185a4b55d713514d91",
    "flex": "5aaf49028ddfe481c3740580860cbc4e0b94fbb6",
    "senlib": "15d57f061f4138609c8dc1ca31073cfdd7c2c47f",
    "comms": "748f4acefa023869696a4228247be61549771a96",
    "llvm": "9a659b398001f4b7bfb1c99f51f69a3e715266d0",
    "runtime": "4acf30579fd4fd71fed533c9a743e0d01072a635",
    "docs": "b68f5aa6f5c1bd1088b0b9debd5874979ec268be",
}


def _artifacts(names, id12):
    return [
        {
            "nevra": f"{n}-*.{id12}.*.x86_64",
            "name": n,
            "id12": id12,
            "publish": "rpm-local",
            "subroot": "next",
            "arch": "x86_64",
        }
        for n in names
    ]


def _built(component, id12, source, deps, names, extra_checkouts=()):
    return {
        "component": component,
        "kind": "rpm",
        "mode": component,
        "arch": "amd64",
        "id12": id12,
        "identity": id12 + "0" * (64 - len(id12)),
        "identity_inputs": {
            "version": "1",
            "source": source,
            "recipe_sha256": "a" * 64,
            "recipe_files": ["Makefile"],
            "dep_ids": [f"{k}@{v}" for k, v in deps.items()],
        },
        "checkouts": [
            {
                "name": "src",
                "repo": f"{GIT}/org/{component}.git",
                "sha": source,
                "branch": "main",
                "recipe": True,
                "identity": True,
            },
            *extra_checkouts,
        ],
        "artifacts": _artifacts(names, id12),
    }


def _dropped(component, id12, source, names, kind="rpm", dep_pins=None):
    """A node already published by an earlier run: no identity_inputs at all."""
    return {
        "component": component,
        "kind": kind,
        "mode": component,
        "id12": id12,
        "checkouts": [
            {
                "name": "src",
                "repo": f"{GIT}/org/{component}.git",
                "sha": source,
                "recipe": True,
                "identity": True,
            }
        ],
        "dep_pins": dep_pins or {},
        "artifacts": _artifacts(names, id12),
    }


def _plan():
    deeptools = _built(
        "deeptools",
        "b591b2c83606",
        SHA["deeptools"],
        {"llvm-build": SHA["llvm"], "llvm-source": SHA["llvm"]},
        ["ibm-deeptools", "ibm-deeptools-devel"],
    )
    # `-devel` / `-headers` are subpackages and must fold onto their parent.
    flex = _built(
        "flex",
        "5cefcccce4cf",
        SHA["flex"],
        {"deeptools-devel": SHA["deeptools"], "senlib-headers": SHA["senlib"]},
        ["ibm-flex", "ibm-flex-devel"],
    )
    comms = _built(
        "spyre-comms",
        "2c1372c287a5",
        SHA["comms"],
        {"flex": SHA["flex"]},
        ["ibm-spyre-comms", "ibm-spyre-comms-devel"],
    )
    # An image node. `base=<identity>` in dep_ids and a bare `base` in dep_pins
    # are layers, not components, so neither is a checkable edge. Its `docs`
    # checkout is a separate repo, so that dep must resolve by checkout name.
    image = {
        "component": "runtime",
        "kind": "image",
        "mode": "runtime-dev",
        "id12": "7bfe15bff5c7",
        "identity_inputs": {
            "source": SHA["runtime"],
            "dep_ids": [
                "base=" + "c" * 64,
                f"deeptools@{SHA['deeptools']}",
                f"runtime-docs@{SHA['docs']}",
            ],
        },
        "dep_pins": {"base": SHA["runtime"], "flex": SHA["flex"]},
        "checkouts": [
            {
                "name": "src",
                "repo": f"{GIT}/org/runtime.git",
                "sha": SHA["runtime"],
                "identity": True,
            },
            {"name": "runtime-docs", "repo": f"{GIT}/other/runtime-docs.git", "sha": SHA["docs"]},
        ],
        "artifacts": [],
    }
    senlib = _dropped(
        "senlib",
        "c7bc40ab1aa0",
        SHA["senlib"],
        ["ibm-senlib-core", "ibm-senlib-dd2", "ibm-senlib-headers"],
    )
    llvm = _dropped("llvm", "080ddeea9a07", SHA["llvm"], [], kind="generic")

    def key(n):
        src = (n.get("identity_inputs") or {}).get("source") or n["id12"]
        return f"{n['component']}|{n['mode']}|{src}"

    nodes = {key(n): n for n in (deeptools, flex, comms, image)}
    dropped = {key(n): n for n in (senlib, llvm)}
    return {"levels": [], "nodes": nodes, "dropped": list(dropped), "dropped_nodes": dropped}


@pytest.fixture(scope="module")
def plan():
    return sr.Plan.parse(_plan(), build=7457, arch="x86_64")


# --- plan parsing ----------------------------------------------------------


def test_parses_both_node_shapes(plan):
    """Dropped nodes carry no identity_inputs but are still part of the set."""
    by = plan.by_component()
    assert {"deeptools", "flex", "senlib", "spyre-comms"} <= set(by)
    assert any(n.dropped for n in by.values()), "fixture should exercise a dropped node"
    # A dropped node has no identity_inputs, so its source must come from
    # checkouts[]; it must still resolve.
    for comp, node in by.items():
        if node.kind == "rpm":
            assert node.source, f"{comp} has no source sha"
            assert node.id12, f"{comp} has no id12"


def test_dropped_node_source_comes_from_checkouts(plan):
    senlib = plan.by_component()["senlib"]
    assert senlib.source == "15d57f061f4138609c8dc1ca31073cfdd7c2c47f"


def test_rpm_names_from_artifacts(plan):
    assert plan.by_component()["senlib"].rpm_names() == [
        "ibm-senlib-core",
        "ibm-senlib-dd2",
        "ibm-senlib-headers",
    ]


def test_real_plan_is_abi_coherent(plan):
    """The invariant holds on a real production plan (the F12/F14 baseline)."""
    pc = plan.by_component()
    edges = sr.abi_edges(plan.sources(), {c: n.deps for c, n in pc.items() if n.deps})
    assert edges, "no edges checked: the check would be vacuous"
    assert [e for e in edges if e.ok] == edges, [
        f"{e.consumer}->{e.producer} wants {e.wants[:12]} has {e.has[:12]}"
        for e in edges
        if not e.ok
    ]


def test_base_image_identity_is_not_an_edge(plan):
    """`base=<identity>` names an identity, not a source sha, so it is skipped."""
    for node in plan.nodes.values():
        assert not any(d.startswith("base") for d in node.deps)


# --- dep-name canonicalization (the F14 regression) ------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("deeptools-devel", "deeptools"),
        ("flex-devel", "flex"),
        ("senlib-headers", "senlib"),
        ("llvm-build", "llvm"),
        ("llvm-source", "llvm"),
        ("deeptools", "deeptools"),
    ],
)
def test_canonical_dep_folds_subpackages(name, expected):
    assert sr.canonical_dep(name) == expected


@pytest.mark.parametrize(
    "name",
    ["torch-spyre-docs", "spyre-perf-suite-src", "spyre_common_headers", "pytorch"],
)
def test_canonical_dep_does_not_fold_separate_repos(name):
    """These look related but are different repos; folding invented 4 phantom
    ABI mismatches on known-good plans."""
    assert sr.canonical_dep(name) == name


def test_a_real_component_is_never_rewritten():
    known = frozenset({"senlib-headers", "senlib"})
    assert sr.canonical_dep("senlib-headers", known) == "senlib-headers"


# --- ABI invariant ---------------------------------------------------------


def test_abi_edge_detects_mismatch():
    bad = sr.AbiEdge("flex", "deeptools", "9a659b398001", "092f4e799b25")
    assert not bad.ok


def test_abi_edge_compares_on_shorter_prefix():
    assert sr.AbiEdge("a", "b", "092f4e799b25", "092f4e799b2523c4d66a").ok


def test_abi_edge_rejects_too_short_a_sha():
    """A 4-char prefix must not be accepted as a match."""
    assert not sr.AbiEdge("a", "b", "092f", "092f4e799b25").ok


def test_unresolved_deps_are_reported_not_passed():
    out = sr.unresolved_deps({"flex": "aaaaaaa"}, {"flex": {"nowhere": "bbbbbbb"}})
    assert out == ["flex -> nowhere"]


# --- lock verification -----------------------------------------------------


FLEX_VER = "2.0.0_rc.1-0.main.87+87.5aaf4902.5cefcccce4cf.5aaf490"
DT_VER = "2.0.0_rc.1-0.master.198+198.092f4e79.b591b2c83606.092f4e7"


def _lock():
    """A committed lock in the shipped v2 shape, with the [abi] attribution."""
    return {
        "version": 2,
        "defaults": {"tree": "next"},
        "packages": {
            "ibm-deeptools": {"version": DT_VER},
            "ibm-flex": {"version": FLEX_VER},
        },
        "overrides": {
            "s390x": {
                "ibm-deeptools": {"version": DT_VER},
                "ibm-flex": {"version": FLEX_VER},
            }
        },
        "abi": {
            "deeptools": {
                "source": SHA["deeptools"],
                "id12": "b591b2c83606",
                "packages": ["ibm-deeptools"],
                "deps": {},
            },
            "flex": {
                "source": SHA["flex"],
                "id12": "5cefcccce4cf",
                "packages": ["ibm-flex"],
                "deps": {"deeptools": SHA["deeptools"]},
            },
        },
    }


def test_valid_lock_verifies():
    edges, bad, _ = sr.verify_lock(_lock())
    assert not bad
    assert len(edges) == 1, "exactly one checkable edge in the fixture"


def test_mismatched_dep_sha_fails():
    lock = _lock()
    lock["abi"]["flex"]["deps"]["deeptools"] = SHA["llvm"]
    _, bad, _ = sr.verify_lock(lock)
    assert any("flex was built against deeptools@" in b for b in bad)


def test_pin_not_carrying_locked_id12_fails():
    """The mixed-build case: a real RPM version from a different build."""
    lock = _lock()
    lock["packages"]["ibm-flex"]["version"] = FLEX_VER.replace("5cefcccce4cf", "2feedcd04810")
    _, bad, _ = sr.verify_lock(lock)
    assert any("does not carry the locked identity 5cefcccce4cf" in b for b in bad)


def test_per_arch_override_is_checked_too():
    """An override is where the old format wildcarded the identity away."""
    lock = _lock()
    lock["overrides"]["s390x"]["ibm-flex"]["version"] = FLEX_VER.replace(
        "5cefcccce4cf", "2feedcd04810"
    )
    _, bad, _ = sr.verify_lock(lock)
    assert any("[s390x]" in b and "5cefcccce4cf" in b for b in bad)


def test_package_pinned_but_unattributed_fails():
    """A pin with no [abi] entry has nothing proving it came from the set."""
    lock = _lock()
    lock["packages"]["ibm-mystery"] = {"version": "1.0-1.deadbeefcafe.x"}
    _, bad, _ = sr.verify_lock(lock)
    assert any("not covered by any [abi] entry" in b for b in bad)


def test_abi_package_missing_from_packages_fails():
    lock = _lock()
    lock["abi"]["flex"]["packages"].append("ibm-flex-devel")
    _, bad, _ = sr.verify_lock(lock)
    assert any("missing from [packages]" in b for b in bad)


def test_a_lock_with_no_abi_section_is_rejected():
    """The old hand-maintained format cannot be proven coherent, so `verify`
    must refuse it rather than pass it silently."""
    lock = _lock()
    del lock["abi"]
    _, bad, _ = sr.verify_lock(lock)
    assert any("no [abi] section" in b for b in bad)


# --- input schema ----------------------------------------------------------


def test_exactly_one_ref_required():
    with pytest.raises(ValueError, match="exactly one"):
        sr.ComponentSpec(commit="a" * 40, branch="main").ref()
    with pytest.raises(ValueError, match="exactly one"):
        sr.ComponentSpec().ref()


def test_commit_must_be_sha_shaped():
    with pytest.raises(ValidationError):
        sr.ComponentSpec(commit="not-a-sha")


def test_unknown_arch_rejected():
    with pytest.raises(ValidationError, match="unknown arch"):
        sr.InputDoc(arches=["risc-v"])


def test_unknown_tree_rejected():
    with pytest.raises(ValidationError, match="must be 'next' or 'prod'"):
        sr.InputDoc(tree="staging")


def test_unknown_key_rejected():
    """A misspelled toml key must fail loudly, not be silently ignored."""
    misspelled = "compon" + "ets"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        sr.InputDoc(**{misspelled: {}})


def test_default_arches_is_x86_only():
    """A 3-arch default would make most orchestrator runs unusable as an oracle,
    since the large majority publish an amd64 plan only."""
    assert sr.DEFAULT_ARCHES == ("x86_64",)
    assert sr.InputDoc().arches == ["x86_64"]


def test_input_can_widen_arches():
    assert sr.InputDoc(arches=["x86_64", "s390x"]).arches == ["x86_64", "s390x"]


@pytest.mark.parametrize("flag", ["-a", "--arches"])
def test_arches_flag_parses_on_resolve_and_check(flag):
    p = sr.build_parser()
    for cmd in ("resolve", "check"):
        args = p.parse_args([cmd, flag, "x86_64", "s390x"])
        assert args.arches == ["x86_64", "s390x"]


def test_arches_flag_defaults_to_none_so_the_input_wins():
    """None, not a list: the toml's own `arches` must survive an unpassed flag."""
    assert sr.build_parser().parse_args(["resolve"]).arches is None


def test_arches_flag_rejects_an_unknown_arch():
    with pytest.raises(SystemExit):
        sr.build_parser().parse_args(["resolve", "--arches", "riscv"])


def test_show_plan_keeps_its_own_single_arch_flag():
    """show-plan takes --arch (singular); --arches is only on resolve/check, so
    the two similar flags never land on the same subcommand."""
    args = sr.build_parser().parse_args(["show-plan", "--arch", "s390x"])
    assert args.arch == "s390x"
    assert not hasattr(args, "arches")


def test_plan_arch_names_differ_from_rpm_arch_names():
    """plan-amd64.json holds the x86_64 RPMs; conflating them 404s."""
    assert sr.PLAN_ARCH["x86_64"] == "amd64"
    assert set(sr.PLAN_ARCH) == set(sr.ARCHES)


def test_missing_nodes_key_is_an_error():
    """A 404 body parses as JSON, so absence of `nodes` must raise."""
    with pytest.raises(sr.ResolveError, match="no 'nodes' key"):
        sr.Plan.parse({"errors": [{"status": 404}]}, build=1, arch="x86_64")


# --- lock rendering --------------------------------------------------------


def test_render_is_deterministic_and_parses():
    import tomllib

    lock = {
        "version": 2,
        "tree": "next",
        "orchestrator_build": 7457,
        "arches": ["x86_64", "s390x"],
        "components": {
            "flex": {
                "ref_kind": "commit",
                "ref": SHA["flex"],
                "source": SHA["flex"],
                "deps": {"deeptools": SHA["deeptools"]},
                "id12": {"x86_64": "5cefcccce4cf", "s390x": "5cefcccce4cf"},
                "packages": {"ibm-flex": {"x86_64": FLEX_VER, "s390x": FLEX_VER}},
            }
        },
    }
    a = sr.render_lock(lock)
    assert a == sr.render_lock(lock)
    d = tomllib.loads(a)
    # The shipped v2 contract the other 28 consumers read.
    assert d["version"] == 2
    assert d["defaults"]["tree"] == "next"
    assert d["packages"]["ibm-flex"]["version"] == FLEX_VER
    assert d["overrides"]["s390x"]["ibm-flex"]["version"] == FLEX_VER
    # and the attribution `verify` needs
    assert d["abi"]["flex"]["id12"] == "5cefcccce4cf"
    assert d["abi"]["flex"]["deps"]["deeptools"] == SHA["deeptools"]


def test_rendered_lock_round_trips_through_verify():
    """End-to-end: render a lock, then prove `verify` accepts it."""
    import tomllib

    lock = {
        "version": 2,
        "tree": "next",
        "orchestrator_build": 7457,
        "arches": ["x86_64", "s390x"],
        "components": {
            "deeptools": {
                "ref_kind": "commit",
                "ref": SHA["deeptools"],
                "source": SHA["deeptools"],
                "deps": {},
                "id12": {"x86_64": "b591b2c83606", "s390x": "b591b2c83606"},
                "packages": {"ibm-deeptools": {"x86_64": DT_VER, "s390x": DT_VER}},
            },
            "flex": {
                "ref_kind": "commit",
                "ref": SHA["flex"],
                "source": SHA["flex"],
                "deps": {"deeptools": SHA["deeptools"]},
                "id12": {"x86_64": "5cefcccce4cf", "s390x": "5cefcccce4cf"},
                "packages": {"ibm-flex": {"x86_64": FLEX_VER, "s390x": FLEX_VER}},
            },
        },
    }
    edges, bad, _ = sr.verify_lock(tomllib.loads(sr.render_lock(lock)))
    assert not bad
    assert len(edges) == 1
