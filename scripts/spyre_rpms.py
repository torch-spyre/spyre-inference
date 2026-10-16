#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.32.0",
#     "pydantic>=2.9.0",
# ]
# ///

"""Resolve git refs to an ABI-compatible set of Spyre RPMs.

WHY THIS EXISTS
---------------
`spyre-rpms.lock` pins full RPM versions and the old resolver picked, per
package, the newest Artifactory file matching a version glob. That is not
ABI-safe: each package is chosen independently, so deeptools from one build can
land beside flex from another.

That is not theoretical. On the `next` tree a single flex source commit
(`e0662966`) has THREE published RPMs, each a different binary:

    ibm-flex-...e0662966.15e5042af71a...   built against one deeptools/senlib
    ibm-flex-...e0662966.bd4f5392d761...   built against another
    ibm-flex-...e0662966.4db4325edbb9...   built against a third

The 12-hex field is the component's *identity* (`id12`), and the orchestrator
derives it from the dependency closure, not just the source:

    id12 = first12(sha256(version, source, recipe_sha256, recipe_files, dep_ids))

So a git commit does NOT uniquely identify a binary, and "newest matching file"
silently mixes incompatible builds. `id12` DOES identify a binary, because two
RPMs sharing an id12 were compiled against the same dependency shas.

HOW ABI COMPATIBILITY IS GUARANTEED
-----------------------------------
A Jenkins orchestrator run publishes a build plan that records, for every
component, its source sha and the exact dep shas it compiled against:

    <ARTIFACTORY_BASE_URL>/artifactory/<ARTIFACTORY_PLAN_PATH>/next/_nodes/orchestrator-<N>/plan-<arch>.json

This tool treats that plan as the ABI oracle. It resolves the git refs in the
input toml against a single plan, then enforces, for every dependency edge:

    consumer's recorded dep sha for <producer>  ==  <producer>'s own source sha

A set satisfying that for every edge was built as one coherent whole. The check
is re-run on the emitted lock by `verify`, needing no network, so CI can prove a
lock is ABI-coherent offline. `resolve` refuses to emit a lock that fails it.

USAGE
-----
    # resolve the input toml into a lock file
    ./scripts/spyre_rpms.py resolve -i spyre-rpms.toml -o spyre-rpms.lock

    # CI lint: regenerate and fail if the committed lock differs
    ./scripts/spyre_rpms.py check -i spyre-rpms.toml -o spyre-rpms.lock

    # offline ABI re-validation of a committed lock (no Artifactory needed)
    ./scripts/spyre_rpms.py verify -o spyre-rpms.lock

    # inspect what a plan contains
    ./scripts/spyre_rpms.py show-plan --orchestrator-build 7471 --arch x86_64

ENVIRONMENT
-----------
    ARTIFACTORY_TOKEN     required for resolve/check/show-plan (bearer token)
    ARTIFACTORY_BASE_URL  required for the same, e.g. https://artifactory.example.com
    ARTIFACTORY_RPM_PATH  Artifactory repo holding the RPMs
    ARTIFACTORY_PLAN_PATH Artifactory repo holding the orchestrator build plans
All four are read only from the environment, never from the toml or argv: the
host and the repo names are CI secrets in this project, not source.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: RPM arch names, as they appear in Artifactory paths and RPM filenames.
ARCHES = ("x86_64", "ppc64le", "s390x")

#: Resolved when neither --arches nor the input's `arches` names any. Most
#: orchestrator runs are amd64-only, so a 3-arch default would make the majority
#: of runs unusable as an ABI oracle.
DEFAULT_ARCHES = ("x86_64",)

#: Jenkins/orchestrator arch names -> RPM arch names. The plan files are named
#: plan-amd64.json while the RPMs live under x86_64/, so the two namespaces must
#: be translated explicitly; conflating them yields a confident 404.
ARCH_LABEL = {"x86_64": "x86_64", "ppc64le": "Power", "s390x": "Z"}

PLAN_ARCH = {"x86_64": "amd64", "ppc64le": "ppc64le", "s390x": "s390x"}

#: Artifactory repo names are deployment config, not source: like the base URL
#: they come from the environment (the repo already carries the RPM one as the
#: ARTIFACTORY_RPM_PATH secret). There is deliberately no default.

#: `next` RPM release field, e.g.
#: 0.main.93+93.e0662966.bd4f5392d761.e066296.el10 -> id12 bd4f5392d761
RE_ID12 = re.compile(r"\.([0-9a-f]{12})\.[0-9a-f]{7}\.el\d+$")

#: A full or abbreviated git sha. Refs that are not shas are resolved via the
#: plan's own checkout records, never by guessing.
RE_SHA = re.compile(r"^[0-9a-f]{7,40}$")

#: A dep edge may name an artifact rather than its producing component. Only
#: these and the `-devel`/`-headers` suffixes are the same source under another
#: name. Lookalikes with their OWN repo (torch-spyre-docs, spyre-perf-suite-src,
#: spyre_common_headers, pytorch) are resolved by checkout name in Plan.parse:
#: folding them onto a parent invents mismatches on healthy plans.
DEP_ALIASES = {
    "llvm-build": "llvm",
    "llvm-source": "llvm",
}

#: Subpackage suffixes stripped before alias lookup.
DEP_SUFFIXES = ("-devel", "-headers")


def canonical_dep(name: str, known: frozenset[str] = frozenset()) -> str:
    """Map a dep edge's name onto the plan component that produces it.

    A name that is already a component is returned untouched.
    """
    if name in known:
        return name
    if name in DEP_ALIASES:
        return DEP_ALIASES[name]
    for suf in DEP_SUFFIXES:
        if name.endswith(suf):
            base = name[: -len(suf)]
            if base in known or not known:
                return DEP_ALIASES.get(base, base)
    return name


#: The lock keeps the shipped `version = 2` schema; what this tool adds is the
#: `[abi]` attribution block, so coherence is keyed off that, not off a bump.
LOCK_FORMAT_VERSION = 2


class ResolveError(RuntimeError):
    """A resolution failed in a way the caller must not paper over."""


# ---------------------------------------------------------------------------
# input schema
# ---------------------------------------------------------------------------


class ComponentSpec(BaseModel):
    """One component's pin, as written in `spyre-rpms.toml`.

    Exactly one of `commit`/`tag`/`branch` is given: they are three spellings of
    "which source", and accepting more than one would leave the precedence rule
    implicit (and thus a silent source of wrong pins).
    """

    model_config = {"extra": "forbid"}

    commit: str | None = None
    tag: str | None = None
    branch: str | None = None
    packages: list[str] = Field(default_factory=list)

    @field_validator("commit")
    @classmethod
    def _sha_shaped(cls, v: str | None) -> str | None:
        if v is not None and not RE_SHA.match(v):
            raise ValueError(f"not a git sha: {v!r}")
        return v

    def ref(self) -> str:
        chosen = [x for x in (self.commit, self.tag, self.branch) if x]
        if len(chosen) != 1:
            raise ValueError("set exactly one of commit/tag/branch")
        return chosen[0]

    def ref_kind(self) -> str:
        if self.commit:
            return "commit"
        return "tag" if self.tag else "branch"


class InputDoc(BaseModel):
    """The whole `spyre-rpms.toml`."""

    model_config = {"extra": "forbid"}

    version: int = 1
    tree: str = "next"
    arches: list[str] = Field(default_factory=lambda: list(DEFAULT_ARCHES))
    orchestrator_build: int | None = None
    components: dict[str, ComponentSpec] = Field(default_factory=dict)

    @field_validator("arches")
    @classmethod
    def _known_arches(cls, v: list[str]) -> list[str]:
        bad = [a for a in v if a not in ARCHES]
        if bad:
            raise ValueError(f"unknown arch(es) {bad}; known: {list(ARCHES)}")
        return v

    @field_validator("tree")
    @classmethod
    def _known_tree(cls, v: str) -> str:
        if v not in ("next", "prod"):
            raise ValueError(f"tree must be 'next' or 'prod', got {v!r}")
        return v


# ---------------------------------------------------------------------------
# plan model
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    """One component in an orchestrator build plan.

    A plan holds two node shapes and BOTH are part of the ABI set:

    * built nodes carry `identity_inputs` (source + dep_ids);
    * dropped nodes (already published by an earlier run, so not rebuilt) carry
      no `identity_inputs` at all, and instead expose their source via
      `checkouts[].sha` and their edges via `dep_pins`.

    Reading only `identity_inputs` silently loses senlib, libaiupti,
    aiu-monitor, llvm and the wheels, so this reads the union of both shapes.
    """

    component: str
    kind: str
    id12: str
    source: str | None
    deps: dict[str, str]
    artifacts: list[dict[str, Any]]
    dropped: bool
    branch: str | None = None

    @classmethod
    def parse(
        cls, raw: dict[str, Any], *, dropped: bool, known: frozenset[str] = frozenset()
    ) -> PlanNode:
        ii = raw.get("identity_inputs") or {}
        source = ii.get("source")
        branch = None
        for co in raw.get("checkouts") or []:
            if co.get("identity") or co.get("recipe"):
                source = source or co.get("sha")
                branch = branch or co.get("branch")
        deps: dict[str, str] = {}
        for dep in ii.get("dep_ids") or []:
            comp, sep, sha = dep.partition("@")
            # Image nodes encode their base layer as `base=<identity>`, which is
            # an identity not a source sha, so it cannot be checked as an edge.
            # An image's base layer is named `base=<identity>` in dep_ids and
            # plain `base` in dep_pins. It is a layer, not a component, so it is
            # never a checkable source edge.
            if sep and RE_SHA.match(sha) and not comp.startswith("base"):
                deps.setdefault(canonical_dep(comp, known), sha)
        for comp, sha in (raw.get("dep_pins") or {}).items():
            if comp.startswith("base"):
                continue
            if isinstance(sha, str) and RE_SHA.match(sha):
                deps.setdefault(canonical_dep(comp, known), sha)
        return cls(
            component=raw["component"],
            kind=raw.get("kind", ""),
            id12=raw.get("id12", ""),
            source=source,
            deps=deps,
            artifacts=list(raw.get("artifacts") or []),
            dropped=dropped,
            branch=branch,
        )

    def rpm_names(self) -> list[str]:
        return sorted({a["name"] for a in self.artifacts if a.get("name")})


@dataclass
class Plan:
    build: int
    arch: str
    nodes: dict[str, PlanNode] = field(default_factory=dict)
    #: Source shas for checkout names that are not themselves plan components
    #: (e.g. torch-spyre-docs, pytorch); used only to check dep edges.
    extra_sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any], *, build: int, arch: str) -> Plan:
        if "nodes" not in raw:
            raise ResolveError(
                f"plan for orchestrator-{build} arch={arch} has no 'nodes' key; "
                f"got keys {sorted(raw)[:8]} (a 404 body parses as JSON too)"
            )
        allraw = dict(raw.get("nodes") or {})
        for key, rawnode in (raw.get("dropped_nodes") or {}).items():
            allraw.setdefault(key, rawnode)
        known = frozenset(rn["component"] for rn in allraw.values() if rn.get("component"))
        nodes: dict[str, PlanNode] = {}
        for key, rawnode in (raw.get("nodes") or {}).items():
            nodes[key] = PlanNode.parse(rawnode, dropped=False, known=known)
        for key, rawnode in (raw.get("dropped_nodes") or {}).items():
            nodes.setdefault(key, PlanNode.parse(rawnode, dropped=True, known=known))

        # A node's `checkouts[]` names extra sources it compiled in (docs, a
        # vendored pytorch tag, a sibling suite). Those names appear as dep
        # edges on OTHER nodes, so publishing them as sources makes those edges
        # checkable without pretending they belong to a parent component.
        extra: dict[str, str] = {}
        for rn in allraw.values():
            for co in rn.get("checkouts") or []:
                nm, sha = co.get("name"), co.get("sha")
                if nm and sha and nm not in known:
                    extra.setdefault(nm, sha)
        return cls(build=build, arch=arch, nodes=nodes, extra_sources=extra)

    def sources(self) -> dict[str, str]:
        """Source sha per dep-edge name: plan components plus checkout names."""
        out = dict(self.extra_sources)
        for comp, node in self.by_component().items():
            if node.source:
                out[comp] = node.source
        return out

    def by_component(self) -> dict[str, PlanNode]:
        """Index by component, preferring a built node over a dropped one."""
        out: dict[str, PlanNode] = {}
        for node in self.nodes.values():
            cur = out.get(node.component)
            if cur is None or (cur.dropped and not node.dropped):
                out[node.component] = node
        return out


# ---------------------------------------------------------------------------
# ABI invariant
# ---------------------------------------------------------------------------


@dataclass
class AbiEdge:
    consumer: str
    producer: str
    wants: str
    has: str

    @property
    def ok(self) -> bool:
        # Shas may be recorded at different lengths across fields, so compare on
        # the shorter prefix rather than demanding equal length.
        n = min(len(self.wants), len(self.has))
        return n >= 7 and self.wants[:n] == self.has[:n]


def abi_edges(sources: dict[str, str], deps: dict[str, dict[str, str]]) -> list[AbiEdge]:
    """Every checkable dependency edge.

    An edge is checkable only when the producer's own source sha is known. A dep
    on a component outside the set is reported separately rather than silently
    counted as a pass.
    """
    out: list[AbiEdge] = []
    for consumer, cdeps in sorted(deps.items()):
        for producer, wants in sorted(cdeps.items()):
            has = sources.get(producer)
            if has:
                out.append(AbiEdge(consumer, producer, wants, has))
    return out


def unresolved_deps(sources: dict[str, str], deps: dict[str, dict[str, str]]) -> list[str]:
    return sorted(
        {
            f"{consumer} -> {producer}"
            for consumer, cdeps in deps.items()
            for producer in cdeps
            if producer not in sources
        }
    )


# ---------------------------------------------------------------------------
# Artifactory client
# ---------------------------------------------------------------------------


class Artifactory:
    """Minimal Artifactory client for AQL and raw file reads."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        rpm_repo: str = "",
        plan_repo: str = "",
        timeout: int = 60,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.rpm_repo = rpm_repo
        self.plan_repo = plan_repo
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}"})

    def aql(self, query: str) -> list[dict[str, Any]]:
        r = self.s.post(
            f"{self.base}/artifactory/api/search/aql",
            data=query.encode(),
            headers={"Content-Type": "text/plain"},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise ResolveError(f"AQL HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        if body.get("errors"):
            # The common one is HTTP 400 "AQL items domain demands the following
            # fields: repo, path and name" when .include() omits them, which a
            # naive wrapper renders as "0 hits" (an absence that is really a
            # broken query).
            raise ResolveError(f"AQL error: {body['errors']}")
        return body.get("results", [])

    def get_json(self, repo_path: str) -> dict[str, Any]:
        url = f"{self.base}/artifactory/{repo_path.lstrip('/')}"
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise ResolveError(f"GET {url} -> HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


def fetch_plan(af: Artifactory, plan_repo: str, build: int, arch: str) -> Plan:
    plan_arch = PLAN_ARCH[arch]
    path = f"{plan_repo}/next/_nodes/orchestrator-{build}/plan-{plan_arch}.json"
    try:
        raw = af.get_json(path)
    except ResolveError as exc:
        raise ResolveError(
            f"cannot read the build plan for orchestrator-{build} arch={arch}\n"
            f"  tried: {path}\n"
            f"  {exc}\n"
            f"  note: an orchestrator run only publishes plans for the arches it "
            f"targeted, so a run scoped to e.g. 's390x ppc64le' has no "
            f"plan-amd64.json."
        ) from exc
    return Plan.parse(raw, build=build, arch=arch)


def latest_orchestrator_build(af: Artifactory, plan_repo: str, arches: Iterable[str]) -> int:
    """Newest orchestrator build publishing a plan for EVERY requested arch.

    Most runs are single-arch, so the newest run overall usually cannot satisfy a
    3-arch lock; this picks the newest that can.
    """
    wanted = {PLAN_ARCH[a] for a in arches}
    rows = af.aql(
        'items.find({"repo":"%s","path":{"$match":"next/_nodes/orchestrator-*"},'
        '"name":{"$match":"plan-*.json"}})'
        '.include("repo","path","name","created")'
        '.sort({"$desc":["created"]}).limit(500)' % plan_repo
    )
    seen: dict[int, set[str]] = {}
    for row in rows:
        m = re.search(r"orchestrator-(\d+)", row.get("path", ""))
        a = re.fullmatch(r"plan-(.+)\.json", row.get("name", ""))
        if m and a:
            seen.setdefault(int(m.group(1)), set()).add(a.group(1))
    for build in sorted(seen, reverse=True):
        if wanted <= seen[build]:
            return build
    raise ResolveError(
        f"no orchestrator run in the last {len(seen)} plan-publishing builds "
        f"covers all of {sorted(wanted)}; pass --orchestrator-build explicitly "
        f"or narrow `arches` in the input"
    )


# ---------------------------------------------------------------------------
# RPM lookup
# ---------------------------------------------------------------------------


@dataclass
class RpmFile:
    name: str
    path: str
    id12: str | None
    created: str


def find_rpms(
    af: Artifactory, rpm_repo: str, tree: str, arch: str, pkg: str, id12: str
) -> list[RpmFile]:
    """RPMs for one package at one identity, in the exact arch directory.

    The path is matched for EQUALITY, not with a `next/*` glob: every `next`
    artifact is mirrored under `next/cache/<arch>`, so a glob returns each file
    twice and an ambiguity check would fire on healthy data.
    """
    path = f"next/{arch}" if tree == "next" else arch
    rows = af.aql(
        'items.find({"repo":"%s","path":"%s","name":{"$match":"%s-*.%s.*.rpm"}})'
        '.include("repo","path","name","created").limit(50)' % (rpm_repo, path, pkg, id12)
    )
    out = []
    for r in rows:
        name = r["name"]
        # `ibm-deeptools-*` also matches `ibm-deeptools-devel-*`, which is a
        # DIFFERENT package, not a second build of this one. Keep only names
        # whose version field starts right after `<pkg>-`, so a subpackage does
        # not read as an ambiguous identity.
        rest = name[len(pkg) + 1 :] if name.startswith(f"{pkg}-") else None
        if rest is None or not rest[:1].isdigit():
            continue
        stem = name.removesuffix(f".{arch}.rpm")
        m = RE_ID12.search(stem)
        out.append(
            RpmFile(
                name=name,
                path=f"{r['path']}/{name}",
                id12=m.group(1) if m else None,
                created=r.get("created", ""),
            )
        )
    return sorted(out, key=lambda f: f.name)


def _version_release(filename: str, pkg: str, arch: str) -> str | None:
    """`ibm-flex-2.0.0_rc.1-0.main.87+....el10.x86_64.rpm` -> the version-release.

    That middle field is exactly what `[packages]` pins, so the lock keeps the
    same shape it has always had.
    """
    stem = filename.removesuffix(f".{arch}.rpm").removesuffix(".el10")
    if not stem.startswith(f"{pkg}-"):
        return None
    return stem[len(pkg) + 1 :] or None


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def _match_ref(node: PlanNode, spec: ComponentSpec) -> bool:
    ref = spec.ref()
    if spec.commit:
        n = min(len(ref), len(node.source or ""))
        return bool(node.source) and n >= 7 and node.source[:n] == ref[:n]
    if spec.branch:
        return node.branch == spec.branch
    return False  # a tag cannot be confirmed from the plan; see resolve()


def resolve(
    doc: InputDoc,
    af: Artifactory,
    *,
    build: int | None,
    arches: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve the input doc into a lock document, or raise ResolveError.

    `arches` overrides the input's own list, so one toml can be resolved for a
    single arch locally and for several in CI.
    """
    arches = list(arches or doc.arches)
    if not arches:
        raise ResolveError("no arches to resolve: pass --arches or set `arches` in the input")
    build = build or doc.orchestrator_build
    if build is None:
        build = latest_orchestrator_build(af, af.plan_repo, arches)
        print(f"selected orchestrator-{build} (newest covering {arches})", file=sys.stderr)

    plans = {a: fetch_plan(af, af.plan_repo, build, a) for a in arches}

    # --- confirm every requested component is in the plan, at the asked ref ---
    locked: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for comp, spec in sorted(doc.components.items()):
        try:
            spec.ref()
        except ValueError as exc:
            problems.append(f"{comp}: {exc}")
            continue
        if spec.tag:
            problems.append(
                f"{comp}: pinned by tag {spec.tag!r}, but a build plan records only "
                f"commit shas and branches, so a tag cannot be confirmed against it. "
                f"Resolve the tag to its commit sha and pin `commit` instead."
            )
            continue

        per_arch: dict[str, PlanNode] = {}
        for arch in arches:
            node = plans[arch].by_component().get(comp)
            if node is None:
                problems.append(
                    f"{comp}: not present in plan orchestrator-{build} arch={arch} "
                    f"(it was neither built nor reused by that run)"
                )
                continue
            if not _match_ref(node, spec):
                problems.append(
                    f"{comp}: plan orchestrator-{build} arch={arch} has "
                    f"{spec.ref_kind()}="
                    f"{(node.source or node.branch or '?')[:40]!r}, "
                    f"input asks for {spec.ref()[:40]!r}"
                )
                continue
            per_arch[arch] = node
        if len(per_arch) == len(arches):
            locked[comp] = {"nodes": per_arch}

    if problems:
        raise ResolveError(
            "cannot resolve the requested pins against a single orchestrator "
            "build, so no ABI-coherent set exists for them:\n  - " + "\n  - ".join(problems)
        )

    # --- ABI invariant, on the plan, before any RPM is named ---
    for arch in arches:
        pc = plans[arch].by_component()
        sources = plans[arch].sources()
        deps = {c: n.deps for c, n in pc.items() if n.deps}
        bad = [e for e in abi_edges(sources, deps) if not e.ok]
        if bad:
            raise ResolveError(
                f"plan orchestrator-{build} arch={arch} is itself ABI-incoherent "
                f"({len(bad)} bad edge(s)); refusing to emit a lock from it:\n  - "
                + "\n  - ".join(
                    f"{e.consumer} was built against {e.producer}@{e.wants[:12]} "
                    f"but the plan's {e.producer} is {e.has[:12]}"
                    for e in bad[:10]
                )
            )

    # --- name the actual RPM files ---
    out_components: dict[str, Any] = {}
    missing: list[str] = []
    for comp, info in sorted(locked.items()):
        nodes: dict[str, PlanNode] = info["nodes"]
        ref_node = nodes[arches[0]]
        pkgs = sorted(set(doc.components[comp].packages) or set(ref_node.rpm_names()))
        if not pkgs:
            # A node with no RPM artifacts (a wheel, an image, a generic) is a
            # legitimate plan member but contributes nothing to an RPM lock.
            continue
        entry: dict[str, Any] = {
            "ref_kind": doc.components[comp].ref_kind(),
            "ref": doc.components[comp].ref(),
            "source": ref_node.source or "",
            "deps": dict(sorted(ref_node.deps.items())),
            "id12": {a: nodes[a].id12 for a in arches},
            "packages": {},
        }
        for pkg in pkgs:
            per_arch_files: dict[str, str] = {}
            for arch in arches:
                files = find_rpms(af, af.rpm_repo, doc.tree, arch, pkg, nodes[arch].id12)
                if not files:
                    missing.append(
                        f"{pkg} @ id12={nodes[arch].id12} arch={arch} "
                        f"(component {comp}, tree={doc.tree})"
                    )
                    continue
                if len({f.name for f in files}) > 1:
                    raise ResolveError(
                        f"{pkg} @ id12={nodes[arch].id12} arch={arch} matched "
                        f"{len(files)} distinct files; an identity must name "
                        f"exactly one build:\n  - " + "\n  - ".join(f.name for f in files[:6])
                    )
                ver = _version_release(files[0].name, pkg, arch)
                if ver is None:
                    raise ResolveError(
                        f"cannot read a version out of {files[0].name!r} for "
                        f"{pkg} on {arch}; the RPM naming may have changed"
                    )
                per_arch_files[arch] = ver
            entry["packages"][pkg] = per_arch_files
        out_components[comp] = entry

    if missing:
        raise ResolveError(
            "these RPMs are not published at the identity the plan recorded, so "
            "the set is incomplete. This is routine while an orchestrator run is "
            "still building (a partially published run is a normal state): wait "
            "for it to finish, or pin an older completed run with "
            "--orchestrator-build. Refusing to substitute a different build, "
            "because that is exactly how an ABI-incompatible set gets assembled:"
            "\n  - " + "\n  - ".join(missing)
        )

    return {
        "version": LOCK_FORMAT_VERSION,
        "generated_by": "scripts/spyre_rpms.py",
        "tree": doc.tree,
        "orchestrator_build": build,
        "arches": arches,
        "components": out_components,
    }


# ---------------------------------------------------------------------------
# lock rendering / verification
# ---------------------------------------------------------------------------


def render_lock(lock: dict[str, Any]) -> str:
    """Render the existing `spyre-rpms.lock` v2 schema.

    The format is deliberately unchanged: 28 files read this lock (notably
    `.github/scripts/resolve_rpms.py`, `scripts/install-pinned-rpms.sh` and the
    `spyre-rpm-install` action), so this stays a drop-in replacement and only the
    way the file is PRODUCED changes.

    What does change is that the per-arch wildcards are now exact. Previously
    ppc64le/s390x pinned `<ver>.<sha8>.*`, leaving the identity unpinned because
    "the builds differ across platforms". They do not: the identity is derived
    from the source and dependency shas, not the platform, so one component has
    the same `id12` on all three arches and every arch can pin it exactly.

    Rendered by hand rather than via a TOML dumper so the bytes stay stable
    across library versions: `check` compares bytes.
    """
    comps = lock["components"]
    arches = lock["arches"]
    primary = arches[0]

    # name -> {arch: version}, where version is the RPM's full version-release.
    pkgs: dict[str, dict[str, str]] = {}
    for entry in comps.values():
        for name, per_arch in entry["packages"].items():
            pkgs.setdefault(name, {}).update(per_arch)

    width = max((len(n) for n in pkgs), default=0)
    L: list[str] = []
    L.append("# Pinned Spyre RPM baseline for CI.")
    L.append("#")
    L.append("# GENERATED FILE -- do not edit by hand. Regenerate with:")
    L.append("#   ./scripts/spyre_rpms.py resolve")
    L.append("# and verify an existing one offline (no Artifactory) with:")
    L.append("#   ./scripts/spyre_rpms.py verify")
    L.append("#")
    L.append(
        f"# Every package below comes from ONE build plan (orchestrator-"
        f"{lock['orchestrator_build']}), and each component's recorded dependency"
    )
    L.append("# shas were checked against the other components in the set, so the")
    L.append("# whole set is ABI-coherent by construction.")
    L.append("#")
    L.append("# All three arches pin the exact build: the identity (the 12-hex field)")
    L.append("# is derived from the source and dependency shas, not the platform, so it")
    L.append("# is the same on x86_64, ppc64le and s390x.")
    L.append("")
    L.append("version = 2")
    L.append("")
    L.append("[defaults]")
    L.append('# Artifactory subdir (<repo>/<tree>/<arch>/); "" is the prod/base tree.')
    L.append(f'tree = "{lock["tree"]}"')
    L.append("")
    L.append(f"# {primary}")
    L.append("[packages]")
    for name in sorted(pkgs):
        v = pkgs[name].get(primary)
        if v:
            L.append(f'{name.ljust(width)} = {{ version = "{v}" }}')
    for arch in arches[1:]:
        L.append("")
        L.append(f"# {ARCH_LABEL.get(arch, arch)}")
        L.append(f"[overrides.{arch}]")
        for name in sorted(pkgs):
            v = pkgs[name].get(arch)
            if v:
                L.append(f'{name.ljust(width)} = {{ version = "{v}" }}')
    L.append("")
    L.append("# The resolved set, for `verify` to re-check without Artifactory.")
    L.append("# `source` is the component's git commit; `deps` are the dependency")
    L.append("# shas it was actually built against.")
    for comp, e in sorted(comps.items()):
        L.append("")
        L.append(f"[abi.{comp}]")
        L.append(f'source = "{e["source"]}"')
        L.append(f'id12 = "{e["id12"][primary]}"')
        L.append("packages = [" + ", ".join(f'"{n}"' for n in sorted(e["packages"])) + "]")
        if e["deps"]:
            L.append(f"[abi.{comp}.deps]")
            for d, sha in sorted(e["deps"].items()):
                L.append(f'{d} = "{sha}"')
    return "\n".join(L).rstrip() + "\n"


def verify_lock(lock: dict[str, Any]) -> tuple[list[AbiEdge], list[str], list[str]]:
    """Re-check a lock's ABI invariant using only the file. Returns (edges, bad, warnings).

    Needs no Artifactory access, so CI can run it on a pull request from a fork
    where secrets are withheld.
    """
    abi = lock.get("abi") or {}
    pkgs = lock.get("packages") or {}
    overrides = lock.get("overrides") or {}
    if not abi:
        return (
            [],
            [
                "this lock carries no [abi] section, so its ABI coherence cannot be "
                "checked. Regenerate it with `./scripts/spyre_rpms.py resolve`."
            ],
            [],
        )

    sources = {c: e["source"] for c, e in abi.items() if e.get("source")}
    deps = {c: (e.get("deps") or {}) for c, e in abi.items()}
    edges = abi_edges(sources, deps)
    bad = [
        f"{e.consumer} was built against {e.producer}@{e.wants[:12]} but the "
        f"lock's {e.producer} is {e.has[:12]}"
        for e in edges
        if not e.ok
    ]
    warnings = [
        f"{edge}: dependency is outside the lock, so its ABI match is unchecked"
        for edge in unresolved_deps(sources, deps)
    ]

    # Every [abi] package must be pinned, and every pin must carry that
    # component's identity on every arch: that is what ties the [packages]
    # table back to the coherent set.
    for comp, e in sorted(abi.items()):
        id12 = e.get("id12", "")
        for name in e.get("packages", []):
            if name not in pkgs:
                bad.append(f"{comp}/{name} is in [abi] but missing from [packages]")
                continue
            seen = {"primary": pkgs[name].get("version", "")}
            for arch, table in overrides.items():
                if name in table:
                    seen[arch] = table[name].get("version", "")
            for where, ver in seen.items():
                if id12 and id12 not in ver:
                    bad.append(
                        f"{comp}/{name} [{where}] pins {ver!r}, which does not "
                        f"carry the locked identity {id12}"
                    )

    # A package pinned but absent from [abi] is unattributed: nothing proves it
    # came from the same build as the rest.
    claimed = {n for e in abi.values() for n in e.get("packages", [])}
    for name in sorted(set(pkgs) - claimed):
        bad.append(f"{name} is pinned in [packages] but not covered by any [abi] entry")

    return edges, bad, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _client(args: argparse.Namespace) -> Artifactory:
    token = os.environ.get("ARTIFACTORY_TOKEN")
    if not token:
        raise ResolveError(
            "ARTIFACTORY_TOKEN is not set. Secrets are read only from the "
            "environment; pass it as a CI secret, never in the toml or on argv."
        )
    base = args.base_url or os.environ.get("ARTIFACTORY_BASE_URL")
    if not base:
        raise ResolveError(
            "set ARTIFACTORY_BASE_URL (or pass --base-url). There is no default: "
            "the host belongs in a CI secret, not in this repo."
        )
    rpm_repo = os.environ.get("ARTIFACTORY_RPM_PATH") or os.environ.get("ARTIFACTORY_RPM_REPO")
    plan_repo = os.environ.get("ARTIFACTORY_PLAN_PATH") or os.environ.get("ARTIFACTORY_PLAN_REPO")
    missing = [
        n
        for n, v in (("ARTIFACTORY_RPM_PATH", rpm_repo), ("ARTIFACTORY_PLAN_PATH", plan_repo))
        if not v
    ]
    if missing:
        raise ResolveError(
            f"set {' and '.join(missing)}. The Artifactory repo names are CI "
            f"secrets in this project, so this script carries no defaults for them."
        )
    return Artifactory(base, token, rpm_repo=rpm_repo, plan_repo=plan_repo)


def load_input(path: str) -> InputDoc:
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ResolveError(f"input file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ResolveError(f"{path} is not valid TOML: {exc}") from None
    try:
        doc = InputDoc(**raw)
    except ValidationError as exc:
        raise ResolveError(f"{path} does not match the expected schema:\n{exc}") from None
    if not doc.components:
        raise ResolveError(f"{path} declares no [components]; nothing to resolve")
    return doc


def load_lock(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ResolveError(f"lock file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ResolveError(f"{path} is not valid TOML: {exc}") from None


def cmd_resolve(args: argparse.Namespace) -> int:
    doc = load_input(args.input)
    lock = resolve(doc, _client(args), build=args.orchestrator_build, arches=args.arches)
    text = render_lock(lock)
    # Verify what will actually be WRITTEN, parsed back as the consumers parse
    # it, not the internal dict: that is the only check that proves the emitted
    # file is coherent.
    _, bad, warnings = verify_lock(tomllib.loads(text))
    if bad:
        raise ResolveError(
            "refusing to write a lock that fails its own ABI check (this is a "
            "bug in this script, not in your input):\n  - " + "\n  - ".join(bad)
        )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        n = sum(len(e["packages"]) for e in lock["components"].values())
        print(
            f"wrote {args.output}: {len(lock['components'])} components, {n} packages, "
            f"{len(lock['arches'])} arches, from orchestrator-{lock['orchestrator_build']}",
            file=sys.stderr,
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    doc = load_input(args.input)
    try:
        with open(args.output, encoding="utf-8") as fh:
            committed = fh.read()
    except FileNotFoundError:
        raise ResolveError(
            f"{args.output} does not exist; run `resolve` and commit the result"
        ) from None
    lock = resolve(doc, _client(args), build=args.orchestrator_build, arches=args.arches)
    fresh = render_lock(lock)
    if fresh == committed:
        print(f"OK: {args.output} matches {args.input}")
        return 0
    diff = difflib.unified_diff(
        committed.splitlines(keepends=True),
        fresh.splitlines(keepends=True),
        fromfile=f"{args.output} (committed)",
        tofile=f"{args.output} (regenerated)",
    )
    sys.stdout.writelines(diff)
    print(
        f"\n::error::{args.output} is stale: regenerate with "
        f"`./scripts/spyre_rpms.py resolve -i {args.input} -o {args.output}` "
        f"and commit the result",
        file=sys.stderr,
    )
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    lock = load_lock(args.output)
    edges, bad, warnings = verify_lock(lock)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if bad:
        print(f"::error::{args.output} is NOT ABI-coherent:", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print(
        f"OK: {args.output} is ABI-coherent "
        f"({len(edges)} dependency edge(s) checked, "
        f"{len(lock.get('abi') or {})} components, "
        f"{len(lock.get('packages') or {})} packages)"
    )
    return 0


def cmd_show_plan(args: argparse.Namespace) -> int:
    af = _client(args)
    build = args.orchestrator_build or latest_orchestrator_build(af, af.plan_repo, [args.arch])
    plan = fetch_plan(af, af.plan_repo, build, args.arch)
    pc = plan.by_component()
    print(f"orchestrator-{build} arch={args.arch}: {len(pc)} components")
    for comp, n in sorted(pc.items()):
        tag = "dropped" if n.dropped else "built"
        src = (n.source or "-")[:12]
        print(f"  {comp:20s} {n.kind:8s} {tag:8s} id12={n.id12 or '-':12s} src={src}")
        for d, sha in sorted(n.deps.items()):
            print(f"        dep {d}@{sha[:12]}")
    sources = plan.sources()
    deps = {c: n.deps for c, n in pc.items() if n.deps}
    edges = abi_edges(sources, deps)
    bad = [e for e in edges if not e.ok]
    print(
        f"\nABI edges: {len(edges)} checked, {len(bad)} mismatched, "
        f"{len(unresolved_deps(sources, deps))} with a producer outside the plan"
    )
    for e in bad:
        print(f"  MISMATCH {e.consumer} wants {e.producer}@{e.wants[:12]}, has {e.has[:12]}")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spyre_rpms.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Resolve git refs to an ABI-compatible set of Spyre RPMs.\n\n"
            "A git commit does NOT uniquely identify an RPM: the same source, "
            "rebuilt against different dependency versions, publishes a "
            "different binary. The orchestrator's `id12` identity does identify "
            "one, because it hashes the dependency shas as well as the source. "
            "This tool resolves refs against a single orchestrator build plan "
            "(the only record of what each binary was actually compiled "
            "against) and refuses to emit a lock whose dependency edges do not "
            "agree."
        ),
        epilog=(
            "environment:\n"
            "  ARTIFACTORY_TOKEN     required for resolve/check/show-plan\n"
            "  ARTIFACTORY_BASE_URL  required for the same (no default)\n\n"
            "examples:\n"
            "  # resolve the input into a lock file\n"
            "  ./scripts/spyre_rpms.py resolve -i spyre-rpms.toml -o spyre-rpms.lock\n\n"
            "  # pin a specific orchestrator run instead of the newest usable one\n"
            "  ./scripts/spyre_rpms.py resolve -i spyre-rpms.toml -o spyre-rpms.lock \\\n"
            "      --orchestrator-build 7471\n\n"
            "  # CI lint: fail if the committed lock is stale\n"
            "  ./scripts/spyre_rpms.py check -i spyre-rpms.toml -o spyre-rpms.lock\n\n"
            "  # offline ABI re-check of a committed lock (no network)\n"
            "  ./scripts/spyre_rpms.py verify -o spyre-rpms.lock\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--base-url",
            default=None,
            help="Artifactory base URL; defaults to $ARTIFACTORY_BASE_URL",
        )
        sp.add_argument(
            "-b",
            "--orchestrator-build",
            type=int,
            default=None,
            help="pin this Jenkins orchestrator build as the ABI oracle; "
            "default is the newest run that published a plan for every "
            "requested arch",
        )

    def arches_flag(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-a",
            "--arches",
            metavar="ARCH",
            nargs="+",
            choices=ARCHES,
            default=None,
            help="arches to resolve (choices: %(choices)s); overrides `arches` "
            "in the input toml. Default: the input's `arches`, else "
            f"{' '.join(DEFAULT_ARCHES)}. Every named arch must resolve from one "
            "orchestrator run, and most runs are amd64-only, so naming all "
            "three narrows the usable runs sharply.",
        )

    sp = sub.add_parser(
        "resolve",
        help="resolve the input toml into a lock file",
        description="Resolve git refs to exact, ABI-coherent RPM paths.",
    )
    sp.add_argument(
        "-i", "--input", default="spyre-rpms.toml", help="input toml (default: %(default)s)"
    )
    sp.add_argument(
        "-o",
        "--output",
        default="spyre-rpms.lock",
        help="lock file to write, or '-' for stdout (default: %(default)s)",
    )
    arches_flag(sp)
    common(sp)
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser(
        "check",
        help="fail if the committed lock differs from a fresh resolve (for CI)",
        description=(
            "Regenerate the lock from the input and compare it to the committed "
            "file, printing a unified diff and exiting 1 if they differ. Use as "
            "a PR lint step."
        ),
    )
    sp.add_argument(
        "-i", "--input", default="spyre-rpms.toml", help="input toml (default: %(default)s)"
    )
    sp.add_argument(
        "-o",
        "--output",
        default="spyre-rpms.lock",
        help="committed lock (default: %(default)s)",
    )
    arches_flag(sp)
    common(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser(
        "verify",
        help="re-check a lock's ABI invariant offline (no network)",
        description=(
            "Re-run the ABI check on a committed lock using only the data in the "
            "file. Needs no Artifactory token, so it is safe in an untrusted CI "
            "context such as pull_request from a fork."
        ),
    )
    sp.add_argument(
        "-o",
        "--output",
        default="spyre-rpms.lock",
        help="lock to verify (default: %(default)s)",
    )
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "show-plan",
        help="print an orchestrator build plan and its ABI edges",
        description="Inspect what a plan records; useful when a resolve fails.",
    )
    sp.add_argument("--arch", default="x86_64", choices=ARCHES, help="(default: %(default)s)")
    common(sp)
    sp.set_defaults(func=cmd_show_plan)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ResolveError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    except requests.RequestException as exc:
        print(f"::error::Artifactory request failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
