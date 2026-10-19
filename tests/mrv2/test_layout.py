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

"""Nothing outside spyre/ imports it (except the opt-in switch's two
env-gated files, added by #1237), and no component imports model_runner.py.
"""

import ast
import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "spyre_inference" / "v1" / "worker" / "spyre"
_SPYRE_INFERENCE_ROOT = _REPO_ROOT / "spyre_inference"

# Nothing is allowed to import the package yet. #1237 adds env-gated branches
# to spyre_inference/__init__.py and platform.py for the opt-in switch; it
# should add those two paths here then, not before the gated imports exist.
_ALLOWED_IMPORTERS: set[pathlib.Path] = set()

_PACKAGE_MODULE = "spyre_inference.v1.worker.spyre"
_PACKAGE_MODULE_PREFIX = _PACKAGE_MODULE + "."


def _module_files() -> list[pathlib.Path]:
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def _non_package_files() -> list[pathlib.Path]:
    return sorted(p for p in _SPYRE_INFERENCE_ROOT.rglob("*.py") if _PACKAGE_ROOT not in p.parents)


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
    return name == "TYPE_CHECKING"


def _runtime_nodes(tree: ast.Module) -> list[ast.AST]:
    """Every node, except those inside an `if TYPE_CHECKING:` body — those
    imports have no runtime effect, so they don't create a real dependency.
    """
    nodes: list[ast.AST] = []

    class _Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            if _is_type_checking_guard(node):
                for child in node.orelse:
                    self.visit(child)
                return
            self.generic_visit(node)

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    _Visitor().visit(tree)
    return nodes


def _containing_package(path: pathlib.Path) -> str:
    """Dotted name of the package a file's directory would be imported as
    (same value for a plain module and for that directory's __init__.py)."""
    return ".".join(path.parent.relative_to(_REPO_ROOT).parts)


def _resolve_relative_import(containing_package: str, level: int, module: str | None) -> str:
    """Absolute name a `from .foo import X` (or `from .. import foo`) would
    resolve to from inside containing_package."""
    return importlib.util.resolve_name("." * level + (module or ""), containing_package)


def _module_name(
    containing_package: str, node: ast.Import | ast.ImportFrom, alias: ast.alias
) -> str:
    if isinstance(node, ast.Import):
        return alias.name
    if node.level:
        return _resolve_relative_import(containing_package, node.level, node.module)
    return node.module or ""


def _is_spyre_package_module(module_name: str) -> bool:
    return module_name == _PACKAGE_MODULE or module_name.startswith(_PACKAGE_MODULE_PREFIX)


def _imports_spyre_package(tree: ast.Module, containing_package: str) -> bool:
    for node in _runtime_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            _is_spyre_package_module(_module_name(containing_package, node, a)) for a in node.names
        ):
            return True
    return False


def _imports_model_runner(tree: ast.Module, containing_package: str) -> bool:
    target = f"{_PACKAGE_MODULE}.model_runner"
    for node in _runtime_nodes(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == target for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            from_module = (
                _resolve_relative_import(containing_package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            if from_module == target:
                return True
            # `from spyre_inference.v1.worker.spyre import model_runner` (or the
            # relative equivalent): the "from" names the package itself, and
            # the submodule is one of the imported names.
            if from_module == _PACKAGE_MODULE and any(a.name == "model_runner" for a in node.names):
                return True
    return False


@pytest.mark.parametrize(
    "path",
    _non_package_files(),
    # Not just cosmetic: the root tests/conftest.py skips any item whose id
    # contains "batched_decode" (unrelated to this package), which would
    # otherwise silently skip, rather than run, this check on
    # batched_decode.py / batched_decode_head_major.py.
    ids=lambda p: str(p.relative_to(_REPO_ROOT)).replace("_", "-"),
)
def test_nothing_outside_the_package_imports_it(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    if _imports_spyre_package(tree, _containing_package(path)):
        assert path in _ALLOWED_IMPORTERS, (
            f"{path.relative_to(_REPO_ROOT)} imports spyre_inference.v1.worker.spyre, "
            "but nothing outside the package may import it until #1237 adds the "
            "opt-in switch's env-gated imports."
        )


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.relative_to(_PACKAGE_ROOT)))
def test_components_never_import_model_runner(path: pathlib.Path) -> None:
    if path.name in {"model_runner.py", "worker.py"}:
        pytest.skip("the coordinator and the worker that constructs it are allowed to")
    tree = ast.parse(path.read_text(), filename=str(path))
    assert not _imports_model_runner(tree, _containing_package(path)), (
        f"{path.relative_to(_PACKAGE_ROOT)} imports model_runner.py; components only "
        "get called BY it, never the other way around."
    )
